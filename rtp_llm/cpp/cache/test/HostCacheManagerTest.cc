#include <gtest/gtest.h>
#include <memory>

#include "rtp_llm/cpp/cache/HostCacheManager.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm {
namespace test {

class HostCacheManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        device_ = createDevice();

        cache_config_ = makeSimpleMhaCacheConfig(
            /*layer_num=*/4,
            /*block_num=*/10,
            /*tokens_per_block=*/4,
            /*dtype=*/rtp_llm::DataType::TYPE_FP16,
            /*local_head_num_kv=*/1,
            /*size_per_head=*/8);

        auto pool_config = BlockPoolConfigHelper::createConfig(cache_config_);
        gpu_pool_        = std::make_shared<BlockPool>(pool_config, device_);
        ASSERT_TRUE(gpu_pool_->init());

        radix_tree_ = gpu_pool_->radixTree();
    }

    // Allocate a GPU block from the pool, insert it into the radix tree with
    // proper reference counting, and release the request reference so it
    // becomes evictable.
    BlockIdxType allocateAndCacheBlock(CacheKeyType cache_key, GroupIdType group_id = 0) {
        auto blocks = gpu_pool_->malloc(1);
        RTP_LLM_CHECK(!blocks.empty());
        BlockIdxType          blk = blocks[0];
        BlockCache::CacheItem item{cache_key, group_id, blk, false};
        radix_tree_->put(item);
        gpu_pool_->blockCacheReference(blk);
        gpu_pool_->requestFree(blk);
        return blk;
    }

    DeviceBase*  device_;
    CacheConfig  cache_config_;
    BlockPoolPtr gpu_pool_;
    RadixTreePtr radix_tree_;
};

TEST_F(HostCacheManagerTest, InitDisabledWithZeroSize) {
    auto hcm = std::make_shared<HostCacheManager>(gpu_pool_, radix_tree_, device_);
    ASSERT_TRUE(hcm->init(0, cache_config_));
    EXPECT_FALSE(hcm->isEnabled());
    EXPECT_EQ(hcm->hostFreeBlocksNum(), 0u);
    EXPECT_EQ(hcm->hostTotalBlocksNum(), 0u);
}

TEST_F(HostCacheManagerTest, InitEnabled) {
    auto hcm = std::make_shared<HostCacheManager>(gpu_pool_, radix_tree_, device_);
    ASSERT_TRUE(hcm->init(16, cache_config_));
    EXPECT_TRUE(hcm->isEnabled());
    EXPECT_GT(hcm->hostTotalBlocksNum(), 0u);
    EXPECT_EQ(hcm->hostFreeBlocksNum(), hcm->hostTotalBlocksNum());
}

TEST_F(HostCacheManagerTest, OffloadBlockTransfersToHost) {
    auto hcm = std::make_shared<HostCacheManager>(gpu_pool_, radix_tree_, device_);
    ASSERT_TRUE(hcm->init(16, cache_config_));

    size_t initial_host_free = hcm->hostFreeBlocksNum();

    allocateAndCacheBlock(100);

    auto evict_result = radix_tree_->evictGPU(1, true);
    ASSERT_EQ(evict_result.offloadable_nodes.size(), 1u);
    auto* node = evict_result.offloadable_nodes[0];
    ASSERT_TRUE(node->isOnGPU());

    ASSERT_TRUE(hcm->offloadBlock(node));

    // After offload, the GPU block data is copied to host and node is marked offloaded
    EXPECT_FALSE(node->isOnGPU());
    EXPECT_TRUE(node->isOnHost());
    EXPECT_EQ(hcm->hostFreeBlocksNum(), initial_host_free - 1);

    // Free the GPU block that was returned by eviction
    gpu_pool_->blockCacheFree(evict_result.offloadable_gpu_blocks);
}

TEST_F(HostCacheManagerTest, OnboardBlockTransfersToGPU) {
    auto hcm = std::make_shared<HostCacheManager>(gpu_pool_, radix_tree_, device_);
    ASSERT_TRUE(hcm->init(16, cache_config_));

    allocateAndCacheBlock(200);

    auto evict_result = radix_tree_->evictGPU(1, true);
    ASSERT_EQ(evict_result.offloadable_nodes.size(), 1u);
    auto* node = evict_result.offloadable_nodes[0];

    // Offload: copy GPU→host, mark node
    ASSERT_TRUE(hcm->offloadBlock(node));
    // Free the GPU block
    gpu_pool_->blockCacheFree(evict_result.offloadable_gpu_blocks);
    ASSERT_TRUE(node->isOnHost());
    ASSERT_FALSE(node->isOnGPU());

    // Onboard: copy host→GPU, mark node
    BlockIdxType gpu_idx = hcm->onboardBlock(node);
    EXPECT_NE(gpu_idx, NULL_BLOCK_IDX);
    EXPECT_TRUE(node->isOnGPU());
    EXPECT_FALSE(node->isOnHost());
}

TEST_F(HostCacheManagerTest, OffloadDisabledReturnsFalse) {
    auto hcm = std::make_shared<HostCacheManager>(gpu_pool_, radix_tree_, device_);
    ASSERT_TRUE(hcm->init(0, cache_config_));

    allocateAndCacheBlock(300);

    auto evict_result = radix_tree_->evictGPU(1, true);
    ASSERT_EQ(evict_result.offloadable_nodes.size(), 1u);
    auto* node = evict_result.offloadable_nodes[0];

    EXPECT_FALSE(hcm->offloadBlock(node));
    EXPECT_TRUE(node->isOnGPU());
}

TEST_F(HostCacheManagerTest, OnboardDisabledReturnsNull) {
    auto hcm = std::make_shared<HostCacheManager>(gpu_pool_, radix_tree_, device_);
    ASSERT_TRUE(hcm->init(0, cache_config_));

    RadixTreeNode dummy_node;
    dummy_node.host_block_idx = 1;
    EXPECT_EQ(hcm->onboardBlock(&dummy_node), NULL_BLOCK_IDX);
}

TEST_F(HostCacheManagerTest, EnsureHostFreeBlocksEvictsFromHost) {
    auto hcm = std::make_shared<HostCacheManager>(gpu_pool_, radix_tree_, device_);
    ASSERT_TRUE(hcm->init(1, cache_config_));

    if (hcm->hostTotalBlocksNum() < 2) {
        GTEST_SKIP() << "Host pool too small for this test";
    }

    allocateAndCacheBlock(400);
    allocateAndCacheBlock(401);

    // Offload two blocks to host
    for (int i = 0; i < 2; ++i) {
        auto evict_result = radix_tree_->evictGPU(1, true);
        if (!evict_result.offloadable_nodes.empty()) {
            auto* node = evict_result.offloadable_nodes[0];
            hcm->offloadBlock(node);
            gpu_pool_->blockCacheFree(evict_result.offloadable_gpu_blocks);
        }
    }

    // Ensure there's space on host by evicting host-resident nodes
    hcm->ensureHostFreeBlocks(1);
    EXPECT_GE(hcm->hostFreeBlocksNum(), 1u);
}

TEST_F(HostCacheManagerTest, OffloadNullNodeReturnsFalse) {
    auto hcm = std::make_shared<HostCacheManager>(gpu_pool_, radix_tree_, device_);
    ASSERT_TRUE(hcm->init(16, cache_config_));
    EXPECT_FALSE(hcm->offloadBlock(nullptr));
}

TEST_F(HostCacheManagerTest, OnboardNullNodeReturnsNull) {
    auto hcm = std::make_shared<HostCacheManager>(gpu_pool_, radix_tree_, device_);
    ASSERT_TRUE(hcm->init(16, cache_config_));
    EXPECT_EQ(hcm->onboardBlock(nullptr), NULL_BLOCK_IDX);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
