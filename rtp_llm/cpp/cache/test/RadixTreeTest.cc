#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/RadixTree.h"

namespace rtp_llm {
namespace test {

class RadixTreeTest: public ::testing::Test {
protected:
    void SetUp() override {
        tree_ = std::make_shared<RadixTree>();
    }

    RadixTreePtr tree_;
};

TEST_F(RadixTreeTest, PutAndMatch) {
    BlockCache::CacheItem item{100, 0, 1, false};
    EXPECT_TRUE(tree_->put(item));
    EXPECT_EQ(tree_->size(), 1u);

    auto result = tree_->match(100, 0);
    EXPECT_EQ(result.matched_index, 1);

    auto miss = tree_->match(999, 0);
    EXPECT_TRUE(isNullBlockIdx(miss.matched_index));
}

TEST_F(RadixTreeTest, PutDuplicateReturnsFalse) {
    BlockCache::CacheItem item{100, 0, 1, false};
    EXPECT_TRUE(tree_->put(item));
    EXPECT_FALSE(tree_->put(item));
    EXPECT_EQ(tree_->size(), 1u);
}

TEST_F(RadixTreeTest, MultipleGroups) {
    BlockCache::CacheItem item1{100, 0, 1, false};
    BlockCache::CacheItem item2{100, 1, 2, false};
    EXPECT_TRUE(tree_->put(item1));
    EXPECT_TRUE(tree_->put(item2));
    EXPECT_EQ(tree_->size(), 2u);

    auto result1 = tree_->match(100, 0);
    EXPECT_EQ(result1.matched_index, 1);

    auto result2 = tree_->match(100, 1);
    EXPECT_EQ(result2.matched_index, 2);
}

TEST_F(RadixTreeTest, PopEvictsLRU) {
    BlockCache::CacheItem item1{100, 0, 1, false};
    BlockCache::CacheItem item2{101, 0, 2, false};
    BlockCache::CacheItem item3{102, 0, 3, false};
    tree_->put(item1);
    tree_->put(item2);
    tree_->put(item3);
    EXPECT_EQ(tree_->size(), 3u);

    // Pop 2 blocks - should evict LRU first (item1, item2)
    auto popped = tree_->pop(2);
    EXPECT_EQ(popped.size(), 2u);
    EXPECT_EQ(tree_->size(), 1u);

    // item3 should still be there
    auto result = tree_->match(102, 0);
    EXPECT_EQ(result.matched_index, 3);
}

TEST_F(RadixTreeTest, PopSkipsResidentBlocks) {
    BlockCache::CacheItem item1{100, 0, 1, true};  // resident
    BlockCache::CacheItem item2{101, 0, 2, false};
    tree_->put(item1);
    tree_->put(item2);

    auto popped = tree_->pop(2);
    EXPECT_EQ(popped.size(), 1u);
    EXPECT_EQ(popped[0], 2);

    // Resident block should remain
    auto result = tree_->match(100, 0);
    EXPECT_EQ(result.matched_index, 1);
}

TEST_F(RadixTreeTest, MatchTouchesNode) {
    BlockCache::CacheItem item1{100, 0, 1, false};
    BlockCache::CacheItem item2{101, 0, 2, false};
    tree_->put(item1);
    tree_->put(item2);

    // Touch item1 by matching it
    tree_->match(100, 0);

    // Pop 1 should evict item2 (LRU) not item1 (touched/MRU)
    auto popped = tree_->pop(1);
    EXPECT_EQ(popped.size(), 1u);
    EXPECT_EQ(popped[0], 2);
}

TEST_F(RadixTreeTest, ContainsWorks) {
    BlockCache::CacheItem item{100, 0, 1, false};
    tree_->put(item);

    EXPECT_TRUE(tree_->contains(100, 0));
    EXPECT_FALSE(tree_->contains(100, 1));
    EXPECT_FALSE(tree_->contains(999, 0));
}

TEST_F(RadixTreeTest, EmptyAndSize) {
    EXPECT_TRUE(tree_->empty());
    EXPECT_EQ(tree_->size(), 0u);

    BlockCache::CacheItem item{100, 0, 1, false};
    tree_->put(item);
    EXPECT_FALSE(tree_->empty());
    EXPECT_EQ(tree_->size(), 1u);
}

TEST_F(RadixTreeTest, CacheSnapshot) {
    BlockCache::CacheItem item1{100, 0, 1, false};
    BlockCache::CacheItem item2{101, 0, 2, false};
    tree_->put(item1);
    tree_->put(item2);

    auto snapshot = tree_->cacheSnapshot(-1);
    EXPECT_EQ(snapshot.values.size(), 2u);
    EXPECT_GE(snapshot.version, 0);

    // Requesting same version should return empty
    auto snapshot2 = tree_->cacheSnapshot(snapshot.version);
    EXPECT_TRUE(snapshot2.values.empty());
}

TEST_F(RadixTreeTest, EvictGPUWithOffload) {
    BlockCache::CacheItem item1{100, 0, 1, false};
    BlockCache::CacheItem item2{101, 0, 2, false};
    tree_->put(item1);
    tree_->put(item2);

    auto result = tree_->evictGPU(1, true);
    EXPECT_EQ(result.offloadable_nodes.size(), 1u);
    EXPECT_EQ(result.offloadable_gpu_blocks.size(), 1u);
    EXPECT_TRUE(result.discarded_gpu_blocks.empty());

    // Node is still in tree (not removed)
    EXPECT_EQ(tree_->size(), 2u);
}

TEST_F(RadixTreeTest, EvictGPUWithoutOffload) {
    BlockCache::CacheItem item1{100, 0, 1, false};
    BlockCache::CacheItem item2{101, 0, 2, false};
    tree_->put(item1);
    tree_->put(item2);

    auto result = tree_->evictGPU(1, false);
    EXPECT_TRUE(result.offloadable_nodes.empty());
    EXPECT_EQ(result.discarded_gpu_blocks.size(), 1u);
    EXPECT_EQ(tree_->size(), 1u);
}

TEST_F(RadixTreeTest, MarkOffloadedAndOnboarded) {
    BlockCache::CacheItem item{100, 0, 1, false};
    tree_->put(item);

    auto evict_result = tree_->evictGPU(1, true);
    ASSERT_EQ(evict_result.offloadable_nodes.size(), 1u);
    auto* node = evict_result.offloadable_nodes[0];

    // Mark as offloaded
    tree_->markOffloaded(node, 42);

    // GPU match should fail now
    auto gpu_result = tree_->match(100, 0);
    EXPECT_TRUE(isNullBlockIdx(gpu_result.matched_index));

    // matchPrefix should find it on host
    auto prefix_result = tree_->matchPrefix({100}, 0, 4);
    EXPECT_EQ(prefix_result.host_nodes.size(), 1u);
    EXPECT_EQ(prefix_result.gpu_blocks.size(), 0u);

    // Mark as onboarded
    tree_->markOnboarded(node, 5);

    // GPU match should succeed
    auto gpu_result2 = tree_->match(100, 0);
    EXPECT_EQ(gpu_result2.matched_index, 5);
}

TEST_F(RadixTreeTest, EvictHost) {
    BlockCache::CacheItem item{100, 0, 1, false};
    tree_->put(item);

    auto evict_result = tree_->evictGPU(1, true);
    ASSERT_EQ(evict_result.offloadable_nodes.size(), 1u);
    auto* node = evict_result.offloadable_nodes[0];
    tree_->markOffloaded(node, 42);

    // Now evict from host
    auto host_evicted = tree_->evictHost(1);
    EXPECT_EQ(host_evicted.size(), 1u);
    EXPECT_EQ(host_evicted[0], 42);

    // Node should be gone
    EXPECT_EQ(tree_->size(), 0u);
}

TEST_F(RadixTreeTest, MatchPrefixGPUOnly) {
    BlockCache::CacheItem item1{100, 0, 1, false};
    BlockCache::CacheItem item2{101, 0, 2, false};
    BlockCache::CacheItem item3{102, 0, 3, false};
    tree_->put(item1);
    tree_->put(item2);
    tree_->put(item3);

    auto result = tree_->matchPrefix({100, 101, 102}, 0, 4);
    EXPECT_EQ(result.gpu_blocks.size(), 3u);
    EXPECT_EQ(result.host_nodes.size(), 0u);
    EXPECT_EQ(result.gpu_reuse_blocks, 3u);
    EXPECT_EQ(result.gpu_reuse_length, 12u);
    EXPECT_EQ(result.total_reuse_length, 12u);
}

TEST_F(RadixTreeTest, MatchPrefixPartialMatch) {
    BlockCache::CacheItem item1{100, 0, 1, false};
    BlockCache::CacheItem item2{101, 0, 2, false};
    tree_->put(item1);
    tree_->put(item2);

    // Third key not in tree
    auto result = tree_->matchPrefix({100, 101, 999}, 0, 4);
    EXPECT_EQ(result.gpu_blocks.size(), 2u);
    EXPECT_EQ(result.gpu_reuse_blocks, 2u);
    EXPECT_EQ(result.gpu_reuse_length, 8u);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
