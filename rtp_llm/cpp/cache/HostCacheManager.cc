#include "rtp_llm/cpp/cache/HostCacheManager.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

HostCacheManager::HostCacheManager(BlockPoolPtr gpu_pool, RadixTreePtr radix_tree, DeviceBase* device):
    gpu_pool_(gpu_pool), radix_tree_(radix_tree), device_(device) {}

bool HostCacheManager::init(size_t host_cache_size_mb, const CacheConfig& cache_config) {
    if (host_cache_size_mb == 0) {
        enabled_ = false;
        return true;
    }

    num_layers_ = cache_config.layer_num;

    auto         host_pool_config = BlockPoolConfigHelper::createConfig(cache_config);
    const size_t bytes_per_block  = host_pool_config.total_size_bytes / cache_config.block_num;
    if (bytes_per_block == 0) {
        RTP_LLM_LOG_ERROR("HostCacheManager init failed: bytes_per_block is 0");
        return false;
    }

    const int64_t host_block_num =
        static_cast<int64_t>(host_cache_size_mb) * 1024 * 1024 / static_cast<int64_t>(bytes_per_block);
    if (host_block_num <= 0) {
        RTP_LLM_LOG_ERROR("HostCacheManager init failed: host_cache_size_mb too small");
        return false;
    }

    // Rebuild config with host block count
    host_pool_config.block_num = static_cast<uint32_t>(host_block_num);
    size_t offset              = 0;
    for (auto& layout : host_pool_config.memory_layouts) {
        layout.block_num = static_cast<uint32_t>(host_block_num);
        layout.kv_block_pool_size_bytes =
            static_cast<size_t>(layout.layer_num) * host_block_num * layout.kv_block_stride_bytes;
        layout.kv_scale_pool_size_bytes =
            static_cast<size_t>(layout.layer_num) * host_block_num * layout.kv_scale_stride_bytes;
        layout.total_size_bytes      = layout.kv_block_pool_size_bytes + layout.kv_scale_pool_size_bytes;
        layout.kv_cache_offset_bytes = offset;
        offset += layout.kv_block_pool_size_bytes;
        layout.kv_scale_offset_bytes = offset;
        offset += layout.kv_scale_pool_size_bytes;
    }
    host_pool_config.total_size_bytes = offset;

    host_pool_ = std::make_shared<BlockPool>(host_pool_config, device_, AllocationType::HOST);
    if (!host_pool_->init()) {
        RTP_LLM_LOG_ERROR("HostCacheManager init failed: host BlockPool init failed");
        return false;
    }

    enabled_ = true;
    RTP_LLM_LOG_INFO("HostCacheManager initialized: host_cache_size_mb=%zu, host_block_num=%ld, layers=%zu",
                     host_cache_size_mb,
                     host_block_num,
                     num_layers_);
    return true;
}

void HostCacheManager::copyBlockBetweenPools(BlockIdxType src_block,
                                             BlockPoolPtr src_pool,
                                             BlockIdxType dst_block,
                                             BlockPoolPtr dst_pool) {
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        auto src_bufs = src_pool->convertIndexToBuffer(static_cast<int>(layer), src_block);
        auto dst_bufs = dst_pool->convertIndexToBuffer(static_cast<int>(layer), dst_block);

        for (size_t i = 0; i < std::min(src_bufs.size(), dst_bufs.size()); ++i) {
            MemoryType src_mem =
                src_pool->where() == MemoryType::MEMORY_GPU ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
            MemoryType dst_mem =
                dst_pool->where() == MemoryType::MEMORY_GPU ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;

            auto src = std::make_shared<Buffer>(
                src_mem, DataType::TYPE_INT8, std::vector<size_t>{src_bufs[i].size_bytes}, src_bufs[i].addr);
            auto dst = std::make_shared<Buffer>(
                dst_mem, DataType::TYPE_INT8, std::vector<size_t>{dst_bufs[i].size_bytes}, dst_bufs[i].addr);
            device_->copy({*dst, *src});
        }
    }
}

bool HostCacheManager::offloadBlock(RadixTreeNode* node) {
    if (!enabled_ || !node || !node->isOnGPU()) {
        return false;
    }

    ensureHostFreeBlocks(1);
    auto host_blocks = host_pool_->malloc(1);
    if (host_blocks.empty()) {
        RTP_LLM_LOG_WARNING("offloadBlock failed: cannot allocate host block");
        return false;
    }

    BlockIdxType host_block_idx = host_blocks[0];
    BlockIdxType gpu_block_idx  = node->gpu_block_idx;

    copyBlockBetweenPools(gpu_block_idx, gpu_pool_, host_block_idx, host_pool_);
    device_->syncAndCheck();

    radix_tree_->markOffloaded(node, host_block_idx);
    return true;
}

BlockIdxType HostCacheManager::onboardBlock(RadixTreeNode* node) {
    if (!enabled_ || !node || !node->isOnHost()) {
        return NULL_BLOCK_IDX;
    }

    auto gpu_blocks = gpu_pool_->malloc(1);
    if (gpu_blocks.empty()) {
        return NULL_BLOCK_IDX;
    }

    BlockIdxType gpu_block_idx  = gpu_blocks[0];
    BlockIdxType host_block_idx = node->host_block_idx;

    copyBlockBetweenPools(host_block_idx, host_pool_, gpu_block_idx, gpu_pool_);
    device_->syncAndCheck();

    host_pool_->requestFree(host_block_idx);
    radix_tree_->markOnboarded(node, gpu_block_idx);
    return gpu_block_idx;
}

void HostCacheManager::ensureHostFreeBlocks(int n) {
    if (!enabled_) {
        return;
    }

    while (host_pool_->freeBlocksNum() < static_cast<size_t>(n)) {
        auto evicted = radix_tree_->evictHost(1);
        if (evicted.empty()) {
            break;
        }
        host_pool_->blockCacheFree(evicted);
    }
}

size_t HostCacheManager::hostFreeBlocksNum() const {
    return enabled_ ? host_pool_->freeBlocksNum() : 0;
}

size_t HostCacheManager::hostTotalBlocksNum() const {
    return enabled_ ? host_pool_->totalBlocksNum() : 0;
}

}  // namespace rtp_llm
