#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/RadixTree.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

class HostCacheManager {
public:
    HostCacheManager(BlockPoolPtr gpu_pool, RadixTreePtr radix_tree, DeviceBase* device);
    ~HostCacheManager() = default;

    bool init(size_t host_cache_size_mb, const CacheConfig& cache_config);

    // Offload a GPU block to host memory (eviction-triggered).
    bool offloadBlock(RadixTreeNode* node);

    // Onboard a host block back to GPU memory (match-triggered).
    // Returns the new GPU block index, or NULL_BLOCK_IDX on failure.
    BlockIdxType onboardBlock(RadixTreeNode* node);

    void ensureHostFreeBlocks(int n);

    size_t hostFreeBlocksNum() const;
    size_t hostTotalBlocksNum() const;
    bool   isEnabled() const {
        return enabled_;
    }

    BlockPoolPtr hostPool() const {
        return host_pool_;
    }

private:
    void
    copyBlockBetweenPools(BlockIdxType src_block, BlockPoolPtr src_pool, BlockIdxType dst_block, BlockPoolPtr dst_pool);

    bool         enabled_ = false;
    BlockPoolPtr gpu_pool_;
    BlockPoolPtr host_pool_;
    RadixTreePtr radix_tree_;
    DeviceBase*  device_;
    size_t       num_layers_ = 0;
};

using HostCacheManagerPtr = std::shared_ptr<HostCacheManager>;

}  // namespace rtp_llm
