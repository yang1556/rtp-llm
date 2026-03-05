#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <unordered_map>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/RadixTree.h"
#include "rtp_llm/cpp/cache/HostCacheManager.h"

namespace rtp_llm {

struct NeedBlocksInfo {
    int common_blocks = 0;  // shared blocks across batches
    int extra_blocks  = 0;  // extra blocks per batch
};

class KVCacheGroup {
public:
    KVCacheGroup(const LayerIdsType& layer_ids,
                 KVCacheSpecPtr      kvcache_spec,
                 BlockPoolPtr        block_pool,
                 int                 group_id,
                 HostCacheManagerPtr host_cache_manager = nullptr):
        layer_ids_(layer_ids),
        kvcache_spec_(std::move(kvcache_spec)),
        block_pool_(block_pool),
        block_cache_(block_pool_->blockCache()),
        radix_tree_(block_pool_->radixTree()),
        host_cache_manager_(host_cache_manager),
        group_id_(group_id),
        seq_size_per_block_(kvcache_spec_->seq_size_per_block) {}

    virtual ~KVCacheGroup() = default;

    bool init();
    virtual bool
    malloc(BlockIndicesType& block_indices, int seq_len, bool enable_reuse_cache = false, int reserve_step = 0) = 0;
    // TODO, match的时候热度不增加，最终匹配成功的时候再去增加热度。
    virtual MatchResult match(const CacheKeysType& cache_keys)      = 0;
    virtual void        free(const BlockIndicesType& block_indices) = 0;
    virtual void
    insertIntoCache(const CacheKeysType& cache_keys, const BlockIndicesType& block_indices, bool is_resident) = 0;
    virtual void
    removeSkippedBlocks(BlockIndicesType& block_indices, bool enable_reuse_cache = false, int reserve_step = 0) = 0;
    virtual int            needBlocksNum(int seq_len, int current_blocks, int reserve_step = 0) const           = 0;
    virtual NeedBlocksInfo getNeedBlocks(
        int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled = false) const = 0;
    virtual void reference(BlockIndicesType& block_indices, const BlockIndicesType& new_block_indices)             = 0;

    void                                   reference(const BlockIndicesType& new_block_indices);
    std::unordered_map<int, torch::Tensor> allLayerCacheBase() const;
    std::unordered_map<int, torch::Tensor> allLayerScaleCacheBase() const;
    BlockAddrInfo                          convertIndexToAddr(int layer_id, int block_id) const;
    std::vector<BlockInfo>                 convertIndexToBuffer(int layer_id, int block_id) const;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const;

    size_t freeBlocksNum() const;
    bool   ensureFreeBlocks(int need_blocks);
    int    seqSizePerBlock() const;
    void   setHostCacheManager(HostCacheManagerPtr host_cache_manager) {
        host_cache_manager_ = host_cache_manager;
    }

protected:
    LayerIdsType        layer_ids_;
    KVCacheSpecPtr      kvcache_spec_;
    BlockPoolPtr        block_pool_;
    BlockCachePtr       block_cache_;
    RadixTreePtr        radix_tree_;
    HostCacheManagerPtr host_cache_manager_;
    int                 group_id_ = 0;

    int                                    seq_size_per_block_;
    std::unordered_map<int, torch::Tensor> global_layer_to_kv_tensors;
    std::unordered_map<int, torch::Tensor> global_layer_to_kv_scale_tensors;
    std::unordered_map<int, int>           global_layer_to_local_layer;
};

using KVCacheGroupPtr = std::shared_ptr<KVCacheGroup>;

}  // namespace rtp_llm
