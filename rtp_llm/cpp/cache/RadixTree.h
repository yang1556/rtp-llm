#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/Types.h"

namespace rtp_llm {

struct RadixTreeNode {
    CacheKeyType cache_key = 0;
    GroupIdType  group_id  = 0;

    BlockIdxType gpu_block_idx  = NULL_BLOCK_IDX;
    BlockIdxType host_block_idx = NULL_BLOCK_IDX;
    bool         is_resident    = false;

    RadixTreeNode* parent = nullptr;
    std::unordered_map<CacheKeyGroupPair,
                       std::unique_ptr<RadixTreeNode>,
                       PairFirstHash<CacheKeyType, GroupIdType>,
                       PairBothEqual<CacheKeyType, GroupIdType>>
        children;

    bool isOnGPU() const {
        return gpu_block_idx != NULL_BLOCK_IDX;
    }
    bool isOnHost() const {
        return host_block_idx != NULL_BLOCK_IDX;
    }
    bool isLeaf() const {
        return children.empty();
    }
    bool isEvictableFromGPU() const {
        return isLeaf() && !is_resident && isOnGPU();
    }
    bool isEvictableFromHost() const {
        return isLeaf() && !is_resident && !isOnGPU() && isOnHost();
    }
};

class RadixTree {
public:
    struct EvictResult {
        BlockIndicesType            discarded_gpu_blocks;
        std::vector<RadixTreeNode*> offloadable_nodes;
        BlockIndicesType            offloadable_gpu_blocks;
    };

    struct ExtendedMatchResult {
        BlockIndicesType            gpu_blocks;
        std::vector<RadixTreeNode*> host_nodes;
        size_t                      gpu_reuse_blocks   = 0;
        size_t                      host_reuse_blocks  = 0;
        size_t                      gpu_reuse_length   = 0;
        size_t                      total_reuse_length = 0;
    };

public:
    RadixTree();
    ~RadixTree() = default;

    // --- BlockCache-compatible interface ---
    bool                      put(BlockCache::CacheItem& item);
    BlockCache::MatchResult   match(CacheKeyType cache_key, int group_id = 0);
    BlockIndicesType          pop(int n);
    bool                      contains(CacheKeyType cache_key, int group_id = 0) const;
    bool                      empty() const;
    size_t                    size() const;
    BlockCache::CacheSnapshot cacheSnapshot(int64_t latest_version) const;

    // --- Extended interface for offload ---
    EvictResult         evictGPU(int n, bool enable_offload);
    BlockIndicesType    evictHost(int n);
    void                markOffloaded(RadixTreeNode* node, BlockIdxType host_block_idx);
    void                markOnboarded(RadixTreeNode* node, BlockIdxType gpu_block_idx);
    ExtendedMatchResult matchPrefix(const CacheKeysType& keys, GroupIdType group_id, int seq_size_per_block);

private:
    using LRUListIter = std::list<RadixTreeNode*>::iterator;

    RadixTreeNode* findNode(CacheKeyType cache_key, GroupIdType group_id) const;
    void           touchNode(RadixTreeNode* node);
    void           addToEvictableGPU(RadixTreeNode* node);
    void           removeFromEvictableGPU(RadixTreeNode* node);
    void           addToEvictableHost(RadixTreeNode* node);
    void           removeFromEvictableHost(RadixTreeNode* node);
    void           updateEvictableStatus(RadixTreeNode* node);
    void           pruneEmptyAncestors(RadixTreeNode* node);

    std::unique_ptr<RadixTreeNode>                  root_;
    std::unordered_map<RadixTreeNode*, LRUListIter> gpu_evict_map_;
    std::list<RadixTreeNode*>                       gpu_evict_list_;  // LRU order, front = oldest
    std::unordered_map<RadixTreeNode*, LRUListIter> host_evict_map_;
    std::list<RadixTreeNode*>                       host_evict_list_;

    // Fast lookup: (cache_key, group_id) → node
    std::unordered_map<CacheKeyGroupPair,
                       RadixTreeNode*,
                       PairFirstHash<CacheKeyType, GroupIdType>,
                       PairBothEqual<CacheKeyType, GroupIdType>>
            node_lookup_;
    size_t  node_count_ = 0;
    int64_t version_    = -1;

    mutable std::mutex mu_;
};

using RadixTreePtr = std::shared_ptr<RadixTree>;

}  // namespace rtp_llm
