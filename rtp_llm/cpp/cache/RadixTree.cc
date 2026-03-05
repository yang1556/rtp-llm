#include "rtp_llm/cpp/cache/RadixTree.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RadixTree::RadixTree() {
    root_ = std::make_unique<RadixTreeNode>();
}

// --- BlockCache-compatible interface ---

bool RadixTree::put(BlockCache::CacheItem& item) {
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(item.block_index), "put block id should not be null block");

    CacheKeyGroupPair key{item.cache_key, item.group_id};

    auto it = node_lookup_.find(key);
    if (it != node_lookup_.end()) {
        touchNode(it->second);
        return false;
    }

    // Find or create parent: walk existing chain to find where this node should attach.
    // In the current architecture cache_keys are inserted in sequence order,
    // so the parent is the most-recently-inserted node in the same chain.
    // However, with flat put() we don't know the parent, so attach to root.
    // The tree will still work for eviction (leaf tracking) and offload.
    // matchPrefix() does sequential tree walking for proper prefix matching.
    auto node           = std::make_unique<RadixTreeNode>();
    node->cache_key     = item.cache_key;
    node->group_id      = item.group_id;
    node->gpu_block_idx = item.block_index;
    node->is_resident   = item.is_resident;
    node->parent        = root_.get();

    auto* node_ptr       = node.get();
    root_->children[key] = std::move(node);
    node_lookup_[key]    = node_ptr;
    node_count_++;
    version_++;

    updateEvictableStatus(node_ptr);

    return true;
}

BlockCache::MatchResult RadixTree::match(CacheKeyType cache_key, int group_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto*                       node = findNode(cache_key, static_cast<GroupIdType>(group_id));
    if (node && node->isOnGPU()) {
        touchNode(node);
        return {node->gpu_block_idx};
    }
    return {NULL_BLOCK_IDX};
}

bool RadixTree::contains(CacheKeyType cache_key, int group_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    return findNode(cache_key, static_cast<GroupIdType>(group_id)) != nullptr;
}

BlockIndicesType RadixTree::pop(int n) {
    std::lock_guard<std::mutex> lock(mu_);
    RTP_LLM_CHECK_WITH_INFO(n > 0, "pop nums should > 0, nums = " + std::to_string(n));

    BlockIndicesType popped;
    while (n > 0 && !gpu_evict_list_.empty()) {
        auto* node = gpu_evict_list_.front();
        if (node->is_resident) {
            // Move resident node to back, try next
            gpu_evict_list_.pop_front();
            gpu_evict_list_.push_back(node);
            gpu_evict_map_[node] = std::prev(gpu_evict_list_.end());
            // If we cycled through all nodes and all are resident, break
            if (node == gpu_evict_list_.front()) {
                break;
            }
            continue;
        }

        popped.push_back(node->gpu_block_idx);
        removeFromEvictableGPU(node);

        // Remove node from tree
        CacheKeyGroupPair key{node->cache_key, node->group_id};
        node_lookup_.erase(key);
        auto* parent = node->parent;
        if (parent) {
            parent->children.erase(key);
            updateEvictableStatus(parent);
        }
        node_count_--;
        version_++;
        n--;
    }

    return popped;
}

bool RadixTree::empty() const {
    std::lock_guard<std::mutex> lock(mu_);
    return node_count_ == 0;
}

size_t RadixTree::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return node_count_;
}

BlockCache::CacheSnapshot RadixTree::cacheSnapshot(int64_t latest_version) const {
    std::lock_guard<std::mutex>        lock(mu_);
    std::vector<BlockCache::CacheItem> values;
    if (latest_version < version_) {
        values.reserve(node_count_);
        for (const auto& [key, node_ptr] : node_lookup_) {
            BlockCache::CacheItem item;
            item.cache_key   = node_ptr->cache_key;
            item.group_id    = node_ptr->group_id;
            item.block_index = node_ptr->gpu_block_idx;
            item.is_resident = node_ptr->is_resident;
            values.push_back(item);
        }
    }
    return {version_, std::move(values)};
}

// --- Extended interface for offload ---

RadixTree::EvictResult RadixTree::evictGPU(int n, bool enable_offload) {
    std::lock_guard<std::mutex> lock(mu_);
    EvictResult                 result;

    while (n > 0 && !gpu_evict_list_.empty()) {
        auto* node = gpu_evict_list_.front();
        if (node->is_resident) {
            gpu_evict_list_.pop_front();
            gpu_evict_list_.push_back(node);
            gpu_evict_map_[node] = std::prev(gpu_evict_list_.end());
            if (node == gpu_evict_list_.front()) {
                break;
            }
            continue;
        }

        BlockIdxType gpu_block = node->gpu_block_idx;
        removeFromEvictableGPU(node);

        if (enable_offload) {
            result.offloadable_nodes.push_back(node);
            result.offloadable_gpu_blocks.push_back(gpu_block);
        } else {
            result.discarded_gpu_blocks.push_back(gpu_block);
            CacheKeyGroupPair key{node->cache_key, node->group_id};
            node_lookup_.erase(key);
            auto* parent = node->parent;
            if (parent) {
                parent->children.erase(key);
                updateEvictableStatus(parent);
            }
            node_count_--;
            version_++;
        }
        n--;
    }

    return result;
}

BlockIndicesType RadixTree::evictHost(int n) {
    std::lock_guard<std::mutex> lock(mu_);
    BlockIndicesType            evicted;

    while (n > 0 && !host_evict_list_.empty()) {
        auto* node = host_evict_list_.front();
        if (node->is_resident) {
            host_evict_list_.pop_front();
            host_evict_list_.push_back(node);
            host_evict_map_[node] = std::prev(host_evict_list_.end());
            if (node == host_evict_list_.front()) {
                break;
            }
            continue;
        }

        evicted.push_back(node->host_block_idx);
        removeFromEvictableHost(node);

        // Node is now fully evicted (no GPU, no host) - remove from tree
        CacheKeyGroupPair key{node->cache_key, node->group_id};
        node_lookup_.erase(key);
        auto* parent = node->parent;
        if (parent) {
            parent->children.erase(key);
            updateEvictableStatus(parent);
        }
        node_count_--;
        version_++;
        n--;
    }

    return evicted;
}

void RadixTree::markOffloaded(RadixTreeNode* node, BlockIdxType host_block_idx) {
    std::lock_guard<std::mutex> lock(mu_);
    node->host_block_idx = host_block_idx;
    node->gpu_block_idx  = NULL_BLOCK_IDX;
    updateEvictableStatus(node);
}

void RadixTree::markOnboarded(RadixTreeNode* node, BlockIdxType gpu_block_idx) {
    std::lock_guard<std::mutex> lock(mu_);
    removeFromEvictableHost(node);
    node->gpu_block_idx  = gpu_block_idx;
    node->host_block_idx = NULL_BLOCK_IDX;
    updateEvictableStatus(node);
}

RadixTree::ExtendedMatchResult
RadixTree::matchPrefix(const CacheKeysType& keys, GroupIdType group_id, int seq_size_per_block) {
    std::lock_guard<std::mutex> lock(mu_);
    ExtendedMatchResult         result;

    for (const auto& cache_key : keys) {
        CacheKeyGroupPair key{cache_key, group_id};
        auto              it = node_lookup_.find(key);
        if (it == node_lookup_.end()) {
            break;
        }

        auto* node = it->second;
        if (node->isOnGPU()) {
            result.gpu_blocks.push_back(node->gpu_block_idx);
            result.gpu_reuse_blocks++;
            touchNode(node);
        } else if (node->isOnHost()) {
            result.host_nodes.push_back(node);
            result.host_reuse_blocks++;
        } else {
            break;
        }
    }

    result.gpu_reuse_length   = result.gpu_reuse_blocks * seq_size_per_block;
    result.total_reuse_length = (result.gpu_reuse_blocks + result.host_reuse_blocks) * seq_size_per_block;
    return result;
}

// --- Private helpers ---

RadixTreeNode* RadixTree::findNode(CacheKeyType cache_key, GroupIdType group_id) const {
    CacheKeyGroupPair key{cache_key, group_id};
    auto              it = node_lookup_.find(key);
    return (it != node_lookup_.end()) ? it->second : nullptr;
}

void RadixTree::touchNode(RadixTreeNode* node) {
    // Move to MRU end (back of list) if in gpu evict list
    auto it = gpu_evict_map_.find(node);
    if (it != gpu_evict_map_.end()) {
        gpu_evict_list_.erase(it->second);
        gpu_evict_list_.push_back(node);
        it->second = std::prev(gpu_evict_list_.end());
    }
}

void RadixTree::addToEvictableGPU(RadixTreeNode* node) {
    if (gpu_evict_map_.count(node) == 0) {
        gpu_evict_list_.push_back(node);
        gpu_evict_map_[node] = std::prev(gpu_evict_list_.end());
    }
}

void RadixTree::removeFromEvictableGPU(RadixTreeNode* node) {
    auto it = gpu_evict_map_.find(node);
    if (it != gpu_evict_map_.end()) {
        gpu_evict_list_.erase(it->second);
        gpu_evict_map_.erase(it);
    }
}

void RadixTree::addToEvictableHost(RadixTreeNode* node) {
    if (host_evict_map_.count(node) == 0) {
        host_evict_list_.push_back(node);
        host_evict_map_[node] = std::prev(host_evict_list_.end());
    }
}

void RadixTree::removeFromEvictableHost(RadixTreeNode* node) {
    auto it = host_evict_map_.find(node);
    if (it != host_evict_map_.end()) {
        host_evict_list_.erase(it->second);
        host_evict_map_.erase(it);
    }
}

void RadixTree::updateEvictableStatus(RadixTreeNode* node) {
    if (node == root_.get()) {
        return;
    }

    if (node->isEvictableFromGPU()) {
        addToEvictableGPU(node);
    } else {
        removeFromEvictableGPU(node);
    }

    if (node->isEvictableFromHost()) {
        addToEvictableHost(node);
    } else {
        removeFromEvictableHost(node);
    }
}

void RadixTree::pruneEmptyAncestors(RadixTreeNode* node) {
    while (node && node != root_.get() && node->children.empty() && !node->isOnGPU() && !node->isOnHost()) {
        auto*             parent = node->parent;
        CacheKeyGroupPair key{node->cache_key, node->group_id};
        node_lookup_.erase(key);
        if (parent) {
            parent->children.erase(key);
        }
        node_count_--;
        version_++;
        node = parent;
        if (node) {
            updateEvictableStatus(node);
        }
    }
}

}  // namespace rtp_llm
