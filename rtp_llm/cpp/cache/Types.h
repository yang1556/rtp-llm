#pragma once

#include <cstddef>
#include <vector>
#include <cstdint>
#include <sstream>
#include <string>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

typedef int32_t          GroupIdType;
typedef std::vector<int> LayerIdsType;

struct BlockAddrInfo {
    void* kv_addr       = nullptr;
    void* kv_scale_addr = nullptr;
};

struct KVCacheInfo {
    size_t                    available_kv_cache = 0;
    size_t                    total_kv_cache     = 0;
    size_t                    block_size         = 0;
    std::vector<CacheKeyType> cached_keys;
    int64_t                   version = -1;
};

struct BlockIdPair {
    BlockIdxType src;
    BlockIdxType dst;
};

struct MatchResult {
    size_t           reuse_length = 0;
    size_t           reuse_blocks = 0;
    BlockIndicesType block_indices;

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "MatchResult reuse_length: " << reuse_length << ", reuse_blocks: " << reuse_blocks
                     << ", block_indices: ";
        for (const auto& v : block_indices) {
            debug_string << v << ", ";
        }
        return debug_string.str();
    }
};

// for p2p connector when TP settings of prefill & decode are different.
struct KVPartitionBytes {
    size_t k_off = 0;
    size_t k_sz  = 0;
    size_t v_off = 0;
    size_t v_sz  = 0;
};

struct MallocInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    int64_t                 request_id          = 0;
    bool                    verbose             = true;  // for failed log
    bool                    reuse_cache         = true;
    bool                    enable_device_cache = true;
};

struct MallocResult {
    bool success;
    int  reuse_len;

    int64_t match_cost_time_us = 0;

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "MallocResult success: " << (success ? "true" : "false") << ", reuse_len: " << reuse_len
                     << ", match_cost_time_us: " << match_cost_time_us;
        return debug_string.str();
    }
};

struct FreeInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;

    int64_t request_id = 0;

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "FreeInfo request_id: " << request_id;
        if (batch_kv_cache_resource) {
            debug_string << ", batch_size: " << batch_kv_cache_resource->batchSize();
            if (batch_kv_cache_resource->batchSize() > 0) {
                debug_string << ", cache_keys: ";
                const auto& cache_keys = batch_kv_cache_resource->cacheKeys(0);
                for (size_t i = 0; i < cache_keys.size() && i < 10; ++i) {  // Limit to first 10
                    debug_string << cache_keys[i] << ", ";
                }
                if (cache_keys.size() > 10) {
                    debug_string << "...";
                }
                debug_string << ", blocks: ";
                const auto& blocks = batch_kv_cache_resource->blocks(0);
                for (size_t i = 0; i < blocks.size() && i < 10; ++i) {  // Limit to first 10
                    debug_string << blocks[i] << ", ";
                }
                if (blocks.size() > 10) {
                    debug_string << "...";
                }
            }
        }
        return debug_string.str();
    }
};

struct InsertInfo {
    BatchKVCacheResourcePtr batch_kv_cache_resource;
    CompleteTokenIdsPtr     complete_token_ids;
    bool                    is_resident;

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "InsertInfo is_resident: " << (is_resident ? "true" : "false");
        if (batch_kv_cache_resource) {
            debug_string << ", batch_size: " << batch_kv_cache_resource->batchSize();
            if (batch_kv_cache_resource->batchSize() > 0) {
                debug_string << ", cache_keys: ";
                const auto& cache_keys = batch_kv_cache_resource->cacheKeys(0);
                for (size_t i = 0; i < cache_keys.size() && i < 10; ++i) {  // Limit to first 10
                    debug_string << cache_keys[i] << ", ";
                }
                if (cache_keys.size() > 10) {
                    debug_string << "...";
                }
                debug_string << ", blocks: ";
                const auto& blocks = batch_kv_cache_resource->blocks(0);
                for (size_t i = 0; i < blocks.size() && i < 10; ++i) {  // Limit to first 10
                    debug_string << blocks[i] << ", ";
                }
                if (blocks.size() > 10) {
                    debug_string << "...";
                }
            }
        }
        return debug_string.str();
    }
};

}  // namespace rtp_llm