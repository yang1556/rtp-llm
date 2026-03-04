#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool KVCacheAllocator::init() {
    RTP_LLM_CHECK_WITH_INFO(doInit(), "kv cache allocator init failed");

    const size_t available_blocks = availableBlocksNum();
    const size_t reserve_blocks =
        static_cast<size_t>(reserve_block_ratio_) * available_blocks / static_cast<size_t>(100);
    reserve_block_num_ = reserve_blocks;
    RTP_LLM_LOG_INFO("KVCacheAllocator set reserve blocks: ratio=%ld%% reserve_blocks=%zu available_blocks=%zu",
                     reserve_block_ratio_,
                     reserve_blocks,
                     available_blocks);

    return true;
}

MallocResult KVCacheAllocator::initMalloc(const MallocInfo& malloc_info) {
    auto init_result = initMallocForCommonLen(malloc_info);
    if (!init_result.success) {
        FreeInfo free_info{malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids};
        free(free_info);
        return init_result;
    }

    auto incr_result = incrMalloc(malloc_info);
    if (!incr_result.success) {
        FreeInfo free_info{malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids};
        free(free_info);
        return incr_result;
    } else {
        reportMetric(malloc_info, init_result);
        return init_result;
    }
}

void KVCacheAllocator::reportMetric(const MallocInfo& malloc_info, const MallocResult& init_result) const {
    if (!metrics_reporter_ || !malloc_info.enable_device_cache) {
        return;
    }
    int64_t gpu_input_length = 0;
    if (malloc_info.batch_kv_cache_resource) {
        size_t cache_keys_count = malloc_info.batch_kv_cache_resource->cacheKeyCount(0);
        gpu_input_length        = static_cast<int64_t>(cache_keys_count) * config_.seq_size_per_block;
    }
    if (gpu_input_length <= 0) {
        return;
    }
    RtpLLMCacheReuseMetricsCollector collector;
    collector.match_cost_time_us = init_result.match_cost_time_us;
    collector.gpu_input_length   = gpu_input_length;
    collector.gpu_reuse_length   = init_result.reuse_len;
    collector.gpu_cache_hit_rate =
        static_cast<float>(static_cast<int64_t>(collector.gpu_reuse_length) * 100 / collector.gpu_input_length);
    kmonitor::MetricsTags tags;
    metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(&tags, &collector);
}

MallocResult KVCacheAllocator::malloc(const MallocInfo& malloc_info) {
    if (!malloc_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false, 0};
    }

    if (!malloc_info.complete_token_ids) {
        RTP_LLM_LOG_ERROR("CompleteTokenIds is null");
        return {false, 0};
    }

    if (malloc_info.batch_kv_cache_resource->curBlocksNum() == 0) {
        return initMalloc(malloc_info);
    } else {
        return incrMalloc(malloc_info);
    }
}

void KVCacheAllocator::blockCopy(int src_block_index, int dest_block_index) {
    BlockIdPair copy_mapping{src_block_index, dest_block_index};
    blockBatchCopy(&copy_mapping, &copy_mapping + 1);
}

void KVCacheAllocator::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    blockBatchCopy(copy_mapping.data(), copy_mapping.data() + copy_mapping.size());
}

void KVCacheAllocator::blockBatchCopy(const Buffer& copy_mapping) {
    RTP_LLM_CHECK(copy_mapping.dim() == 2 && copy_mapping.shape()[1] == 2);
    const auto* begin_ptr = (const BlockIdPair*)copy_mapping.data();
    size_t      copy_num  = copy_mapping.shape()[0];
    blockBatchCopy(begin_ptr, begin_ptr + copy_num);
}

void KVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    using CopyType = BatchCopyParams::CopyType;

    if (end_ptr == begin_ptr) {
        return;
    }

    BatchCopyParams copy_params;

    const size_t copy_num = (end_ptr - begin_ptr) * config_.layer_num;

    size_t copy_nums[CopyType::TYPE_SIZE] = {};
    auto   copy_type                      = BatchCopyParams::get_copy_type(
        allocation_type_ == AllocationType::DEVICE ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU,
        allocation_type_ == AllocationType::DEVICE ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU);
    copy_nums[copy_type] += copy_num;  // for kv

    for (size_t i = 0; i < CopyType::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<CopyType>(i), copy_nums[i]);
    }

    auto&  spec                = config_.cache_specs[0];
    size_t kv_block_size_bytes = spec->block_size_bytes();

    for (auto it = begin_ptr; it != end_ptr; ++it) {
        auto [src_block_index, dest_block_index] = *it;

        for (int layer_id = 0; layer_id < config_.layer_num; layer_id++) {
            auto src_addr_info = convertIndexToAddr(layer_id, src_block_index);
            auto dst_addr_info = convertIndexToAddr(layer_id, dest_block_index);

            RTP_LLM_CHECK_WITH_INFO(src_addr_info.kv_addr && dst_addr_info.kv_addr,
                                    "Failed to get block address for layer %d, src_block %d, dst_block %d",
                                    layer_id,
                                    src_block_index,
                                    dest_block_index);

            copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, kv_block_size_bytes, copy_type);

            if (src_addr_info.kv_scale_addr && dst_addr_info.kv_scale_addr) {
                copy_params.add(dst_addr_info.kv_scale_addr,
                                src_addr_info.kv_scale_addr,
                                static_cast<size_t>(config_.kv_scale_stride_bytes),
                                copy_type);
            }
        }
    }

    device_->batchCopy(copy_params);
}

size_t KVCacheAllocator::freeBlocksNum() const {
    return block_pool_->freeBlocksNum();
}

int64_t KVCacheAllocator::getMrCostTimeMs() const {
    return block_pool_->getMrCostTimeMs();
}

size_t KVCacheAllocator::availableBlocksNum() const {
    return block_pool_->availableBlocksNum();
}

size_t KVCacheAllocator::availableTokensNum() const {
    return block_pool_->availableBlocksNum() * seqSizePerBlock();
}

size_t KVCacheAllocator::totalBlocksNum() const {
    return block_pool_->totalBlocksNum();
}

size_t KVCacheAllocator::maxAvailableTokensNum() const {
    return block_pool_->totalBlocksNum() * seqSizePerBlock();
}

void KVCacheAllocator::regUserMr(size_t model_id) {
    block_pool_->regUserMr(model_id);
}

std::vector<std::pair<BufferPtr, size_t>> KVCacheAllocator::getAllBuffers() const {
    std::vector<std::pair<BufferPtr, size_t>> results;

    CacheLayerLayout layout = allLayerCacheBase();
    results.reserve(layout.layers_to_kv_buffer_ptrs.size());

    for (const auto& buf : layout.layers_to_kv_buffer_ptrs) {
        if (!buf || buf->sizeBytes() == 0) {
            continue;
        }
        const size_t kv_block_stride_bytes = config_.kv_block_stride_bytes;
        results.emplace_back(buf, kv_block_stride_bytes);
    }

    for (const auto& buf : layout.layers_to_scale_buffer_ptrs) {
        if (!buf || buf->sizeBytes() == 0) {
            continue;
        }
        const size_t kv_scale_stride_bytes = config_.kv_scale_stride_bytes;
        results.emplace_back(buf, kv_scale_stride_bytes);
    }

    return results;
}

}  // namespace rtp_llm
