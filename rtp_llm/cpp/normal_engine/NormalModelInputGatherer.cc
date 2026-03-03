#include <algorithm>
#include <cstring>
#include <sstream>
#include "torch/all.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/normal_engine/NormalModelInputGatherer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {

NormalModelInputGatherer::NormalModelInputGatherer(const NormalModelInputGathererConfig& config): config_(config) {}

absl::StatusOr<GptModelInputs> NormalModelInputGatherer::gather(const StreamGroups& stream_groups) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    auto context_streams = stream_groups.contextStreams();
    auto decode_streams  = stream_groups.decodeStreams();
    RTP_LLM_LOG_DEBUG(
        "context_streams size = %d, decode_streams size = %d", context_streams.size(), decode_streams.size());
    GptModelInputs model_input;
    const size_t   current_tokens_size      = stream_groups.modelExecuteTokenSize();
    const size_t   total_batch_size         = stream_groups.totalModelBatchSize();
    const size_t   total_decode_batch_size  = stream_groups.totalDecodeBatchSize();
    const size_t   total_context_batch_size = stream_groups.totalContextBatchSize();
    const size_t   total_block_copy_num     = stream_groups.totalBlockUpdateCopyNum();
    const size_t   max_blocks_num           = stream_groups.curBlocksNum();
    const size_t   multimodal_features_len  = stream_groups.mmFeaturesLen();

    const bool has_multimodal_input = config_.is_multimodal && stream_groups.has_multimodal_input();
    const bool need_cal_position_id = (config_.mm_position_ids_style != PositionIdsStyle::DEFAULT) || config_.has_positional_encoding;

    size_t num_layers = 0;
    if (model_input.kv_cache_layer_to_group.defined()) {
        num_layers = model_input.kv_cache_layer_to_group.numel();
    } else {
        num_layers = config_.layer_to_kv_cache_group_id.size();
    }

    // Use pinned_memory(true) in TensorOptions to leverage PyTorch's CachingHostAllocator,
    // which reuses pinned memory blocks across calls instead of cudaHostAlloc/Free each time.
    static const auto pinned_i32  = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    static const auto pinned_i64  = torch::TensorOptions(torch::kInt64).pinned_memory(true);
    static const auto pinned_bool = torch::TensorOptions(torch::kBool).pinned_memory(true);

    model_input.combo_tokens = torch::empty({(int64_t)current_tokens_size}, pinned_i32);
    if (max_blocks_num) {
        model_input.kv_cache_kernel_block_id = torch::zeros({(int64_t)config_.kv_cache_group_nums,
                                                             (int64_t)total_batch_size,
                                                             (int64_t)(max_blocks_num * config_.kernel_blocks_per_kv_block)},
                                                            pinned_i32);
        model_input.kv_cache_block_id        = torch::zeros(
            {(int64_t)config_.kv_cache_group_nums, (int64_t)total_batch_size, (int64_t)max_blocks_num}, pinned_i32);
        model_input.kv_cache_layer_to_group = torch::empty({(int64_t)config_.num_layers}, pinned_i32);
        model_input.kv_cache_group_types    = torch::empty({(int64_t)config_.kv_cache_group_nums}, pinned_i32);
        model_input.kv_cache_update_mapping = torch::empty({(int64_t)total_block_copy_num, 2}, pinned_i32);
        model_input.cache_keys = torch::empty({(int64_t)total_context_batch_size, (int64_t)max_blocks_num}, pinned_i64);
    }
    model_input.request_id            = torch::empty({(int64_t)total_context_batch_size}, pinned_i64);
    model_input.request_pd_separation = torch::empty({(int64_t)total_context_batch_size}, pinned_bool);
    model_input.input_lengths         = torch::empty({(int64_t)total_batch_size}, pinned_i32);
    model_input.sequence_lengths      = torch::empty({(int64_t)total_decode_batch_size}, pinned_i32);
    model_input.lm_output_indexes     = torch::empty({(int64_t)total_batch_size}, pinned_i32);
    model_input.lm_output_lengths     = torch::empty({(int64_t)total_batch_size}, pinned_i32);
    model_input.prefix_lengths        = torch::empty({(int64_t)total_context_batch_size}, pinned_i32);
    if (need_cal_position_id) {
        model_input.combo_position_ids =
            torch::empty({(int64_t)(current_tokens_size * config_.position_id_len_factor)}, pinned_i32);
    }
    if (has_multimodal_input) {
        model_input.text_tokens_mask = torch::empty({(int64_t)current_tokens_size}, pinned_i32);
        model_input.mm_features_locs = torch::empty({(int64_t)multimodal_features_len}, pinned_i32);
    }
    model_input.kv_block_stride_bytes     = config_.block_stride_bytes;
    model_input.kv_scale_stride_bytes     = config_.scale_stride_bytes;
    model_input.seq_size_per_block        = config_.seq_size_per_block;
    model_input.kernel_seq_size_per_block = config_.kernel_seq_size_per_block;
    model_input.pd_separation             = config_.role_type == RoleType::PREFILL;
    model_input.warmup                    = config_.warm_up;
    model_input.decode_entrance           = config_.decode_entrance;
    model_input.is_fake_stream            = stream_groups.isFakeStream();

    int* merged_tokens      = model_input.combo_tokens.data_ptr<int32_t>();
    int* input_lengths      = model_input.input_lengths.data_ptr<int32_t>();
    int* sequence_lengths   = model_input.sequence_lengths.data_ptr<int32_t>();
    int* lm_output_indexes  = model_input.lm_output_indexes.data_ptr<int32_t>();
    int* lm_output_lengths  = model_input.lm_output_lengths.data_ptr<int32_t>();
    int* prefix_lengths     = model_input.prefix_lengths.data_ptr<int32_t>();
    int* combo_position_ids = need_cal_position_id ? model_input.combo_position_ids.data_ptr<int32_t>() : nullptr;
    int* merged_text_mask   = has_multimodal_input ? model_input.text_tokens_mask.data_ptr<int32_t>() : nullptr;
    int* mm_features_locs   = has_multimodal_input ? model_input.mm_features_locs.data_ptr<int32_t>() : nullptr;
    int  batch_idx          = 0;
    int  input_vocab_size   = config_.input_vocab_size ? config_.input_vocab_size : config_.vocab_size;

    if (model_input.kv_cache_layer_to_group.defined()) {
        std::memcpy(model_input.kv_cache_layer_to_group.data_ptr(),
                    config_.layer_to_kv_cache_group_id.data(),
                    static_cast<size_t>(num_layers) * sizeof(int32_t));
    }

    if (model_input.kv_cache_group_types.defined()) {
        auto* dst = model_input.kv_cache_group_types.data_ptr<int32_t>();
        for (size_t g = 0; g < config_.kv_cache_group_nums; ++g) {
            dst[g] = static_cast<int32_t>(config_.kv_cache_group_types[g]);
        }
    }

    auto*      kv_cache_update_mapping = model_input.kv_cache_update_mapping.defined() ?
                                             (BlockIdPair*)model_input.kv_cache_update_mapping.data_ptr() :
                                             nullptr;
    const auto add_cache_update_copy   = [&](const auto& update_mapping) {
        size_t update_copy_num = update_mapping.size();
        std::memcpy(kv_cache_update_mapping, update_mapping.data(), update_copy_num * sizeof(BlockIdPair));
        kv_cache_update_mapping += update_copy_num;
    };

    if (merged_text_mask) {
        std::fill(merged_text_mask, merged_text_mask + current_tokens_size, 1);
    }

    for (const auto& stream : decode_streams) {
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto current_batch_size     = stream->currentBatchSize();

        auto& kv_cache = *stream->kvCachePtr();
        RTP_LLM_LOG_DEBUG("decode kv_cache: %s", kv_cache.debugString().c_str());
        RTP_LLM_LOG_DEBUG("decode stream: %s", stream->debugString().c_str());

        for (auto i = 0; i < current_batch_size; ++i) {
            model_input.trace_ids.push_back(stream->traceId());

            auto currentTokens = stream->currentExecuteTokens(i);
            if (currentTokens[0] >= input_vocab_size) {
                std::ostringstream error_msg;
                error_msg << "stream [" << stream->streamId() << "] token_id " << currentTokens[0]
                          << " exceed vocab_size " << input_vocab_size;
                return absl::InvalidArgumentError(error_msg.str());
            }
            merged_tokens[batch_idx]    = currentTokens[0];
            input_lengths[batch_idx]    = stream->inputLength();
            sequence_lengths[batch_idx] = stream->seqLength() - 1;  // need remove
            if (need_cal_position_id) {
                stream->generateNextPositionId(combo_position_ids + batch_idx * config_.position_id_len_factor);
            }
            lm_output_indexes[batch_idx] = batch_idx;
            lm_output_lengths[batch_idx] = 1;
            if (max_blocks_num) {
                RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_kernel_block_id.dim() == 3,
                                        "hybrid kv_cache_kernel_block_id must be 3-D");
                RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_block_id.dim() == 3,
                                        "hybrid kv_cache_block_id must be 3-D");
                const size_t batch           = model_input.kv_cache_kernel_block_id.size(1);
                int32_t*     kernel_dst_base = model_input.kv_cache_kernel_block_id.data_ptr<int32_t>();
                int32_t*     store_dst_base  = model_input.kv_cache_block_id.data_ptr<int32_t>();
                for (int gid = 0; gid < kv_cache.groupNums(); ++gid) {
                    auto&    kernel_blocks = kv_cache.kernelBlocks(i, gid);
                    int32_t* kernel_dst    = kernel_dst_base
                                          + (static_cast<size_t>(gid) * batch + static_cast<size_t>(batch_idx))
                                                * max_blocks_num * config_.kernel_blocks_per_kv_block;
                    std::memcpy(kernel_dst, kernel_blocks.data(), kernel_blocks.size() * sizeof(int32_t));

                    auto&    physical_blocks = kv_cache.blocks(i, gid);
                    int32_t* store_dst =
                        store_dst_base
                        + (static_cast<size_t>(gid) * batch + static_cast<size_t>(batch_idx)) * max_blocks_num;
                    std::memcpy(store_dst, physical_blocks.data(), physical_blocks.size() * sizeof(int32_t));
                }
            }
            batch_idx += 1;
        }

        if (max_blocks_num) {
            add_cache_update_copy(stream->streamCacheResource().getKVBlockUpdateMapping());
        }

        stream->step();
    }

    std::vector<torch::Tensor> gathered_mm_features;
    int                        token_idx          = batch_idx;
    int                        cum_output_seq_len = batch_idx;
    int                        mm_feature_index   = 0;

    for (const auto& stream : context_streams) {
        // context stream也需要batch运行是为了perf test的场景
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto current_batch_size     = stream->currentBatchSize();

        auto& kv_cache = *stream->kvCachePtr();
        if (config_.enable_detail_log) {
            RTP_LLM_LOG_DEBUG("context kv_cache: %s", kv_cache.debugString().c_str());
            RTP_LLM_LOG_DEBUG("context stream: %s", stream->debugString().c_str());
        } else {
            RTP_LLM_LOG_TRACE("context kv_cache: %s", kv_cache.debugString().c_str());
            RTP_LLM_LOG_TRACE("context stream: %s", stream->debugString().c_str());
        }

        // TODO(xinfei.sxf) deal with adjusted common seq len.
        for (auto i = 0; i < current_batch_size; ++i) {
            model_input.trace_ids.push_back(stream->traceId());

            auto input_tokens = stream->currentExecuteTokens(i);
            auto input_masks  = stream->textTokensMask();
            memcpy(merged_tokens + token_idx, input_tokens.data(), input_tokens.size() * sizeof(int));
            cum_output_seq_len += input_tokens.size();

            for (int index = 0; index < input_tokens.size(); ++index) {
                if (input_tokens[index] >= input_vocab_size && (index >= input_masks.size() || input_masks[index])) {
                    std::ostringstream error_msg;
                    error_msg << "stream [" << stream->streamId() << "] token_id " << input_tokens[index]
                              << " exceed vocab_size " << input_vocab_size;
                    return absl::InvalidArgumentError(error_msg.str());
                }
            }

            input_lengths[batch_idx]                            = input_tokens.size();
            prefix_lengths[batch_idx - total_decode_batch_size] = stream->prefixLength();
            lm_output_indexes[batch_idx]                        = cum_output_seq_len - 1;
            lm_output_lengths[batch_idx]                        = 1;

            if (has_multimodal_input) {
                std::vector<torch::Tensor> mm_features = stream->multimodalFeatures();
                torch::Tensor              mm_locs     = stream->multimodalLocations();
                if (mm_locs.defined()) {
                    auto* mm_locs_data = mm_locs.data_ptr<int>();
                    for (int i = 0; i < mm_locs.numel(); ++i) {
                        mm_features_locs[mm_feature_index] = mm_locs_data[i] + token_idx - stream->reuseLength();
                        mm_feature_index++;
                    }
                    for (auto& mm_feature : mm_features) {
                        if (!mm_feature.is_cuda()) {
                            gathered_mm_features.emplace_back(mm_feature.to(torch::kCUDA));
                        } else {
                            gathered_mm_features.emplace_back(mm_feature);
                        }
                    }
                    auto text_token_mask = stream->textTokensMask();
                    memcpy(merged_text_mask + token_idx, text_token_mask.data(), text_token_mask.size() * sizeof(int));
                }
            }

            if (need_cal_position_id) {
                auto context_pos_ids = stream->generateContextPositionIds();
                int  reuse_offset    = stream->reuseLength() * config_.position_id_len_factor;
                memcpy(combo_position_ids + token_idx * config_.position_id_len_factor,
                       context_pos_ids.data_ptr<int>() + reuse_offset,
                       (context_pos_ids.numel() - reuse_offset) * sizeof(int));
            }
            if (max_blocks_num) {
                RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_kernel_block_id.dim() == 3,
                                        "hybrid kv_cache_kernel_block_id must be 3-D");
                RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_block_id.dim() == 3,
                                        "hybrid kv_cache_block_id must be 3-D");
                const size_t batch           = model_input.kv_cache_kernel_block_id.size(1);
                int32_t*     kernel_dst_base = model_input.kv_cache_kernel_block_id.data_ptr<int32_t>();
                int32_t*     store_dst_base  = model_input.kv_cache_block_id.data_ptr<int32_t>();
                for (int gid = 0; gid < kv_cache.groupNums(); ++gid) {
                    auto&    kernel_blocks = kv_cache.kernelBlocks(i, gid);
                    int32_t* kernel_dst    = kernel_dst_base
                                          + (static_cast<size_t>(gid) * batch + static_cast<size_t>(batch_idx))
                                                * max_blocks_num * config_.kernel_blocks_per_kv_block;
                    std::memcpy(kernel_dst, kernel_blocks.data(), kernel_blocks.size() * sizeof(int32_t));

                    auto&    physical_blocks = kv_cache.blocks(i, gid);
                    int32_t* store_dst =
                        store_dst_base
                        + (static_cast<size_t>(gid) * batch + static_cast<size_t>(batch_idx)) * max_blocks_num;
                    std::memcpy(store_dst, physical_blocks.data(), physical_blocks.size() * sizeof(int32_t));
                }
                if (config_.role_type == RoleType::PREFILL && stream->hasCacheKeys()) {
                    std::memcpy(model_input.cache_keys.data_ptr<int64_t>()
                                    + (batch_idx - total_decode_batch_size) * model_input.cache_keys.size(1),
                                stream->cacheKeys(i).data(),
                                stream->cacheKeys(i).size() * sizeof(int64_t));
                }
            }
            *(model_input.request_id.data_ptr<int64_t>() + (batch_idx - total_decode_batch_size)) = stream->streamId();
            *(reinterpret_cast<bool*>(model_input.request_pd_separation.data_ptr())
              + (batch_idx - total_decode_batch_size)) = stream->queryPdSep();
            batch_idx += 1;
            token_idx += input_tokens.size();
        }

        if (max_blocks_num) {
            add_cache_update_copy(stream->streamCacheResource().getKVBlockUpdateMapping());
        }

        stream->step();
    }

    if (config_.is_multimodal && gathered_mm_features.size() > 0) {
        model_input.multimodal_features = std::move(gathered_mm_features);
    }
    return model_input;
}

}  // namespace rtp_llm
