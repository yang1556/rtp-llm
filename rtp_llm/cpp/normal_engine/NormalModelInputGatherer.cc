#include <algorithm>
#include <cstring>
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/normal_engine/NormalModelInputGatherer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {

namespace {

enum class GatherContextMode {
    DECODE,
    CONTEXT
};

struct GatherModelInputContext {
    int                                    input_vocab_size;
    bool                                   need_cal_position_id;
    size_t                                 max_blocks_num;
    int*                                   merged_tokens;
    int*                                   input_lengths;
    int*                                   lora_ids;
    int*                                   lora_input_lengths;
    int*                                   lm_output_indexes;
    int*                                   lm_output_lengths;
    int*                                   combo_position_ids;
    BlockIdPair*                           kv_cache_update_mapping;
    int                                    batch_idx;
    int*                                   sequence_lengths;
    bool                                   has_multimodal_input;
    size_t                                 total_decode_batch_size;
    int*                                   prefix_lengths;
    int*                                   merged_text_mask;
    int*                                   mm_features_locs;
    int                                    token_idx;
    int                                    cum_output_seq_len;
    int                                    mm_feature_index;
    std::vector<BufferPtr>*                gathered_mm_features;
    BufferPtr                              kv_cache_block_id;
    BufferPtr                              cache_keys;
    int64_t*                               request_id;
    bool*                                  request_pd_separation;
    std::optional<std::vector<BufferPtr>>* multimodal_features;
};

GatherModelInputContext createGatherContext(const NormalModelInputGathererConfig& config,
                                            GptModelInputs&                       model_input,
                                            const StreamGroups&                   stream_groups,
                                            GatherContextMode                     mode,
                                            std::vector<BufferPtr>*               gathered_mm_features = nullptr) {
    GatherModelInputContext ctx{};
    ctx.input_vocab_size = config.input_vocab_size ? config.input_vocab_size : static_cast<int>(config.vocab_size);
    ctx.need_cal_position_id =
        (config.mm_position_ids_style != PositionIdsStyle::DEFAULT) || config.has_positional_encoding;
    ctx.max_blocks_num        = stream_groups.curBlocksNum();
    ctx.merged_tokens         = (int*)model_input.combo_tokens->data();
    ctx.input_lengths         = (int*)model_input.input_lengths->data();
    ctx.sequence_lengths      = (int*)model_input.sequence_lengths->data();
    ctx.lora_ids              = (int*)model_input.lora_ids->data();
    ctx.lora_input_lengths    = (int*)model_input.lora_input_lengths->data();
    ctx.lm_output_indexes     = (int*)model_input.lm_output_indexes->data();
    ctx.lm_output_lengths     = (int*)model_input.lm_output_lengths->data();
    ctx.combo_position_ids    = ctx.need_cal_position_id ? (int*)model_input.combo_position_ids->data() : nullptr;
    ctx.has_multimodal_input  = config.is_multimodal && stream_groups.has_multimodal_input();
    ctx.prefix_lengths        = (int*)model_input.prefix_lengths->data();
    ctx.merged_text_mask      = ctx.has_multimodal_input ? (int*)model_input.text_tokens_mask->data() : nullptr;
    ctx.mm_features_locs      = ctx.has_multimodal_input ? (int*)model_input.mm_features_locs->data() : nullptr;
    ctx.gathered_mm_features  = gathered_mm_features;
    ctx.kv_cache_block_id     = model_input.kv_cache_block_id;
    ctx.cache_keys            = model_input.cache_keys;
    ctx.request_id            = (int64_t*)model_input.request_id->data();
    ctx.request_pd_separation = (bool*)model_input.request_pd_separation->data();
    ctx.multimodal_features   = &model_input.multimodal_features;

    size_t kv_cache_mapping_offset = 0;
    if (mode == GatherContextMode::DECODE) {
        ctx.batch_idx = 0;
    } else {
        ctx.total_decode_batch_size = stream_groups.totalDecodeBatchSize();
        ctx.batch_idx               = static_cast<int>(ctx.total_decode_batch_size);
        ctx.token_idx               = ctx.batch_idx;
        ctx.cum_output_seq_len      = ctx.batch_idx;
        ctx.mm_feature_index        = 0;
        for (const auto& stream : stream_groups.decodeStreams()) {
            kv_cache_mapping_offset += stream->streamCacheResource().getKVBlockUpdateMapping().size();
        }
    }
    ctx.kv_cache_update_mapping =
        model_input.kv_cache_update_mapping ?
            (BlockIdPair*)model_input.kv_cache_update_mapping->data() + kv_cache_mapping_offset :
            nullptr;
    return ctx;
}

void copyKvCacheBlockIdsToModelInput(const rtp_llm::BufferPtr&   kv_cache_block_id,
                                     const BatchKVCacheResource& kv_cache,
                                     int                         stream_batch_idx,
                                     int                         model_batch_idx,
                                     size_t                      max_blocks_num) {
    if (!kv_cache_block_id || max_blocks_num == 0) {
        return;
    }
    for (int gid = 0; gid < kv_cache.groupNums(); ++gid) {
        auto& blocks = kv_cache.blocks(stream_batch_idx, gid);
        RTP_LLM_CHECK_WITH_INFO(kv_cache_block_id->shape().size() == 3, "hybrid kv_cache_block_id must be 3-D");
        const size_t batch    = kv_cache_block_id->shape()[1];
        int32_t*     dst_base = kv_cache_block_id->data<int32_t>();
        int32_t*     dst =
            dst_base + (static_cast<size_t>(gid) * batch + static_cast<size_t>(model_batch_idx)) * max_blocks_num;
        std::memcpy(dst, blocks.data(), blocks.size() * sizeof(int32_t));
    }
}

void gatherMultimodalFeaturesForContextBatch(const GenerateStreamPtr& stream,
                                             GatherModelInputContext& ctx,
                                             rtp_llm::DeviceBase*     device) {
    if (!ctx.has_multimodal_input) {
        return;
    }
    std::vector<torch::Tensor> mm_features = stream->multimodalFeatures();
    rtp_llm::BufferPtr         mm_locs     = stream->multimodalLocations();
    if (mm_locs == nullptr) {
        return;
    }
    for (int i = 0; i < mm_locs->size(); ++i) {
        ctx.mm_features_locs[ctx.mm_feature_index] =
            *mm_locs->dataWithOffset<int>(i) + ctx.token_idx - stream->reuseLength();
        ctx.mm_feature_index++;
    }
    for (auto& mm_feature : mm_features) {
        auto feature_buffer = torchTensor2Buffer(mm_feature);
        if (feature_buffer->where() != rtp_llm::MemoryType::MEMORY_GPU) {
            ctx.gathered_mm_features->emplace_back(device->clone({*feature_buffer}));
        } else {
            ctx.gathered_mm_features->emplace_back(feature_buffer);
        }
    }
    auto text_token_mask = stream->textTokensMask();
    memcpy(ctx.merged_text_mask + ctx.token_idx, text_token_mask.data(), text_token_mask.size() * sizeof(int));
}

}  // namespace

NormalModelInputGatherer::NormalModelInputGatherer(const NormalModelInputGathererConfig& config): config_(config) {}

GptModelInputs NormalModelInputGatherer::allocateModelInputBuffers(const StreamGroups& stream_groups) const {
    const size_t current_tokens_size      = stream_groups.modelExecuteTokenSize();
    const size_t total_batch_size         = stream_groups.totalModelBatchSize();
    const size_t total_decode_batch_size  = stream_groups.totalDecodeBatchSize();
    const size_t total_context_batch_size = stream_groups.totalContextBatchSize();
    const size_t total_block_copy_num     = stream_groups.totalBlockUpdateCopyNum();
    const size_t max_blocks_num           = stream_groups.curBlocksNum();
    const size_t multimodal_features_len  = stream_groups.mmFeaturesLen();
    const bool   has_multimodal_input     = config_.is_multimodal && stream_groups.has_multimodal_input();
    const bool   need_cal_position_id =
        (config_.mm_position_ids_style != PositionIdsStyle::DEFAULT) || config_.has_positional_encoding;

    auto*          device_ = config_.device;
    GptModelInputs model_input;
    model_input.combo_tokens      = CACHED_HOST_BUF(TYPE_INT32, {current_tokens_size});
    model_input.input_lengths     = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size});
    model_input.sequence_lengths  = CACHED_HOST_BUF(TYPE_INT32, {total_decode_batch_size});
    model_input.lm_output_indexes = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size});
    model_input.lm_output_lengths = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size});
    model_input.prefix_lengths    = CACHED_HOST_BUF(TYPE_INT32, {total_context_batch_size});
    if (need_cal_position_id) {
        model_input.combo_position_ids =
            CACHED_HOST_BUF(TYPE_INT32, {current_tokens_size * config_.position_id_len_factor});
    }
    model_input.lora_ids           = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size});
    model_input.lora_input_lengths = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size});
    if (max_blocks_num) {
        model_input.kv_cache_block_id = CACHED_HOST_BUF(
            TYPE_INT32, {static_cast<size_t>(config_.kv_cache_group_nums), total_batch_size, max_blocks_num});
        model_input.kv_cache_layer_to_group = CACHED_HOST_BUF(TYPE_INT32, {config_.num_layers});
        model_input.kv_cache_group_types =
            CACHED_HOST_BUF(TYPE_INT32, {static_cast<size_t>(config_.kv_cache_group_nums)});
        model_input.kv_cache_update_mapping = CACHED_HOST_BUF(TYPE_INT32, {total_block_copy_num, 2});
        model_input.cache_keys              = CACHED_HOST_BUF(TYPE_INT64, {total_context_batch_size, max_blocks_num});
    }
    if (has_multimodal_input) {
        model_input.text_tokens_mask = CACHED_HOST_BUF(TYPE_INT32, {current_tokens_size});
        std::fill((int*)model_input.text_tokens_mask->data(),
                  (int*)model_input.text_tokens_mask->data() + current_tokens_size,
                  1);
        model_input.mm_features_locs = CACHED_HOST_BUF(TYPE_INT32, {multimodal_features_len});
    }
    model_input.request_id            = CACHED_HOST_BUF(TYPE_INT64, {total_context_batch_size});
    model_input.request_pd_separation = CACHED_HOST_BUF(TYPE_BOOL, {total_context_batch_size});
    model_input.kv_block_stride_bytes = config_.block_stride_bytes;
    model_input.kv_scale_stride_bytes = config_.scale_stride_bytes;
    model_input.seq_size_per_block    = config_.seq_size_per_block;
    model_input.pd_separation         = config_.role_type == RoleType::PREFILL;
    model_input.decode_entrance       = config_.decode_entrance;
    model_input.warmup                = config_.warm_up;
    model_input.is_fake_stream        = stream_groups.isFakeStream();
    return model_input;
}

void NormalModelInputGatherer::initializeKvCacheMetadata(GptModelInputs& model_input) const {
    if (model_input.kv_cache_layer_to_group && !config_.layer_to_kv_cache_group_id.empty()) {
        std::memcpy(model_input.kv_cache_layer_to_group->data(),
                    config_.layer_to_kv_cache_group_id.data(),
                    config_.num_layers * sizeof(int32_t));
    }
    if (model_input.kv_cache_group_types) {
        auto* dst = model_input.kv_cache_group_types->data<int32_t>();
        for (size_t g = 0; g < config_.kv_cache_group_nums; ++g) {
            dst[g] = static_cast<int32_t>(config_.kv_cache_group_types[g]);
        }
    }
}

absl::Status NormalModelInputGatherer::processDecodeStreams(GptModelInputs&     model_input,
                                                            const StreamGroups& stream_groups) const {
    auto ctx            = createGatherContext(config_, model_input, stream_groups, GatherContextMode::DECODE);
    auto decode_streams = stream_groups.decodeStreams();

    const auto add_cache_update_copy = [&](const auto& update_mapping) {
        size_t update_copy_num = update_mapping.size();
        std::memcpy(ctx.kv_cache_update_mapping, update_mapping.data(), update_copy_num * sizeof(BlockIdPair));
        ctx.kv_cache_update_mapping += update_copy_num;
    };

    for (const auto& stream : decode_streams) {
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto  current_batch_size    = stream->currentBatchSize();
        auto& kv_cache              = *stream->kvCachePtr();
        RTP_LLM_LOG_DEBUG("decode stream: %s", stream->debugString().c_str());
        RTP_LLM_LOG_DEBUG("decode kv_cache: %s", kv_cache.debugString().c_str());

        for (auto i = 0; i < current_batch_size; ++i) {
            model_input.trace_ids.push_back(stream->traceId());
            auto currentTokens = stream->currentExecuteTokens(i);
            if (currentTokens[0] >= ctx.input_vocab_size) {
                std::ostringstream error_msg;
                error_msg << "stream [" << stream->streamId() << "] token_id " << currentTokens[0]
                          << " exceed vocab_size " << ctx.input_vocab_size;
                return absl::InvalidArgumentError(error_msg.str());
            }
            ctx.merged_tokens[ctx.batch_idx]    = currentTokens[0];
            ctx.input_lengths[ctx.batch_idx]    = stream->inputLength();
            ctx.sequence_lengths[ctx.batch_idx] = stream->seqLength() - 1;
            if (ctx.need_cal_position_id) {
                stream->generateNextPositionId(ctx.combo_position_ids + ctx.batch_idx * config_.position_id_len_factor,
                                               config_.device);
            }
            ctx.lora_ids[ctx.batch_idx]           = stream->loraId();
            ctx.lora_input_lengths[ctx.batch_idx] = 1;
            ctx.lm_output_indexes[ctx.batch_idx]  = ctx.batch_idx;
            ctx.lm_output_lengths[ctx.batch_idx]  = 1;
            copyKvCacheBlockIdsToModelInput(ctx.kv_cache_block_id, kv_cache, i, ctx.batch_idx, ctx.max_blocks_num);
            ctx.batch_idx += 1;
        }
        if (ctx.max_blocks_num) {
            add_cache_update_copy(stream->streamCacheResource().getKVBlockUpdateMapping());
        }
        stream->step();
    }
    return absl::OkStatus();
}

absl::Status NormalModelInputGatherer::processContextStreams(GptModelInputs&     model_input,
                                                             const StreamGroups& stream_groups) const {
    std::vector<rtp_llm::BufferPtr> gathered_mm_features;
    auto                            ctx =
        createGatherContext(config_, model_input, stream_groups, GatherContextMode::CONTEXT, &gathered_mm_features);
    auto context_streams = stream_groups.contextStreams();

    const auto add_cache_update_copy = [&](const auto& update_mapping) {
        size_t update_copy_num = update_mapping.size();
        std::memcpy(ctx.kv_cache_update_mapping, update_mapping.data(), update_copy_num * sizeof(BlockIdPair));
        ctx.kv_cache_update_mapping += update_copy_num;
    };

    for (const auto& stream : context_streams) {
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto  current_batch_size    = stream->currentBatchSize();
        auto& kv_cache              = *stream->kvCachePtr();
        if (config_.enable_detail_log) {
            RTP_LLM_LOG_DEBUG("context kv_cache: %s", kv_cache.debugString().c_str());
            RTP_LLM_LOG_DEBUG("context stream: %s", stream->debugString().c_str());
        } else {
            RTP_LLM_LOG_TRACE("context kv_cache: %s", kv_cache.debugString().c_str());
            RTP_LLM_LOG_TRACE("context stream: %s", stream->debugString().c_str());
        }

        for (auto i = 0; i < current_batch_size; ++i) {
            const auto prefill_batch_idx = ctx.batch_idx - ctx.total_decode_batch_size;
            model_input.trace_ids.push_back(stream->traceId());
            auto input_tokens = stream->currentExecuteTokens(i);
            auto input_masks  = stream->textTokensMask();
            memcpy(ctx.merged_tokens + ctx.token_idx, input_tokens.data(), input_tokens.size() * sizeof(int));
            ctx.cum_output_seq_len += input_tokens.size();

            for (int index = 0; index < input_tokens.size(); ++index) {
                if (input_tokens[index] >= ctx.input_vocab_size
                    && (index >= input_masks.size() || input_masks[index])) {
                    std::ostringstream error_msg;
                    error_msg << "stream [" << stream->streamId() << "] token_id " << input_tokens[index]
                              << " exceed vocab_size " << ctx.input_vocab_size;
                    return absl::InvalidArgumentError(error_msg.str());
                }
            }

            ctx.input_lengths[ctx.batch_idx]      = input_tokens.size();
            ctx.prefix_lengths[prefill_batch_idx] = stream->prefixLength();
            ctx.lm_output_indexes[ctx.batch_idx]  = ctx.cum_output_seq_len - 1;
            ctx.lm_output_lengths[ctx.batch_idx]  = 1;
            gatherMultimodalFeaturesForContextBatch(stream, ctx, config_.device);

            if (ctx.need_cal_position_id) {
                auto context_pos_ids = stream->generateContextPositionIds(config_.device);
                memcpy(ctx.combo_position_ids + ctx.token_idx * config_.position_id_len_factor,
                       context_pos_ids->dataWithOffset<int>(stream->reuseLength() * config_.position_id_len_factor),
                       (context_pos_ids->size() - stream->reuseLength() * config_.position_id_len_factor)
                           * context_pos_ids->typeSize());
            }
            ctx.lora_ids[ctx.batch_idx]           = stream->loraId();
            ctx.lora_input_lengths[ctx.batch_idx] = ctx.input_lengths[ctx.batch_idx];
            copyKvCacheBlockIdsToModelInput(ctx.kv_cache_block_id, kv_cache, i, ctx.batch_idx, ctx.max_blocks_num);
            if (ctx.max_blocks_num && config_.role_type == RoleType::PREFILL) {
                std::memcpy((*ctx.cache_keys)[prefill_batch_idx].data(),
                            stream->cacheKeys(i).data(),
                            stream->cacheKeys(i).size() * sizeof(int64_t));
            }
            ctx.request_id[prefill_batch_idx]            = stream->streamId();
            ctx.request_pd_separation[prefill_batch_idx] = stream->queryPdSep();
            ctx.batch_idx += 1;
            ctx.token_idx += input_tokens.size();
        }
        if (ctx.max_blocks_num) {
            add_cache_update_copy(stream->streamCacheResource().getKVBlockUpdateMapping());
        }
        stream->step();
    }
    if (config_.is_multimodal && gathered_mm_features.size() > 0) {
        *ctx.multimodal_features = std::move(gathered_mm_features);
    }
    return absl::OkStatus();
}

absl::StatusOr<GptModelInputs> NormalModelInputGatherer::gather(const StreamGroups& stream_groups) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_LOG_DEBUG("context_streams size = %d, decode_streams size = %d",
                      stream_groups.contextStreams().size(),
                      stream_groups.decodeStreams().size());
    auto model_input = allocateModelInputBuffers(stream_groups);
    initializeKvCacheMetadata(model_input);
    RETURN_IF_STATUS_ERROR(processDecodeStreams(model_input, stream_groups));
    RETURN_IF_STATUS_ERROR(processContextStreams(model_input, stream_groups));
    return model_input;
}

}  // namespace rtp_llm
