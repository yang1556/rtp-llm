#include "rtp_llm/cpp/models/ModelUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <cstring>
#include <numeric>

namespace rtp_llm {

BufferPtr sliceKvCacheBlockIdByBatch(const BufferPtr&     kv_cache_block_id,
                                     size_t               batch_offset,
                                     size_t               batch_size,
                                     rtp_llm::DeviceBase* device) {
    if (!kv_cache_block_id) {
        return nullptr;
    }
    const auto& shape = kv_cache_block_id->shape();
    if (shape.size() == 2) {
        return kv_cache_block_id->slice(batch_offset, batch_size);
    }
    if (shape.size() == 3) {
        const size_t group      = shape[0];
        const size_t batch      = shape[1];
        const size_t max_blocks = shape[2];
        RTP_LLM_CHECK_WITH_INFO(batch_offset + batch_size <= batch,
                                "sliceKvCacheBlockIdByBatch out of range: offset=%zu size=%zu batch=%zu",
                                batch_offset,
                                batch_size,
                                batch);
        auto           out          = device->allocateBuffer(BufferParams(
            rtp_llm::DataType::TYPE_INT32, {group, batch_size, max_blocks}, rtp_llm::AllocationType::HOST));
        const int32_t* src_base     = kv_cache_block_id->data<int32_t>();
        int32_t*       dst_base     = out->data<int32_t>();
        const size_t   src_stride_g = batch * max_blocks;
        const size_t   dst_stride_g = batch_size * max_blocks;
        for (size_t g = 0; g < group; ++g) {
            const int32_t* src = src_base + g * src_stride_g + batch_offset * max_blocks;
            int32_t*       dst = dst_base + g * dst_stride_g;
            std::memcpy(dst, src, dst_stride_g * sizeof(int32_t));
        }
        return out;
    }
    return kv_cache_block_id;
}

BufferPtr tpSyncEmbeddingOrLogits(rtp_llm::DeviceBase*             device,
                                  const rtp_llm::DeviceProperties& device_props,
                                  const BufferPtr&                 buffer) {
    const auto tp_size      = device_props.tp_size;
    const auto tp_rank      = device_props.tp_rank;
    const auto buffer_shape = buffer->shape();
    const auto local_size   = buffer->size();
    auto all_data = device->allocateBuffer(BufferParams(buffer->type(), {buffer_shape[0], buffer_shape[1] * tp_size}));
    auto buffer_view = buffer->reshape({buffer->size()});
    auto all_data_1d = all_data->reshape({all_data->size()});
    device->copy({all_data_1d.view(local_size * tp_rank, local_size), buffer_view});
    device->allGather({{all_data}});
    device->checkError();
    auto ret = device->transpose({all_data->reshape({tp_size, buffer_shape[0], buffer_shape[1]})});
    device->checkError();
    ret->updateShape({buffer_shape[0], buffer_shape[1] * tp_size});
    return ret;
}

void holdInputsHostBuffers(ModelBufferHolder& buffer_holder, const GptModelInputs& inputs) {
    buffer_holder.hold_host(inputs.combo_tokens);
    buffer_holder.hold_host(inputs.input_lengths);
    buffer_holder.hold_host(inputs.sequence_lengths);
    buffer_holder.hold_host(inputs.lm_output_indexes);
    buffer_holder.hold_host(inputs.lm_output_lengths);
    buffer_holder.hold_host(inputs.prefix_lengths);
    buffer_holder.hold_host(inputs.combo_position_ids);
    buffer_holder.hold_host(inputs.combo_tokens_type_ids);
    buffer_holder.hold_host(inputs.last_hidden_states);
    buffer_holder.hold_host(inputs.lora_ids);
    buffer_holder.hold_host(inputs.lora_input_lengths);
    buffer_holder.hold_host(inputs.attention_mask);
    buffer_holder.hold_host(inputs.kv_cache_block_id);
    buffer_holder.hold_host(inputs.kv_cache_layer_to_group);
    buffer_holder.hold_host(inputs.kv_cache_group_types);
    buffer_holder.hold_host(inputs.kv_cache_update_mapping);
    if (inputs.multimodal_features.has_value()) {
        for (auto& mm_feature : inputs.multimodal_features.value()) {
            buffer_holder.hold_host(mm_feature);
        }
    }
    buffer_holder.hold_host(inputs.text_tokens_mask);
    buffer_holder.hold_host(inputs.mm_features_locs);
    if (inputs.input_embeddings.has_value()) {
        for (auto& input_embedding : inputs.input_embeddings.value()) {
            buffer_holder.hold_host(input_embedding);
        }
    }
    buffer_holder.hold_host(inputs.input_embeddings_locs);
    buffer_holder.hold_host(inputs.request_id);
    buffer_holder.hold_host(inputs.request_pd_separation);
    buffer_holder.hold_host(inputs.cache_keys);
}

MicroBatchPlan
planMicroBatches(const GptModelInputs& inputs, size_t layer_num, const rtp_llm::DeviceProperties& device_props) {
    if (!int(device_props.enable_layer_micro_batch)) {
        RTP_LLM_LOG_DEBUG("micro batch disable when enable_layer_micro_batch is false");
        return {false, {}};
    }
    const auto& input_lengths      = inputs.input_lengths;
    const auto& sequence_lengths   = inputs.sequence_lengths;
    const auto  decoder_batch_size = sequence_lengths->shape()[0];
    const auto  context_batch_size = input_lengths->shape()[0] - decoder_batch_size;

    if (decoder_batch_size + context_batch_size < 2) {
        RTP_LLM_LOG_DEBUG("micro batch disable when batch size %ld is less than 2",
                          decoder_batch_size + context_batch_size);
        return {false, {}};
    }

    if (context_batch_size && decoder_batch_size) {
        if (layer_num == 1) {
            size_t total_token_num = decoder_batch_size;
            for (size_t i = 0; i < context_batch_size; i++) {
                total_token_num += input_lengths->data<int32_t>()[i + decoder_batch_size];
            }
            size_t context_batch_0_size = 0;
            size_t context_batch_1_size = 0;
            size_t decode_batch_0_size  = 0;
            size_t decode_batch_1_size  = 0;
            if (total_token_num > decoder_batch_size * 2) {
                decode_batch_0_size        = decoder_batch_size;
                decode_batch_1_size        = 0;
                size_t acc_token_num       = decoder_batch_size;
                size_t context_split_point = 0;
                for (context_split_point = 0; context_split_point < context_batch_size; context_split_point++) {
                    acc_token_num += input_lengths->data<int32_t>()[context_split_point + decoder_batch_size];
                    if (acc_token_num * 2 >= total_token_num) {
                        break;
                    }
                }
                context_batch_0_size = context_split_point;
                context_batch_1_size = context_batch_size - context_split_point;
            } else {
                decode_batch_0_size  = total_token_num / 2;
                decode_batch_1_size  = decoder_batch_size - total_token_num / 2;
                context_batch_0_size = 0;
                context_batch_1_size = context_batch_size;
            }
            return MicroBatchPlan{
                true, {{context_batch_0_size, decode_batch_0_size}, {context_batch_1_size, decode_batch_1_size}}};
        } else {
            RTP_LLM_LOG_DEBUG("split context in micro batch 0, decode in micro batch 1 disabled!");
            return {false, {}};
        }
    }

    const auto batch_size_to_split = context_batch_size ? context_batch_size : decoder_batch_size;
    const auto micro_batch_0_size  = (batch_size_to_split + 1) / 2;
    const auto micro_batch_1_size  = batch_size_to_split - micro_batch_0_size;
    return context_batch_size ? MicroBatchPlan{true, {{micro_batch_0_size, 0}, {micro_batch_1_size, 0}}} :
                                MicroBatchPlan{true, {{0, micro_batch_0_size}, {0, micro_batch_1_size}}};
}

std::pair<std::vector<GptModelInputs>, std::vector<TokenSliceInfo>> splitInputsIntoMicroBatches(
    const GptModelInputs& inputs, const MicroBatchPlan& micro_batch_plan, rtp_llm::DeviceBase* device) {
    std::vector<GptModelInputs> micro_batch_inputs;
    std::vector<TokenSliceInfo> token_slice_recipes;
    size_t                      sliced_token_idx       = 0;
    size_t                      sliced_lm_output_index = 0;
    size_t                      sliced_batch_idx       = 0;
    size_t                      decode_batch_idx       = 0;
    size_t                      prefill_batch_idx      = 0;

    if (!micro_batch_plan.enable) {
        micro_batch_inputs.push_back(inputs);
        GptModelInputs fake_inputs;
        fake_inputs.kv_cache_block_id = nullptr;
        fake_inputs.combo_tokens      = inputs.combo_tokens->slice(0, 1);
        fake_inputs.input_lengths =
            device->allocateBuffer(BufferParams(DataType::TYPE_INT32, {1}, AllocationType::HOST));
        fake_inputs.input_lengths->data<int32_t>()[0] = 1;
        fake_inputs.sequence_lengths =
            device->allocateBuffer(BufferParams(DataType::TYPE_INT32, {0}, AllocationType::HOST));
        fake_inputs.prefix_lengths =
            device->allocateBuffer(BufferParams(DataType::TYPE_INT32, {1}, AllocationType::HOST));
        fake_inputs.prefix_lengths->data<int32_t>()[0] = 0;
        micro_batch_inputs.push_back(fake_inputs);
    } else {
        for (size_t i = 0; i < micro_batch_plan.batch_infos.size(); ++i) {
            const auto& p_micro_batch_size = micro_batch_plan.batch_infos[i].prefill_num;
            const auto& d_micro_batch_size = micro_batch_plan.batch_infos[i].decoder_num;

            if (d_micro_batch_size && p_micro_batch_size) {
                GptModelInputs micro_model_inputs = inputs;
                size_t         total_batch_size   = d_micro_batch_size + p_micro_batch_size;
                micro_model_inputs.input_lengths  = inputs.input_lengths->slice(sliced_batch_idx, total_batch_size);
                micro_model_inputs.sequence_lengths =
                    inputs.sequence_lengths->slice(decode_batch_idx, d_micro_batch_size);
                micro_model_inputs.kv_cache_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_block_id, sliced_batch_idx, total_batch_size, device);
                micro_model_inputs.prefix_lengths = inputs.prefix_lengths->slice(prefill_batch_idx, p_micro_batch_size);
                micro_model_inputs.attention_mask =
                    inputs.attention_mask ? inputs.attention_mask->slice(sliced_batch_idx, total_batch_size) : nullptr;
                micro_model_inputs.lm_output_lengths =
                    inputs.lm_output_lengths->slice(sliced_batch_idx, total_batch_size);
                int32_t slice_token_num =
                    std::accumulate(micro_model_inputs.input_lengths->data<int32_t>() + d_micro_batch_size,
                                    micro_model_inputs.input_lengths->data<int32_t>() + total_batch_size,
                                    0)
                    + d_micro_batch_size;
                int32_t slice_lm_output_num =
                    std::accumulate(micro_model_inputs.lm_output_lengths->data<int32_t>(),
                                    micro_model_inputs.lm_output_lengths->data<int32_t>() + total_batch_size,
                                    0);
                micro_model_inputs.lm_output_indexes =
                    inputs.lm_output_indexes->slice(sliced_lm_output_index, slice_lm_output_num);
                micro_model_inputs.combo_tokens = inputs.combo_tokens->slice(sliced_token_idx, slice_token_num);
                micro_model_inputs.request_id =
                    inputs.request_id ? inputs.request_id->slice(prefill_batch_idx, p_micro_batch_size) : nullptr;
                micro_model_inputs.request_pd_separation =
                    inputs.request_pd_separation ?
                        inputs.request_pd_separation->slice(prefill_batch_idx, p_micro_batch_size) :
                        nullptr;
                micro_model_inputs.cache_keys =
                    inputs.cache_keys ? inputs.cache_keys->slice(prefill_batch_idx, p_micro_batch_size) : nullptr;
                token_slice_recipes.emplace_back(TokenSliceInfo{sliced_token_idx, (size_t)slice_token_num});
                micro_batch_inputs.push_back(micro_model_inputs);
                sliced_lm_output_index += slice_lm_output_num;
                sliced_token_idx += slice_token_num;
                sliced_batch_idx += total_batch_size;
                prefill_batch_idx += p_micro_batch_size;
                decode_batch_idx += d_micro_batch_size;
            } else if (d_micro_batch_size) {
                GptModelInputs micro_model_inputs = inputs;
                micro_model_inputs.combo_tokens   = inputs.combo_tokens->slice(sliced_token_idx, d_micro_batch_size);
                micro_model_inputs.input_lengths  = inputs.input_lengths->slice(sliced_batch_idx, d_micro_batch_size);
                micro_model_inputs.sequence_lengths =
                    inputs.sequence_lengths->slice(decode_batch_idx, d_micro_batch_size);
                micro_model_inputs.attention_mask =
                    inputs.attention_mask ? inputs.attention_mask->slice(sliced_batch_idx, d_micro_batch_size) :
                                            nullptr;
                micro_model_inputs.kv_cache_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_block_id, sliced_batch_idx, d_micro_batch_size, device);
                micro_model_inputs.prefix_lengths =
                    device->allocateBuffer(BufferParams(DataType::TYPE_INT32, {0}, AllocationType::HOST), {});
                micro_model_inputs.lm_output_indexes =
                    inputs.lm_output_indexes->slice(sliced_batch_idx, d_micro_batch_size);
                token_slice_recipes.emplace_back(TokenSliceInfo{sliced_token_idx, d_micro_batch_size});
                micro_batch_inputs.push_back(micro_model_inputs);
                sliced_token_idx += d_micro_batch_size;
                sliced_batch_idx += d_micro_batch_size;
                decode_batch_idx += d_micro_batch_size;
                sliced_lm_output_index += d_micro_batch_size;
            } else {
                GptModelInputs micro_model_inputs = inputs;
                micro_model_inputs.input_lengths  = inputs.input_lengths->slice(sliced_batch_idx, p_micro_batch_size);
                micro_model_inputs.kv_cache_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_block_id, sliced_batch_idx, p_micro_batch_size, device);
                micro_model_inputs.prefix_lengths = inputs.prefix_lengths->slice(prefill_batch_idx, p_micro_batch_size);
                micro_model_inputs.attention_mask =
                    inputs.attention_mask ? inputs.attention_mask->slice(sliced_batch_idx, p_micro_batch_size) :
                                            nullptr;
                micro_model_inputs.sequence_lengths =
                    device->allocateBuffer(BufferParams(DataType::TYPE_INT32, {0}, AllocationType::HOST), {});
                micro_model_inputs.lm_output_lengths =
                    inputs.lm_output_lengths->slice(sliced_batch_idx, p_micro_batch_size);
                int32_t slice_token_num =
                    std::accumulate(micro_model_inputs.input_lengths->data<int32_t>(),
                                    micro_model_inputs.input_lengths->data<int32_t>() + p_micro_batch_size,
                                    0);
                int32_t slice_lm_output_num =
                    std::accumulate(micro_model_inputs.lm_output_lengths->data<int32_t>(),
                                    micro_model_inputs.lm_output_lengths->data<int32_t>() + p_micro_batch_size,
                                    0);
                micro_model_inputs.lm_output_indexes =
                    inputs.lm_output_indexes->slice(sliced_lm_output_index, slice_lm_output_num);
                micro_model_inputs.combo_tokens = inputs.combo_tokens->slice(sliced_token_idx, slice_token_num);
                micro_model_inputs.request_id =
                    inputs.request_id ? inputs.request_id->slice(prefill_batch_idx, p_micro_batch_size) : nullptr;
                micro_model_inputs.request_pd_separation =
                    inputs.request_pd_separation ?
                        inputs.request_pd_separation->slice(prefill_batch_idx, p_micro_batch_size) :
                        nullptr;
                micro_model_inputs.cache_keys =
                    inputs.cache_keys ? inputs.cache_keys->slice(prefill_batch_idx, p_micro_batch_size) : nullptr;
                token_slice_recipes.emplace_back(TokenSliceInfo{sliced_token_idx, (size_t)slice_token_num});
                micro_batch_inputs.push_back(micro_model_inputs);
                sliced_lm_output_index += slice_lm_output_num;
                sliced_token_idx += slice_token_num;
                sliced_batch_idx += p_micro_batch_size;
                prefill_batch_idx += p_micro_batch_size;
            }
        }
    }
    return {micro_batch_inputs, token_slice_recipes};
}

void tpSyncModelInputs(GptModelInputs& inputs, rtp_llm::DeviceBase* device) {
    if (device->getDeviceProperties().tp_size <= 1) {
        return;
    }

    const size_t shape_hints_size = GptModelInputIndex::gptModelInputLength;
    auto         shape_hints      = device->allocateBuffer(
        BufferParams(rtp_llm::DataType::TYPE_INT32, {shape_hints_size}, rtp_llm::AllocationType::HOST));
    auto shape_hints_ptr                              = shape_hints->data<int32_t>();
    shape_hints_ptr[GptModelInputIndex::comboTokens]  = inputs.combo_tokens.get() ? inputs.combo_tokens->size() : 0;
    shape_hints_ptr[GptModelInputIndex::inputLengths] = inputs.input_lengths.get() ? inputs.input_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::sequenceLengths] =
        inputs.sequence_lengths.get() ? inputs.sequence_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::prefixLengths] =
        inputs.prefix_lengths.get() ? inputs.prefix_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::maxBlocksPerBatch] =
        inputs.kv_cache_block_id.get() ? inputs.kv_cache_block_id->shape()[2] : 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheGroupNum] =
        inputs.kv_cache_block_id.get() ? inputs.kv_cache_block_id->shape()[0] : 1;
    shape_hints_ptr[GptModelInputIndex::kvCacheLayerToGroupLen] =
        inputs.kv_cache_layer_to_group.get() ? inputs.kv_cache_layer_to_group->size() : 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheGroupTypesLen] =
        inputs.kv_cache_group_types.get() ? inputs.kv_cache_group_types->size() : 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheUpdateCopyNum] =
        inputs.kv_cache_update_mapping.get() ? inputs.kv_cache_update_mapping->shape()[0] : 0;
    shape_hints_ptr[GptModelInputIndex::lmOutputIndexes] =
        inputs.lm_output_indexes.get() ? inputs.lm_output_indexes->size() : 0;
    shape_hints_ptr[GptModelInputIndex::lmOutputLengthes] =
        inputs.lm_output_lengths.get() ? inputs.lm_output_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::comboPositionIds] =
        inputs.combo_position_ids.get() ? inputs.combo_position_ids->size() : 0;
    shape_hints_ptr[GptModelInputIndex::loraIds] = inputs.lora_ids.get() ? inputs.lora_ids->size() : 0;
    shape_hints_ptr[GptModelInputIndex::loraInputLengths] =
        inputs.lora_input_lengths.get() ? inputs.lora_input_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::textTokensMask] =
        inputs.text_tokens_mask.get() ? inputs.text_tokens_mask->size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesLocs] =
        inputs.mm_features_locs.get() ? inputs.mm_features_locs->size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] =
        inputs.multimodal_features.has_value() ? inputs.multimodal_features.value().size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesSize] =
        shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] ? inputs.multimodal_features.value()[0]->shape()[1] : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesDtype] =
        shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] ?
            (std::uint8_t)inputs.multimodal_features.value()[0]->type() :
            0;
    shape_hints_ptr[GptModelInputIndex::needAllLogits] = inputs.need_all_logits;
    shape_hints_ptr[GptModelInputIndex::mtpHiddenStates] =
        inputs.last_hidden_states.get() ? inputs.last_hidden_states->size() : 0;
    shape_hints_ptr[GptModelInputIndex::mtpHiddenStatesDtype] =
        shape_hints_ptr[GptModelInputIndex::mtpHiddenStates] ? (std::uint8_t)inputs.last_hidden_states->type() : 0;
    shape_hints_ptr[GptModelInputIndex::skipRun] = inputs.skip_run;
    shape_hints_ptr[GptModelInputIndex::gptModelRequestLength] =
        inputs.request_id.get() ? inputs.request_id->size() : 0;
    shape_hints_ptr[GptModelInputIndex::isFakeStream] = inputs.is_fake_stream;
    device->broadcast({{shape_hints}, 0});
    device->syncCommunication(false);
    device->syncAndCheck();

    rtp_llm::BufferPtr mm_features_shape;
    int32_t*           mm_features_shape_ptr = nullptr;
    inputs.need_all_logits                   = shape_hints_ptr[GptModelInputIndex::needAllLogits];
    inputs.skip_run                          = shape_hints_ptr[GptModelInputIndex::skipRun];
    inputs.is_fake_stream                    = shape_hints_ptr[GptModelInputIndex::isFakeStream];
    if (inputs.skip_run) {
        return;
    }
    const size_t mm_features_num = shape_hints_ptr[GptModelInputIndex::mmFeaturesNum];
    if (mm_features_num) {
        mm_features_shape =
            device->allocateBuffer(BufferParams(rtp_llm::DataType::TYPE_INT32,
                                                {(size_t)shape_hints_ptr[GptModelInputIndex::mmFeaturesNum]},
                                                rtp_llm::AllocationType::HOST));
        mm_features_shape_ptr = mm_features_shape->data<int32_t>();
        for (auto i = 0; i < mm_features_num; ++i) {
            mm_features_shape_ptr[i] =
                inputs.multimodal_features.has_value() ? inputs.multimodal_features.value()[i]->shape()[0] : 0;
        }
        device->broadcast({{mm_features_shape}, 0});
        device->syncCommunication(false);
        device->syncAndCheck();
    }

    auto   max_blocks              = (size_t)shape_hints_ptr[GptModelInputIndex::maxBlocksPerBatch];
    auto   kv_cache_group_num      = (size_t)shape_hints_ptr[GptModelInputIndex::kvCacheGroupNum];
    auto   layer_to_group_len      = (size_t)shape_hints_ptr[GptModelInputIndex::kvCacheLayerToGroupLen];
    auto   group_types_len         = (size_t)shape_hints_ptr[GptModelInputIndex::kvCacheGroupTypesLen];
    auto   combo_position_ids_size = shape_hints_ptr[GptModelInputIndex::comboPositionIds];
    auto   text_tokens_mask_size   = shape_hints_ptr[GptModelInputIndex::textTokensMask];
    auto   mm_features_locs_size   = shape_hints_ptr[GptModelInputIndex::mmFeaturesLocs];
    auto   hidden_states_size      = shape_hints_ptr[GptModelInputIndex::mtpHiddenStates];
    size_t request_length          = shape_hints_ptr[GptModelInputIndex::gptModelRequestLength];

    if (device->getDeviceProperties().tp_rank) {
        auto context_batch_size = (size_t)shape_hints_ptr[GptModelInputIndex::prefixLengths];

        inputs.combo_tokens =
            device->allocateBuffer(BufferParams(rtp_llm::DataType::TYPE_INT32,
                                                {(size_t)shape_hints_ptr[GptModelInputIndex::comboTokens]},
                                                rtp_llm::AllocationType::HOST));
        inputs.input_lengths =
            device->allocateBuffer(BufferParams(rtp_llm::DataType::TYPE_INT32,
                                                {(size_t)shape_hints_ptr[GptModelInputIndex::inputLengths]},
                                                rtp_llm::AllocationType::HOST));
        inputs.sequence_lengths =
            device->allocateBuffer(BufferParams(rtp_llm::DataType::TYPE_INT32,
                                                {(size_t)shape_hints_ptr[GptModelInputIndex::sequenceLengths]},
                                                rtp_llm::AllocationType::HOST));
        inputs.prefix_lengths = device->allocateBuffer(
            BufferParams(rtp_llm::DataType::TYPE_INT32, {context_batch_size}, rtp_llm::AllocationType::HOST));
        if (max_blocks != 0) {
            inputs.kv_cache_block_id = device->allocateBuffer(BufferParams(
                rtp_llm::DataType::TYPE_INT32,
                std::vector<size_t>{
                    kv_cache_group_num, (size_t)shape_hints_ptr[GptModelInputIndex::inputLengths], max_blocks},
                rtp_llm::AllocationType::HOST));
            if (inputs.pd_separation) {
                inputs.cache_keys = device->allocateBuffer(BufferParams(
                    rtp_llm::DataType::TYPE_INT64, {context_batch_size, max_blocks}, rtp_llm::AllocationType::HOST));
            }
            inputs.kv_cache_update_mapping = device->allocateBuffer(
                BufferParams(rtp_llm::DataType::TYPE_INT32,
                             {(size_t)shape_hints_ptr[GptModelInputIndex::kvCacheUpdateCopyNum], 2},
                             rtp_llm::AllocationType::HOST));
        }
        if (layer_to_group_len) {
            inputs.kv_cache_layer_to_group = device->allocateBuffer(
                BufferParams(rtp_llm::DataType::TYPE_INT32, {layer_to_group_len}, rtp_llm::AllocationType::HOST));
        }
        if (group_types_len) {
            inputs.kv_cache_group_types = device->allocateBuffer(
                BufferParams(rtp_llm::DataType::TYPE_INT32, {group_types_len}, rtp_llm::AllocationType::HOST));
        }
        inputs.request_id = device->allocateBuffer(
            BufferParams(rtp_llm::DataType::TYPE_INT64, {request_length}, rtp_llm::AllocationType::HOST));
        inputs.request_pd_separation = device->allocateBuffer(
            BufferParams(rtp_llm::DataType::TYPE_BOOL, {request_length}, rtp_llm::AllocationType::HOST));
        inputs.lm_output_indexes =
            device->allocateBuffer(BufferParams(rtp_llm::DataType::TYPE_INT32,
                                                {(size_t)shape_hints_ptr[GptModelInputIndex::lmOutputIndexes]},
                                                rtp_llm::AllocationType::HOST));
        inputs.lm_output_lengths =
            device->allocateBuffer(BufferParams(rtp_llm::DataType::TYPE_INT32,
                                                {(size_t)shape_hints_ptr[GptModelInputIndex::lmOutputLengthes]},
                                                rtp_llm::AllocationType::HOST));
        if (combo_position_ids_size) {
            inputs.combo_position_ids = device->allocateBuffer(BufferParams(
                rtp_llm::DataType::TYPE_INT32, {(size_t)combo_position_ids_size}, rtp_llm::AllocationType::HOST));
        }
        if (shape_hints_ptr[GptModelInputIndex::loraIds]) {
            inputs.lora_ids =
                device->allocateBuffer(BufferParams(rtp_llm::DataType::TYPE_INT32,
                                                    {(size_t)shape_hints_ptr[GptModelInputIndex::loraIds]},
                                                    rtp_llm::AllocationType::HOST));
        }
        if (shape_hints_ptr[GptModelInputIndex::loraInputLengths]) {
            inputs.lora_input_lengths =
                device->allocateBuffer(BufferParams(rtp_llm::DataType::TYPE_INT32,
                                                    {(size_t)shape_hints_ptr[GptModelInputIndex::loraInputLengths]},
                                                    rtp_llm::AllocationType::HOST));
        }
        if (shape_hints_ptr[GptModelInputIndex::mtpHiddenStates]) {
            auto hidden_states_dim0 = (size_t)shape_hints_ptr[GptModelInputIndex::comboTokens];
            auto hidden_states_dim1 = (size_t)hidden_states_size / hidden_states_dim0;
            RTP_LLM_CHECK(hidden_states_size % hidden_states_dim0 == 0);
            inputs.last_hidden_states = device->allocateBuffer(
                BufferParams((rtp_llm::DataType)shape_hints_ptr[GptModelInputIndex::mtpHiddenStatesDtype],
                             {hidden_states_dim0, hidden_states_dim1},
                             rtp_llm::AllocationType::DEVICE));
        }
        if (text_tokens_mask_size) {
            inputs.text_tokens_mask = device->allocateBuffer(BufferParams(
                rtp_llm::DataType::TYPE_INT32, {(size_t)text_tokens_mask_size}, rtp_llm::AllocationType::HOST));
        }
        if (mm_features_locs_size) {
            inputs.mm_features_locs = device->allocateBuffer(BufferParams(
                rtp_llm::DataType::TYPE_INT32, {(size_t)mm_features_locs_size}, rtp_llm::AllocationType::HOST));
        }
        if (mm_features_num) {
            std::vector<rtp_llm::BufferPtr> mm_features;
            for (auto mm_index = 0; mm_index < mm_features_num; ++mm_index) {
                mm_features.emplace_back(device->allocateBuffer(
                    BufferParams((rtp_llm::DataType)shape_hints_ptr[GptModelInputIndex::mmFeaturesDtype],
                                 {(size_t)mm_features_shape_ptr[mm_index],
                                  (size_t)shape_hints_ptr[GptModelInputIndex::mmFeaturesSize]},
                                 rtp_llm::AllocationType::DEVICE)));
            }
            inputs.multimodal_features = std::move(mm_features);
        }
    }

    std::vector<rtp_llm::BufferPtr> buffers;
    buffers.emplace_back(inputs.combo_tokens);
    buffers.emplace_back(inputs.input_lengths);
    buffers.emplace_back(inputs.sequence_lengths);
    buffers.emplace_back(inputs.prefix_lengths);
    if (max_blocks) {
        buffers.emplace_back(inputs.kv_cache_block_id);
        if (inputs.kv_cache_layer_to_group) {
            buffers.emplace_back(inputs.kv_cache_layer_to_group);
        }
        if (inputs.kv_cache_group_types) {
            buffers.emplace_back(inputs.kv_cache_group_types);
        }
        if (inputs.pd_separation) {
            buffers.emplace_back(inputs.cache_keys);
        }
        buffers.emplace_back(inputs.kv_cache_update_mapping);
    }
    buffers.emplace_back(inputs.request_id);
    buffers.emplace_back(inputs.request_pd_separation);
    buffers.emplace_back(inputs.lm_output_indexes);
    buffers.emplace_back(inputs.lm_output_lengths);
    if (combo_position_ids_size) {
        buffers.emplace_back(inputs.combo_position_ids);
    }
    buffers.emplace_back(inputs.lora_ids);
    buffers.emplace_back(inputs.lora_input_lengths);
    if (text_tokens_mask_size) {
        buffers.emplace_back(inputs.text_tokens_mask);
    }
    if (mm_features_locs_size) {
        buffers.emplace_back(inputs.mm_features_locs);
    }
    if (mm_features_num) {
        for (auto& mm_feature : inputs.multimodal_features.value()) {
            buffers.emplace_back(mm_feature);
        }
    }
    if (hidden_states_size) {
        buffers.emplace_back(inputs.last_hidden_states);
    }
    device->broadcast({buffers, 0});
    device->syncAndCheck();
}

}  // namespace rtp_llm
