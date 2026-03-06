#include "rtp_llm/cpp/models/ModelUtils.h"
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
        auto out = device->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {group, batch_size, max_blocks}, rtp_llm::AllocationType::HOST});
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
    auto       all_data     = device->allocateBuffer({buffer->type(), {buffer_shape[0], buffer_shape[1] * tp_size}});
    auto       buffer_view  = buffer->reshape({buffer->size()});
    auto       all_data_1d  = all_data->reshape({all_data->size()});
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
        fake_inputs.input_lengths     = device->allocateBuffer({DataType::TYPE_INT32, {1}, AllocationType::HOST});
        fake_inputs.input_lengths->data<int32_t>()[0] = 1;
        fake_inputs.sequence_lengths = device->allocateBuffer({DataType::TYPE_INT32, {0}, AllocationType::HOST});
        fake_inputs.prefix_lengths   = device->allocateBuffer({DataType::TYPE_INT32, {1}, AllocationType::HOST});
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
                    device->allocateBuffer({DataType::TYPE_INT32, {0}, AllocationType::HOST}, {});
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
                    device->allocateBuffer({DataType::TYPE_INT32, {0}, AllocationType::HOST}, {});
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

}  // namespace rtp_llm
