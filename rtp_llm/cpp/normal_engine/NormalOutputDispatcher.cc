#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"

namespace rtp_llm {

NormalOutputDispatcher::NormalOutputDispatcher(const NormalOutputDispatcherConfig& config): config_(config) {}

absl::Status NormalOutputDispatcher::dispatch(const StreamGroups& stream_groups,
                                              const MergedOutput& merge_outputs) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto& sampler_output    = merge_outputs.sampler_output;
    const auto& new_all_token_ids = sampler_output.token_ids;
    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", new_all_token_ids->debugStringWithData<int32_t>().c_str());
    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    RTP_LLM_CHECK(total_batch_size_out == new_all_token_ids->shape()[0]);
    int   batch_idx_in     = 0;
    int   batch_idx_out    = 0;
    int   token_offset     = 0;
    bool  return_all_probs = stream_groups.needReturnAllProbs();
    auto* device_          = config_.device;
    auto  new_tokens_all   = CACHED_HOST_BUF(TYPE_INT32, {(size_t)total_batch_size_out, (size_t)1});

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();

        dispatchSingleStream(
            stream, merge_outputs, batch_idx_in, batch_idx_out, token_offset, return_all_probs, new_tokens_all);

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
        token_offset += stream->currentExecuteTokenSize();
    }

    RTP_LLM_LOG_DEBUG("dispatch done");
    return absl::OkStatus();
}

void NormalOutputDispatcher::dispatchSingleStream(GenerateStreamPtr   stream,
                                                  const MergedOutput& merge_outputs,
                                                  int                 batch_idx_in,
                                                  int                 batch_idx_out,
                                                  int                 token_offset,
                                                  bool                return_all_probs,
                                                  const BufferPtr&    new_tokens_all) const {
    const auto&  model_output      = merge_outputs.model_output;
    const auto&  sampler_output    = merge_outputs.sampler_output;
    const auto&  new_all_token_ids = sampler_output.token_ids;
    const size_t token_stride      = new_all_token_ids->shape()[1];

    auto cur_batch_size  = stream->currentBatchSize();
    auto next_batch_size = stream->nextBatchSize();
    auto token_size      = stream->currentExecuteTokenSize();

    auto batch_new_all_token_ids = new_all_token_ids->slice(batch_idx_out, next_batch_size);

    bool has_beam_search = stream->currentNumBeams() > 1 || stream->nextNumBeams() > 1;
    bool has_var_batch   = stream->currentBatchSize() != stream->nextBatchSize();

    BufferPtr src_batch_indices;
    if (has_beam_search) {
        src_batch_indices = sampler_output.beam_index->slice(batch_idx_out, next_batch_size);
    } else if (has_var_batch) {
        src_batch_indices = config_.device->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {(size_t)next_batch_size}, rtp_llm::AllocationType::HOST}, {});
        config_.device->bufMemset(*src_batch_indices, 0);
    }
    const auto get_src_idx = [&](int32_t dst_idx) {
        return src_batch_indices ? src_batch_indices->data<int32_t>()[dst_idx] : dst_idx;
    };

    BufferPtr batch_hidden_states = nullptr;
    if (stream->generateConfig()->return_hidden_states) {
        batch_hidden_states = model_output.hidden_states->slice(batch_idx_in, cur_batch_size);
    }

    BufferPtr batch_logits = nullptr;
    if (stream->returnLogits() || stream->calculateSoftmaxProbs() || has_beam_search) {
        batch_logits = model_output.logits->slice(batch_idx_in, cur_batch_size);
    }

    BufferPtr all_probs = nullptr;
    if (return_all_probs) {
        all_probs = sampler_output.all_probs->slice(batch_idx_out, next_batch_size, false);
        all_probs->updateParent(sampler_output.all_probs);
    }

    BufferPtr batch_cum_log_probs;
    if (sampler_output.cum_log_probs) {
        batch_cum_log_probs = sampler_output.cum_log_probs->slice(batch_idx_out, next_batch_size);
    }

    BufferPtr loss;
    if (stream->calculateLoss()) {
        auto               all_logits = model_output.all_logits->view(token_offset, token_size - 1);
        auto               tokens     = stream->currentExecuteTokens(0);
        rtp_llm::BufferPtr label      = config_.device->clone(
            {{rtp_llm::MemoryType::MEMORY_CPU, rtp_llm::DataType::TYPE_INT32, {tokens.size() - 1}, tokens.data() + 1}});
        loss = config_.device->loss({all_logits, *label});
    }

    BufferPtr all_hidden_states = nullptr;
    if (stream->needReturnHiddenStates()) {
        all_hidden_states = model_output.all_hidden_states->slice(token_offset, token_size, false);
        all_hidden_states->updateParent(model_output.all_hidden_states);
    }

    BufferPtr new_tokens = new_tokens_all->slice(batch_idx_out, next_batch_size);
    for (size_t i = 0; i < next_batch_size; ++i) {
        new_tokens->data<int32_t>()[i] =
            new_all_token_ids->data<int32_t>()[(batch_idx_out + i) * token_stride + token_stride - 1];
    }

    BufferPtr batch_softmax_result;
    BufferPtr current_softmax_result;
    if (stream->calculateSoftmaxProbs()) {
        current_softmax_result = config_.device->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {(size_t)next_batch_size, (size_t)1}, rtp_llm::AllocationType::HOST}, {});
        batch_softmax_result = config_.device->softmax(
            {batch_logits, std::nullopt, std::nullopt, 1.0f, DataType::TYPE_FP32, std::nullopt});
        for (int i = 0; i < next_batch_size; ++i) {
            config_.device->copy({(*current_softmax_result)[i],
                                  (*batch_softmax_result)[get_src_idx(i)].view(new_tokens->data<int32_t>()[i], 1)});
        }
    }

    for (int i = 0; i < cur_batch_size; ++i) {
        if (sampler_output.success && !(*(sampler_output.success->dataWithOffset<bool>(batch_idx_in + i)))) {
            stream->setStop(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
        }
    }

    RTP_LLM_LOG_DEBUG(
        "stream [%ld], new_tokens = [%s]", stream->streamId(), new_tokens->debugStringWithData<int32_t>().c_str());

    stream->update({has_beam_search ? batch_new_all_token_ids : new_tokens,
                    1,
                    batch_hidden_states,
                    batch_logits,
                    current_softmax_result,
                    batch_cum_log_probs,
                    all_probs,
                    loss,
                    src_batch_indices,
                    all_hidden_states});
}

}  // namespace rtp_llm
