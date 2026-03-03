#include <algorithm>
#include <cstring>
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/normal_engine/NormalSamplerInputGatherer.h"

namespace rtp_llm {

NormalSamplerInputGatherer::NormalSamplerInputGatherer(const NormalSamplerInputGathererConfig& config):
    config_(config) {}

absl::StatusOr<SamplerInputs> NormalSamplerInputGatherer::gather(const StreamGroups&    stream_groups,
                                                                 const GptModelOutputs& model_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_CHECK(!stream_groups.empty());
    auto all_streams = stream_groups.allStreams();

    SamplerInputs sampler_inputs = allocateSamplerInputs(stream_groups);
    fillSamplerCommonInputs(sampler_inputs, all_streams);
    fillSamplerExtraInputs(sampler_inputs, stream_groups, model_output);
    fillSamplerLogitsProcessor(sampler_inputs, all_streams);

    RTP_LLM_LOG_DEBUG("gatherSamplerInput done");
    return std::move(sampler_inputs);
}

SamplerInputs NormalSamplerInputGatherer::allocateSamplerInputs(const StreamGroups& stream_groups,
                                                                size_t              propose_step) const {
    auto          total_batch_size_in  = stream_groups.totalSamplerBatchSizeIn();
    auto          total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    auto*         device_              = config_.device;
    SamplerInputs sampler_inputs;
    sampler_inputs.step             = stream_groups.maxSeqLen() + propose_step;
    sampler_inputs.batch_size       = total_batch_size_in;
    sampler_inputs.batch_size_out   = total_batch_size_out;
    sampler_inputs.sequence_lengths = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size_in});
    sampler_inputs.logits_processor_states_ptr.reset();
    sampler_inputs.input_lengths        = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size_in});
    sampler_inputs.num_beams_in         = CACHED_HOST_BUF(TYPE_UINT64, {total_batch_size_in});
    sampler_inputs.num_beams_out        = CACHED_HOST_BUF(TYPE_UINT64, {total_batch_size_in});
    sampler_inputs.top_k                = CACHED_HOST_BUF(TYPE_UINT32, {total_batch_size_in});
    sampler_inputs.top_p                = CACHED_HOST_BUF(TYPE_FP32, {total_batch_size_in});
    sampler_inputs.temperature          = CACHED_HOST_BUF(TYPE_FP32, {total_batch_size_in});
    sampler_inputs.repetition_penalty   = CACHED_HOST_BUF(TYPE_FP32, {total_batch_size_in});
    sampler_inputs.presence_penalty     = CACHED_HOST_BUF(TYPE_FP32, {total_batch_size_in});
    sampler_inputs.frequency_penalty    = CACHED_HOST_BUF(TYPE_FP32, {total_batch_size_in});
    sampler_inputs.no_repeat_ngram_size = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size_in});
    sampler_inputs.do_sample            = CACHED_HOST_BUF(TYPE_BOOL, {total_batch_size_in});
    sampler_inputs.finished_mask        = CACHED_HOST_BUF(TYPE_BOOL, {total_batch_size_in});
    if (stream_groups.needReturnCumLogProbs()) {
        sampler_inputs.cum_log_probs = config_.device->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {total_batch_size_in}, rtp_llm::AllocationType::HOST}, {});
    }
    sampler_inputs.token_ids = config_.device->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {total_batch_size_in, sampler_inputs.step + 1}, rtp_llm::AllocationType::HOST},
        {});
    sampler_inputs.generator.resize(total_batch_size_in);
    return sampler_inputs;
}

void NormalSamplerInputGatherer::fillSamplerCommonInputs(SamplerInputs&                sampler_inputs,
                                                         std::list<GenerateStreamPtr>& all_streams,
                                                         bool                          score_batch,
                                                         size_t                        propose_step) const {
    int*      input_lengths        = sampler_inputs.input_lengths->data<int32_t>();
    int*      sequence_lengths     = sampler_inputs.sequence_lengths->data<int32_t>();
    uint64_t* num_beams_in         = sampler_inputs.num_beams_in->data<uint64_t>();
    uint64_t* num_beams_out        = sampler_inputs.num_beams_out->data<uint64_t>();
    uint32_t* top_k                = sampler_inputs.top_k->data<uint32_t>();
    float*    top_p                = sampler_inputs.top_p->data<float>();
    float*    temperature          = sampler_inputs.temperature->data<float>();
    float*    repetition_penalty   = sampler_inputs.repetition_penalty->data<float>();
    float*    presence_penalty     = sampler_inputs.presence_penalty->data<float>();
    float*    frequency_penalty    = sampler_inputs.frequency_penalty->data<float>();
    int32_t*  no_repeat_ngram_size = sampler_inputs.no_repeat_ngram_size->data<int32_t>();
    bool*     do_sample            = sampler_inputs.do_sample->data<bool>();

    int batch_idx = 0;
    for (auto& stream : all_streams) {
        int sampler_batch_size;
        if (score_batch) {
            sampler_batch_size = stream->scoreLen();
        } else if (stream->needTilingForSampling()) {
            sampler_batch_size = stream->nextBatchSize();
        } else {
            sampler_batch_size = stream->currentBatchSize();
        }
        if (sampler_inputs.cum_log_probs) {
            const auto& cum_log_probs = stream->cumLogProbs();
            memcpy(sampler_inputs.cum_log_probs->dataWithOffset<float>(batch_idx),
                   cum_log_probs->data(),
                   cum_log_probs->sizeBytes());
        }
        for (int i = 0; i < sampler_batch_size; ++i) {
            input_lengths[batch_idx]      = stream->inputLength();
            sequence_lengths[batch_idx]   = stream->seqLength() + propose_step;
            num_beams_in[batch_idx]       = stream->currentNumBeams();
            num_beams_out[batch_idx]      = stream->nextNumBeams();
            top_k[batch_idx]              = stream->generateConfig()->top_k;
            top_p[batch_idx]              = stream->generateConfig()->top_p;
            temperature[batch_idx]        = stream->generateConfig()->temperature;
            repetition_penalty[batch_idx] = stream->generateConfig()->repetition_penalty;
            presence_penalty[batch_idx]   = stream->generateConfig()->presence_penalty;
            frequency_penalty[batch_idx]  = stream->generateConfig()->frequency_penalty;
            do_sample[batch_idx]          = stream->generateConfig()->do_sample;
            if (!do_sample[batch_idx]) {
                top_k[batch_idx]       = 1;
                top_p[batch_idx]       = 1;
                temperature[batch_idx] = 1;
            }
            no_repeat_ngram_size[batch_idx]     = stream->generateConfig()->no_repeat_ngram_size.value_or(0);
            sampler_inputs.generator[batch_idx] = stream->getGenerator();
            batch_idx += 1;
        }
    }
}

void NormalSamplerInputGatherer::fillSamplerExtraInputs(SamplerInputs&         sampler_inputs,
                                                        const StreamGroups&    stream_groups,
                                                        const GptModelOutputs& model_output) const {
    auto   all_streams                = stream_groups.allStreams();
    auto   total_batch_size_in        = stream_groups.totalSamplerBatchSizeIn();
    bool   return_all_probs           = stream_groups.needReturnAllProbs();
    size_t total_decode_batch_size_in = 0;
    int    batch_idx                  = 0;
    bool   return_logits              = false;
    bool   calculate_softmax_probs    = false;
    bool   need_tiling                = false;

    for (auto& stream : all_streams) {
        const auto& complete_token_ids = stream->completeTokenIds();
        auto        complete_seq_len   = complete_token_ids->shape()[1];
        auto        seq_len            = stream->seqLength();
        auto        current_batch_size = stream->currentBatchSize();
        auto        sampler_batch_size =
            stream->needTilingForSampling() ? stream->nextBatchSize() : stream->currentBatchSize();

        for (int i = 0; i < sampler_batch_size; ++i) {
            int cur_batch = std::min(i, current_batch_size - 1);
            memcpy(sampler_inputs.token_ids->dataWithOffset<int32_t>((batch_idx) * (sampler_inputs.step + 1)),
                   complete_token_ids->dataWithOffset<int32_t>(cur_batch * complete_seq_len),
                   seq_len * sizeof(int));
            sampler_inputs.finished_mask->data<bool>()[batch_idx] = stream->isDoneWithoutLock(i);
            batch_idx += 1;
        }
        need_tiling |= stream->needTilingForSampling();
        if (!stream->isContextStream()) {
            total_decode_batch_size_in += sampler_batch_size;
        }
        return_logits |= stream->returnLogits();
        calculate_softmax_probs |= stream->calculateSoftmaxProbs();
        RTP_LLM_LOG_DEBUG("stream [%ld], complete token ids = [%s]",
                          stream->streamId(),
                          complete_token_ids->debugStringWithData<int32_t>(sampler_inputs.step).c_str());
        RTP_LLM_LOG_DEBUG("stream [%ld], sampler inputs token ids = [%s]",
                          stream->streamId(),
                          sampler_inputs.token_ids->debugStringWithData<int32_t>().c_str());
    }

    auto vocab_size           = model_output.logits->shape()[1];
    sampler_inputs.vocab_size = vocab_size;
    if (return_all_probs) {
        sampler_inputs.all_probs = config_.device->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {total_batch_size_in, vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
        config_.device->bufMemset(*sampler_inputs.all_probs, 0);
    }

    if (need_tiling) {
        sampler_inputs.logits = config_.device->allocateBuffer(
            {model_output.logits->type(), {total_batch_size_in, vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
        config_.device->copy({sampler_inputs.logits->view(0, total_decode_batch_size_in),
                              model_output.logits->view(0, total_decode_batch_size_in)});
        size_t input_offset = 0, logits_offset = 0;
        for (auto& stream : stream_groups.contextStreams()) {
            auto sampler_batch_size =
                stream->needTilingForSampling() ? stream->nextBatchSize() : stream->currentBatchSize();
            for (int i = 0; i < sampler_batch_size; ++i) {
                config_.device->copy(
                    {sampler_inputs.logits->view(input_offset, 1), model_output.logits->view(logits_offset, 1)});
                input_offset += 1;
            }
            logits_offset += 1;
        }
    } else if (return_logits || calculate_softmax_probs) {
        sampler_inputs.logits = config_.device->clone({*model_output.logits, rtp_llm::AllocationType::DEVICE});
    } else {
        sampler_inputs.logits = model_output.logits;
    }
}

void NormalSamplerInputGatherer::fillSamplerLogitsProcessor(SamplerInputs&                sampler_inputs,
                                                            std::list<GenerateStreamPtr>& all_streams,
                                                            bool                          score_batch) const {
    (void)score_batch;
    LogitsProcessorStatesPtr state_ptr = std::make_shared<LogitsProcessorStates>();
    std::for_each(all_streams.begin(), all_streams.end(), [&state_ptr, idx = 0](auto& stream) mutable {
        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            state_ptr->insert(processor, idx, idx + stream->currentBatchSize());
        }
        idx += stream->currentBatchSize();
    });
    sampler_inputs.logits_processor_states_ptr = state_ptr;
}

}  // namespace rtp_llm
