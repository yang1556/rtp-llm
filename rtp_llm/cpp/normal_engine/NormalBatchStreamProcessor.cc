#include <memory>
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"

namespace rtp_llm {

NormalBatchStreamProcessor::NormalBatchStreamProcessor(
    const ModelConfig&                 model_config,
    const PDSepConfig&                 pd_sep_config,
    const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
    const CacheConfig&                 cache_config,
    bool                               warm_up) {
    NormalModelInputGathererConfig model_gatherer_config;
    model_gatherer_config.input_vocab_size       = model_config.input_vocab_size;
    model_gatherer_config.vocab_size             = model_config.vocab_size;
    model_gatherer_config.num_layers             = model_config.num_layers;
    model_gatherer_config.position_id_len_factor = model_config.attn_config.rope_config.index_factor;
    model_gatherer_config.mm_position_ids_style =
        static_cast<PositionIdsStyle>(model_config.mm_model_config.mm_position_ids_style);
    model_gatherer_config.has_positional_encoding    = model_config.has_positional_encoding;
    model_gatherer_config.is_multimodal              = model_config.mm_model_config.is_multimodal;
    model_gatherer_config.role_type                  = pd_sep_config.role_type;
    model_gatherer_config.block_stride_bytes         = cache_config.kv_block_stride_bytes;
    model_gatherer_config.scale_stride_bytes         = cache_config.kv_scale_stride_bytes;
    model_gatherer_config.seq_size_per_block         = cache_config.seq_size_per_block;
    model_gatherer_config.kv_cache_group_nums        = cache_config.groupNums();
    model_gatherer_config.layer_to_kv_cache_group_id = cache_config.layer_to_group_id;
    model_gatherer_config.kv_cache_group_types       = cache_config.group_types;
    model_gatherer_config.decode_entrance            = pd_sep_config.decode_entrance;
    model_gatherer_config.warm_up                    = warm_up;
    model_gatherer_config.enable_detail_log          = profiling_debug_logging_config.enable_detail_log;
    model_gatherer_config.device                     = rtp_llm::DeviceFactory::getDefaultDevice();

    device_                 = model_gatherer_config.device;
    vocab_size_             = model_gatherer_config.vocab_size;
    model_input_gatherer_   = std::make_unique<NormalModelInputGatherer>(model_gatherer_config);
    sampler_input_gatherer_ = std::make_unique<NormalSamplerInputGatherer>(NormalSamplerInputGathererConfig{device_});
    output_dispatcher_      = std::make_unique<NormalOutputDispatcher>(NormalOutputDispatcherConfig{device_});
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups& stream_groups,
                                                  const MergedOutput& merge_outputs) const {
    return output_dispatcher_->dispatch(stream_groups, merge_outputs);
}

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    return model_input_gatherer_->gather(stream_groups);
}

absl::StatusOr<SamplerInputs>
NormalBatchStreamProcessor::gatherSamplerInput(const StreamGroups&    stream_groups,
                                               const GptModelOutputs& model_output) const {
    return sampler_input_gatherer_->gather(stream_groups, model_output);
}

SamplerInputs NormalBatchStreamProcessor::allocateSamplerInputs(const StreamGroups& stream_groups,
                                                                size_t              propose_step) const {
    return sampler_input_gatherer_->allocateSamplerInputs(stream_groups, propose_step);
}

void NormalBatchStreamProcessor::fillSamplerCommonInputs(SamplerInputs&                sampler_inputs,
                                                         std::list<GenerateStreamPtr>& all_streams,
                                                         bool                          score_batch,
                                                         size_t                        propose_step) const {
    sampler_input_gatherer_->fillSamplerCommonInputs(sampler_inputs, all_streams, score_batch, propose_step);
}

}  // namespace rtp_llm
