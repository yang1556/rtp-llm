#pragma once

#include <list>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"

namespace rtp_llm {

struct NormalSamplerInputGathererConfig {
    rtp_llm::DeviceBase* device = nullptr;
};

class NormalSamplerInputGatherer {
public:
    explicit NormalSamplerInputGatherer(const NormalSamplerInputGathererConfig& config);

    absl::StatusOr<SamplerInputs> gather(const StreamGroups& stream_groups, const GptModelOutputs& model_output) const;

    SamplerInputs allocateSamplerInputs(const StreamGroups& stream_groups, size_t propose_step = 0) const;
    void          fillSamplerCommonInputs(SamplerInputs&                sampler_inputs,
                                          std::list<GenerateStreamPtr>& all_streams,
                                          bool                          score_batch  = false,
                                          size_t                        propose_step = 0) const;

private:
    void fillSamplerExtraInputs(SamplerInputs&         sampler_inputs,
                                const StreamGroups&    stream_groups,
                                const GptModelOutputs& model_output) const;
    void fillSamplerLogitsProcessor(SamplerInputs&                sampler_inputs,
                                    std::list<GenerateStreamPtr>& all_streams,
                                    bool                          score_batch = false) const;

    NormalSamplerInputGathererConfig config_;
};

}  // namespace rtp_llm
