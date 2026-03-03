#pragma once

#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"

namespace rtp_llm {

struct NormalOutputDispatcherConfig {
    rtp_llm::DeviceBase* device = nullptr;
};

class NormalOutputDispatcher {
public:
    explicit NormalOutputDispatcher(const NormalOutputDispatcherConfig& config);

    absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const;

private:
    void dispatchSingleStream(GenerateStreamPtr   stream,
                              const MergedOutput& merge_outputs,
                              int                 batch_idx_in,
                              int                 batch_idx_out,
                              int                 token_offset,
                              bool                return_all_probs,
                              const BufferPtr&    new_tokens_all) const;

    NormalOutputDispatcherConfig config_;
};

}  // namespace rtp_llm
