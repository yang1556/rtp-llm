#include "rtp_llm/cpp/speculative_engine/SpeculativeSmartScheduler.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {

absl::StatusOr<std::list<GenerateStreamPtr>> SpeculativeSmartScheduler::schedule(size_t reserve_step) {
    if (!pending_sp_run_streams_.empty()) {
        std::list<GenerateStreamPtr> moved_pending_sp_run_streams;
        moved_pending_sp_run_streams.splice(moved_pending_sp_run_streams.end(), pending_sp_run_streams_);
        return moved_pending_sp_run_streams;
    }
    CHECK_AND_RETURN_REF(streams, SmartScheduler::schedule(reserve_step));
    if (streams.empty()) {
        return streams;
    }
    // printf("stream size:%ld\n",streams.size());
    if (streams.size() < 16) {
        for (auto& stream : streams) {
            if (stream->forceDisableSpRun()) {
                stream->setForceDisableSpRun(false);
                stream->setNeedConvertPropose(true);
                // printf("enale sp\n");
                // if (stream->getLastHiddenStates() != nullptr){
                //     printf("stream has lasthiddenstates\n");
                // }
            }
        }
    } else {
        for (auto& stream : streams) {
            stream->setForceDisableSpRun(true);
            // printf("disbale sp\n");
        }
    }
    std::list<GenerateStreamPtr> normal_run_streams;
    for (auto& stream : streams) {
        if (stream->disableSpRun()) {
            normal_run_streams.emplace_back(stream);
        } else {
            pending_sp_run_streams_.emplace_back(stream);
        }
    }
    return normal_run_streams;
}

}  // namespace rtp_llm
