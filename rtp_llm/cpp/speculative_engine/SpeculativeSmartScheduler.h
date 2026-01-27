#pragma once

#include "rtp_llm/cpp/engine_base/schedulers/SmartScheduler.h"

namespace rtp_llm {

class SpeculativeSmartScheduler: virtual public SmartScheduler {
public:
    explicit SpeculativeSmartScheduler(const rtp_llm::GptInitParameter&     params,
                                       const std::shared_ptr<CacheManager>& cache_manager,
                                       const kmonitor::MetricsReporterPtr   metrics_reporter = nullptr,
                                       const int                            max_score_len    = 1):
        FIFOScheduler(params, cache_manager, metrics_reporter, max_score_len),
        SmartScheduler(params, cache_manager, metrics_reporter, max_score_len) {}

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule(size_t reserve_step = 0) override;
    bool                                         empty() override {
        return pending_sp_run_streams_.empty() && SmartScheduler::empty();
    };

private:
    std::list<GenerateStreamPtr> pending_sp_run_streams_;
};

}  // namespace rtp_llm
