#pragma once

#include "torch/csrc/autograd/profiler_kineto.h"

namespace rtp_llm {

namespace tpi = torch::profiler::impl;

class TorchProfile {
public:
    TorchProfile(const std::string& prefix, std::string user_torch_cuda_profiler_dir = "");
    ~TorchProfile();
    void start();
    void stop();

protected:
    std::string                 dest_dir_;
    static size_t               count;
    std::string                 prefix_;
    tpi::ProfilerConfig         config_ = tpi::ProfilerConfig(tpi::ProfilerState::KINETO);
    std::set<tpi::ActivityType> activities_{tpi::ActivityType::CPU, tpi::ActivityType::CUDA};
    bool                        stoped_ = true;
};

}  // namespace rtp_llm