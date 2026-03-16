#include "rtp_llm/cpp/engine_base/TorchProfiler.h"
#include <string>
#include "rtp_llm/cpp/config/StaticConfig.h"
namespace rtp_llm {
namespace tap = torch::autograd::profiler;

size_t TorchProfile::count = 0;

TorchProfile::TorchProfile(const std::string& prefix, std::string user_torch_cuda_profiler_dir): prefix_(prefix) {
    dest_dir_ = (user_torch_cuda_profiler_dir != "") ? user_torch_cuda_profiler_dir : ".";
    tap::prepareProfiler(config_, activities_);
}

TorchProfile::~TorchProfile() {
    if (!stoped_) {
        stoped_ = true;
        stop();
    }
}

void TorchProfile::start() {
    count += 1;
    stoped_ = false;
    tap::enableProfiler(config_, activities_);
}

void TorchProfile::stop() {
    std::unique_ptr<tap::ProfilerResult> res       = tap::disableProfiler();
    std::string                          file_name = dest_dir_ + "/" + prefix_ + std::to_string(count) + ".json";
    res->save(file_name);
    stoped_ = true;
}

}  // namespace rtp_llm
