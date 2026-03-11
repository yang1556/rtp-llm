#pragma once

#include "rtp_llm/cpp/devices/Weights.h"

namespace rtp_llm {

std::unique_ptr<const rtp_llm::Weights> loadWeightsFromDir(std::string dir_path);

}  // namespace rtp_llm
