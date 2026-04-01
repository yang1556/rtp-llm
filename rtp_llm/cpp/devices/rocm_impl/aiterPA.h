#pragma once
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
// #include "aiter_meta/csrc/include/attention.h"
#include "attention.h"
#include <pybind11/pybind11.h>
#include <Python.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace py = pybind11;

static inline void pa_fwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int           block_tables_stride0,
    int           max_qlen = 1,
    std::optional<torch::Tensor> K_QScale = std::nullopt,
    std::optional<torch::Tensor> V_QScale = std::nullopt,
    std::optional<torch::Tensor> out_ = std::nullopt,
    std::optional<torch::Tensor> qo_indptr = std::nullopt,
    int                          high_precision = 1,
    std::optional<std::string>   kernelName = std::nullopt) {
    throw std::runtime_error("pa_fwd symbol is unavailable in current aiter package");
}

namespace rtp_llm {

class AiterWrapper {
public:
    AiterWrapper(const DeviceInitParams& params);
    void runTritonPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_mtp, hipStream_t stream);
    void runHipPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_tmp, hipStream_t stream);
private:
    py::module_ pa_gluon_aot_api;
    py::module_ hip_pa_api;
    py::object  pa_gluon_load_libs;
    py::object  hip_pa_load_libs;
    bool        use_asm_pa_;
    bool        pa_python_available_ = false;
};

void runAiterAsmPA(const AttentionModuleParams& params,
                rtp_llm::DeviceBase* device, Buffer& q_tmp);
void runAiterPA(const AttentionModuleParams& params,
                rtp_llm::DeviceBase* device, Buffer& q_tmp);
}  // namespace rtp_llm
