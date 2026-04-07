#pragma once

#include "ATen/core/TensorBody.h"
#include <torch/version.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

class CaptureMemoryHold {
public:
    CaptureMemoryHold() {}

    CaptureMemoryHold(at::Tensor hidden_states, const torch_ext::PyModelInputs& inputs):
        all_layers_output_(std::move(hidden_states)), py_model_inputs_(inputs) {}

    void setHiddenStates(at::Tensor hidden_states) {
        all_layers_output_ = std::move(hidden_states);
    }

public:
    py::object               attn_pyobj_{py::none()};
    at::Tensor               all_layers_output_;
    torch_ext::PyModelInputs py_model_inputs_;
};

class GraphInstance {
public:
#if (TORCH_VERSION_MAJOR > 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 8)
    explicit GraphInstance(bool keep_graph = false): graph_(keep_graph) {}
#else
    explicit GraphInstance(bool keep_graph = false): graph_() {
        (void)keep_graph;
    }
#endif
    at::cuda::CUDAGraph graph_;
    CaptureMemoryHold   mem_hold_;
};

// RAII: temporarily set the device graph stream for capture, then restore the previous stream on scope exit.
class CudaGraphStreamGuard {
public:
    explicit CudaGraphStreamGuard(rtp_llm::cuda_graph::GraphStream capture_stream):
        origin_stream_(rtp_llm::cuda_graph::graphGetCurrentStream()) {
        rtp_llm::cuda_graph::graphSetCurrentStream(capture_stream);
        RTP_LLM_LOG_INFO("Set graph stream for capture. origin_stream=%p, capture_stream=%p",
                         reinterpret_cast<void*>(origin_stream_.stream()),
                         reinterpret_cast<void*>(capture_stream.stream()));
    }
    ~CudaGraphStreamGuard() {
        rtp_llm::cuda_graph::graphSetCurrentStream(origin_stream_);
        RTP_LLM_LOG_INFO("Restore graph stream after capture. restored_stream=%p",
                         reinterpret_cast<void*>(origin_stream_.stream()));
    }

    CudaGraphStreamGuard(const CudaGraphStreamGuard&)            = delete;
    CudaGraphStreamGuard& operator=(const CudaGraphStreamGuard&) = delete;
    CudaGraphStreamGuard(CudaGraphStreamGuard&&)                 = delete;
    CudaGraphStreamGuard& operator=(CudaGraphStreamGuard&&)      = delete;

private:
    rtp_llm::cuda_graph::GraphStream origin_stream_;
};

class CudaGraphCaptureGuard {
public:
    explicit CudaGraphCaptureGuard(rtp_llm::cuda_graph::GraphNcclCaptureContext* ctx = nullptr): ctx_(ctx) {
        rtp_llm::cuda_graph::enter_graph_capture(ctx_);
    }

    ~CudaGraphCaptureGuard() {
        rtp_llm::cuda_graph::exit_graph_capture(ctx_);
    }

    CudaGraphCaptureGuard(const CudaGraphCaptureGuard&)            = delete;
    CudaGraphCaptureGuard& operator=(const CudaGraphCaptureGuard&) = delete;
    CudaGraphCaptureGuard(CudaGraphCaptureGuard&&)                 = delete;
    CudaGraphCaptureGuard& operator=(CudaGraphCaptureGuard&&)      = delete;

private:
    rtp_llm::cuda_graph::GraphNcclCaptureContext* ctx_{nullptr};
};
