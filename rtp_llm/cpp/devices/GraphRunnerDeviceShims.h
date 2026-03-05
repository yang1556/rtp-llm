#pragma once

// Backend-agnostic graph runner device operations (declarations only).
// Implementations live in GraphRunnerDeviceShims.cc to avoid pulling
// ROCm/CUDA runtime headers (e.g. cuda_shims.h) into every TU that includes
// GraphBaseRunner.h, which would conflict with PyTorch includes.
//
// For stream/device type: use event_device_type() or GRAPH_DEVICE_TYPE macro.
// Do not add ATen HIP/CUDA includes here (use explicit at::hip::* vs at::cuda::*
// in .cc with #if USING_ROCM).

#include <functional>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace py = pybind11;

#if USING_ROCM
#define GRAPH_DEVICE_TYPE c10::DeviceType::HIP
#else
#define GRAPH_DEVICE_TYPE c10::DeviceType::CUDA
#endif

namespace rtp_llm {
namespace graph_runner {

// NCCL context for graph capture (ROCm uses it; CUDA ignores it).
struct GraphNcclCaptureContext {
    int64_t comm_handle{0};
    int     rank{0};
    int     world_size{1};
};

const char*     debug_file_prefix();
c10::DeviceType event_device_type();
void            memcpy_async(const torch::Tensor& src, torch::Tensor& dst, size_t size);
void            device_synchronize();
void            record_forward_event(torch::Event& event);
void            synchronize_forward_stream();
void            with_capture_stream(const std::function<void()>& fn);
bool            should_skip_decode_capture(py::object py_instance, bool is_prefill_mode);
void            before_capture_stream(py::object py_instance, int key, const char* key_type);
void            enter_graph_capture(GraphNcclCaptureContext* ctx);
void            exit_graph_capture(GraphNcclCaptureContext* ctx);
int             kv_block_cols(int max_seq_len, int seq_size_per_block);
torch::Tensor   sequence_lengths_plus_one_tensor(int max_bs, const at::TensorOptions& opts);

}  // namespace graph_runner
}  // namespace rtp_llm
