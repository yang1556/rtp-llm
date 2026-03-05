#include "rtp_llm/cpp/devices/GraphRunnerDeviceShims.h"

#include <cstring>

#include "rtp_llm/cpp/devices/GraphStreamLife.h"
#include "rtp_llm/cpp/utils/Logger.h"

#if USING_ROCM
#include "rtp_llm/cpp/rocm/hip_capture_check.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#else
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

namespace rtp_llm {
namespace graph_runner {

const char* debug_file_prefix() {
#if USING_ROCM
    return "hip_graph_tokens";
#else
    return "cuda_graph_tokens";
#endif
}

c10::DeviceType event_device_type() {
#if USING_ROCM
    return c10::DeviceType::HIP;
#else
    return c10::DeviceType::CUDA;
#endif
}

void memcpy_async(const torch::Tensor& src, torch::Tensor& dst, size_t size) {
    if (!src.defined() || src.numel() <= 0) {
        return;
    }
#if USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();
    if (src.is_cuda() && dst.is_cuda()) {
        ROCM_CHECK(hipMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, hipMemcpyDeviceToDevice, stream));
    } else if (!src.is_cuda() && !dst.is_cuda()) {
        std::memcpy(dst.data_ptr(), src.data_ptr(), size);
    } else if (src.is_cuda() && !dst.is_cuda()) {
        ROCM_CHECK(hipMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, hipMemcpyDeviceToHost, stream));
    } else {
        ROCM_CHECK(hipMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, hipMemcpyHostToDevice, stream));
    }
#else
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (src.is_cuda() && dst.is_cuda()) {
        check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyDeviceToDevice, stream));
    } else if (!src.is_cuda() && !dst.is_cuda()) {
        std::memcpy(dst.data_ptr(), src.data_ptr(), size);
    } else if (src.is_cuda() && !dst.is_cuda()) {
        check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyDeviceToHost, stream));
    } else {
        check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyHostToDevice, stream));
    }
#endif
}

void device_synchronize() {
#if USING_ROCM
    ROCM_CHECK(hipDeviceSynchronize());
#else
    check_cuda_value(cudaDeviceSynchronize());
#endif
}

void record_forward_event(torch::Event& event) {
#if USING_ROCM
    event.record(at::hip::getCurrentHIPStream());
#else
    event.record(at::cuda::getCurrentCUDAStream());
#endif
}

void synchronize_forward_stream() {
#if USING_ROCM
    auto stream = at::hip::getCurrentHIPStream();
    ROCM_CHECK(hipStreamSynchronize(stream.stream()));
#else
    (void)0;
#endif
}

void with_capture_stream(const std::function<void()>& fn) {
#if USING_ROCM
    at::hip::HIPStream capture_stream = at::hip::getStreamFromPool(true);
    GraphStreamLife    life(capture_stream);
    fn();
#else
    at::cuda::CUDAStream capture_stream = at::cuda::getStreamFromPool(true);
    GraphStreamLife      life(capture_stream);
    fn();
#endif
}

bool should_skip_decode_capture(py::object py_instance, bool is_prefill_mode) {
#if USING_ROCM
    if (is_prefill_mode) {
        return false;
    }
    py::gil_scoped_acquire gil;
    bool                   has_kv_cache = true;
    if (py::hasattr(py_instance, "kv_cache")) {
        has_kv_cache = !py_instance.attr("kv_cache").is_none();
    }
    if (!has_kv_cache) {
        RTP_LLM_LOG_WARNING("HIP graph capture is enabled but kv_cache is not available. "
                            "Skipping decode graph capture for this instance.");
    }
    return !has_kv_cache;
#else
    (void)py_instance;
    (void)is_prefill_mode;
    return false;
#endif
}

void before_capture_stream(py::object py_instance, int key, const char* key_type) {
#if USING_ROCM
    (void)py_instance;
    py::gil_scoped_acquire gil;
    try {
        py::module_ torch_dist = py::module_::import("torch.distributed");
        if (torch_dist.attr("is_initialized")().cast<bool>()) {
            RTP_LLM_LOG_INFO("Executing torch.distributed.barrier() before graph capture for %s %d", key_type, key);
            torch_dist.attr("barrier")();
            RTP_LLM_LOG_INFO("torch.distributed.barrier() completed for %s %d", key_type, key);
        }
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING("Failed to execute torch.distributed.barrier(): %s", e.what());
    }
#else
    (void)py_instance;
    (void)key;
    (void)key_type;
#endif
}

void enter_graph_capture(GraphNcclCaptureContext* ctx) {
#if USING_ROCM
    rocm::CaptureCheck::in_hip_graph_capture = true;
    // Always notify collective_torch so _in_graph_capture is set and TP collectives
    // use direct RCCL (no torch.distributed watchdog). Comm is either passed here
    // or already registered via set_graph_capture_nccl_comm in PyWrappedModel (decoupled).
    try {
        py::module_ collective_torch = py::module_::import("rtp_llm.models_py.distributed.collective_torch");
        if (ctx && ctx->comm_handle != 0) {
            collective_torch.attr("enter_graph_capture_mode")(ctx->comm_handle, ctx->world_size, ctx->rank);
        } else {
            collective_torch.attr("enter_graph_capture_mode")(0, 0, 0);
        }
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING("Failed to enter graph capture mode: %s", e.what());
    }
#else
    (void)ctx;
    CaptureCheck::in_cuda_graph_capture = true;
#endif
}

void exit_graph_capture(GraphNcclCaptureContext* ctx) {
#if USING_ROCM
    rocm::CaptureCheck::in_hip_graph_capture = false;
    try {
        py::module_ collective_torch = py::module_::import("rtp_llm.models_py.distributed.collective_torch");
        collective_torch.attr("exit_graph_capture_mode")();
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING("Failed to exit graph capture mode: %s", e.what());
    }
#else
    (void)ctx;
    CaptureCheck::in_cuda_graph_capture = false;
#endif
}

int kv_block_cols(int max_seq_len, int seq_size_per_block) {
#if USING_ROCM
    return (max_seq_len + seq_size_per_block - 1) / seq_size_per_block;
#else
    return (max_seq_len + seq_size_per_block - 1) / seq_size_per_block + 1;
#endif
}

torch::Tensor sequence_lengths_plus_one_tensor(int max_bs, const at::TensorOptions& opts) {
#if USING_ROCM
    return torch::full({max_bs}, 2, opts);
#else
    return torch::zeros({max_bs}, opts);
#endif
}

}  // namespace graph_runner
}  // namespace rtp_llm
