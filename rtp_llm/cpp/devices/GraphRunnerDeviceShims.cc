#include "rtp_llm/cpp/devices/GraphRunnerDeviceShims.h"

namespace rtp_llm {
namespace graph_runner {

#if USING_ROCM
py::module_& getCollectiveTorchModule() {
    RTP_LLM_CHECK_WITH_INFO(PyGILState_Check(), "getCollectiveTorchModule requires GIL to be held");
    static py::module_ collective_torch = py::module_::import("rtp_llm.models_py.distributed.collective_torch");
    return collective_torch;
}
#endif

void register_graph_capture_nccl_comm(void* nccl_comm, int world_size, int rank) {
#if USING_ROCM
    if (nccl_comm == nullptr || world_size <= 1) {
        try {
            py::module_& collective_torch = getCollectiveTorchModule();
            collective_torch.attr("set_hipgraph_capture_nccl_comm")(static_cast<uintptr_t>(0), 0, rank);
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_WARNING("Failed to clear NCCL comm for graph capture: %s", e.what());
        }
        return;
    }
    try {
        py::module_& collective_torch = getCollectiveTorchModule();
        collective_torch.attr("set_hipgraph_capture_nccl_comm")(
            reinterpret_cast<uintptr_t>(nccl_comm), world_size, rank);
        RTP_LLM_LOG_INFO("Registered NCCL comm for graph capture (rank=%d, world_size=%d)", rank, world_size);
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING("Failed to register NCCL comm: %s", e.what());
    }
#else
    (void)nccl_comm;
    (void)world_size;
    (void)rank;
#endif
}

void enter_graph_capture(GraphNcclCaptureContext* ctx) {
#if USING_ROCM
    rocm::setHipGraphCaptureEnabled(true);
    try {
        py::module_& collective_torch = getCollectiveTorchModule();
        if (ctx && ctx->comm_handle != 0) {
            collective_torch.attr("enter_hipgraph_capture_mode")(ctx->comm_handle, ctx->world_size, ctx->rank);
        } else {
            collective_torch.attr("enter_hipgraph_capture_mode")(0, 0, 0);
        }
    } catch (const py::error_already_set& e) {
        rocm::setHipGraphCaptureEnabled(false);
        const int rank = ctx ? ctx->rank : -1;
        try {
            py::module_& collective_torch = getCollectiveTorchModule();
            collective_torch.attr("set_hipgraph_capture_nccl_comm")(static_cast<uintptr_t>(0), 0, rank);
        } catch (const py::error_already_set& clear_e) {
            RTP_LLM_LOG_WARNING("Failed to clear NCCL comm after enter_graph_capture failure: %s", clear_e.what());
        }
        RTP_LLM_LOG_WARNING("Failed to enter graph capture mode: %s", e.what());
        throw;
    }
#else
    (void)ctx;
    CaptureCheck::in_cuda_graph_capture = true;
#endif
}

void exit_graph_capture(GraphNcclCaptureContext* ctx) {
#if USING_ROCM
    try {
        py::module_& collective_torch = getCollectiveTorchModule();
        collective_torch.attr("exit_hipgraph_capture_mode")();
    } catch (const py::error_already_set& e) {
        const unsigned long long comm_handle = ctx ? static_cast<unsigned long long>(ctx->comm_handle) : 0ULL;
        const int                rank        = ctx ? ctx->rank : -1;
        const int                world_size  = ctx ? ctx->world_size : -1;
        RTP_LLM_LOG_WARNING("Failed to exit graph capture mode (comm_handle=%llu, rank=%d, world_size=%d): %s",
                            comm_handle,
                            rank,
                            world_size,
                            e.what());
        try {
            py::module_& collective_torch = getCollectiveTorchModule();
            collective_torch.attr("set_hipgraph_capture_nccl_comm")(static_cast<uintptr_t>(0), 0, rank);
        } catch (const py::error_already_set& clear_e) {
            RTP_LLM_LOG_WARNING("Failed to clear NCCL comm after exit_graph_capture failure: %s", clear_e.what());
        }
    }
    rocm::setHipGraphCaptureEnabled(false);
#else
    (void)ctx;
    CaptureCheck::in_cuda_graph_capture = false;
#endif
}

}  // namespace graph_runner
}  // namespace rtp_llm
