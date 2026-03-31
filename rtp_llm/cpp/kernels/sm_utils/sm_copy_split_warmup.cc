#include "rtp_llm/cpp/kernels/sm_utils/sm_copy_split_warmup.h"

#include "rtp_llm/cpp/kernels/sm_utils/sm_copy_kernel.h"

#include <cuda_runtime.h>

namespace rtp_llm {

void warmup_sm_copy_split_kernels_visible_cuda_devices() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
        return;
    }
    int prev_device = 0;
    (void)cudaGetDevice(&prev_device);
    for (int i = 0; i < n; ++i) {
        if (cudaSetDevice(i) != cudaSuccess) {
            continue;
        }
        cudaStream_t stream{};
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            continue;
        }
        (void)sDevMPS::warmup_sm_copy_split_kernels(stream);
        (void)cudaStreamSynchronize(stream);
        (void)cudaStreamDestroy(stream);
    }
    (void)cudaSetDevice(prev_device);
}

}  // namespace rtp_llm
