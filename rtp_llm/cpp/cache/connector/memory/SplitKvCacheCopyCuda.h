#pragma once

#include "rtp_llm/cpp/devices/OpData.h"
#include <memory>

namespace rtp_llm {

/// CUDA split KV H2D/D2H copy (scatter/gather) for merged host slabs → per-layer kv + kv_scale GPU buffers.
/// Implemented in the cache connector layer; uses sm_copy kernels (not DeviceOps).
class SplitKvCacheCopyCuda {
public:
    SplitKvCacheCopyCuda();
    ~SplitKvCacheCopyCuda();

    /// @param cuda_stream `cudaStream_t` as void* to keep this header CUDA-free for most TUs.
    void run(void* cuda_stream, const MultiCopyParams& params);

private:
    void ensureBuffers(size_t staging_bytes, size_t ptr_table_bytes, void* cuda_stream);

    void*  staging_{nullptr};
    void*  ptr0_{nullptr};
    void*  ptr1_{nullptr};
    size_t staging_cap_{0};
    size_t ptr_table_cap_{0};
    int    buffer_device_{-1};     // device on which staging_/ptr* were allocated
    void*  copy_stream_{nullptr};  // CudaDevice::noBlockCopyStream; used for cudaFreeAsync in dtor
};

}  // namespace rtp_llm
