#include "rtp_llm/cpp/cache/connector/memory/SplitKvCacheCopyCuda.h"

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/kernels/sm_utils/sm_copy_kernel.h"

#include <cuda_runtime.h>
#include <vector>

namespace rtp_llm {

SplitKvCacheCopyCuda::SplitKvCacheCopyCuda() = default;
SplitKvCacheCopyCuda::~SplitKvCacheCopyCuda() {
    if (buffer_device_ < 0) {
        return;
    }
    check_cuda_value(cudaSetDevice(buffer_device_));
    if (copy_stream_) {
        auto stream = static_cast<cudaStream_t>(copy_stream_);
        check_cuda_value(cudaStreamSynchronize(stream));
        if (staging_) {
            check_cuda_value(cudaFreeAsync(staging_, stream));
            staging_ = nullptr;
        }
        if (ptr0_) {
            check_cuda_value(cudaFreeAsync(ptr0_, stream));
            ptr0_ = nullptr;
        }
        if (ptr1_) {
            check_cuda_value(cudaFreeAsync(ptr1_, stream));
            ptr1_ = nullptr;
        }
        check_cuda_value(cudaStreamSynchronize(stream));
    } else {
        if (staging_) {
            check_cuda_value(cudaFree(staging_));
            staging_ = nullptr;
        }
        if (ptr0_) {
            check_cuda_value(cudaFree(ptr0_));
            ptr0_ = nullptr;
        }
        if (ptr1_) {
            check_cuda_value(cudaFree(ptr1_));
            ptr1_ = nullptr;
        }
    }
    staging_cap_   = 0;
    ptr_table_cap_ = 0;
}

void SplitKvCacheCopyCuda::ensureBuffers(size_t staging_bytes, size_t ptr_table_bytes, void* cuda_stream) {
    auto      stream = static_cast<cudaStream_t>(cuda_stream);
    const int dev    = buffer_device_;
    RUNTIME_ASSERT_OP_ARG(dev >= 0, "split KV copy: buffer device not set before ensureBuffers");
    check_cuda_value(cudaSetDevice(dev));
    if (staging_bytes > staging_cap_) {
        if (staging_) {
            check_cuda_value(cudaStreamSynchronize(stream));
            check_cuda_value(cudaFreeAsync(staging_, stream));
            check_cuda_value(cudaStreamSynchronize(stream));
            staging_     = nullptr;
            staging_cap_ = 0;
        }
        check_cuda_value(cudaMalloc(&staging_, staging_bytes));
        staging_cap_ = staging_bytes;
    }
    if (ptr_table_bytes > ptr_table_cap_) {
        if (ptr0_ || ptr1_) {
            check_cuda_value(cudaStreamSynchronize(stream));
            if (ptr0_) {
                check_cuda_value(cudaFreeAsync(ptr0_, stream));
                ptr0_ = nullptr;
            }
            if (ptr1_) {
                check_cuda_value(cudaFreeAsync(ptr1_, stream));
                ptr1_ = nullptr;
            }
            check_cuda_value(cudaStreamSynchronize(stream));
            ptr_table_cap_ = 0;
        }
        check_cuda_value(cudaMalloc(&ptr0_, ptr_table_bytes));
        check_cuda_value(cudaMalloc(&ptr1_, ptr_table_bytes));
        ptr_table_cap_ = ptr_table_bytes;
    }
}

void SplitKvCacheCopyCuda::run(void* cuda_stream, const MultiCopyParams& params) {
    copy_stream_ = cuda_stream;
    auto stream  = static_cast<cudaStream_t>(cuda_stream);
    RUNTIME_ASSERT_OP_ARG(params.multi_src.size() == params.multi_dst.size(),
                          "multi_src and multi_dst must have the same size");
    if (params.multi_src.empty()) {
        check_cuda_error();
        return;
    }
    RUNTIME_ASSERT_OP_ARG(params.multi_src.size() % params.batch_size == 0,
                          "split KV copy: multi_src size not divisible by batch_size");
    const size_t actual_batch_size = params.batch_size;
    const size_t block_nums        = params.multi_src.size() / actual_batch_size;
    const size_t scatter_count     = actual_batch_size / 2;
    RUNTIME_ASSERT_OP_ARG(params.kv_cache_size > 0 && params.kv_scale_size > 0,
                          "split KV copy: kv_cache_size and kv_scale_size must be set");
    RUNTIME_ASSERT_OP_ARG(params.block_size == scatter_count * (params.kv_cache_size + params.kv_scale_size),
                          "split KV copy: block_size mismatch");
    const bool src_is_host = params.multi_src[0]
                             && (params.multi_src[0]->where() == MemoryType::MEMORY_CPU
                                 || params.multi_src[0]->where() == MemoryType::MEMORY_CPU_PINNED);
    const bool dst_is_gpu  = params.multi_dst[0] && params.multi_dst[0]->where() == MemoryType::MEMORY_GPU;
    const bool src_is_gpu  = params.multi_src[0] && params.multi_src[0]->where() == MemoryType::MEMORY_GPU;
    const bool dst_is_host = params.multi_dst[0]
                             && (params.multi_dst[0]->where() == MemoryType::MEMORY_CPU
                                 || params.multi_dst[0]->where() == MemoryType::MEMORY_CPU_PINNED);

    cudaPointerAttributes attr{};
    if (src_is_host && dst_is_gpu) {
        check_cuda_value(cudaPointerGetAttributes(&attr, params.multi_dst[0]->data()));
    } else if (src_is_gpu && dst_is_host) {
        check_cuda_value(cudaPointerGetAttributes(&attr, params.multi_src[0]->data()));
    } else {
        RTP_LLM_FAIL("split KV copy: unsupported layout (only split H2D or D2H host<->GPU)");
    }
    RUNTIME_ASSERT_OP_ARG(attr.type == cudaMemoryTypeDevice, "split KV copy: expected device pointer");
    const int ptr_device = attr.device;
    check_cuda_value(cudaSetDevice(ptr_device));
    buffer_device_ = ptr_device;

    const size_t ptr_table_bytes = scatter_count * sizeof(void*);
    RUNTIME_ASSERT_OP_ARG(ptr_table_bytes > 0, "split KV copy: ptr_table_bytes is 0");
    ensureBuffers(params.block_size, ptr_table_bytes, cuda_stream);

    if (src_is_host && dst_is_gpu) {
        void* const        d_dst_kv_cache_ptrs = ptr0_;
        void* const        d_dst_kv_scale_ptrs = ptr1_;
        void* const        d_src_staging       = staging_;
        std::vector<void*> h_dst_kv_cache(scatter_count), h_dst_kv_scale(scatter_count);
        for (size_t b = 0; b < block_nums; ++b) {
            const size_t base = b * actual_batch_size;
            check_cuda_value(cudaMemcpyAsync(
                d_src_staging, params.multi_src[base]->data(), params.block_size, cudaMemcpyHostToDevice, stream));
            for (size_t j = 0; j < scatter_count; ++j) {
                h_dst_kv_cache[j] = params.multi_dst[base + j * 2]->data();
                h_dst_kv_scale[j] = params.multi_dst[base + j * 2 + 1]->data();
            }
            check_cuda_value(cudaMemcpyAsync(
                d_dst_kv_cache_ptrs, h_dst_kv_cache.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
            check_cuda_value(cudaMemcpyAsync(
                d_dst_kv_scale_ptrs, h_dst_kv_scale.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
            sDevMPS::launch_scatter_copy_split(d_src_staging,
                                               static_cast<void**>(d_dst_kv_cache_ptrs),
                                               static_cast<void**>(d_dst_kv_scale_ptrs),
                                               params.kv_cache_size,
                                               params.kv_scale_size,
                                               static_cast<int>(scatter_count),
                                               0,
                                               stream);
        }
        check_cuda_value(cudaStreamSynchronize(stream));
    } else {
        void* const              d_src_kv_cache_ptrs = ptr0_;
        void* const              d_src_kv_scale_ptrs = ptr1_;
        void* const              d_dst_staging       = staging_;
        std::vector<const void*> h_src_kv_cache(scatter_count), h_src_kv_scale(scatter_count);
        for (size_t b = 0; b < block_nums; ++b) {
            const size_t base = b * actual_batch_size;
            for (size_t j = 0; j < scatter_count; ++j) {
                h_src_kv_cache[j] = params.multi_src[base + j * 2]->data();
                h_src_kv_scale[j] = params.multi_src[base + j * 2 + 1]->data();
            }
            check_cuda_value(cudaMemcpyAsync(
                d_src_kv_cache_ptrs, h_src_kv_cache.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
            check_cuda_value(cudaMemcpyAsync(
                d_src_kv_scale_ptrs, h_src_kv_scale.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
            sDevMPS::launch_gather_copy_split(static_cast<const void**>(d_src_kv_cache_ptrs),
                                              static_cast<const void**>(d_src_kv_scale_ptrs),
                                              params.kv_cache_size,
                                              params.kv_scale_size,
                                              d_dst_staging,
                                              static_cast<int>(scatter_count),
                                              0,
                                              stream);
            check_cuda_value(cudaMemcpyAsync(
                params.multi_dst[base]->data(), d_dst_staging, params.block_size, cudaMemcpyDeviceToHost, stream));
        }
        check_cuda_value(cudaStreamSynchronize(stream));
    }
    check_cuda_error();
}

}  // namespace rtp_llm
