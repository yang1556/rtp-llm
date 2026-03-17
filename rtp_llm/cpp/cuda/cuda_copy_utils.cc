#include "rtp_llm/cpp/cuda/cuda_copy_utils.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/kernels/sm_utils/sm_copy_kernel.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <vector>

namespace rtp_llm {

namespace {

int calculateBlockNum(size_t bytes_per_src, size_t num_srcs) {
    constexpr size_t bytes_per_block = 256;
    size_t           total_bytes     = bytes_per_src * num_srcs;
    size_t           num_blocks      = (total_bytes + bytes_per_block - 1) / bytes_per_block;
    constexpr size_t max_blocks      = 65535;
    return static_cast<int>(std::min(num_blocks, max_blocks));
}

}  // namespace

bool CudaCopyUtils::multiCopyWithGatherScatter(const std::vector<BufferPtr>& multi_src,
                                               const std::vector<BufferPtr>& multi_dst,
                                               cudaStream_t                  stream,
                                               size_t                        block_size,
                                               size_t                        batch_size) {
    if (multi_src.size() != multi_dst.size() || multi_src.empty())
        return false;

    const size_t actual_batch_size = batch_size;
    if (multi_src.size() % actual_batch_size != 0)
        return false;
    const size_t block_nums = multi_src.size() / actual_batch_size;

    // Assume uniform src/dst memory type; check first element only for path selection.
    const bool src_is_host =
        multi_src[0]
        && (multi_src[0]->where() == MemoryType::MEMORY_CPU || multi_src[0]->where() == MemoryType::MEMORY_CPU_PINNED);
    const bool dst_is_gpu = multi_dst[0] && multi_dst[0]->where() == MemoryType::MEMORY_GPU;
    const bool src_is_gpu = multi_src[0] && multi_src[0]->where() == MemoryType::MEMORY_GPU;
    const bool dst_is_host =
        multi_dst[0]
        && (multi_dst[0]->where() == MemoryType::MEMORY_CPU || multi_dst[0]->where() == MemoryType::MEMORY_CPU_PINNED);

    if (src_is_host && dst_is_gpu) {
        // H2D: CPU contiguous per block -> GPU non-contiguous. dst_ptrs must be void** in device memory.
        const size_t block_total_bytes = block_size;
        const size_t scatter_count     = actual_batch_size / 2;
        const size_t size_per_dst      = block_total_bytes / scatter_count;
        const size_t ptr_table_bytes   = scatter_count * sizeof(void*);
        void*        d_dst_ptrs        = nullptr;
        if (cudaMalloc(&d_dst_ptrs, ptr_table_bytes) != cudaSuccess)
            return false;
        std::vector<void*> h_dst_ptrs(scatter_count);
        for (size_t b = 0; b < block_nums; ++b) {
            const size_t base    = b * actual_batch_size;
            void*        src_dev = nullptr;
            if (cudaHostGetDevicePointer(&src_dev, multi_src[base]->data(), 0) != cudaSuccess) {
                cudaStreamSynchronize(stream);
                cudaFree(d_dst_ptrs);
                return false;
            }
            for (size_t j = 0; j < scatter_count; ++j)
                h_dst_ptrs[j] = multi_dst[base + j * 2]->data();
            if (cudaMemcpyAsync(d_dst_ptrs, h_dst_ptrs.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream)
                != cudaSuccess) {
                cudaStreamSynchronize(stream);
                cudaFree(d_dst_ptrs);
                return false;
            }
            int block_num = calculateBlockNum(size_per_dst, scatter_count);
            sDevMPS::launch_scatter_copy(src_dev,
                                         0,
                                         size_per_dst,
                                         static_cast<void**>(d_dst_ptrs),
                                         static_cast<int>(scatter_count),
                                         block_num,
                                         stream);
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            cudaFree(d_dst_ptrs);
            return false;
        }
        cudaFree(d_dst_ptrs);
        return true;
    }

    if (src_is_gpu && dst_is_host) {
        // D2H: GPU non-contiguous -> CPU contiguous. src_ptrs must be const void** in device memory.
        const size_t block_total_bytes = block_size;
        const size_t gather_count      = actual_batch_size / 2;
        const size_t size_per_src      = block_total_bytes / gather_count;
        const size_t ptr_table_bytes   = gather_count * sizeof(void*);
        void*        d_src_ptrs        = nullptr;
        if (cudaMalloc(&d_src_ptrs, ptr_table_bytes) != cudaSuccess)
            return false;
        std::vector<const void*> h_src_ptrs(gather_count);
        for (size_t b = 0; b < block_nums; ++b) {
            const size_t base = b * actual_batch_size;
            for (size_t j = 0; j < gather_count; ++j)
                h_src_ptrs[j] = multi_src[base + j * 2]->data();
            if (cudaMemcpyAsync(d_src_ptrs, h_src_ptrs.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream)
                != cudaSuccess) {
                cudaStreamSynchronize(stream);
                cudaFree(d_src_ptrs);
                return false;
            }
            void* dst_dev = nullptr;
            if (cudaHostGetDevicePointer(&dst_dev, multi_dst[base]->data(), 0) != cudaSuccess) {
                cudaStreamSynchronize(stream);
                cudaFree(d_src_ptrs);
                return false;
            }
            int block_num = calculateBlockNum(size_per_src, gather_count);
            sDevMPS::launch_gather_copy(static_cast<const void**>(d_src_ptrs),
                                        0,
                                        size_per_src,
                                        dst_dev,
                                        static_cast<int>(gather_count),
                                        block_num,
                                        stream);
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            cudaFree(d_src_ptrs);
            return false;
        }
        cudaFree(d_src_ptrs);
        return true;
    }

    return false;
}

}  // namespace rtp_llm
