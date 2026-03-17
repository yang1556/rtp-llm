#include "rtp_llm/cpp/kernels/sm_utils/sm_copy_kernel.h"

namespace sDevMPS {

// 这部分宏和全局函数调整到这里，避免和RTP冲突
#if !__NVCC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

#define WARP_SIZE 32
#define THREADS_PER_BLOCK 512

static __host__ __device__ constexpr int divUp(int x, int y) {
    return (x + y - 1) / y;
}

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err)           \
                      << std::endl;                                                                                    \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

__global__ void gather_copy_kernel(const void** src_ptrs, size_t offset, size_t size, void* dst, int num_srcs) {
    if (blockIdx.x >= num_srcs)
        return;
    const size_t tid         = threadIdx.x;
    const size_t num_threads = blockDim.x;

    const int    bytes_int4        = sizeof(int4);
    const size_t num_elements_int4 = size / bytes_int4;
    const size_t total_bytes_int4  = bytes_int4 * num_elements_int4;

    const size_t remaining_bytes = size - total_bytes_int4;

    const int    bytes_int2        = sizeof(int2);
    const size_t num_elements_int2 = remaining_bytes / bytes_int2;
    const size_t total_bytes_int2  = bytes_int2 * num_elements_int2;

    const size_t remaining_bytes_char = remaining_bytes - total_bytes_int2;

    for (int src_idx = blockIdx.x; src_idx < num_srcs; src_idx += gridDim.x) {
        const char* src_base = reinterpret_cast<const char*>(src_ptrs[src_idx]) + offset;
        char*       dst_base = reinterpret_cast<char*>(dst) + src_idx * size;

        size_t element_idx = tid;
#pragma unroll 4
        while (element_idx < num_elements_int4) {
            reinterpret_cast<int4*>(dst_base)[element_idx] = reinterpret_cast<const int4*>(src_base)[element_idx];
            element_idx += num_threads;
        }

        if (remaining_bytes == 0)
            continue;

        char*       dst_base_int2 = dst_base + total_bytes_int4;
        const char* src_base_int2 = src_base + total_bytes_int4;

        element_idx = tid;
#pragma unroll 2
        while (element_idx < num_elements_int2) {
            reinterpret_cast<int2*>(dst_base_int2)[element_idx] =
                reinterpret_cast<const int2*>(src_base_int2)[element_idx];
            element_idx += num_threads;
        }

        if (tid < remaining_bytes_char) {
            dst_base_int2[total_bytes_int2 + tid] = src_base_int2[total_bytes_int2 + tid];
        }
    }
}

__global__ void scatter_copy_kernel(const void* src, size_t offset, size_t size, void** dst_ptrs, int num_dsts) {
    if (blockIdx.x >= num_dsts)
        return;
    const size_t tid         = threadIdx.x;
    const size_t num_threads = blockDim.x;

    const int    bytes_int4        = sizeof(int4);
    const size_t num_elements_int4 = size / bytes_int4;
    const size_t total_bytes_int4  = bytes_int4 * num_elements_int4;

    const size_t remaining_bytes = size - total_bytes_int4;

    const int    bytes_int2        = sizeof(int2);
    const size_t num_elements_int2 = remaining_bytes / bytes_int2;
    const size_t total_bytes_int2  = bytes_int2 * num_elements_int2;

    const size_t remaining_bytes_char = remaining_bytes - total_bytes_int2;

    for (int dst_idx = blockIdx.x; dst_idx < num_dsts; dst_idx += gridDim.x) {
        const char* src_base = reinterpret_cast<const char*>(src) + dst_idx * size;
        char*       dst_base = reinterpret_cast<char*>(dst_ptrs[dst_idx]) + offset;

        size_t element_idx = tid;
#pragma unroll 4
        while (element_idx < num_elements_int4) {
            reinterpret_cast<int4*>(dst_base)[element_idx] = reinterpret_cast<const int4*>(src_base)[element_idx];
            element_idx += num_threads;
        }

        if (remaining_bytes == 0)
            continue;

        char*       dst_base_int2 = dst_base + total_bytes_int4;
        const char* src_base_int2 = src_base + total_bytes_int4;

        element_idx = tid;
#pragma unroll 2
        while (element_idx < num_elements_int2) {
            reinterpret_cast<int2*>(dst_base_int2)[element_idx] =
                reinterpret_cast<const int2*>(src_base_int2)[element_idx];
            element_idx += num_threads;
        }

        if (tid < remaining_bytes_char) {
            dst_base_int2[total_bytes_int2 + tid] = src_base_int2[total_bytes_int2 + tid];
        }
    }
}

// Scatter from contiguous src to num_dsts pairs (dst_kv_cache[i], dst_kv_scale[i]).
// Src layout: [kv0_cache, kv0_scale, kv1_cache, kv1_scale, ...]; stride = kv_cache_size + kv_scale_size per dst.
__global__ void scatter_copy_split_kernel(const void* src,
                                          void**      dst_kv_cache_ptrs,
                                          void**      dst_kv_scale_ptrs,
                                          size_t      kv_cache_size,
                                          size_t      kv_scale_size,
                                          int         num_dsts) {
    if (blockIdx.x >= num_dsts)
        return;
    const size_t tid         = threadIdx.x;
    const size_t num_threads = blockDim.x;
    const size_t stride      = kv_cache_size + kv_scale_size;

    const int bytes_int4 = sizeof(int4);

    auto copy_region = [&](const char* src_base, char* dst_base, size_t region_size) {
        const size_t num_elements_int4    = region_size / bytes_int4;
        const size_t total_bytes_int4     = bytes_int4 * num_elements_int4;
        const size_t remaining_bytes      = region_size - total_bytes_int4;
        const int    bytes_int2           = sizeof(int2);
        const size_t num_elements_int2    = remaining_bytes / bytes_int2;
        const size_t total_bytes_int2     = bytes_int2 * num_elements_int2;
        const size_t remaining_bytes_char = remaining_bytes - total_bytes_int2;

        size_t element_idx = tid;
#pragma unroll 4
        while (element_idx < num_elements_int4) {
            reinterpret_cast<int4*>(dst_base)[element_idx] = reinterpret_cast<const int4*>(src_base)[element_idx];
            element_idx += num_threads;
        }
        if (remaining_bytes == 0)
            return;
        char*       dst_int2 = dst_base + total_bytes_int4;
        const char* src_int2 = src_base + total_bytes_int4;
        element_idx          = tid;
#pragma unroll 2
        while (element_idx < num_elements_int2) {
            reinterpret_cast<int2*>(dst_int2)[element_idx] = reinterpret_cast<const int2*>(src_int2)[element_idx];
            element_idx += num_threads;
        }
        if (tid < remaining_bytes_char)
            dst_int2[total_bytes_int2 + tid] = src_int2[total_bytes_int2 + tid];
    };

    for (int dst_idx = blockIdx.x; dst_idx < num_dsts; dst_idx += gridDim.x) {
        const char* src_base = reinterpret_cast<const char*>(src) + dst_idx * stride;
        if (kv_cache_size > 0 && dst_kv_cache_ptrs[dst_idx] != nullptr) {
            copy_region(src_base, reinterpret_cast<char*>(dst_kv_cache_ptrs[dst_idx]), kv_cache_size);
        }
        if (kv_scale_size > 0 && dst_kv_scale_ptrs[dst_idx] != nullptr) {
            copy_region(src_base + kv_cache_size, reinterpret_cast<char*>(dst_kv_scale_ptrs[dst_idx]), kv_scale_size);
        }
    }
}

// Gather from num_srcs pairs (src_kv_cache[i], src_kv_scale[i]) to contiguous dst.
// Dst layout: [kv0_cache, kv0_scale, kv1_cache, kv1_scale, ...].
__global__ void gather_copy_split_kernel(const void** src_kv_cache_ptrs,
                                         const void** src_kv_scale_ptrs,
                                         size_t       kv_cache_size,
                                         size_t       kv_scale_size,
                                         void*        dst,
                                         int          num_srcs) {
    if (blockIdx.x >= num_srcs)
        return;
    const size_t tid         = threadIdx.x;
    const size_t num_threads = blockDim.x;
    const size_t stride      = kv_cache_size + kv_scale_size;

    const int bytes_int4 = sizeof(int4);

    auto copy_region = [&](const char* src_base, char* dst_base, size_t region_size) {
        const size_t num_elements_int4    = region_size / bytes_int4;
        const size_t total_bytes_int4     = bytes_int4 * num_elements_int4;
        const size_t remaining_bytes      = region_size - total_bytes_int4;
        const int    bytes_int2           = sizeof(int2);
        const size_t num_elements_int2    = remaining_bytes / bytes_int2;
        const size_t total_bytes_int2     = bytes_int2 * num_elements_int2;
        const size_t remaining_bytes_char = remaining_bytes - total_bytes_int2;

        size_t element_idx = tid;
#pragma unroll 4
        while (element_idx < num_elements_int4) {
            reinterpret_cast<int4*>(dst_base)[element_idx] = reinterpret_cast<const int4*>(src_base)[element_idx];
            element_idx += num_threads;
        }
        if (remaining_bytes == 0)
            return;
        char*       dst_int2 = dst_base + total_bytes_int4;
        const char* src_int2 = src_base + total_bytes_int4;
        element_idx          = tid;
#pragma unroll 2
        while (element_idx < num_elements_int2) {
            reinterpret_cast<int2*>(dst_int2)[element_idx] = reinterpret_cast<const int2*>(src_int2)[element_idx];
            element_idx += num_threads;
        }
        if (tid < remaining_bytes_char)
            dst_int2[total_bytes_int2 + tid] = src_int2[total_bytes_int2 + tid];
    };

    for (int src_idx = blockIdx.x; src_idx < num_srcs; src_idx += gridDim.x) {
        char* dst_base = reinterpret_cast<char*>(dst) + src_idx * stride;
        if (kv_cache_size > 0 && src_kv_cache_ptrs[src_idx] != nullptr) {
            copy_region(reinterpret_cast<const char*>(src_kv_cache_ptrs[src_idx]), dst_base, kv_cache_size);
        }
        if (kv_scale_size > 0 && src_kv_scale_ptrs[src_idx] != nullptr) {
            copy_region(
                reinterpret_cast<const char*>(src_kv_scale_ptrs[src_idx]), dst_base + kv_cache_size, kv_scale_size);
        }
    }
}

__global__ void n2n_copy_kernel(const void** src_ptrs,
                                void**       dst_ptrs,
                                const size_t element_size,
                                const int* __restrict__ src_indices,
                                const int* __restrict__ dst_indices,
                                const int copy_count,
                                const int num_tensors) {
    const int nBlocks  = gridDim.x;
    const int nThreads = blockDim.x;
    const int nWarps   = nThreads / WARP_SIZE;
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;

    const int tensors_per_block = divUp(num_tensors, nBlocks);
    const int start_tensor_idx  = blockIdx.x * tensors_per_block;
    const int end_tensor_idx    = min(start_tensor_idx + tensors_per_block, num_tensors);

    const int elems_per_warp = divUp(copy_count, nWarps);
    const int start_elem     = warp_id * elems_per_warp;
    const int end_elem       = min(start_elem + elems_per_warp, copy_count);

    const int bytes_int4         = sizeof(int4);
    const int chunks_per_element = divUp(element_size, bytes_int4);
    const int chunks_per_thread  = divUp(chunks_per_element, WARP_SIZE);

    for (int tensor_idx = start_tensor_idx; tensor_idx < end_tensor_idx; tensor_idx++) {
        const void* src_tensor = src_ptrs[tensor_idx];
        void*       dst_tensor = dst_ptrs[tensor_idx];

        for (int elem_idx = start_elem; elem_idx < end_elem; elem_idx++) {
            const char* src_base = static_cast<const char*>(src_tensor) + src_indices[elem_idx] * element_size;
            char*       dst_base = static_cast<char*>(dst_tensor) + dst_indices[elem_idx] * element_size;

#pragma unroll
            for (int chunk_idx = 0; chunk_idx < chunks_per_thread; chunk_idx++) {
                const int    global_chunk_idx = lane_id + chunk_idx * WARP_SIZE;
                const size_t offset           = global_chunk_idx * bytes_int4;

                if (offset >= element_size)
                    break;

                const size_t copy_size = (offset + bytes_int4 <= element_size) ? bytes_int4 : element_size - offset;

                const uint4* src_ptr = reinterpret_cast<const uint4*>(src_base + offset);
                uint4*       dst_ptr = reinterpret_cast<uint4*>(dst_base + offset);

                if (copy_size == bytes_int4) {
                    *dst_ptr = *src_ptr;
                } else {
                    char*       dst_char = reinterpret_cast<char*>(dst_ptr);
                    const char* src_char = reinterpret_cast<const char*>(src_ptr);
                    for (size_t i = 0; i < copy_size; i++) {
                        dst_char[i] = src_char[i];
                    }
                }
            }
        }
    }
}

void launch_gather_copy(
    const void** src_ptrs, size_t offset, size_t size, void* dst, int num_srcs, int block_num, cudaStream_t stream) {
    if (block_num == 0) {
        block_num = num_srcs;
    }
    gather_copy_kernel<<<block_num, THREADS_PER_BLOCK, 0, stream>>>(src_ptrs, offset, size, dst, num_srcs);
}

void launch_scatter_copy(
    const void* src, size_t offset, size_t size, void** dst_ptrs, int num_dsts, int block_num, cudaStream_t stream) {
    if (block_num == 0) {
        block_num = num_dsts;
    }
    scatter_copy_kernel<<<block_num, THREADS_PER_BLOCK, 0, stream>>>(src, offset, size, dst_ptrs, num_dsts);
}

void launch_scatter_copy_split(const void*  src,
                               void**       dst_kv_cache_ptrs,
                               void**       dst_kv_scale_ptrs,
                               size_t       kv_cache_size,
                               size_t       kv_scale_size,
                               int          num_dsts,
                               int          block_num,
                               cudaStream_t stream) {
    if (block_num == 0) {
        block_num = num_dsts;
    }
    scatter_copy_split_kernel<<<block_num, THREADS_PER_BLOCK, 0, stream>>>(
        src, dst_kv_cache_ptrs, dst_kv_scale_ptrs, kv_cache_size, kv_scale_size, num_dsts);
}

void launch_gather_copy_split(const void** src_kv_cache_ptrs,
                              const void** src_kv_scale_ptrs,
                              size_t       kv_cache_size,
                              size_t       kv_scale_size,
                              void*        dst,
                              int          num_srcs,
                              int          block_num,
                              cudaStream_t stream) {
    // Do not call cudaSetDevice here: caller must set the current device; changing it inside this
    // helper can race multi-GPU / multi-thread paths and mis-associate launches with streams.
    if (block_num == 0) {
        block_num = num_srcs;
    }
    gather_copy_split_kernel<<<block_num, THREADS_PER_BLOCK, 0, stream>>>(
        src_kv_cache_ptrs, src_kv_scale_ptrs, kv_cache_size, kv_scale_size, dst, num_srcs);
}

bool warmup_sm_copy_split_kernels(cudaStream_t stream) {
    constexpr size_t kv_cache_size = 32;
    constexpr size_t kv_scale_size = 32;
    constexpr int    num           = 1;
    constexpr int    block_num     = 1;
    constexpr size_t ptr_bytes     = sizeof(void*);

    void* scatter_src       = nullptr;
    void* dst_kv            = nullptr;
    void* dst_scale         = nullptr;
    void* d_dst_kv_table    = nullptr;
    void* d_dst_scale_table = nullptr;
    void* gather_dst        = nullptr;
    void* src_kv_buf        = nullptr;
    void* src_scale_buf     = nullptr;
    void* d_src_kv_table    = nullptr;
    void* d_src_scale_table = nullptr;

    auto free_all = [&]() {
        cudaFree(scatter_src);
        cudaFree(dst_kv);
        cudaFree(dst_scale);
        cudaFree(d_dst_kv_table);
        cudaFree(d_dst_scale_table);
        cudaFree(gather_dst);
        cudaFree(src_kv_buf);
        cudaFree(src_scale_buf);
        cudaFree(d_src_kv_table);
        cudaFree(d_src_scale_table);
        (void)cudaGetLastError();
    };

#define WARMUP_CHK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e_internal_wu_ = (call);                                                                           \
        if (e_internal_wu_ != cudaSuccess) {                                                                           \
            free_all();                                                                                                \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

    WARMUP_CHK(cudaMalloc(&scatter_src, kv_cache_size + kv_scale_size));
    WARMUP_CHK(cudaMalloc(&dst_kv, kv_cache_size));
    WARMUP_CHK(cudaMalloc(&dst_scale, kv_scale_size));
    WARMUP_CHK(cudaMalloc(&d_dst_kv_table, ptr_bytes));
    WARMUP_CHK(cudaMalloc(&d_dst_scale_table, ptr_bytes));
    WARMUP_CHK(cudaMalloc(&gather_dst, kv_cache_size + kv_scale_size));
    WARMUP_CHK(cudaMalloc(&src_kv_buf, kv_cache_size));
    WARMUP_CHK(cudaMalloc(&src_scale_buf, kv_scale_size));
    WARMUP_CHK(cudaMalloc(&d_src_kv_table, ptr_bytes));
    WARMUP_CHK(cudaMalloc(&d_src_scale_table, ptr_bytes));

#undef WARMUP_CHK

    {
        void* h_kv[1] = {dst_kv};
        void* h_sc[1] = {dst_scale};
        if (cudaMemcpyAsync(d_dst_kv_table, h_kv, ptr_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess
            || cudaMemcpyAsync(d_dst_scale_table, h_sc, ptr_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            free_all();
            return false;
        }
    }

    launch_scatter_copy_split(scatter_src,
                              reinterpret_cast<void**>(d_dst_kv_table),
                              reinterpret_cast<void**>(d_dst_scale_table),
                              kv_cache_size,
                              kv_scale_size,
                              num,
                              block_num,
                              stream);

    {
        void* h_sk[1] = {src_kv_buf};
        void* h_ss[1] = {src_scale_buf};
        if (cudaMemcpyAsync(d_src_kv_table, h_sk, ptr_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess
            || cudaMemcpyAsync(d_src_scale_table, h_ss, ptr_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            free_all();
            return false;
        }
    }

    launch_gather_copy_split(reinterpret_cast<const void**>(d_src_kv_table),
                             reinterpret_cast<const void**>(d_src_scale_table),
                             kv_cache_size,
                             kv_scale_size,
                             gather_dst,
                             num,
                             block_num,
                             stream);

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        free_all();
        return false;
    }

    free_all();
    return true;
}

}  // namespace sDevMPS