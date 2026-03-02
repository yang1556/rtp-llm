#include "rtp_llm/cpp/kernels/sparse_logits.h"
#include "rtp_llm/cpp/kernels/logits_util.h"

namespace rtp_llm {

template<typename T>
__global__ void fill_mem_value(const int batch_size, const int sparse_vs, T* sparse_logits, int* sparse_index) {
    int batch_idx = blockIdx.y;
    int vocab_idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if (batch_idx < batch_size && vocab_idx < sparse_vs) {
        int global_idx            = batch_idx * sparse_vs + vocab_idx;
        sparse_logits[global_idx] = NegativeInfinity<T>();
        sparse_index[global_idx]  = -1;
    }
}

// Batch version kernel for processing multiple beams
template<typename T>
__global__ void sparse_logits_kernel(T*   logits,
                                     T*   sparse_logits,
                                     int* sparse_index,
                                     const int* __restrict__ batch_idx,
                                     const int* __restrict__ vocab_idx,
                                     int batch_size,
                                     int vocab_size,
                                     int weight_size,
                                     int sparse_vs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < weight_size) {
        int b_idx      = 0;
        int offset_idx = 0;
        int total_size = batch_size * 2;
        int start_idx  = 0;
        for (int i = 0; i < total_size; i += 2) {
            offset_idx = idx - start_idx;  // todo fix
            if (idx < batch_idx[i]) {
                b_idx = batch_idx[i + 1];
                break;
            }
            start_idx = batch_idx[i];
        }

        int v_idx = vocab_idx[idx];
        if (b_idx < batch_size && v_idx < vocab_size) {
            int global_idx            = b_idx * vocab_size + v_idx;
            int sparse_idx            = b_idx * sparse_vs + offset_idx;
            sparse_logits[sparse_idx] = logits[global_idx];
            sparse_index[sparse_idx]  = v_idx;
        }
    }
}

template<typename T>
void invokeSparseLogits(T*   logits,
                        T*   sparse_logits,
                        int* sparse_index,
                        const int* __restrict__ batch_idx,
                        const int* __restrict__ vocab_idx,
                        int          batch_size,
                        int          vocab_size,
                        int          weight_size,
                        int          sparse_vs,
                        cudaStream_t stream) {
    dim3 block, grid;

    block.x = 256;
    block.y = 1;
    block.z = 1;
    grid.y  = batch_size;
    grid.z  = 1;
    grid.x  = (sparse_vs + block.x - 1) / block.x;
    fill_mem_value<<<grid, block, 0, stream>>>(batch_size, sparse_vs, sparse_logits, sparse_index);
    check_cuda_error();

    grid.y = 1;
    grid.x = (weight_size + block.x - 1) / block.x;
    sparse_logits_kernel<<<grid, block, 0, stream>>>(
        logits, sparse_logits, sparse_index, batch_idx, vocab_idx, batch_size, vocab_size, weight_size, sparse_vs);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

template void invokeSparseLogits<float>(float* logits,
                                        float* sparse_logits,
                                        int*   sparse_index,
                                        const int* __restrict__ batch_idx,
                                        const int* __restrict__ vocab_idx,
                                        int          batch_size,
                                        int          vocab_size,
                                        int          weight_size,
                                        int          sparse_vs,
                                        cudaStream_t stream);
template void invokeSparseLogits<half>(half* logits_batch,
                                       half* sparse_logits,
                                       int*  sparse_index,
                                       const int* __restrict__ batch_idx,
                                       const int* __restrict__ vocab_idx,
                                       int          batch_size,
                                       int          vocab_size,
                                       int          weight_size,
                                       int          sparse_vs,
                                       cudaStream_t stream);
template void invokeSparseLogits<__nv_bfloat16>(__nv_bfloat16* logits_batch,
                                                __nv_bfloat16* sparse_logits,
                                                int*           sparse_index,
                                                const int* __restrict__ batch_idx,
                                                const int* __restrict__ vocab_idx,
                                                int          batch_size,
                                                int          vocab_size,
                                                int          weight_size,
                                                int          sparse_vs,
                                                cudaStream_t stream);

}  // namespace rtp_llm