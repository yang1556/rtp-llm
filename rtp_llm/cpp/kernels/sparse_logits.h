#pragma once

#include <stdint.h>
#if USING_CUDA
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

template<typename T>
void invokeSparseLogits(T*   logits,
                        T*   sparse_logits,
                        int* sparse_index,
                        const int* __restrict__ batch_idx,
                        const int* __restrict__ vocab_idx,
                        int batch_size,
                        int vocab_size,
                        int weight_size,
                        int sparse_vs,
#if USING_CUDA
                        cudaStream_t stream);
#elif USING_ROCM
                        hipStream_t stream);
#endif

}  // namespace rtp_llm
