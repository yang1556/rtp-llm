#include "rtp_llm/cpp/kernels/block_zero_kernels.h"

namespace rtp_llm {

__global__ void zero_incomplete_kv_cache_blocks_kernel(
    const void* const* __restrict__ layer_base_addrs,
    const int32_t* __restrict__ kv_cache_block_id,
    const int32_t* __restrict__ token_counts,
    const int32_t* __restrict__ layer_to_group,
    size_t batch_size,
    size_t layer_num,
    size_t batch_dim,
    size_t max_blocks_per_batch,
    size_t block_stride_bytes,
    size_t seq_size_per_block)
{
    const size_t batch_idx = blockIdx.x;
    const size_t layer_idx = blockIdx.y;

    if (batch_idx >= batch_size || layer_idx >= layer_num)
        return;

    const int32_t tokens = token_counts[batch_idx];
    if (tokens <= 0)
        return;

    if ((tokens - 1) % static_cast<int32_t>(seq_size_per_block) != 0)
        return;

    const size_t group_idx = layer_to_group ? static_cast<size_t>(layer_to_group[layer_idx]) : 0;

    const size_t last_block_index = static_cast<size_t>(tokens - 1) / seq_size_per_block;
    if (last_block_index >= max_blocks_per_batch)
        return;

    const int32_t block_id =
        kv_cache_block_id[group_idx * batch_dim * max_blocks_per_batch
                          + batch_idx * max_blocks_per_batch
                          + last_block_index];

    if (block_id <= 0)
        return;

    const void* base = layer_base_addrs[layer_idx];
    if (!base)
        return;

    char* dst = static_cast<char*>(const_cast<void*>(base))
                + static_cast<size_t>(block_id) * block_stride_bytes;

    const size_t n_uint4   = block_stride_bytes / sizeof(uint4);
    const size_t remainder = block_stride_bytes % sizeof(uint4);

    uint4*      dst4  = reinterpret_cast<uint4*>(dst);
    const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);

    for (size_t i = threadIdx.x; i < n_uint4; i += blockDim.x) {
        dst4[i] = zero4;
    }

    if (remainder > 0) {
        const size_t tail_start = n_uint4 * sizeof(uint4);
        for (size_t i = threadIdx.x; i < remainder; i += blockDim.x) {
            dst[tail_start + i] = 0;
        }
    }
}

void invokeZeroIncompleteKvCacheBlocks(const void* const* layer_base_addrs,
                                       const int32_t*     kv_cache_block_id,
                                       const int32_t*     token_counts,
                                       const int32_t*     layer_to_group,
                                       size_t             batch_size,
                                       size_t             layer_num,
                                       size_t             batch_dim,
                                       size_t             max_blocks_per_batch,
                                       size_t             block_stride_bytes,
                                       size_t             seq_size_per_block,
                                       cudaStream_t       stream) {
    if (batch_size == 0 || layer_num == 0 || block_stride_bytes == 0)
        return;

    constexpr int kThreads = 256;
    dim3          grid(static_cast<unsigned>(batch_size), static_cast<unsigned>(layer_num));
    dim3          block(kThreads);

    zero_incomplete_kv_cache_blocks_kernel<<<grid, block, 0, stream>>>(
        layer_base_addrs,
        kv_cache_block_id,
        token_counts,
        layer_to_group,
        batch_size,
        layer_num,
        batch_dim,
        max_blocks_per_batch,
        block_stride_bytes,
        seq_size_per_block);
}

}  // namespace rtp_llm
