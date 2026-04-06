#pragma once

#include <cstddef>
#include <cstdint>

#if USING_CUDA
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

/// Zero the latest incomplete KV cache block for each (batch, layer) pair.
///
/// A block is considered "incomplete" when `(total_tokens - 1) % seq_size_per_block == 0`,
/// i.e. the current token is the first to land in a fresh block.  Mid-block positions are
/// skipped with a single integer modulo — no warp divergence since all threads in a block
/// share the same batch_idx.
///
/// @param layer_base_addrs       Device array [layer_num], int64 cast to void*.
/// @param kv_cache_block_id      Device array [G, batch_dim, max_blocks_per_batch], int32.
/// @param token_counts           Device array [batch_size], int32. Total tokens per batch element.
/// @param layer_to_group         Device array [layer_num], int32. Null => all layers map to group 0.
/// @param batch_size             Number of batch elements in token_counts.
/// @param layer_num              Number of layers.
/// @param batch_dim              Second dimension of kv_cache_block_id (>= batch_size).
/// @param max_blocks_per_batch   Third dimension of kv_cache_block_id.
/// @param block_stride_bytes     Per-block stride in bytes (same for all layers).
/// @param seq_size_per_block     Tokens per block.
/// @param stream                 CUDA stream.
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
                                       cudaStream_t       stream);

}  // namespace rtp_llm
