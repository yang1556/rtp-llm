#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include <cuda_runtime.h>
#include <vector>

namespace rtp_llm {

class CudaCopyUtils {
public:
    /// Dispatches by transfer type: H2D -> launch_scatter_copy, D2H -> launch_gather_copy, D2D -> both.
    /// Loop is by block (grouped by buffer size); each launch copies size_per bytes per buffer. Returns false if mixed
    /// or unsupported.
    /// When block_size and batch_size are both > 0 (e.g. from KVCacheMemoryConnector), block_total_bytes = block_size;
    /// else derived from buffers and kLayerNums.
    static bool multiCopyWithGatherScatter(const std::vector<BufferPtr>& multi_src,
                                           const std::vector<BufferPtr>& multi_dst,
                                           cudaStream_t                  stream,
                                           size_t                        block_size = 0,
                                           size_t                        batch_size = 0);
};

}  // namespace rtp_llm
