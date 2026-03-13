#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/models/GptModelTypes.h"
#include <utility>
#include <vector>

namespace rtp_llm {

/** Slice kv_cache_block_id by batch. Handles 2-D [batch, max_blocks] and 3-D [group, batch, max_blocks]. */
BufferPtr sliceKvCacheBlockIdByBatch(const BufferPtr&     kv_cache_block_id,
                                     size_t               batch_offset,
                                     size_t               batch_size,
                                     rtp_llm::DeviceBase* device);

/** TP sync for embedding or logits (all-gather + transpose). */
BufferPtr tpSyncEmbeddingOrLogits(rtp_llm::DeviceBase*             device,
                                  const rtp_llm::DeviceProperties& device_props,
                                  const BufferPtr&                 buffer);

/** Hold all host buffers from inputs in buffer_holder to avoid unexpected H2D copy. */
void holdInputsHostBuffers(ModelBufferHolder& buffer_holder, const GptModelInputs& inputs);

/** Compute micro-batch split plan from inputs. */
MicroBatchPlan
planMicroBatches(const GptModelInputs& inputs, size_t layer_num, const rtp_llm::DeviceProperties& device_props);

/** Split inputs into micro-batches according to plan. Requires device for sliceKvCacheBlockIdByBatch and buffer
 * allocation. */
std::pair<std::vector<GptModelInputs>, std::vector<TokenSliceInfo>> splitInputsIntoMicroBatches(
    const GptModelInputs& inputs, const MicroBatchPlan& micro_batch_plan, rtp_llm::DeviceBase* device);

}  // namespace rtp_llm
