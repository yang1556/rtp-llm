#pragma once

#include "rtp_llm/cpp/models/GptModelTypes.h"
#include <optional>
#include <string>
#include <mutex>
#include <memory>
#include "rtp_llm/cpp/core/Types.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include "rtp_llm/models_py/bindings/OpDefsUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/GraphBase.h"
#if USING_CUDA
#include <c10/cuda/CUDAStream.h>
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#endif

#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"

namespace py = pybind11;

namespace rtp_llm {

class PyWrappedModel: public IGptModel {
public:
    // py_instance is `py_model` indeedly.
    PyWrappedModel(const GptModelInitParams& params, py::object py_instance, bool is_prefill_cuda_graph_mode = false);
    ~PyWrappedModel();

    GptModelOutputs forward(const GptModelInputs& inputs) override;
    void            releaseBuffers() override;

    GptModelOutputs forwardMicroBatched(const GptModelInputs& inputs);

private:
    std::optional<PyCacheStoreInputs> prepareWriteCacheParams(const GptModelInputs& inputs);

    // Helper functions to reduce code duplication
    torch_ext::PyAttentionInputs   buildPyAttentionInputs(const GptModelInputs& inputs);
    torch_ext::BertEmbeddingInputs buildBertEmbeddingInputs(const GptModelInputs& inputs);
    void                           setupKVCacheForAttentionInputs(torch_ext::PyAttentionInputs& py_attn_inputs,
                                                                  const GptModelInputs&         inputs,
                                                                  BufferPtr&                    kv_cache_block_id_device,
                                                                  std::vector<BufferPtr>*       kv_cache_block_id_device_by_group = nullptr);
    GptModelOutputs                callForwardPostLayers(BufferPtr             hidden_states,
                                                         const GptModelInputs& inputs,
                                                         bool                  skip_final_layernorm,
                                                         size_t                num_valid_tokens = -1);
    torch::Tensor                  tensorHoldHostAndToCuda(const torch::Tensor& tensor);

    // Post-layers and micro-batch helpers (implemented in PyWrappedModel.cc, no GptModel dependency)
    MicroBatchPlan planMicroBatches(const GptModelInputs& inputs);
    std::pair<std::vector<GptModelInputs>, std::vector<TokenSliceInfo>>
                    splitInputsIntoMicroBatches(const GptModelInputs& inputs, const MicroBatchPlan& micro_batch_plan);
    GptModelOutputs forwardPostLayers(BufferPtr             hidden,
                                      bool                  has_context_request,
                                      bool                  need_all_logits,
                                      const BufferPtr&      lm_output_indexes,
                                      bool                  enable_sp,
                                      size_t                token_num,
                                      const GptModelInputs& inputs,
                                      BufferPtr             merged_eagle3_hidden,
                                      bool                  skip_final_layernorm = false);
    void            holdInputsHostBuffers(const GptModelInputs& inputs);
    BufferPtr       tpSyncEmbeddingOrLogits(const BufferPtr& buffer);

    // State
    rtp_llm::DeviceBase*                     device_{nullptr};
    const rtp_llm::DeviceProperties          device_props_{};
    GptModelDescription                      description_{};
    rtp_llm::Weights                         weights_;
    size_t                                   model_id_{0};
    size_t                                   layer_num_{0};
    std::optional<rtp_llm::CacheLayerLayout> kv_cache_layer_layout_;

    GraphBase*                                 graph_runner_{nullptr};
    py::object                                 py_model_;
    py::object                                 held_attn_pyobj_;
    bool                                       enable_cuda_graph_{false};
    bool                                       is_prefill_cuda_graph_mode_{false};
    std::unique_ptr<IContextParallelProcessor> context_parallel_processor_{nullptr};
};

}  // namespace rtp_llm
