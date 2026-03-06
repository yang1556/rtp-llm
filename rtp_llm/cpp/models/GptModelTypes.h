#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/Weights.h"
#include "rtp_llm/cpp/cache/BufferTypes.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <torch/extension.h>

namespace rtp_llm {

struct GptModelDescription {
    rtp_llm::AttentionConfigs attention_conf;
    rtp_llm::FfnConfigs       ffn_conf;
    rtp_llm::NormType         norm_type;
    DataType                  data_type;
    rtp_llm::QScheme          act_qscheme            = rtp_llm::QScheme::NoQuantize;
    const DataType            compute_type           = rtp_llm::DataType::TYPE_INVALID;
    double                    layernorm_eps          = 1e-5;
    size_t                    vocab_size             = 0;
    bool                      post_layernorm         = false;
    double                    input_embedding_scalar = 1;
    double                    residual_scalar        = 1;
    bool                      reverse_e_h_norm       = false;
};

struct GptModelInitParams {
    rtp_llm::DeviceBase*                  device;
    const rtp_llm::Weights                weights;
    const GptModelDescription             description;
    const std::optional<CacheLayerLayout> kv_cache_layer_layout;
    size_t                                model_id;
};

struct MicroBatchInfo {
    size_t prefill_num;
    size_t decoder_num;
};

struct MicroBatchPlan {
    bool                        enable = false;
    std::vector<MicroBatchInfo> batch_infos;
};

struct TokenSliceInfo {
    size_t offset = 0;
    size_t count  = 0;
};

struct ModelBufferHolder {
    std::vector<BufferPtr>     buffers;
    std::vector<torch::Tensor> tensors;

    void hold_host(const BufferPtr& buffer) {
        if (buffer && buffer->where() != MemoryType::MEMORY_GPU) {
            buffers.push_back(buffer);
        }
    }

    void hold_host(const torch::Tensor& tensor) {
        if (tensor.defined() && tensor.device().is_cpu()) {
            tensors.push_back(tensor);
        }
    }

    void hold(const BufferPtr& buffer) {
        if (buffer) {
            buffers.push_back(buffer);
        }
    }

    void hold(const torch::Tensor& tensor) {
        if (tensor.defined()) {
            tensors.push_back(tensor);
        }
    }

    void release() {
        buffers.clear();
        tensors.clear();
    }
};

/** Abstract model interface. PyWrappedModel and GptModel both implement this. */
class IGptModel {
public:
    virtual ~IGptModel()                                          = default;
    virtual GptModelOutputs forward(const GptModelInputs& inputs) = 0;
    virtual void            releaseBuffers() {}
};

}  // namespace rtp_llm
