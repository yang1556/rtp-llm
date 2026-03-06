#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/ModelUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/utils.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include <cstdint>
#include <stdexcept>
#include <mutex>
#include <vector>
#include <algorithm>
#include <numeric>
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <cstdlib>
#include <iostream>
#include <cstring>
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"

namespace rtp_llm {

PyWrappedModel::PyWrappedModel(const GptModelInitParams& params,
                               py::object                py_instance,
                               bool                      is_prefill_cuda_graph_mode):
    device_(params.device),
    device_props_(params.device->getDeviceProperties()),
    description_(params.description),
    weights_(params.weights),
    model_id_(params.model_id),
    layer_num_(params.weights.layers.size()),
    kv_cache_layer_layout_(params.kv_cache_layer_layout),
    enable_cuda_graph_(params.device->initParams().hw_kernel_config.enable_cuda_graph),
    is_prefill_cuda_graph_mode_(is_prefill_cuda_graph_mode) {
    if (setenv("PYTHONUNBUFFERED", "TRUE", 1) != 0) {
        RTP_LLM_LOG_WARNING("Failed to set PYTHONUNBUFFERED environment variable on POSIX.");
    } else {
        RTP_LLM_LOG_INFO("Set PYTHONUNBUFFERED=TRUE for Python interpreter.");
    }

    py::gil_scoped_acquire          gil;
    torch_ext::PyModelInitResources init_resources;

    if (params.kv_cache_layer_layout.has_value()) {
        torch_ext::KVCache kv_cache;
        kv_cache.seq_size_per_block = params.description.attention_conf.tokens_per_block;
        const auto& layout          = params.kv_cache_layer_layout.value();
        kv_cache.kv_cache_base_by_layer.reserve(layout.layers_to_kv_buffer_ptrs.size());
        for (const auto& buf : layout.layers_to_kv_buffer_ptrs) {
            if (buf) {
                kv_cache.kv_cache_base_by_layer.push_back(Buffer2torchTensor(buf, false));
            } else {
                kv_cache.kv_cache_base_by_layer.push_back(torch::Tensor());
            }
        }
        kv_cache.kv_scale_base_by_layer.reserve(layout.layers_to_scale_buffer_ptrs.size());
        for (const auto& buf : layout.layers_to_scale_buffer_ptrs) {
            if (buf) {
                kv_cache.kv_scale_base_by_layer.push_back(Buffer2torchTensor(buf, false));
            } else {
                kv_cache.kv_scale_base_by_layer.push_back(torch::Tensor());
            }
        }
        init_resources.kv_cache = kv_cache;
    }

    py::object py_init_result;
    py_model_                 = py_instance;
    auto py_initialize_method = py_model_.attr("initialize");
    py_init_result            = py_initialize_method(init_resources);
    if (enable_cuda_graph_) {
#if USING_CUDA
        c10::ScalarType dtype         = dataTypeToTorchType(description_.data_type);
        const auto&     device_params = params.device->initParams();
        GraphParams     graph_params;
        graph_params.enable_cuda_graph            = device_params.hw_kernel_config.enable_cuda_graph;
        graph_params.enable_cuda_graph_debug_mode = device_params.hw_kernel_config.enable_cuda_graph_debug_mode;
        graph_params.is_prefill_cuda_graph_mode   = is_prefill_cuda_graph_mode;
        graph_params.max_seq_len                  = device_params.max_seq_len;
        graph_params.tokens_per_block             = device_params.tokens_per_block;
        graph_params.hidden_size                  = device_params.hidden_size;
        graph_params.model_data_type              = dtype;
        graph_params.max_context_batch_size = device_params.runtime_config.fifo_scheduler_config.max_context_batch_size;
        graph_params.concurrency_limit      = device_params.concurrency_config.concurrency_limit;
        graph_params.prefill_capture_seq_lens   = device_params.hw_kernel_config.prefill_capture_seq_lens;
        graph_params.decode_capture_batch_sizes = device_params.hw_kernel_config.decode_capture_batch_sizes;
        graph_params.kv_cache_layer_to_group    = device_params.kv_cache_layer_to_group;
        graph_params.kv_cache_group_num         = device_params.kv_cache_group_num;

        if (is_prefill_cuda_graph_mode) {
            graph_params.num_tokens_per_bs = device_params.max_seq_len;
        } else if (device_params.sp_config.gen_num_per_cycle > 1 && !params.model_id) {
            graph_params.num_tokens_per_bs = device_params.sp_config.gen_num_per_cycle + 1;
        } else {
            graph_params.num_tokens_per_bs = 1;
        }

        graph_runner_ = new CudaGraphRunner(graph_params, py_instance);
        RTP_LLM_CHECK_WITH_INFO(graph_runner_ != nullptr, "graph_runner_ can't be nullptr in PyWrapper");
#else
        RTP_LLM_CHECK_WITH_INFO(false, "CUDA Graph is only supported on CUDA platform for now");
#endif
        if (weights_.position_encoding) {
            graph_runner_->setPositionEncoding(Buffer2torchTensor(weights_.position_encoding->kernel, false).cuda());
        }
        if (weights_.token_type_embedding) {
            graph_runner_->setTokenTypeEmbedding(
                Buffer2torchTensor(weights_.token_type_embedding->kernel, false).cuda());
        }
        graph_runner_->setInputEmbeddingScalar(description_.input_embedding_scalar);
        RTP_LLM_CHECK_WITH_INFO(graph_runner_ != nullptr, "graph_runner_ can't be null");
        py_init_result = py_instance.attr("initialize")(init_resources);
        RTP_LLM_LOG_INFO("allocation records before capture:");
        params.device->traceMemoryUsage();
        graph_runner_->initCapture();
        RTP_LLM_LOG_INFO("allocation records after capture:");
        params.device->traceMemoryUsage();
    }

    auto py_init_success = py_init_result.cast<bool>();
    if (!py_init_success) {
        throw std::runtime_error("PyWrappedModel constructor: Python model initialization failed.");
    }

    if (params.device->getDeviceProperties().enable_prefill_cp) {
        context_parallel_processor_ = ContextParallelProcessorFactory::create(ProcessorType::ZIG_ZAG);
        RTP_LLM_LOG_INFO("Context parallel processor initialized with ZIG_ZAG strategy.");
    }

    RTP_LLM_LOG_INFO("PyWrappedModel initialized done.");
}

void PyWrappedModel::releaseBuffers() {
    buffer_holder_.release();
}

BufferPtr PyWrappedModel::tpSyncEmbeddingOrLogits(const BufferPtr& buffer) {
    return rtp_llm::tpSyncEmbeddingOrLogits(device_, device_props_, buffer);
}

MicroBatchPlan PyWrappedModel::planMicroBatches(const GptModelInputs& inputs) {
    return rtp_llm::planMicroBatches(inputs, layer_num_, device_props_);
}

std::pair<std::vector<GptModelInputs>, std::vector<TokenSliceInfo>>
PyWrappedModel::splitInputsIntoMicroBatches(const GptModelInputs& inputs, const MicroBatchPlan& micro_batch_plan) {
    return rtp_llm::splitInputsIntoMicroBatches(inputs, micro_batch_plan, device_);
}

void PyWrappedModel::holdInputsHostBuffers(const GptModelInputs& inputs) {
    rtp_llm::holdInputsHostBuffers(buffer_holder_, inputs);
}

GptModelOutputs PyWrappedModel::forwardPostLayers(BufferPtr             input,
                                                  bool                  has_context_request,
                                                  bool                  need_all_logits,
                                                  const BufferPtr&      lm_output_indexes,
                                                  bool                  enable_sp,
                                                  size_t                token_num,
                                                  const GptModelInputs& inputs,
                                                  BufferPtr             merged_eagle3_hidden,
                                                  bool                  skip_final_layernorm) {
    (void)inputs;
    DevicePerfWrapper wrapper(device_, "forwardPostLayers");
    BufferPtr         all_gather_output = nullptr;
    if (enable_sp && device_props_.tp_size > 1) {
        all_gather_output = device_->allocateBuffer(
            {input->type(), {input->shape()[0] * device_props_.tp_size, input->shape()[1]}}, {"all_gather_output"});
        size_t m                 = all_gather_output->shape()[0];
        int    m_split           = device_props_.m_split;
        size_t overlap_comm_type = device_props_.overlap_comm_type;
        if (overlap_comm_type == 1 && m_split > 0) {
            size_t token_idx    = 0;
            size_t ag_token_idx = 0;
            size_t m_chunk      = m / m_split;
            if (m > 128) {
                m_chunk = (m / m_split + 127) & ~127;
            }
            while (token_idx < m) {
                const auto micro_batch_tokens      = std::min(m - token_idx, m_chunk);
                const auto ag_micro_batch_tokens   = micro_batch_tokens / device_props_.tp_size;
                auto       micro_batch_recv_buffer = all_gather_output->slice(token_idx, micro_batch_tokens);
                auto       micro_ag_send_buffer    = input->slice(ag_token_idx, ag_micro_batch_tokens);
                device_->allGather({{micro_batch_recv_buffer}, ParallelMode::TP, {micro_ag_send_buffer}, false});
                token_idx += micro_batch_tokens;
                ag_token_idx += ag_micro_batch_tokens;
            }
        } else {
            device_->allGather({{all_gather_output}, ParallelMode::TP, {input}, false});
        }
        size_t pad_mod_num = device_props_.tp_size * std::max((size_t)1, (size_t)device_props_.m_split);
        if (token_num % pad_mod_num != 0) {
            input = device_->clone({all_gather_output->view(0, token_num), AllocationType::DEVICE});
        } else {
            input = all_gather_output;
        }
    }

    auto hidden = input;
    if (weights_.final_layernorm && !skip_final_layernorm) {
        auto final_layernorm = device_->layernorm(LayernormParams(hidden,
                                                                  nullptr,
                                                                  rtp_llm::mayGetRef(weights_.final_layernorm),
                                                                  std::nullopt,
                                                                  std::nullopt,
                                                                  std::nullopt,
                                                                  0.f,
                                                                  description_.layernorm_eps,
                                                                  true,
                                                                  false,
                                                                  description_.norm_type));
        hidden               = std::move(final_layernorm.output);
    }
    printBufferData(*hidden, "final_hidden");

    const auto& lm_head = weights_.lm_head;
    if (lm_head) {
        printBufferData(*lm_output_indexes, "lm_output_indexes");
        buffer_holder_.hold_host(lm_output_indexes);
        BufferPtr lm_output_indexes_device = device_->clone({*lm_output_indexes, AllocationType::DEVICE});
        auto      last_hidden =
            has_context_request && !need_all_logits ? device_->select({*hidden, *lm_output_indexes_device}) : hidden;
        printBufferData(*last_hidden, "last_hidden");
        auto logits = device_->gemm(GemmParams(*last_hidden,
                                               *(lm_head->kernel),
                                               std::nullopt,
                                               nullptr,
                                               rtp_llm::DataType::TYPE_FP32,
                                               rtp_llm::DataType::TYPE_FP32,
                                               TransposeOperation::NONE,
                                               TransposeOperation::TRANSPOSE));
        printBufferData(*logits, "logits");
        if (device_props_.tp_size > 1) {
            logits = tpSyncEmbeddingOrLogits(logits);
        }
        if (device_->initParams().profile_debug_logging_config.check_nan) {
            (void)device_->checkNAN(*last_hidden);
            (void)device_->checkNAN(*logits);
        }
        BufferPtr softmax_result;
        if (need_all_logits) {
            auto last_logits = device_->select({*logits, *lm_output_indexes_device});
            return {std::move(last_logits),
                    std::move(last_hidden),
                    std::move(hidden),
                    std::move(logits),
                    std::move(softmax_result)};
        }
        hidden = merged_eagle3_hidden ? merged_eagle3_hidden : hidden;
        return {std::move(logits), std::move(last_hidden), std::move(hidden), nullptr, std::move(softmax_result)};
    } else {
        return {nullptr, nullptr, std::move(hidden)};
    }
}

torch::Tensor PyWrappedModel::tensorHoldHostAndToCuda(const torch::Tensor& tensor) {
    if (tensor.device().is_cuda()) {
        return tensor;
    }

    buffer_holder_.hold_host(tensor);
    return tensor.to(torch::kCUDA, /*non_blocking=*/true, /*copy=*/false);
}

PyWrappedModel::~PyWrappedModel() {
    try {
        py::gil_scoped_acquire gil;
        // Always release py_model_ since it's always initialized now
        py_model_.release();
        if (graph_runner_ != nullptr) {
            delete graph_runner_;
            graph_runner_ = nullptr;
        }
        RTP_LLM_LOG_INFO("PyWrappedModel destroyed, Python object instance released.");
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python error during PyWrappedModel destruction: %s", e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("C++ error during PyWrappedModel destruction: %s", e.what());
    }
}

// Helper function to build PyAttentionInputs from GptModelInputs
torch_ext::PyAttentionInputs PyWrappedModel::buildPyAttentionInputs(const GptModelInputs& inputs) {
    DevicePerfWrapper            wrapper(device_, "py model buildPyAttentionInputs");
    torch_ext::PyAttentionInputs py_attn_inputs;
    py_attn_inputs.prefix_lengths   = Buffer2torchTensor(inputs.prefix_lengths, false);
    py_attn_inputs.sequence_lengths = Buffer2torchTensor(inputs.sequence_lengths, false);
    py_attn_inputs.input_lengths    = Buffer2torchTensor(inputs.input_lengths, false);

    if (inputs.kv_cache_block_id) {
        py_attn_inputs.kv_cache_block_id_host = Buffer2torchTensor(inputs.kv_cache_block_id);
    }
    if (inputs.kv_cache_layer_to_group) {
        py_attn_inputs.kv_cache_layer_to_group = Buffer2torchTensor(inputs.kv_cache_layer_to_group, false);
    }

    // Calculate cu_seqlens
    int    batch_size         = py_attn_inputs.input_lengths.size(0);
    size_t context_batch_size = py_attn_inputs.prefix_lengths.size(0);
    size_t decode_batch_size  = py_attn_inputs.sequence_lengths.size(0);
    py_attn_inputs.dtype      = dataTypeToTorchType(description_.data_type);
    py_attn_inputs.is_prefill = !decode_batch_size;
    RTP_LLM_CHECK_WITH_INFO(
        context_batch_size + decode_batch_size == batch_size,
        "batch size check failed context_batch_size[%ld] decode_batch_size[%ld] total_batch_size[%ld]",
        context_batch_size,
        decode_batch_size,
        batch_size);

    if (context_batch_size > 0) {
        torch::Tensor cu_seqlens =
            torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
        torch::Tensor cu_kv_seqlens =
            torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));

        cu_seqlens.slice(0, 1, context_batch_size + 1) = py_attn_inputs.input_lengths.cumsum(0);
        cu_kv_seqlens.slice(0, 1, context_batch_size + 1) =
            py_attn_inputs.input_lengths.add(py_attn_inputs.prefix_lengths).cumsum(0);

        py_attn_inputs.context_total_kv_length = cu_kv_seqlens[context_batch_size].item<int>();
        py_attn_inputs.total_tokens            = cu_seqlens[batch_size].item<int>();
        py_attn_inputs.cu_seqlens              = tensorHoldHostAndToCuda(cu_seqlens);
        py_attn_inputs.cu_kv_seqlens           = tensorHoldHostAndToCuda(cu_kv_seqlens);
    } else {
        py_attn_inputs.total_tokens = 0;
        py_attn_inputs.cu_seqlens =
            torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        py_attn_inputs.cu_kv_seqlens =
            torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        torch::Tensor decode_cu_seqlens = torch::arange(
            0, py_attn_inputs.sequence_lengths.size(0) + 1, 1, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
        py_attn_inputs.decode_cu_seqlens_host = decode_cu_seqlens;
        py_attn_inputs.decode_cu_seqlens_d    = tensorHoldHostAndToCuda(decode_cu_seqlens);
    }

    // create device tensors
    py_attn_inputs.prefix_lengths_d          = tensorHoldHostAndToCuda(py_attn_inputs.prefix_lengths);
    py_attn_inputs.sequence_lengths_plus_1_d = tensorHoldHostAndToCuda(py_attn_inputs.sequence_lengths + 1);
    py_attn_inputs.input_lengths_d           = tensorHoldHostAndToCuda(py_attn_inputs.input_lengths);
    return py_attn_inputs;
}

// Helper function to setup KV cache for attention inputs
void PyWrappedModel::setupKVCacheForAttentionInputs(torch_ext::PyAttentionInputs& py_attn_inputs,
                                                    const GptModelInputs&         inputs,
                                                    BufferPtr&                    kv_cache_block_id_device,
                                                    std::vector<BufferPtr>*       kv_cache_block_id_device_by_group) {
    DevicePerfWrapper wrapper(device_, "py model setupKVCacheForAttentionInputs");
    if (!inputs.kv_cache_block_id) {
        return;
    }
    const auto& shape = inputs.kv_cache_block_id->shape();
    RTP_LLM_CHECK_WITH_INFO(shape.size() == 3, "kv_cache_block_id shape should be 3");
    // New layout: [group, batch, max_blocks]
    // build per-group contiguous 2-D tables on device.
    const size_t group = shape[0];
    kv_cache_block_id_device_by_group->clear();
    kv_cache_block_id_device_by_group->reserve(group);

    py_attn_inputs.kv_cache_block_id_host_by_group.clear();
    py_attn_inputs.kv_cache_block_id_device_by_group.clear();
    py_attn_inputs.kv_cache_block_id_host_by_group.reserve(group);
    py_attn_inputs.kv_cache_block_id_device_by_group.reserve(group);

    for (size_t g = 0; g < group; ++g) {
        // group view: [batch, max_blocks] on HOST
        const auto group_view = (*inputs.kv_cache_block_id)[g];
        py_attn_inputs.kv_cache_block_id_host_by_group.push_back(Buffer2torchTensor(group_view, false));

        auto dev_group = device_->clone({group_view, AllocationType::DEVICE, {"kv_cache_block_id_group_device"}});
        kv_cache_block_id_device_by_group->push_back(dev_group);
        py_attn_inputs.kv_cache_block_id_device_by_group.push_back(Buffer2torchTensor(dev_group, false));

        if (g == 0) {
            kv_cache_block_id_device = dev_group;
        }
    }

    // Legacy 2-D fields default to group 0.
    // NOTE: keep host/device 2-D fields consistent to avoid shape mismatch in CUDA graph replay path.
    py_attn_inputs.kv_cache_block_id_device = py_attn_inputs.kv_cache_block_id_device_by_group[0];
    py_attn_inputs.kv_cache_block_id_host   = py_attn_inputs.kv_cache_block_id_host_by_group[0];
}

// Helper function to build BertEmbeddingInputs from GptModelInputs
torch_ext::BertEmbeddingInputs PyWrappedModel::buildBertEmbeddingInputs(const GptModelInputs& inputs) {
    DevicePerfWrapper              wrapper(device_, "py model buildBertEmbeddingInputs");
    torch_ext::BertEmbeddingInputs bert_embedding_inputs;

    // Convert combo_position_ids from Buffer to torch::Tensor
    if (inputs.combo_position_ids) {
        bert_embedding_inputs.combo_position_ids = Buffer2torchTensor(inputs.combo_position_ids, false).cuda();
    }

    // Convert combo_tokens_type_ids from Buffer to torch::Tensor
    if (inputs.combo_tokens_type_ids) {
        {
            DevicePerfWrapper wrapper(device_, "py model combo_tokens.cuda()");
            bert_embedding_inputs.combo_tokens_type_ids =
                Buffer2torchTensor(inputs.combo_tokens_type_ids, false).cuda();
        }
    }

    // Get position_encoding from model weights (no clone needed for weights)
    if (weights_.position_encoding) {
        DevicePerfWrapper wrapper(device_, "py model weights_.position_encoding->kernel");
        bert_embedding_inputs.position_encoding = Buffer2torchTensor(weights_.position_encoding->kernel, false);
    }

    // Get token_type_embedding from model weights (no clone needed for weights)
    if (weights_.token_type_embedding) {
        DevicePerfWrapper wrapper(device_, "py model weights_.token_type_embedding->kernel");
        bert_embedding_inputs.token_type_embedding = Buffer2torchTensor(weights_.token_type_embedding->kernel, false);
    }

    // Set input_embedding_scalar
    bert_embedding_inputs.input_embedding_scalar = description_.input_embedding_scalar;
    return bert_embedding_inputs;
}

GptModelOutputs PyWrappedModel::callForwardPostLayers(BufferPtr             hidden_states,
                                                      const GptModelInputs& inputs,
                                                      bool                  skip_final_layernorm,
                                                      size_t                num_valid_tokens) {
    size_t num_input_tokens =
        num_valid_tokens != static_cast<size_t>(-1) ? num_valid_tokens : inputs.combo_tokens->shape()[0];
    return forwardPostLayers(hidden_states,
                             inputs.input_lengths->shape()[0] != inputs.sequence_lengths->shape()[0],
                             false,
                             inputs.lm_output_indexes,
                             false,
                             num_input_tokens,
                             inputs,
                             nullptr,
                             skip_final_layernorm);
}

std::optional<PyCacheStoreInputs> PyWrappedModel::prepareWriteCacheParams(const GptModelInputs& inputs) {
    std::optional<PyCacheStoreInputs> params;
    if (!inputs.warmup && inputs.pd_separation) {
        const auto           decoder_batch_size = inputs.sequence_lengths->shape()[0];
        const auto           context_batch_size = inputs.input_lengths->shape()[0] - decoder_batch_size;
        std::vector<int64_t> cache_keys_vec;
        if (inputs.cache_keys) {
            cache_keys_vec = rtp_llm::buffer2vector<int64_t>(*inputs.cache_keys);
        }
        torch::Tensor kv_cache_layer_to_group = inputs.kv_cache_layer_to_group ?
                                                    Buffer2torchTensor(inputs.kv_cache_layer_to_group, false) :
                                                    torch::Tensor();
        torch::Tensor kv_cache_group_types =
            inputs.kv_cache_group_types ? Buffer2torchTensor(inputs.kv_cache_group_types, false) : torch::Tensor();
        PyCacheStoreInputs cache_store_inputs{context_batch_size,
                                              decoder_batch_size,
                                              Buffer2torchTensor(inputs.request_id, false),
                                              Buffer2torchTensor(inputs.request_pd_separation, false),
                                              kv_cache_layer_to_group,
                                              kv_cache_group_types,
                                              transVectorToString(cache_keys_vec),
                                              inputs.seq_size_per_block,
                                              inputs.kv_block_stride_bytes,
                                              inputs.kv_scale_stride_bytes,
                                              inputs.pd_separation,
                                              model_id_,
                                              inputs.decode_entrance,
                                              inputs.warmup,
                                              description_.attention_conf.use_mla
                                                  && device_->mla_ops_type != rtp_llm::MlaOpsType::MHA};
        params = cache_store_inputs;
    }
    return params;
}

GptModelOutputs PyWrappedModel::forwardMicroBatched(const GptModelInputs& inputs) {
    py::object py_forward_method = py_model_.attr("forward_micro_batch");
    if (device_props_.ffn_as_service) {
        py::object py_outputs_obj = py_forward_method(std::vector<PyModelInputs>{});
        return GptModelOutputs({nullptr, nullptr, nullptr, nullptr, nullptr});
    }

    auto micro_batch_plan  = planMicroBatches(inputs);
    auto [split_inputs, _] = splitInputsIntoMicroBatches(inputs, micro_batch_plan);
    std::vector<PyModelInputs> input_list;
    input_list.reserve(split_inputs.size());
    std::vector<BufferPtr>              kv_cache_block_ids_device(split_inputs.size());
    std::vector<std::vector<BufferPtr>> kv_cache_block_ids_device_by_group(split_inputs.size());

    for (size_t i = 0; i < split_inputs.size(); ++i) {
        const auto& micro_inputs          = split_inputs[i].kv_cache_block_id ? split_inputs[i] : split_inputs[0];
        auto        py_attn_inputs        = buildPyAttentionInputs(micro_inputs);
        auto        bert_embedding_inputs = buildBertEmbeddingInputs(micro_inputs);
        setupKVCacheForAttentionInputs(
            py_attn_inputs, micro_inputs, kv_cache_block_ids_device[i], &kv_cache_block_ids_device_by_group[i]);

        calculatePaddingOffset(py_attn_inputs);
        py_attn_inputs.padding_offset = tensorHoldHostAndToCuda(py_attn_inputs.padding_offset);

        torch::Tensor token_ids = Buffer2torchTensor(micro_inputs.combo_tokens).cuda();
        torch::Tensor input_hiddens =
            inputs.last_hidden_states ? Buffer2torchTensor(inputs.last_hidden_states, false) : torch::empty({0});
        input_list.emplace_back(PyModelInputs{token_ids, input_hiddens, py_attn_inputs, bert_embedding_inputs});
    }

    py::object py_outputs_obj   = py_forward_method(input_list);
    auto       py_model_outputs = py_outputs_obj.cast<std::vector<PyModelOutputs>>();
    RTP_LLM_CHECK_WITH_INFO(py_model_outputs.size() == input_list.size(),
                            "py_model_outputs.size:%d != micro_batch_inputs.size:%d",
                            py_model_outputs.size(),
                            input_list.size());

    // TODO: merge hidden states in one buffer
    BufferPtr hidden_states = nullptr;
    if (!micro_batch_plan.enable) {
        RTP_LLM_CHECK_WITH_INFO(py_model_outputs[0].hidden_states.size(0) == inputs.combo_tokens->shape()[0],
                                "py_model_outputs[0].hidden_states.size(0):%d != inputs.combo_tokens->shape()[0]:%d",
                                py_model_outputs[0].hidden_states.size(0),
                                inputs.combo_tokens->shape()[0]);
        hidden_states = torchTensor2Buffer(py_model_outputs[0].hidden_states);
    } else {
        hidden_states =
            device_->allocateBuffer({description_.data_type,
                                     {inputs.combo_tokens->shape()[0],
                                      description_.attention_conf.head_num * description_.attention_conf.size_per_head},
                                     AllocationType::DEVICE});
        int offset = 0;
        for (int i = 0; i < py_model_outputs.size(); i++) {
            RTP_LLM_CHECK_WITH_INFO(
                offset + py_model_outputs[i].hidden_states.size(0) <= inputs.combo_tokens->shape()[0],
                "offset + py_model_outputs[i].hidden_states.size(0):%d > inputs.combo_tokens->shape()[0]:%d",
                offset + py_model_outputs[i].hidden_states.size(0),
                inputs.combo_tokens->shape()[0]);
            auto hidden_states_slice = hidden_states->slice(offset, offset + py_model_outputs[i].hidden_states.size(0));
            auto py_model_output     = py_model_outputs[i];
            device_->copy({*hidden_states_slice, *torchTensor2Buffer(py_model_output.hidden_states)});
            offset += py_model_outputs[i].hidden_states.size(0);
        }
        RTP_LLM_CHECK_WITH_INFO(offset == inputs.combo_tokens->shape()[0],
                                "total out hidden size:%d != inputs.combo_tokens->shape()[0]:%d",
                                offset,
                                inputs.combo_tokens->shape()[0]);
    }

    RTP_LLM_LOG_DEBUG("Python object instance forward method called successfully.");

    return callForwardPostLayers(hidden_states, inputs, false);
}

GptModelOutputs PyWrappedModel::forward(const GptModelInputs& inputs) {
    DevicePerfWrapper wrapper(device_, "py model forward");
    holdInputsHostBuffers(inputs);
    py::gil_scoped_acquire gil;
    try {
        RTP_LLM_LOG_DEBUG("Calling forward method on Python object instance.");

        if (int(device_props_.enable_layer_micro_batch)) {
            return forwardMicroBatched(inputs);
        }
        PyContextParallelParams cp_params;
        if (device_props_.enable_prefill_cp) {
            context_parallel_processor_->handleInputs(device_, const_cast<GptModelInputs&>(inputs), cp_params);
        }

        torch::Tensor token_ids;
        token_ids = tensorHoldHostAndToCuda(Buffer2torchTensor(inputs.combo_tokens, false));

        torch::Tensor input_hiddens =
            inputs.last_hidden_states ? Buffer2torchTensor(inputs.last_hidden_states, false) : torch::empty({0});

        auto attention_inputs      = buildPyAttentionInputs(inputs);
        auto bert_embedding_inputs = buildBertEmbeddingInputs(inputs);

        if (device_props_.enable_prefill_cp) {
            attention_inputs.context_parallel_info = cp_params;
        }

        BufferPtr              kv_cache_block_id_device;
        std::vector<BufferPtr> kv_cache_block_id_device_by_group;
        if (!inputs.warmup && inputs.pd_separation) {
            attention_inputs.cache_store_inputs = prepareWriteCacheParams(inputs);
        }
        setupKVCacheForAttentionInputs(
            attention_inputs, inputs, kv_cache_block_id_device, &kv_cache_block_id_device_by_group);

        calculatePaddingOffset(attention_inputs);
        attention_inputs.padding_offset = tensorHoldHostAndToCuda(attention_inputs.padding_offset);

        auto py_model_inputs = PyModelInputs({token_ids, input_hiddens, attention_inputs, bert_embedding_inputs});
        PyModelOutputs py_model_outputs;
        BufferPtr      hidden_states;

        // Cast the Python object to PyModelOutputs and extract hidden states
        CudaGraphState graph_state;
        if (enable_cuda_graph_ && graph_runner_->canRun(py_model_inputs, graph_state)) {
            DevicePerfWrapper wrapper(device_, "cuda graph python forward");
            py_model_inputs.attention_inputs.is_s_padded = true;
            py_model_outputs                             = graph_runner_->forward(py_model_inputs, graph_state);
            hidden_states = device_->clone({*torchTensor2Buffer(py_model_outputs.hidden_states)});
        } else {
            DevicePerfWrapper wrapper(device_, "normal forward");
            auto              attn_pyobj       = py_model_.attr("prepare_fmha_impl")(py_model_inputs, false);
            auto              py_model_forward = py_model_.attr("forward");
            auto              outputs          = py_model_forward(py_model_inputs, attn_pyobj);
            py_model_outputs                   = outputs.cast<PyModelOutputs>();
            hidden_states                      = device_->clone({*torchTensor2Buffer(py_model_outputs.hidden_states)});
        }

        RTP_LLM_LOG_DEBUG("Python object instance forward method called successfully.");
        if (device_props_.enable_prefill_cp) {
            size_t num_valid_tokens =
                context_parallel_processor_->handleOutputs(device_, hidden_states, inputs, cp_params);
            return callForwardPostLayers(hidden_states, inputs, true, num_valid_tokens);
        }
        return callForwardPostLayers(hidden_states, inputs, true);

    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python error during forward call on Python instance: %s", e.what());
        throw std::runtime_error(std::string("pybind11 error during forward call on Python instance: ") + e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("C++ error during forward call on Python instance: %s", e.what());
        throw std::runtime_error(std::string("C++ error during forward call on Python instance: ") + e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("An unknown error occurred during forward call on Python instance.");
        throw std::runtime_error("An unknown error occurred during forward call on Python instance.");
    }
}

}  // namespace rtp_llm
