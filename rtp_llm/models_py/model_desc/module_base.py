import logging
import os
import types
from typing import Any, Optional

import torch
from torch import Tensor, nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules import AttnImplFactory
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.ops.compute_ops import (
    DeviceType,
    KVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
    get_device,
)
from rtp_llm.utils.model_weight import W


class GptModelBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config,
        weight: ModelWeights,
        max_generate_batch_size: int,
        fmha_config=None,  # Optional FMHAConfig
        py_hw_kernel_config=None,  # Optional HWKernelConfig
        device_resource_config: Optional[
            DeviceResourceConfig
        ] = None,  # Optional DeviceResourceConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weight = weight
        self.fmha_config = fmha_config
        self.py_hw_kernel_config = py_hw_kernel_config
        self.micro_batch_size: int = (
            1
            if device_resource_config
            and device_resource_config.enable_layer_micro_batch == 0
            else 2
        )
        self.layer_num: int = config.num_layers
        self.vocab_size: int = config.vocab_size

        self.kv_cache: Optional[KVCache] = None
        self.device_type: DeviceType = get_device().get_device_type()

        ## (batch_size -> fmha_params)
        self.params_dict: dict[int, Any] = {}

        if os.environ.get("TORCH_PROFILE_OUTPUT_DIR"):
            self._install_profiler_wrapper()

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.kv_cache = init_resource.kv_cache
        if self.kv_cache is not None:
            num_layers = len(self.kv_cache.kv_cache_base_by_layer)
            layer0_shape = (
                self.kv_cache.kv_cache_base_by_layer[0].shape
                if num_layers > 0
                and self.kv_cache.kv_cache_base_by_layer[0] is not None
                else None
            )
            num_scale_layers = len(self.kv_cache.kv_scale_base_by_layer)
            logging.info(
                f"GptModelBase initialized with "
                f"num_kv_layers={num_layers}, "
                f"layer0_kv_cache_shape={layer0_shape}, "
                f"num_scale_layers={num_scale_layers}, "
            )
        return True

    ## for cuda graph attn kernel params' fill
    def fill_params(
        self,
        sequence_lengths: Tensor,
        input_lengths: Tensor,
        kv_cache_block_id_host: Tensor,
        replay_batch_size: int,
        capture_batch_size: int,
        seq_size_per_block: int,
    ):
        assert capture_batch_size in self.params_dict
        params_ptr = self.params_dict[capture_batch_size]
        assert params_ptr is not None
        params_ptr.fillParams(
            sequence_lengths,
            input_lengths,
            kv_cache_block_id_host,
            replay_batch_size,
            seq_size_per_block,
        )

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        fmha_impl = AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self.weight,
            inputs.attention_inputs,
            self.fmha_config,
            is_cuda_graph,
        )
        return fmha_impl

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")

    def _install_profiler_wrapper(self) -> None:
        """Replace self.forward with a profiled version that captures tensor shapes."""
        self._real_forward = self.__class__.forward.__get__(self, self.__class__)

        def _profiled_forward(model_self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
            return model_self._forward_with_profiler(inputs, fmha_impl)

        self.forward = types.MethodType(_profiled_forward, self)

    def _build_profiler(self) -> "torch.profiler.profile":
        warmup_steps = int(os.environ.get("TORCH_PROFILE_WARMUP", "2"))
        active_steps = int(os.environ.get("TORCH_PROFILE_ACTIVE", "1"))
        output_dir = os.environ.get("TORCH_PROFILE_OUTPUT_DIR", "/tmp/rtp_llm_profile")
        os.makedirs(output_dir, exist_ok=True)

        def on_trace_ready(profiler: "torch.profiler.profile") -> None:
            output_path = os.path.join(output_dir, f"trace_step{profiler.step_num}.json")
            profiler.export_chrome_trace(output_path)
            logging.info(f"[torch.profiler] trace saved to {output_path}")

        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=warmup_steps,
                active=active_steps,
                repeat=1,
            ),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            on_trace_ready=on_trace_ready,
        )

    def _forward_with_profiler(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        warmup_steps = int(os.environ.get("TORCH_PROFILE_WARMUP", "2"))
        active_steps = int(os.environ.get("TORCH_PROFILE_ACTIVE", "1"))
        total_steps = warmup_steps + active_steps

        if not hasattr(self, "_profiler_state"):
            self._profiler_state: dict[str, Any] = {
                "profiler": None,
                "step": 0,
                "done": False,
            }

        state = self._profiler_state

        if state["done"]:
            return self._real_forward(inputs, fmha_impl)

        if state["profiler"] is None:
            state["profiler"] = self._build_profiler()
            state["profiler"].__enter__()

        with torch.profiler.record_function("model_forward"):
            outputs = self._real_forward(inputs, fmha_impl)

        state["profiler"].step()
        state["step"] += 1

        if state["step"] >= total_steps:
            state["profiler"].__exit__(None, None, None)
            state["done"] = True

        return outputs
