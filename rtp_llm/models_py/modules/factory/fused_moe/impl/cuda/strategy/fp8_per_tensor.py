"""CUDA FP8 PerTensor quantization strategies"""

from typing import Any

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class CudaFp8PerTensorEpLowLatencyStrategy(MoeStrategy):
    """CUDA FP8 PerTensor EP low latency strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy == "fp8_per_tensor_ep_low_latency" or config.moe_strategy == "auto")
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        # Support both native FP8 and mixed precision mode (W4A8 model with high layer index)
        is_native_fp8 = quant_method in ["FP8_PER_TENSOR_COMPRESSED", "FP8_DYNAMIC_PER_TENSOR"]
        is_mixed_fp8 = (
            quant_method == "W4A8_INT4_PER_CHANNEL"
            and resolver.is_mixed_precision_fp8_layer(config)
        )
        checker.check(is_native_fp8 or is_mixed_fp8)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
            CutlassBatchedExpertsFp8,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=CutlassBatchedExpertsFp8,
            quant_config=quant_config,
        )


class CudaFp8PerTensorEpNormalStrategy(MoeStrategy):
    """CUDA FP8 PerTensor EP normal mode strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy == "fp8_per_tensor_ep_normal" or config.moe_strategy == "auto")
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        # Support both native FP8 and mixed precision mode
        is_native_fp8 = quant_method in ["FP8_PER_TENSOR_COMPRESSED", "FP8_DYNAMIC_PER_TENSOR"]
        is_mixed_fp8 = (
            quant_method == "W4A8_INT4_PER_CHANNEL"
            and resolver.is_mixed_precision_fp8_layer(config)
        )
        checker.check(is_native_fp8 or is_mixed_fp8)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
            CutlassExpertsFp8,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterFp8PerTensor,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=DeepepNormalRouterFp8PerTensor,
            executor_class=CutlassExpertsFp8,
            quant_config=quant_config,
        )


class CudaFp8PerTensorNoDPStrategy(MoeStrategy):
    """CUDA FP8 PerTensor single GPU strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy == "fp8_per_tensor_no_dp" or config.moe_strategy == "auto")
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        # Support both native FP8 and mixed precision mode
        is_native_fp8 = quant_method in ["FP8_PER_TENSOR_COMPRESSED", "FP8_DYNAMIC_PER_TENSOR"]
        is_mixed_fp8 = (
            quant_method == "W4A8_INT4_PER_CHANNEL"
            and resolver.is_mixed_precision_fp8_layer(config)
        )
        checker.check(is_native_fp8 or is_mixed_fp8)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
            CutlassExpertsFp8,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerTensor,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp8PerTensor,
            executor_class=CutlassExpertsFp8,
            quant_config=quant_config,
        )
