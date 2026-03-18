"""Rocm FP8 PerBlock quantization strategies"""

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy

class RocmFp8PerBlockPureTPStrategy(MoeStrategy):
    """Rocm FP8 PerBlock pure TP strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
            RocmExpertsFp8PerBlock,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.pure_tp_router import (
            PureTpRouterNoQuant,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fnuz,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=PureTpRouterNoQuant,
            executor_class=RocmExpertsFp8PerBlock,
            quant_config=quant_config,
        )