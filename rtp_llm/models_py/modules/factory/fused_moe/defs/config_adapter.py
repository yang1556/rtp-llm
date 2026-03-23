"""
Adapter to provide a unified interface from individual config objects.
This allows Router and Executor classes to work with specific config objects.
"""

from typing import Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import QuantizationConfig
from rtp_llm.ops import MoeConfig, ParallelismConfig


class MoEConfigAdapter:
    """
    Adapter class that provides a unified interface
    from individual configuration objects.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        moe_config: Optional[MoeConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        enable_cuda_graph: bool = False,
        layer_idx: Optional[int] = None,
    ):
        self.model_config = model_config
        self.parallelism_config = parallelism_config
        self.moe_config = moe_config or MoeConfig()
        self.quant_config = quant_config

        # Provide shortcut access to commonly used attributes
        self.ep_size = parallelism_config.ep_size
        self.ep_rank = parallelism_config.ep_rank
        self.tp_size = parallelism_config.get_attn_tp_size()
        self.tp_rank = parallelism_config.get_attn_tp_rank()
        self.dp_size = parallelism_config.dp_size
        self.dp_rank = parallelism_config.dp_rank
        self.world_size = parallelism_config.world_size
        # Calculate local_rank from world_rank and local_world_size
        self.local_rank = parallelism_config.local_rank

        self.expert_num = model_config.expert_num
        self.moe_k = model_config.moe_k
        self.moe_topk_group = model_config.moe_topk_group
        self.hidden_size = model_config.hidden_size
        self.data_type = model_config.data_type
        self.head_num = model_config.attn_config.head_num
        self.ll_num_max_token = moe_config.ll_num_max_token if moe_config else 0
        self.masked_max_token_num = moe_config.masked_max_token_num if moe_config else 0
        self.moe_strategy = (self.moe_config.moe_strategy
                             if self.moe_config else "auto")
        self.enable_cuda_graph = enable_cuda_graph

        # Layer index for mixed precision support
        self.layer_idx: Optional[int] = layer_idx
        # w4a8_max_layer_num from model_config (for mixed precision MoE)
        self.w4a8_max_layer_num: int = getattr(model_config, 'w4a8_max_layer_num', -1)

    @property
    def activation_type(self):
        """Access activation_type from model_config when needed."""
        return self.model_config.activation_type
