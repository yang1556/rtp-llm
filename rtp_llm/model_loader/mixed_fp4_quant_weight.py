import copy
from typing import Any, List

from rtp_llm.config.quant_config import ModelOptFp4Config, QuantizationConfig
from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnAtomicWeight,
    _linear_attn_split_stratey,
)
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)

from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkvz_transpose_reorder,
    merge_ba_transpose_reorder,
    transpose,
)

IN_PROJ_QKV_SUFFIX = ".in_proj_qkv."
IN_PROJ_Z_SUFFIX = ".in_proj_z."
IN_PROJ_A_SUFFIX = ".in_proj_a."
IN_PROJ_B_SUFFIX = ".in_proj_b."

def mixed_fp4_gpt_style_tp_strategy():
    tp_strategy = copy.deepcopy(W.gpt_style_tp_strategy)
    tp_strategy.update(_linear_attn_split_stratey)
    return tp_strategy

class MixedFp4PerGroupAtomicWeight(AtomicWeight):
    gpt_style_tp_strategy = mixed_fp4_gpt_style_tp_strategy()

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _get_split_func(self):
        return self.gpt_style_tp_strategy[self.name]

class MixedFp4PerGroupLinearAttnAtomicWeight(
    LinearAttnAtomicWeight, MixedFp4PerGroupAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class MixedFp4PerGroupFfnAtomicWeight(FfnAtomicWeight, MixedFp4PerGroupAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class MixedFp4PerGroupMoeAtomicWeight(MoeAtomicWeight, MixedFp4PerGroupAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


def create_mixed_fp4_per_group_weight(
    src_weight_info: WeightModule, *args: Any, **kwargs: Any
) -> MixedFp4PerGroupAtomicWeight:
    if isinstance(src_weight_info, LinearAttnAtomicWeight):
        return MixedFp4PerGroupLinearAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return MixedFp4PerGroupAtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")


class MixedFp4Weight(CompositeWeight, QuantWeight):
    unquantized_weight_list = [
        W.linear_attn_qkvz_w,
        W.linear_attn_ba_w,
        W.linear_attn_alog,
        W.linear_attn_dt_b,
        W.linear_attn_conv1d_w,
        W.linear_attn_out_w,
        W.linear_attn_norm_w,
    ]

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, ModelOptFp4Config
        ):
            return False
        name = src_weight_info.name
        return name in cls.unquantized_weight_list

    def __init__(
        self,
        src_weight_info: WeightModule,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        kernel: WeightModule = None
        
        if src_weight_info.name == W.linear_attn_qkvz_w:
            kernel = self._get_linear_attn_qkvz_weight(src_weight_info)
        elif src_weight_info.name == W.linear_attn_ba_w:
            kernel = self._get_linear_attn_ba_weight(src_weight_info)
        elif src_weight_info.name == W.linear_attn_out_w:
            kernel = self._get_linear_attn_out_weight(src_weight_info)
        elif isinstance(src_weight_info, LinearAttnAtomicWeight):
            kernel = self._get_linear_attn_common_weight(src_weight_info)

        sub_weights = {kernel.name: kernel}

        super().__init__(sub_weights, quant_config=quant_config, *args, **kwargs)
        self.kernel = sub_weights.get(kernel.name)

    def _get_linear_attn_qkvz_weight(self, src_weight_info: LinearAttnAtomicWeight):
        assert src_weight_info.name == W.linear_attn_qkvz_w
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 2
        qkv_weight = next((w for w in weights if IN_PROJ_QKV_SUFFIX in w.name), None)
        z_weight = next((w for w in weights if IN_PROJ_Z_SUFFIX in w.name), None)

        if qkv_weight is None or z_weight is None:
            raise ValueError("Missing required weights: in_proj_qkv or in_proj_z")

        merge_weights = [qkv_weight, z_weight]
            
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            src_weight_info.name,
            merge_weights,
            merge_qkvz_transpose_reorder,
            config=src_weight_info.config,
        )
        return kernel

    def _get_linear_attn_ba_weight(self, src_weight_info: LinearAttnAtomicWeight):
        assert src_weight_info.name == W.linear_attn_ba_w
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 2
        a_weight = next((w for w in weights if IN_PROJ_A_SUFFIX in w.name), None)
        b_weight = next((w for w in weights if IN_PROJ_B_SUFFIX in w.name), None)

        if a_weight is None or b_weight is None:
            raise ValueError("Missing required weights: in_proj_a or in_proj_b")

        merge_weights = [b_weight, a_weight]
            
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            src_weight_info.name,
            merge_weights,
            merge_ba_transpose_reorder,
            config=src_weight_info.config,
        )
        return kernel

    def _get_linear_attn_out_weight(self, src_weight_info: LinearAttnAtomicWeight):
        assert src_weight_info.name == W.linear_attn_out_w
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1
            
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            src_weight_info.name,
            weights,
            transpose,
            config=src_weight_info.config,
        )
        return kernel
    
    def _get_linear_attn_common_weight(self, src_weight_info: LinearAttnAtomicWeight):
        assert src_weight_info.name in self.unquantized_weight_list
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            src_weight_info.name,
            weights,
            identity,
            config=src_weight_info.config,
        )
        return kernel
