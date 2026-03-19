import functools
import copy
from typing import Any, List

from rtp_llm.config.quant_config import ModelOptFp4Config, QuantizationConfig
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnAtomicWeight,
    _linear_attn_split_stratey,
)
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)

from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    merge_qkv_hf,
    identity,
    merge_qkvz_transpose_reorder,
    merge_ba_transpose_reorder,
    transpose,
    plus_one,
    split_q_gate,
)

from rtp_llm.utils.util import check_with_info

IN_PROJ_QKV_SUFFIX = ".in_proj_qkv."
IN_PROJ_Z_SUFFIX = ".in_proj_z."
IN_PROJ_A_SUFFIX = ".in_proj_a."
IN_PROJ_B_SUFFIX = ".in_proj_b."
SELF_ATTN_Q_SUFFIX = ".self_attn.q_proj."
SELF_ATTN_K_SUFFIX = ".self_attn.k_proj."
SELF_ATTN_V_SUFFIX = ".self_attn.v_proj."

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

class MixedFp4PerGroupAttnAtomicWeight(
    AttnAtomicWeight, MixedFp4PerGroupAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


def create_mixed_fp4_per_group_weight(
    src_weight_info: WeightModule, *args: Any, **kwargs: Any
) -> MixedFp4PerGroupAtomicWeight:
    if isinstance(src_weight_info, LinearAttnAtomicWeight):
        return MixedFp4PerGroupLinearAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AttnAtomicWeight):
        return MixedFp4PerGroupAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return MixedFp4PerGroupAtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")


class MixedFp4Weight(CompositeWeight, QuantWeight):
    unquantized_weight_list = [
        W.attn_qkv_w,
        W.attn_o_w,
        W.attn_gate_w,
        W.q_ln_gamma,
        W.k_ln_gamma,
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
        return name in cls.unquantized_weight_list and quant_config.mixed_attention

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
        elif src_weight_info.name == W.attn_qkv_w:
            kernel = self._get_qkv_weight(src_weight_info)
        elif src_weight_info.name == W.attn_o_w:
            kernel = self._get_mha_attn_out_weight(src_weight_info)
        elif src_weight_info.name == W.attn_gate_w:
            kernel = self._get_mha_attn_gate_weight(src_weight_info)
        elif src_weight_info.name == W.q_ln_gamma:
            kernel = self._get_q_norm_weight(src_weight_info)
        elif src_weight_info.name == W.k_ln_gamma:
            kernel = self._get_k_norm_weight(src_weight_info)

        sub_weights = {kernel.name: kernel}

        super().__init__(sub_weights, quant_config=quant_config, *args, **kwargs)
        self.kernel = sub_weights.get(kernel.name)

    def _get_qkv_weight(self, src_weight_info: AttnAtomicWeight):
        assert src_weight_info.name == W.attn_qkv_w
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 3
        head_num = src_weight_info.config.head_num
        head_dim = src_weight_info.config.size_per_head
        q_name = next((w.name for w in weights if SELF_ATTN_Q_SUFFIX in w.name), None)
        k_weight = next((w for w in weights if SELF_ATTN_K_SUFFIX in w.name), None)
        v_weight = next((w for w in weights if SELF_ATTN_V_SUFFIX in w.name), None)
        q_weight = CkptWeightInfo(
                        q_name,
                        functools.partial(
                            split_q_gate,
                            head_num=head_num,
                            head_dim=head_dim,
                            part=0,
                        ),
                   )
        qkv_list = [q_weight, k_weight, v_weight]
        
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.attn_qkv_w,
            qkv_list,
            merge_qkv_hf,
            config=src_weight_info.config,
        )

        return kernel

    def _get_mha_attn_gate_weight(self, src_weight_info: AttnAtomicWeight):
        assert src_weight_info.name == W.attn_gate_w
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1
        head_num = src_weight_info.config.head_num
        head_dim = src_weight_info.config.size_per_head
        q_name = weights[0].name
        q_weight = [CkptWeightInfo(
                        q_name,
                        functools.partial(
                            split_q_gate,
                            head_num=head_num,
                            head_dim=head_dim,
                            part=1,
                        ),
                   )]
        
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.attn_gate_w,
            q_weight,
            transpose,
            config=src_weight_info.config,
        )

        return kernel

    def _get_q_norm_weight(self, src_weight_info: AtomicWeight):
        assert src_weight_info.name == W.q_ln_gamma
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1
            
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            src_weight_info.name,
            weights,
            plus_one,
            config=src_weight_info.config,
        )
        return kernel

    def _get_k_norm_weight(self, src_weight_info: AtomicWeight):
        assert src_weight_info.name == W.k_ln_gamma
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1
            
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            src_weight_info.name,
            weights,
            plus_one,
            config=src_weight_info.config,
        )
        return kernel

    def _get_mha_attn_out_weight(self, src_weight_info: AttnAtomicWeight):
        check_with_info(
            src_weight_info.name == W.attn_o_w,
            "src_weight_info.name != W.attn_o_w, actual: {}".format(
                src_weight_info.name
            ),
        )
        check_with_info(
            isinstance(src_weight_info, AttnAtomicWeight),
            "src_weight_info is not AttnAtomicWeight, actual: {}".format(
                src_weight_info
            ),
        )

        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.attn_o_w,
            src_weight_info.weights,
            transpose,
            config=src_weight_info.config,
        )

        return kernel

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
