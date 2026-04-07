"""Multi-shape kernel test for CuteDSL FP4 executor on SM100.

Tests CutedslFp4Executor across various (E, M, K, N) shapes.
The existing cutedsl_fp4_executor_test covers one fixed shape;
this test covers smaller/larger/edge cases.
"""
import unittest
import torch

try:
    import pytest
    pytestmark = [pytest.mark.gpu(type="SM100_ARM")]
except ImportError:
    pytest = None

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import FusedMoEQuantConfig
from rtp_llm.ops import ParallelismConfig, MoeConfig
from rtp_llm.utils.model_weight import W

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


def _make_config(E, hidden, inter, top_k, max_batch=256):
    mc = ModelConfig()
    mc.attn_config.head_num = 2
    mc.attn_config.size_per_head = 128
    mc.num_layers = 2
    mc.max_seq_len = 2048
    mc.vocab_size = 500000
    mc.expert_num = E
    mc.hidden_size = hidden
    mc.moe_inter_size = inter
    mc.moe_k = top_k

    pc = ParallelismConfig()
    pc.world_size = 1
    pc.dp_size = 1
    pc.tp_size = 1
    pc.ep_size = 1
    pc.dp_rank = 0
    pc.tp_rank = 0
    pc.ep_rank = 0
    pc.world_rank = 0
    pc.local_rank = 0
    pc.local_world_size = 1

    moe_cfg = MoeConfig()
    moe_cfg.ll_num_max_token = max_batch

    return MoEConfigAdapter(model_config=mc, parallelism_config=pc, moe_config=moe_cfg)


class TestCutedslFp4Kernel(unittest.TestCase):

    def _run_test(self, E, M_per_expert, hidden_size, inter_size, top_k=8):
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("SM100+ required")

        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import (
            CutedslFp4Executor,
        )
        from flashinfer import scaled_fp4_grouped_quantize

        K = hidden_size
        N = inter_size
        device = "cuda"

        config = _make_config(E, K, N, top_k, max_batch=M_per_expert)

        # Generate weights
        w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)

        w1_amax = w1_bf16.abs().amax(dim=(1, 2)).float()
        w2_amax = w2_bf16.abs().amax(dim=(1, 2)).float()
        w1_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

        w1_fp4, w1_bs = scaled_fp4_grouped_quantize(
            w1_bf16, torch.full((E,), 2 * N, dtype=torch.int32, device=device), w1_gs)
        w2_fp4, w2_bs = scaled_fp4_grouped_quantize(
            w2_bf16, torch.full((E,), K, dtype=torch.int32, device=device), w2_gs)

        weights = {
            W.moe_w1: w1_fp4.permute(2, 0, 1),
            W.moe_w2: w2_fp4.permute(2, 0, 1),
            W.moe_s1: w1_bs,
            W.moe_s2: w2_bs,
            W.moe_w1_s2: 1.0 / w1_gs,
            W.moe_w2_s2: 1.0 / w2_gs,
            W.moe_w1_i_s: torch.ones(E, dtype=torch.float32, device=device),
            W.moe_w2_i_s: torch.ones(E, dtype=torch.float32, device=device),
        }

        executor = CutedslFp4Executor(
            config,
            FusedMoEQuantConfig(quant_dtype=torch.uint8, per_act_token_quant=False,
                                per_out_ch_quant=False, block_shape=[16, 16]),
            weights,
        )

        # Input
        expert_x = torch.randn(E, M_per_expert, K, device=device, dtype=torch.bfloat16) * 0.1
        masked_m = torch.full((E,), M_per_expert, dtype=torch.int32, device=device)

        payload = ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_origin_dtype=torch.bfloat16,
            expert_x_scale=None,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=masked_m,
                expert_num_tokens_cpu=None,
            ),
        )

        result = executor.execute(payload, "silu", None, None, False, None)
        out = result.fused_expert_output

        self.assertTrue(torch.isfinite(out).all(), "Output contains NaN/Inf")
        out_norm = out.norm().item()
        self.assertGreater(out_norm, 1e-4, f"Output near-zero: norm={out_norm}")
        print(f"  E={E} M/E={M_per_expert} K={K} N={N} output_norm={out_norm:.4f}")

    def test_small(self):
        self._run_test(E=8, M_per_expert=16, hidden_size=4096, inter_size=1536)

    def test_medium(self):
        self._run_test(E=8, M_per_expert=128, hidden_size=4096, inter_size=1536)

    def test_large(self):
        self._run_test(E=8, M_per_expert=256, hidden_size=4096, inter_size=1536)


if __name__ == "__main__":
    unittest.main()
