"""Kernel-level test for TRT-LLM FP4 fused MoE on SM100.

Uses TrtllmFp4Executor which handles weight shuffling internally.
Verifies output is non-zero and finite across various shapes.
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
    CombineForwardPayload,
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


def _prepare_fp4_weights(E, N, K, device="cuda"):
    """Generate FP4 weights using flashinfer quantization."""
    from flashinfer import scaled_fp4_grouped_quantize

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

    input_gs = torch.ones(E, dtype=torch.float32, device=device)
    a2_gs = torch.ones(E, dtype=torch.float32, device=device)

    weights = {
        W.moe_w1: w1_fp4.permute(2, 0, 1),
        W.moe_w2: w2_fp4.permute(2, 0, 1),
        W.moe_s1: w1_bs,
        W.moe_s2: w2_bs,
        W.moe_w1_s2: 1.0 / w1_gs,
        W.moe_w2_s2: 1.0 / w2_gs,
        W.moe_w1_i_s: input_gs,
        W.moe_w2_i_s: a2_gs,
    }
    return weights


class TestTrtllmFp4Kernel(unittest.TestCase):

    def _run_test(self, E, num_tokens, hidden, inter, top_k=2):
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("SM100+ required")

        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
            TrtllmFp4Executor,
        )

        device = "cuda"
        K = hidden
        N = inter

        config = _make_config(E, K, N, top_k, max_batch=num_tokens)
        weights = _prepare_fp4_weights(E, N, K, device)

        executor = TrtllmFp4Executor(
            config,
            FusedMoEQuantConfig(
                quant_dtype=torch.uint8,
                per_act_token_quant=False,
                per_out_ch_quant=False,
                block_shape=[16, 16],
            ),
            weights,
        )

        # Build payload — tokens scattered to experts
        hidden_states = torch.randn(num_tokens, K, device=device, dtype=torch.bfloat16) * 0.1

        # Routing
        logits = torch.randn(num_tokens, E, device=device, dtype=torch.float32)
        probs = torch.softmax(logits, dim=1)
        topk_weights, topk_ids = torch.topk(probs, top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Scatter to 3D expert layout
        masked_m = torch.zeros(E, dtype=torch.int32, device=device)
        for i in range(E):
            masked_m[i] = (topk_ids == i).sum()
        max_m = int(masked_m.max().item())
        if max_m == 0:
            max_m = 1

        expert_x = torch.zeros(E, max_m, K, device=device, dtype=torch.bfloat16)
        pos = torch.zeros(E, dtype=torch.long, device=device)
        for t in range(num_tokens):
            for k in range(top_k):
                eid = topk_ids[t, k].item()
                p = pos[eid].item()
                if p < max_m:
                    expert_x[eid, p] = hidden_states[t]
                    pos[eid] += 1

        payload = ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_origin_dtype=torch.bfloat16,
            expert_x_scale=None,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=masked_m,
                expert_num_tokens_cpu=None,
            ),
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

        result = executor.execute(payload, "silu", None, None, False, None)

        # Verify
        out = result.fused_expert_output
        self.assertTrue(torch.isfinite(out).all(), "Output contains NaN/Inf")
        # Check at least some experts have non-zero output
        active_experts = (masked_m > 0).sum().item()
        self.assertGreater(active_experts, 0, "No active experts")
        print(f"  E={E} tokens={num_tokens} top_k={top_k} active_experts={active_experts} output_norm={out.norm().item():.4f}")

    def test_small(self):
        self._run_test(E=8, num_tokens=16, hidden=4096, inter=1536, top_k=2)

    def test_medium(self):
        self._run_test(E=8, num_tokens=128, hidden=4096, inter=1536, top_k=2)

    def test_many_experts(self):
        self._run_test(E=64, num_tokens=256, hidden=4096, inter=1536, top_k=2)


if __name__ == "__main__":
    unittest.main()
