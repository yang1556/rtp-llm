"""Kernel-level test for TRT-LLM FP4 fused MoE on SM100.

Tests trtllm_fp4_block_scale_routed_moe end-to-end (scatter+GEMM1+act+GEMM2+gather).
"""
import unittest
import torch

try:
    import pytest
    pytestmark = [pytest.mark.gpu(type="SM100_ARM")]
except ImportError:
    pytest = None


def _dequant_fp4(packed_uint8, blockscale_e4m3, global_scale, block_size=16):
    """Dequantize NV-FP4 packed uint8 to bfloat16."""
    from flashinfer import e2m1_and_ufp8sf_scale_to_float
    return e2m1_and_ufp8sf_scale_to_float(packed_uint8, blockscale_e4m3, global_scale)


class TestTrtllmFp4Kernel(unittest.TestCase):

    def _run_test(self, E, num_tokens, hidden_size, inter_size, top_k=2):
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("SM100+ required")

        from flashinfer import fp4_quantize, ActivationType
        from flashinfer.fp4_quantization import block_scale_interleave
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )
        from flashinfer.utils import device_support_pdl

        K = hidden_size
        N = inter_size
        device = "cuda"

        # Generate input
        hidden = torch.randn(num_tokens, K, device=device, dtype=torch.bfloat16) * 0.1

        # Routing: softmax → topk → renormalize
        logits = torch.randn(num_tokens, E, device=device, dtype=torch.float32)
        probs = torch.softmax(logits, dim=1)
        topk_weights, topk_ids = torch.topk(probs, top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Generate bf16 weights
        w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)

        # FP4 quantize weights
        def _quantize_and_shuffle_w1(w_bf16):
            E_local = w_bf16.shape[0]
            gs = torch.empty(E_local, device=device, dtype=torch.float32)
            all_fp4 = []
            all_bs = []
            for i in range(E_local):
                amax = w_bf16[i].abs().amax().float()
                gs[i] = 448.0 * 6.0 / amax.clamp(min=1e-4)
                fp4, bs = fp4_quantize(w_bf16[i], gs[i])
                all_fp4.append(fp4)
                all_bs.append(bs)
            fp4_stacked = torch.stack(all_fp4)  # [E, 2N, K//2]
            bs_stacked = torch.stack(all_bs)

            # Shuffle for TRT-LLM
            perm = _maybe_get_cached_w3_w1_permute_indices(N, K, {})
            fp4_shuffled = fp4_stacked[:, perm, :]
            bs_shuffled = block_scale_interleave(bs_stacked[:, perm, :], N * 2)
            return fp4_shuffled, bs_shuffled, gs

        def _quantize_and_shuffle_w2(w_bf16):
            E_local = w_bf16.shape[0]
            gs = torch.empty(E_local, device=device, dtype=torch.float32)
            all_fp4 = []
            all_bs = []
            for i in range(E_local):
                amax = w_bf16[i].abs().amax().float()
                gs[i] = 448.0 * 6.0 / amax.clamp(min=1e-4)
                fp4, bs = fp4_quantize(w_bf16[i], gs[i])
                all_fp4.append(fp4)
                all_bs.append(bs)
            fp4_stacked = torch.stack(all_fp4)
            bs_stacked = torch.stack(all_bs)

            perm = get_w2_permute_indices_with_cache(K, N, {})
            fp4_shuffled = fp4_stacked[:, perm, :]
            bs_shuffled = block_scale_interleave(bs_stacked[:, perm, :], K)
            return fp4_shuffled, bs_shuffled, gs

        w1_fp4, w1_bs, w1_gs = _quantize_and_shuffle_w1(w1_bf16)
        w2_fp4, w2_bs, w2_gs = _quantize_and_shuffle_w2(w2_bf16)

        # Quantize input
        input_gs = torch.empty(E, device=device, dtype=torch.float32)
        amax = hidden.abs().amax()
        for i in range(E):
            input_gs[i] = 448.0 * 6.0 / amax.clamp(min=1e-4)
        hidden_fp4, hidden_bs = fp4_quantize(hidden, input_gs[0])

        # Compute scales
        g1_alphas = input_gs * (1.0 / w1_gs)
        c_global_sf = torch.ones(E, device=device, dtype=torch.float32)
        g1_scale_c = g1_alphas / c_global_sf
        a2_input_scale = c_global_sf
        g2_alphas = a2_input_scale * (1.0 / w2_gs)

        # Pack routing: (topk_ids << 16) | topk_weights_uint16
        topk_weights_uint16 = topk_weights.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
        packed = (topk_ids.to(torch.int32) << 16) | topk_weights_uint16

        # Call kernel
        result = trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed,
            routing_bias=None,
            hidden_states=hidden_fp4,
            hidden_states_scale=hidden_bs.view(torch.float8_e4m3fn),
            gemm1_weights=w1_fp4,
            gemm1_weights_scale=w1_bs.view(torch.float8_e4m3fn),
            gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
            gemm2_weights=w2_fp4,
            gemm2_weights_scale=w2_bs.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=g1_scale_c,
            output1_scale_gate_scalar=g1_alphas,
            output2_scale_scalar=g2_alphas,
            num_experts=E, top_k=top_k,
            n_group=None, topk_group=None,
            intermediate_size=N,
            local_expert_offset=0, local_num_experts=E,
            routed_scaling_factor=None,
            routing_method_type=1,  # Renormalize
            do_finalize=True,
            enable_pdl=device_support_pdl(),
            activation_type=ActivationType.Gelu if False else ActivationType.Silu,
            output=None,
        )[0]

        # Verify output is non-zero and finite
        self.assertTrue(torch.isfinite(result).all(), "Output contains NaN/Inf")
        self.assertGreater(result.abs().mean().item(), 1e-6, "Output is near-zero")
        print(f"  E={E} tokens={num_tokens} top_k={top_k} output_mean={result.abs().mean().item():.6f}")

    def test_small(self):
        self._run_test(E=8, num_tokens=16, hidden_size=4096, inter_size=1536, top_k=2)

    def test_medium(self):
        self._run_test(E=8, num_tokens=128, hidden_size=4096, inter_size=1536, top_k=2)

    def test_large_experts(self):
        self._run_test(E=64, num_tokens=256, hidden_size=4096, inter_size=1536, top_k=2)


if __name__ == "__main__":
    unittest.main()
