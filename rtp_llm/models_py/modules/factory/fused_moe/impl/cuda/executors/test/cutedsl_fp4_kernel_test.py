"""Kernel-level functional test for CuteDSL FP4 grouped GEMM on SM100.

Directly calls flashinfer_cutedsl_moe_masked (FC1+SiLU+FC2) and verifies
output against dequantized bf16 reference.
"""
import unittest
import torch

try:
    import pytest
    pytestmark = [pytest.mark.gpu(type="SM100_ARM")]
except ImportError:
    pytest = None

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
kE2M1ToFloat = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _break_fp4_bytes(a):
    m, n = a.shape
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n * 2).to(torch.bfloat16)


def _convert_swizzled_to_linear(a_sf_swizzled, m, k, block_size=16):
    m_tiles = (m + 127) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def _dequantize_fp4(tensor_fp4, tensor_sf, global_scale, block_size=16):
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = _break_fp4_bytes(tensor_fp4)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = _convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_f32 = tensor_sf.to(torch.float32) / global_scale
    out = (tensor_f32.float() * tensor_sf_f32.unsqueeze(-1)).reshape(m, k)
    return out.to(torch.bfloat16)


def _ref_moe(a_bf16, w1_bf16, w2_bf16, masked_m, E, K, N):
    """Reference MoE: FC1(gate+up) → SiLU → FC2, per-expert."""
    from flashinfer import fp4_quantize
    m = a_bf16.shape[1]
    out = torch.zeros(E, m, K, device=a_bf16.device, dtype=torch.bfloat16)
    for i in range(E):
        nm = masked_m[i].item()
        if nm == 0:
            continue
        x = a_bf16[i, :nm].float()
        w1 = w1_bf16[i].float()  # [2N, K]
        w2 = w2_bf16[i].float()  # [K, N]
        fc1 = x @ w1.T  # [nm, 2N]
        gate = fc1[:, :N]
        up = fc1[:, N:]
        act = torch.sigmoid(gate) * gate * up  # SiLU(gate) * up
        # Simulate intermediate FP4 requant
        inter_gs = torch.tensor(1.0, device=a_bf16.device)
        inter_q, inter_bs = fp4_quantize(act.to(torch.bfloat16), inter_gs)
        act_dq = _dequantize_fp4(inter_q, inter_bs, inter_gs).float()
        fc2 = act_dq @ w2.T  # [nm, K]
        out[i, :nm] = fc2.to(torch.bfloat16)
    return out


class TestCutedslFp4Kernel(unittest.TestCase):

    def _run_test(self, E, M_per_expert, hidden_size, inter_size):
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("SM100+ required")

        from rtp_llm.models_py.kernels.cuda.fp4_kernel import flashinfer_cutedsl_moe_masked
        from flashinfer import scaled_fp4_grouped_quantize

        K = hidden_size
        N = inter_size
        device = "cuda"

        # Create bf16 weights
        w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)

        # Quantize weights to FP4
        w1_amax = w1_bf16.abs().amax(dim=(1, 2)).float()
        w2_amax = w2_bf16.abs().amax(dim=(1, 2)).float()
        w1_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

        w1_fp4, w1_bs = scaled_fp4_grouped_quantize(
            w1_bf16,
            torch.full((E,), 2 * N, dtype=torch.int32, device=device),
            w1_gs,
        )
        w2_fp4, w2_bs = scaled_fp4_grouped_quantize(
            w2_bf16,
            torch.full((E,), K, dtype=torch.int32, device=device),
            w2_gs,
        )

        # Weight permute for CuteDSL: [E, N, K//2] → stored as permuted
        w1_perm = w1_fp4.permute(2, 0, 1)  # logical [m, k//2, E]
        w2_perm = w2_fp4.permute(2, 0, 1)

        # Input
        hidden = torch.randn(E, M_per_expert, K, device=device, dtype=torch.bfloat16) * 0.1
        masked_m = torch.full((E,), M_per_expert, dtype=torch.int32, device=device)

        # Scales
        input_gs = torch.ones(E, dtype=torch.float32, device=device)
        a2_gs = torch.ones(E, dtype=torch.float32, device=device)
        w1_alpha = (input_gs * (1.0 / w1_gs)).to(torch.float32)
        w2_alpha = (a2_gs * (1.0 / w2_gs)).to(torch.float32)

        # Execute kernel
        result = flashinfer_cutedsl_moe_masked(
            hidden_states=(hidden, None),
            input_global_scale=input_gs,
            w1=w1_perm, w1_blockscale=w1_bs, w1_alpha=w1_alpha,
            w2=w2_perm, a2_global_scale=a2_gs,
            w2_blockscale=w2_bs, w2_alpha=w2_alpha,
            masked_m=masked_m,
        )  # [E, M, K]

        # Reference
        ref = _ref_moe(hidden, w1_bf16, w2_bf16, masked_m, E, K, N)

        # Compare valid tokens
        for i in range(E):
            nm = masked_m[i].item()
            if nm == 0:
                continue
            cos = torch.nn.functional.cosine_similarity(
                result[i, :nm].float().flatten(),
                ref[i, :nm].float().flatten(), dim=0
            )
            print(f"  E={E} M/E={M_per_expert} expert={i} cos_sim={cos.item():.4f}")
            self.assertGreater(cos.item(), 0.5, f"cos_sim too low for expert {i}: {cos.item():.4f}")

    def test_small(self):
        self._run_test(E=8, M_per_expert=16, hidden_size=4096, inter_size=1536)

    def test_medium(self):
        self._run_test(E=8, M_per_expert=128, hidden_size=4096, inter_size=1536)

    def test_large(self):
        self._run_test(E=8, M_per_expert=256, hidden_size=4096, inter_size=1536)


if __name__ == "__main__":
    unittest.main()
