"""FP4 vs FP8 MoE benchmark on SM100.

Compares CuteDSL FP4, TRT-LLM FP4, and FlashInfer FP8 groupwise.
"""
import time
import unittest
import torch

try:
    import pytest
    pytestmark = [pytest.mark.gpu(type="SM100_ARM")]
except ImportError:
    pytest = None

BLOCK_SIZE = 128
WARMUP_ITERS = 5
BENCH_ITERS = 30
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


def _bench_time(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return ((time.perf_counter() - start) / iters) * 1000


def _compute_moe_tflops(total_tokens, N, K, avg_ms):
    """FC1(2N*K) + FC2(K*N) FLOPS."""
    flops = total_tokens * (2 * N) * K * 2 + total_tokens * K * N * 2
    return (flops / (avg_ms / 1000)) / 1e12


# ===== CuteDSL FP4 =====

def _bench_cutedsl_fp4(E, M_per_expert, K, N):
    from rtp_llm.models_py.kernels.cuda.fp4_kernel import flashinfer_cutedsl_moe_masked
    from flashinfer import scaled_fp4_grouped_quantize

    device = "cuda"
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

    w1_perm = w1_fp4.permute(2, 0, 1)
    w2_perm = w2_fp4.permute(2, 0, 1)

    hidden = torch.randn(E, M_per_expert, K, device=device, dtype=torch.bfloat16) * 0.1
    masked_m = torch.full((E,), M_per_expert, dtype=torch.int32, device=device)

    input_gs = torch.ones(E, dtype=torch.float32, device=device)
    a2_gs = torch.ones(E, dtype=torch.float32, device=device)
    w1_alpha = (input_gs * (1.0 / w1_gs)).to(torch.float32)
    w2_alpha = (a2_gs * (1.0 / w2_gs)).to(torch.float32)

    def run():
        return flashinfer_cutedsl_moe_masked(
            hidden_states=(hidden, None),
            input_global_scale=input_gs,
            w1=w1_perm, w1_blockscale=w1_bs, w1_alpha=w1_alpha,
            w2=w2_perm, a2_global_scale=a2_gs,
            w2_blockscale=w2_bs, w2_alpha=w2_alpha,
            masked_m=masked_m,
        )

    try:
        avg_ms = _bench_time(run)
        total_tokens = E * M_per_expert
        return avg_ms, _compute_moe_tflops(total_tokens, N, K, avg_ms), None
    except Exception as e:
        return None, None, str(e)[:200]


# ===== TRT-LLM FP4 =====

def _bench_trtllm_fp4(E, num_tokens, K, N, top_k):
    from flashinfer import fp4_quantize, ActivationType
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.utils import device_support_pdl

    device = "cuda"
    hidden = torch.randn(num_tokens, K, device=device, dtype=torch.bfloat16) * 0.1

    # Routing
    logits = torch.randn(num_tokens, E, device=device, dtype=torch.float32)
    probs = torch.softmax(logits, dim=1)
    topk_w, topk_ids = torch.topk(probs, top_k, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    # Weights
    w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    cache = {}
    w1_gs = torch.empty(E, device=device, dtype=torch.float32)
    w2_gs = torch.empty(E, device=device, dtype=torch.float32)
    w1_all_fp4, w1_all_bs = [], []
    w2_all_fp4, w2_all_bs = [], []
    for i in range(E):
        a1 = w1_bf16[i].abs().amax().float().clamp(min=1e-4)
        w1_gs[i] = 448.0 * 6.0 / a1
        fp4, bs = fp4_quantize(w1_bf16[i], w1_gs[i])
        w1_all_fp4.append(fp4); w1_all_bs.append(bs)

        a2 = w2_bf16[i].abs().amax().float().clamp(min=1e-4)
        w2_gs[i] = 448.0 * 6.0 / a2
        fp4, bs = fp4_quantize(w2_bf16[i], w2_gs[i])
        w2_all_fp4.append(fp4); w2_all_bs.append(bs)

    w1_fp4 = torch.stack(w1_all_fp4)
    w1_bs_t = torch.stack(w1_all_bs)
    w2_fp4 = torch.stack(w2_all_fp4)
    w2_bs_t = torch.stack(w2_all_bs)

    perm1 = _maybe_get_cached_w3_w1_permute_indices(N, K, cache)
    w1_fp4 = w1_fp4[:, perm1, :]
    w1_bs_t = block_scale_interleave(w1_bs_t[:, perm1, :], N * 2)

    perm2 = get_w2_permute_indices_with_cache(K, N, cache)
    w2_fp4 = w2_fp4[:, perm2, :]
    w2_bs_t = block_scale_interleave(w2_bs_t[:, perm2, :], K)

    # Input quantize
    input_gs = torch.empty(E, device=device, dtype=torch.float32)
    amax = hidden.abs().amax().float().clamp(min=1e-4)
    for i in range(E):
        input_gs[i] = 448.0 * 6.0 / amax
    hidden_fp4, hidden_bs = fp4_quantize(hidden, input_gs[0])

    g1_alphas = input_gs * (1.0 / w1_gs)
    c_global_sf = torch.ones(E, device=device, dtype=torch.float32)
    g1_scale_c = g1_alphas / c_global_sf
    g2_alphas = c_global_sf * (1.0 / w2_gs)

    topk_w_u16 = topk_w.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    packed = (topk_ids.to(torch.int32) << 16) | topk_w_u16

    def run():
        return trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed, routing_bias=None,
            hidden_states=hidden_fp4,
            hidden_states_scale=hidden_bs.view(torch.float8_e4m3fn),
            gemm1_weights=w1_fp4, gemm1_weights_scale=w1_bs_t.view(torch.float8_e4m3fn),
            gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
            gemm2_weights=w2_fp4, gemm2_weights_scale=w2_bs_t.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=g1_scale_c,
            output1_scale_gate_scalar=g1_alphas,
            output2_scale_scalar=g2_alphas,
            num_experts=E, top_k=top_k,
            n_group=None, topk_group=None,
            intermediate_size=N,
            local_expert_offset=0, local_num_experts=E,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=True,
            enable_pdl=device_support_pdl(),
            activation_type=ActivationType.Silu,
            output=None,
        )[0]

    try:
        avg_ms = _bench_time(run)
        total_tokens = num_tokens * top_k
        return avg_ms, _compute_moe_tflops(total_tokens, N, K, avg_ms), None
    except Exception as e:
        return None, None, str(e)[:200]


# ===== FlashInfer FP8 =====

def _per_block_quantize_fp8(tensor, block_size=128):
    has_batch = tensor.dim() == 3
    if has_batch:
        E, N_dim, K_dim = tensor.shape
        flat = tensor.reshape(-1, K_dim).float()
    else:
        N_dim, K_dim = tensor.shape
        flat = tensor.float()
    N_total = flat.shape[0]
    n_blocks = (N_total + block_size - 1) // block_size
    k_blocks = (K_dim + block_size - 1) // block_size
    N_pad = n_blocks * block_size
    K_pad = k_blocks * block_size
    padded = torch.zeros(N_pad, K_pad, device=tensor.device, dtype=torch.float32)
    padded[:N_total, :K_dim] = flat
    viewed = padded.view(n_blocks, block_size, k_blocks, block_size)
    amax = viewed.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = amax / 448.0
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    scale_exp = scale.unsqueeze(1).unsqueeze(3).expand_as(viewed)
    quantized = (viewed / scale_exp).reshape(N_pad, K_pad)
    fp8 = quantized[:N_total, :K_dim].to(torch.float8_e4m3fn)
    if has_batch:
        fp8 = fp8.view(E, N_dim, K_dim)
        scale = scale.view(E, N_dim // block_size if N_dim % block_size == 0 else n_blocks // E, k_blocks)
    return fp8, scale


def _bench_flashinfer_fp8(E, M_per_expert, K, N):
    from flashinfer.gemm import group_gemm_fp8_nt_groupwise
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.flashinfer_fp8_groupwise_executor import _recompute_float32_scales
    from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul

    device = "cuda"
    total_M = M_per_expert * E

    w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w1_fp8, _ = _per_block_quantize_fp8(w1_bf16)
    w2_fp8, _ = _per_block_quantize_fp8(w2_bf16)
    w1_scale_mn = _recompute_float32_scales(w1_fp8).permute(0, 2, 1).contiguous()
    w2_scale_mn = _recompute_float32_scales(w2_fp8).permute(0, 2, 1).contiguous()

    grouped_input = torch.randn(total_M, K, device=device, dtype=torch.bfloat16) * 0.1
    M_padded = ((M_per_expert + 3) // 4) * 4
    m_indptr = torch.arange(0, E + 1, dtype=torch.int32, device=device) * M_padded

    def run():
        inp_fp8, inp_scale = sgl_per_token_group_quant_fp8(
            grouped_input, group_size=BLOCK_SIZE,
            column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False)
        inp_scale_mn = inp_scale.T.contiguous()
        fc1 = group_gemm_fp8_nt_groupwise(
            a=inp_fp8, b=w1_fp8, a_scale=inp_scale_mn, b_scale=w1_scale_mn,
            m_indptr=m_indptr, scale_major_mode="MN", out_dtype=torch.bfloat16)
        act = torch.empty((total_M, N), device=device, dtype=torch.bfloat16)
        silu_and_mul(act, fc1)
        fc2_fp8, fc2_scale = sgl_per_token_group_quant_fp8(
            act, group_size=BLOCK_SIZE,
            column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False)
        fc2_scale_mn = fc2_scale.T.contiguous()
        return group_gemm_fp8_nt_groupwise(
            a=fc2_fp8, b=w2_fp8, a_scale=fc2_scale_mn, b_scale=w2_scale_mn,
            m_indptr=m_indptr, scale_major_mode="MN", out_dtype=torch.bfloat16)

    try:
        avg_ms = _bench_time(run)
        return avg_ms, _compute_moe_tflops(total_M, N, K, avg_ms), None
    except Exception as e:
        return None, None, str(e)[:200]


# ===== Comparison =====

SCENARIOS = [
    # (label, E, M_per_expert, N, K, top_k, type)
    ("Decode-2tok/exp",     8,    2, 2048, 7168, 2, "decode"),
    ("Decode-8tok/exp",     8,    8, 2048, 7168, 2, "decode"),
    ("Decode-16tok/exp",    8,   16, 2048, 7168, 2, "decode"),
    ("Prefill-32tok/exp",   8,   32, 2048, 7168, 2, "prefill"),
    ("Prefill-128tok/exp",  8,  128, 2048, 7168, 2, "prefill"),
    ("Prefill-512tok/exp",  8,  512, 2048, 7168, 2, "prefill"),
    ("ManyExp-8tok/exp",   64,    8, 2048, 7168, 2, "decode"),
    ("DSv3-32tok/exp",    256,   32, 2048, 7168, 8, "prefill"),
]


class TestFp4Benchmark(unittest.TestCase):

    def test_fp4_vs_fp8_comparison(self):
        """Compare FP4 (CuteDSL + TRT-LLM) vs FP8 (FlashInfer) on SM100."""
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("SM100+ required")

        print("\n" + "=" * 130)
        print(f"  FP4 vs FP8 MoE Benchmark — SM100 (Full MoE: FC1+SiLU+FC2)")
        print(f"  N=2048 (inter), K=7168 (hidden)")
        print(f"  Warmup={WARMUP_ITERS}, Iters={BENCH_ITERS}")
        print("=" * 130)

        header = (f"{'Scenario':<22} {'Type':<8} {'E':>4} {'M/E':>5} {'TotM':>7} | "
                  f"{'CuteDSL ms':>10} {'TF':>7} | "
                  f"{'TRT-LLM ms':>10} {'TF':>7} | "
                  f"{'FI-FP8 ms':>10} {'TF':>7} | "
                  f"{'Best':>12}")
        print(header)
        print("-" * 130)

        errors = []
        for label, E, M_per_exp, N, K, top_k, stype in SCENARIOS:
            total_tokens = E * M_per_exp

            # CuteDSL FP4
            cd_ms, cd_tf, cd_err = _bench_cutedsl_fp4(E, M_per_exp, K, N)
            if cd_err:
                errors.append(f"[CuteDSL] {label}: {cd_err}")

            # TRT-LLM FP4 (uses total tokens, not M/E — it does routing internally)
            num_tokens_for_trtllm = total_tokens // top_k  # actual input tokens before topk expansion
            tl_ms, tl_tf, tl_err = _bench_trtllm_fp4(E, num_tokens_for_trtllm, K, N, top_k)
            if tl_err:
                errors.append(f"[TRT-LLM] {label}: {tl_err}")

            # FlashInfer FP8
            fi_ms, fi_tf, fi_err = _bench_flashinfer_fp8(E, M_per_exp, K, N)
            if fi_err:
                errors.append(f"[FI-FP8] {label}: {fi_err}")

            cd_s = f"{cd_ms:.3f}" if cd_ms else "ERR"
            cd_t = f"{cd_tf:.1f}" if cd_tf else "—"
            tl_s = f"{tl_ms:.3f}" if tl_ms else "ERR"
            tl_t = f"{tl_tf:.1f}" if tl_tf else "—"
            fi_s = f"{fi_ms:.3f}" if fi_ms else "ERR"
            fi_t = f"{fi_tf:.1f}" if fi_tf else "—"

            candidates = []
            if cd_ms: candidates.append(("CuteDSL", cd_ms))
            if tl_ms: candidates.append(("TRT-LLM", tl_ms))
            if fi_ms: candidates.append(("FI-FP8", fi_ms))

            if candidates:
                best_name, best_ms = min(candidates, key=lambda x: x[1])
                best_s = best_name
            else:
                best_s = "N/A"

            print(f"{label:<22} {stype:<8} {E:>4} {M_per_exp:>5} {total_tokens:>7} | "
                  f"{cd_s:>10} {cd_t:>7} | "
                  f"{tl_s:>10} {tl_t:>7} | "
                  f"{fi_s:>10} {fi_t:>7} | "
                  f"{best_s:>12}")

        print("=" * 130)
        print("CuteDSL=FlashInfer CuteDSL FP4 masked, TRT-LLM=TRT-LLM FP4 fused, FI-FP8=FlashInfer FP8 groupwise")

        if errors:
            print()
            for e in errors:
                print(f"  {e}")


if __name__ == "__main__":
    unittest.main()
