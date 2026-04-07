"""Multi-implementation FP8 MoE GEMM benchmark on SM100 (B200/GB200).

Compares FlashInfer group_gemm_fp8_nt_groupwise vs DeepGEMM masked/contiguous
for decode and prefill scenarios with DeepSeek-V3 model dimensions.

Run via GB200 CI with tag SM100_ARM.
"""

import time
import unittest

import torch

try:
    import pytest
    pytestmark = [pytest.mark.gpu(type="SM100_ARM"), pytest.mark.fp8_sm100]
except ImportError:
    pytest = None

BLOCK_SIZE = 128
WARMUP_ITERS = 10
BENCH_ITERS = 50


def _per_block_quantize_fp8(tensor, block_size=128):
    """Per-block FP8 quantization matching FlashInfer convention."""
    has_batch = tensor.dim() == 3
    if has_batch:
        E, N, K = tensor.shape
        flat = tensor.reshape(-1, K).float()
    else:
        N, K = tensor.shape
        flat = tensor.float()

    N_total = flat.shape[0]
    n_blocks = (N_total + block_size - 1) // block_size
    k_blocks = (K + block_size - 1) // block_size

    N_pad = n_blocks * block_size
    K_pad = k_blocks * block_size
    padded = torch.zeros(N_pad, K_pad, device=tensor.device, dtype=torch.float32)
    padded[:N_total, :K] = flat

    viewed = padded.view(n_blocks, block_size, k_blocks, block_size)
    amax = viewed.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = amax / 448.0
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))

    scale_expanded = scale.unsqueeze(1).unsqueeze(3).expand_as(viewed)
    quantized = (viewed / scale_expanded).reshape(N_pad, K_pad)
    fp8 = quantized[:N_total, :K].to(torch.float8_e4m3fn)

    if has_batch:
        fp8 = fp8.view(E, N, K)
        scale = scale.view(E, N // block_size if N % block_size == 0 else n_blocks // E,
                          k_blocks)
    return fp8, scale


def _recompute_float32_scales(weight_fp8, block_size=128):
    """Recompute per-block float32 scales from FP8 data."""
    from rtp_llm.models_py.utils.math import align, ceil_div

    has_batch = weight_fp8.dim() == 3
    if has_batch:
        E, N, K = weight_fp8.shape
        w_flat = weight_fp8.reshape(-1, K)
    else:
        N, K = weight_fp8.shape
        w_flat = weight_fp8

    N_total = w_flat.shape[0]
    N_padded = align(N_total, block_size)
    K_padded = align(K, block_size)

    w_padded = torch.zeros(
        (N_padded, K_padded), dtype=w_flat.dtype, device=w_flat.device
    )
    w_padded[:N_total, :K] = w_flat

    w_view = w_padded.view(N_padded // block_size, block_size, K_padded // block_size, block_size)
    w_amax = w_view.float().abs().amax(dim=(1, 3)).clamp(min=1e-4)

    sf = w_amax / 448.0
    sf = torch.pow(2.0, torch.ceil(torch.log2(sf.abs())))

    if has_batch:
        sf = sf.view(E, ceil_div(N, block_size), ceil_div(K, block_size))

    return sf


def _compute_tflops(total_padded, N, K, avg_ms):
    """Compute TFLOPS for FC1+FC2 MoE GEMM."""
    flops_fc1 = total_padded * N * K * 2
    flops_fc2 = total_padded * K * (N // 2) * 2
    total_flops = flops_fc1 + flops_fc2
    return (total_flops / (avg_ms / 1000)) / 1e12


def _bench_time(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Warmup + benchmark, return avg ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return ((time.perf_counter() - start) / iters) * 1000


# ===== FlashInfer Benchmark =====

def _bench_flashinfer(E, num_tokens, K, N, top_k):
    """Benchmark FlashInfer group_gemm_fp8_nt_groupwise (FC1+activation+FC2)."""
    from flashinfer.gemm import group_gemm_fp8_nt_groupwise
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul

    device = "cuda"

    w1_bf16 = (torch.randn(E, N, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N // 2, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w1_fp8, _ = _per_block_quantize_fp8(w1_bf16)
    w2_fp8, _ = _per_block_quantize_fp8(w2_bf16)

    w1_scale_mn = _recompute_float32_scales(w1_fp8).permute(0, 2, 1).contiguous()
    w2_scale_mn = _recompute_float32_scales(w2_fp8).permute(0, 2, 1).contiguous()

    hidden_states = torch.randn(num_tokens, K, device=device, dtype=torch.bfloat16) * 0.1
    topk_ids = torch.zeros(num_tokens, top_k, device=device, dtype=torch.int32)
    for i in range(num_tokens):
        for k in range(top_k):
            topk_ids[i, k] = (i * top_k + k) % E

    num_per_expert = [0] * E
    for i in range(num_tokens):
        for k in range(top_k):
            num_per_expert[topk_ids[i, k].item()] += 1

    padded_tokens = [((t + 3) // 4) * 4 for t in num_per_expert]
    total_padded = sum(padded_tokens)

    m_indptr = torch.zeros(E + 1, dtype=torch.int32, device=device)
    for i in range(E):
        m_indptr[i + 1] = m_indptr[i] + padded_tokens[i]

    # Pre-build grouped input
    M = num_tokens
    flat_ids = topk_ids.view(-1)
    tok_idx = torch.arange(M, device=device, dtype=torch.int32).unsqueeze(1).expand(-1, top_k).reshape(-1)
    _, sort_order = flat_ids.sort(stable=True)
    sorted_tok = tok_idx[sort_order]
    grouped_bf16 = hidden_states[sorted_tok.long()]

    grouped_input = torch.zeros((total_padded, K), device=device, dtype=torch.bfloat16)
    offset = 0
    src_offset = 0
    for i in range(E):
        actual = num_per_expert[i]
        padded = padded_tokens[i]
        if actual > 0:
            grouped_input[offset:offset + actual] = grouped_bf16[src_offset:src_offset + actual]
        src_offset += actual
        offset += padded

    def run():
        inp_fp8, inp_scale = sgl_per_token_group_quant_fp8(
            grouped_input, group_size=BLOCK_SIZE,
            column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False,
        )
        inp_scale_mn = inp_scale.T.contiguous()

        fc1_out = group_gemm_fp8_nt_groupwise(
            a=inp_fp8, b=w1_fp8, a_scale=inp_scale_mn, b_scale=w1_scale_mn,
            m_indptr=m_indptr, scale_major_mode="MN", out_dtype=torch.bfloat16,
        )

        fc1_act = torch.empty((total_padded, N // 2), device=device, dtype=torch.bfloat16)
        silu_and_mul(fc1_act, fc1_out)

        fc2_fp8, fc2_scale = sgl_per_token_group_quant_fp8(
            fc1_act, group_size=BLOCK_SIZE,
            column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False,
        )
        fc2_scale_mn = fc2_scale.T.contiguous()

        fc2_out = group_gemm_fp8_nt_groupwise(
            a=fc2_fp8, b=w2_fp8, a_scale=fc2_scale_mn, b_scale=w2_scale_mn,
            m_indptr=m_indptr, scale_major_mode="MN", out_dtype=torch.bfloat16,
        )
        return fc2_out

    avg_ms = _bench_time(run)
    return avg_ms, total_padded


# ===== DeepGEMM Masked Benchmark =====

def _bench_deepgemm_masked(E, num_tokens, K, N, top_k):
    """Benchmark DeepGEMM masked (decode-optimized) FC1+activation+FC2."""
    from deep_gemm import m_grouped_fp8_gemm_nt_masked
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
    from rtp_llm.models_py.utils.math import align

    device = "cuda"
    alignment = align(num_tokens * top_k // E + 16, BLOCK_SIZE)  # per-expert max tokens

    w1_bf16 = (torch.randn(E, N, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N // 2, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w1_fp8, w1_scale = _per_block_quantize_fp8(w1_bf16)
    w2_fp8, w2_scale = _per_block_quantize_fp8(w2_bf16)

    # DeepGEMM masked needs 3D input [E, alignment, K]
    input_3d = torch.randn(E, alignment, K, device=device, dtype=torch.bfloat16) * 0.1

    # Per-expert token counts
    tokens_per_expert = num_tokens * top_k // E
    num_recv = torch.full((E,), tokens_per_expert, dtype=torch.int32, device=device)
    expected_m = alignment

    def run():
        # Quantize input to FP8
        inp_flat = input_3d.view(-1, K)
        inp_fp8, inp_scale = sgl_per_token_group_quant_fp8(
            inp_flat, group_size=BLOCK_SIZE,
            column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False,
        )
        inp_fp8_3d = inp_fp8.view(E, alignment, K)
        # DeepGEMM expects scale as tuple (fp8, scale)
        inp_scale_flat = inp_scale  # [E*alignment, K//128]

        upgate = torch.empty((E, alignment, N), device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (inp_fp8_3d, inp_scale_flat.view(E * alignment, -1)),
            w1_fp8,
            upgate, num_recv, expected_m,
            disable_ue8m0_cast=True,
        )

        down_in = torch.empty((E, alignment, N // 2), device=device, dtype=torch.bfloat16)
        silu_and_mul(down_in.view(-1, N // 2), upgate.view(-1, N))

        down_fp8, down_scale = sgl_per_token_group_quant_fp8(
            down_in.view(-1, N // 2), group_size=BLOCK_SIZE,
            column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False,
        )

        output = torch.empty((E, alignment, K), device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (down_fp8.view(E, alignment, N // 2), down_scale.view(E * alignment, -1)),
            w2_fp8,
            output, num_recv, expected_m,
            disable_ue8m0_cast=True,
        )
        return output

    total_padded = E * alignment
    try:
        avg_ms = _bench_time(run)
    except Exception as e:
        return None, total_padded, str(e)
    return avg_ms, total_padded, None


# ===== DeepGEMM Contiguous Benchmark =====

def _bench_deepgemm_contiguous(E, num_tokens, K, N, top_k):
    """Benchmark DeepGEMM contiguous (prefill-optimized) FC1+activation+FC2."""
    from deep_gemm import m_grouped_fp8_gemm_nt_contiguous
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul

    device = "cuda"
    total_tokens = num_tokens * top_k

    w1_bf16 = (torch.randn(E, N, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N // 2, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w1_fp8, _ = _per_block_quantize_fp8(w1_bf16)
    w2_fp8, _ = _per_block_quantize_fp8(w2_bf16)

    # Contiguous: flat [total_tokens, K] with expert_offsets
    input_flat = torch.randn(total_tokens, K, device=device, dtype=torch.bfloat16) * 0.1
    tokens_per_expert = total_tokens // E
    expert_offsets = torch.arange(0, E + 1, dtype=torch.int32, device=device) * tokens_per_expert

    def run():
        inp_fp8, inp_scale = sgl_per_token_group_quant_fp8(
            input_flat, group_size=BLOCK_SIZE,
            column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False,
        )

        upgate = torch.empty((total_tokens, N), device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_contiguous(
            (inp_fp8, inp_scale),
            w1_fp8, upgate, expert_offsets,
            disable_ue8m0_cast=True,
        )

        down_in = torch.empty((total_tokens, N // 2), device=device, dtype=torch.bfloat16)
        silu_and_mul(down_in, upgate)

        down_fp8, down_scale = sgl_per_token_group_quant_fp8(
            down_in, group_size=BLOCK_SIZE,
            column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False,
        )

        output = torch.empty((total_tokens, K), device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_contiguous(
            (down_fp8, down_scale),
            w2_fp8, output, expert_offsets,
            disable_ue8m0_cast=True,
        )
        return output

    try:
        avg_ms = _bench_time(run)
    except Exception as e:
        return None, total_tokens, str(e)
    return avg_ms, total_tokens, None


# ===== Comparison Test =====

SCENARIOS = [
    # (label, expert_num, num_tokens, hidden_size, moe_inter_size, top_k, scenario_type)
    ("Decode-16tok",   8,   16,  7168, 2048, 2, "decode"),
    ("Decode-64tok",   8,   64,  7168, 2048, 2, "decode"),
    ("Prefill-256tok", 8,  256,  7168, 2048, 2, "prefill"),
    ("Prefill-1024tok",8, 1024,  7168, 2048, 2, "prefill"),
    ("Prefill-4096tok",8, 4096,  7168, 2048, 2, "prefill"),
    ("ManyExp-64E",   64,  512,  7168, 2048, 2, "prefill"),
    ("DSv3-256E",    256, 1024,  7168, 2048, 8, "prefill"),
]


class TestMoeFp8Benchmark(unittest.TestCase):
    """Compare FlashInfer vs DeepGEMM on SM100 for FP8 MoE."""

    def test_comparison_table(self):
        """Run all scenarios and print comparison table."""
        results = []

        for label, E, tokens, K, inter, top_k, stype in SCENARIOS:
            N = inter * 2
            row = {"label": label, "E": E, "tokens": tokens, "top_k": top_k, "type": stype}

            # FlashInfer
            fi_ms, fi_padded = _bench_flashinfer(E, tokens, K, N, top_k)
            fi_tflops = _compute_tflops(fi_padded, N, K, fi_ms)
            row["fi_ms"] = fi_ms
            row["fi_tflops"] = fi_tflops

            # DeepGEMM Masked (decode-oriented)
            dg_masked_ms, dg_masked_pad, dg_masked_err = _bench_deepgemm_masked(E, tokens, K, N, top_k)
            if dg_masked_ms is not None:
                dg_masked_tflops = _compute_tflops(dg_masked_pad, N, K, dg_masked_ms)
                row["dgm_ms"] = dg_masked_ms
                row["dgm_tflops"] = dg_masked_tflops
            else:
                row["dgm_ms"] = None
                row["dgm_err"] = dg_masked_err

            # DeepGEMM Contiguous (prefill-oriented)
            dg_cont_ms, dg_cont_pad, dg_cont_err = _bench_deepgemm_contiguous(E, tokens, K, N, top_k)
            if dg_cont_ms is not None:
                dg_cont_tflops = _compute_tflops(dg_cont_pad, N, K, dg_cont_ms)
                row["dgc_ms"] = dg_cont_ms
                row["dgc_tflops"] = dg_cont_tflops
            else:
                row["dgc_ms"] = None
                row["dgc_err"] = dg_cont_err

            results.append(row)

        # Print comparison table
        print("\n" + "=" * 110)
        print(f"  FP8 MoE GEMM Benchmark — SM100 (FlashInfer Groupwise vs DeepGEMM)")
        print(f"  Model dims: hidden=7168, inter=2048 (DeepSeek-V3)")
        print(f"  Warmup={WARMUP_ITERS}, Iters={BENCH_ITERS}")
        print("=" * 110)

        header = f"{'Scenario':<20} {'Type':<8} {'E':>4} {'Tok':>5} {'K':>2} | " \
                 f"{'FI ms':>7} {'FI TF':>7} | " \
                 f"{'DGM ms':>7} {'DGM TF':>7} | " \
                 f"{'DGC ms':>7} {'DGC TF':>7} | " \
                 f"{'Best':>10}"
        print(header)
        print("-" * 110)

        for r in results:
            fi_ms_s = f"{r['fi_ms']:.3f}"
            fi_tf_s = f"{r['fi_tflops']:.1f}"

            dgm_ms_s = f"{r['dgm_ms']:.3f}" if r.get('dgm_ms') else "ERR"
            dgm_tf_s = f"{r.get('dgm_tflops', 0):.1f}" if r.get('dgm_ms') else "—"

            dgc_ms_s = f"{r['dgc_ms']:.3f}" if r.get('dgc_ms') else "ERR"
            dgc_tf_s = f"{r.get('dgc_tflops', 0):.1f}" if r.get('dgc_ms') else "—"

            # Determine best (lowest latency)
            candidates = [("FI", r['fi_ms'])]
            if r.get('dgm_ms'):
                candidates.append(("DGM", r['dgm_ms']))
            if r.get('dgc_ms'):
                candidates.append(("DGC", r['dgc_ms']))
            best_name, best_ms = min(candidates, key=lambda x: x[1])
            speedup_vs_fi = r['fi_ms'] / best_ms if best_name != "FI" else 1.0
            if best_name == "FI":
                best_s = "FI"
            else:
                best_s = f"{best_name} {speedup_vs_fi:.2f}x"

            print(f"{r['label']:<20} {r['type']:<8} {r['E']:>4} {r['tokens']:>5} {r['top_k']:>2} | "
                  f"{fi_ms_s:>7} {fi_tf_s:>7} | "
                  f"{dgm_ms_s:>7} {dgm_tf_s:>7} | "
                  f"{dgc_ms_s:>7} {dgc_tf_s:>7} | "
                  f"{best_s:>10}")

        print("=" * 110)
        print("FI=FlashInfer groupwise, DGM=DeepGEMM masked, DGC=DeepGEMM contiguous")
        print("TF=TFLOPS, Best=lowest latency implementation")
        print()

        # Print errors if any
        for r in results:
            if r.get('dgm_err'):
                print(f"  [DGM ERROR] {r['label']}: {r['dgm_err'][:100]}")
            if r.get('dgc_err'):
                print(f"  [DGC ERROR] {r['label']}: {r['dgc_err'][:100]}")


if __name__ == "__main__":
    unittest.main()
