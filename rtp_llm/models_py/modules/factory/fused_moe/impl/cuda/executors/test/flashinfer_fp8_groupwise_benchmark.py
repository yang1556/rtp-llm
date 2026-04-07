"""Multi-implementation FP8 MoE single-FC GEMM benchmark on SM100.

Compares FlashInfer group_gemm_fp8_nt_groupwise vs DeepGEMM
m_grouped_fp8_gemm_nt_contiguous / m_grouped_fp8_gemm_nt_masked.
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
    """Per-block FP8 quantization for FlashInfer."""
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


def _compute_gemm_tflops(M, N, K, avg_ms):
    return (2 * M * N * K / (avg_ms / 1000)) / 1e12


# ===== FlashInfer =====

def _bench_flashinfer(E, M_per_expert, N, K):
    from flashinfer.gemm import group_gemm_fp8_nt_groupwise
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.flashinfer_fp8_groupwise_executor import _recompute_float32_scales

    device = "cuda"
    total_M = M_per_expert * E

    b_bf16 = (torch.randn(E, N, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    b_fp8, _ = _per_block_quantize_fp8(b_bf16)
    b_scale = _recompute_float32_scales(b_fp8).permute(0, 2, 1).contiguous()

    a_bf16 = torch.randn(total_M, K, device=device, dtype=torch.bfloat16) * 0.1
    a_fp8, a_scale = sgl_per_token_group_quant_fp8(
        a_bf16, group_size=BLOCK_SIZE,
        column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False,
    )
    a_scale_mn = a_scale.T.contiguous()

    M_padded = ((M_per_expert + 3) // 4) * 4
    m_indptr = torch.arange(0, E + 1, dtype=torch.int32, device=device) * M_padded

    def run():
        return group_gemm_fp8_nt_groupwise(
            a=a_fp8, b=b_fp8, a_scale=a_scale_mn, b_scale=b_scale,
            m_indptr=m_indptr, scale_major_mode="MN", out_dtype=torch.bfloat16,
        )

    avg_ms = _bench_time(run)
    return avg_ms, _compute_gemm_tflops(total_M, N, K, avg_ms)


# ===== DeepGEMM Contiguous =====

def _bench_deepgemm_contiguous(E, M_per_expert, N, K):
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import m_grouped_fp8_gemm_nt_contiguous
    from rtp_llm.test.utils.numeric_util import per_token_cast_to_fp8, per_block_cast_to_fp8
    from rtp_llm.models_py.utils.math import align, ceil_div

    device = "cuda"
    mk_align = 128  # get_mk_alignment_for_contiguous_layout()

    # Each expert gets M_per_expert tokens, aligned
    aligned_m = align(M_per_expert, mk_align)
    total_M = aligned_m * E

    a_bf16 = torch.randn(total_M, K, device=device, dtype=torch.bfloat16) * 0.1
    b_bf16 = torch.randn(E, N, K, device=device, dtype=torch.bfloat16) * 0.1

    # Quantize using DeepGEMM's own functions (UE8M0)
    a_fp8 = per_token_cast_to_fp8(a_bf16, use_ue8m0=True)  # returns (fp8, scale) tuple
    b_fp8_data = torch.empty(E, N, K, device=device, dtype=torch.float8_e4m3fn)
    b_fp8_scale = torch.empty(E, ceil_div(N, 128), ceil_div(K, 128), device=device, dtype=torch.float32)
    for i in range(E):
        b_fp8_data[i], b_fp8_scale[i] = per_block_cast_to_fp8(b_bf16[i], use_ue8m0=True)
    b_fp8 = (b_fp8_data, b_fp8_scale)

    # m_indices: expert id for each token row
    m_indices = torch.empty(total_M, device=device, dtype=torch.int32)
    for i in range(E):
        start = i * aligned_m
        m_indices[start:start + M_per_expert] = i
        m_indices[start + M_per_expert:start + aligned_m] = -1  # padding

    output = torch.empty(total_M, N, device=device, dtype=torch.bfloat16)

    def run():
        m_grouped_fp8_gemm_nt_contiguous(a_fp8, b_fp8, output, m_indices)
        return output

    try:
        avg_ms = _bench_time(run)
        return avg_ms, _compute_gemm_tflops(E * M_per_expert, N, K, avg_ms), None
    except Exception as e:
        return None, None, str(e)[:300]


# ===== DeepGEMM Masked =====

def _bench_deepgemm_masked(E, M_per_expert, N, K):
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import m_grouped_fp8_gemm_nt_masked
    from rtp_llm.test.utils.numeric_util import per_token_cast_to_fp8, per_block_cast_to_fp8
    from rtp_llm.models_py.utils.math import align, ceil_div

    device = "cuda"
    max_m = align(M_per_expert, BLOCK_SIZE)

    a_bf16 = torch.randn(E, max_m, K, device=device, dtype=torch.bfloat16) * 0.1
    b_bf16 = torch.randn(E, N, K, device=device, dtype=torch.bfloat16) * 0.1

    # Quantize per expert
    a_fp8_data = torch.empty(E, max_m, K, device=device, dtype=torch.float8_e4m3fn)
    a_fp8_scale = torch.empty(E, max_m, ceil_div(K, 128), device=device, dtype=torch.float32)
    b_fp8_data = torch.empty(E, N, K, device=device, dtype=torch.float8_e4m3fn)
    b_fp8_scale = torch.empty(E, ceil_div(N, 128), ceil_div(K, 128), device=device, dtype=torch.float32)
    for i in range(E):
        a_fp8_data[i], a_fp8_scale[i] = per_token_cast_to_fp8(a_bf16[i], use_ue8m0=True)
        b_fp8_data[i], b_fp8_scale[i] = per_block_cast_to_fp8(b_bf16[i], use_ue8m0=True)

    a_fp8 = (a_fp8_data, a_fp8_scale)
    b_fp8 = (b_fp8_data, b_fp8_scale)

    masked_m = torch.full((E,), M_per_expert, device=device, dtype=torch.int32)
    output = torch.empty(E, max_m, N, device=device, dtype=torch.bfloat16)

    def run():
        m_grouped_fp8_gemm_nt_masked(a_fp8, b_fp8, output, masked_m, M_per_expert)
        return output

    try:
        avg_ms = _bench_time(run)
        return avg_ms, _compute_gemm_tflops(E * M_per_expert, N, K, avg_ms), None
    except Exception as e:
        return None, None, str(e)[:300]


# ===== Test =====

SCENARIOS = [
    # (label, E, M_per_expert, N, K, type)
    ("Decode-2tok/exp",     8,    2, 4096, 7168, "decode"),
    ("Decode-8tok/exp",     8,    8, 4096, 7168, "decode"),
    ("Decode-16tok/exp",    8,   16, 4096, 7168, "decode"),
    ("Prefill-32tok/exp",   8,   32, 4096, 7168, "prefill"),
    ("Prefill-128tok/exp",  8,  128, 4096, 7168, "prefill"),
    ("Prefill-512tok/exp",  8,  512, 4096, 7168, "prefill"),
    ("ManyExp-8tok/exp",   64,    8, 4096, 7168, "decode"),
    ("DSv3-32tok/exp",    256,   32, 4096, 7168, "prefill"),
]


class TestMoeFp8Benchmark(unittest.TestCase):

    def test_raw_gemm_comparison(self):
        """Single FC GEMM: FlashInfer vs DeepGEMM contiguous vs masked."""
        print("\n" + "=" * 120)
        print(f"  FP8 MoE Single-FC GEMM Benchmark — SM100")
        print(f"  N=4096 (moe_inter*2), K=7168 (hidden)")
        print(f"  Warmup={WARMUP_ITERS}, Iters={BENCH_ITERS}")
        print("=" * 120)

        header = (f"{'Scenario':<22} {'Type':<8} {'E':>4} {'M/E':>5} {'TotM':>7} | "
                  f"{'FI ms':>7} {'FI TF':>7} | "
                  f"{'DGC ms':>7} {'DGC TF':>7} | "
                  f"{'DGM ms':>7} {'DGM TF':>7} | "
                  f"{'Best':>12}")
        print(header)
        print("-" * 120)

        errors = []
        for label, E, M_per_exp, N, K, stype in SCENARIOS:
            total_M = E * M_per_exp

            fi_ms, fi_tf = _bench_flashinfer(E, M_per_exp, N, K)

            dgc_ms, dgc_tf, dgc_err = _bench_deepgemm_contiguous(E, M_per_exp, N, K)
            if dgc_err:
                errors.append(f"[DGC] {label}: {dgc_err}")

            dgm_ms, dgm_tf, dgm_err = _bench_deepgemm_masked(E, M_per_exp, N, K)
            if dgm_err:
                errors.append(f"[DGM] {label}: {dgm_err}")

            fi_s = f"{fi_ms:.3f}"
            fi_t = f"{fi_tf:.1f}"
            dgc_s = f"{dgc_ms:.3f}" if dgc_ms else "ERR"
            dgc_t = f"{dgc_tf:.1f}" if dgc_tf else "—"
            dgm_s = f"{dgm_ms:.3f}" if dgm_ms else "ERR"
            dgm_t = f"{dgm_tf:.1f}" if dgm_tf else "—"

            candidates = [("FI", fi_ms)]
            if dgc_ms:
                candidates.append(("DGC", dgc_ms))
            if dgm_ms:
                candidates.append(("DGM", dgm_ms))
            best_name, best_ms = min(candidates, key=lambda x: x[1])
            best_s = best_name if best_name == "FI" else f"{best_name} {fi_ms/best_ms:.2f}x"

            print(f"{label:<22} {stype:<8} {E:>4} {M_per_exp:>5} {total_M:>7} | "
                  f"{fi_s:>7} {fi_t:>7} | "
                  f"{dgc_s:>7} {dgc_t:>7} | "
                  f"{dgm_s:>7} {dgm_t:>7} | "
                  f"{best_s:>12}")

        print("=" * 120)
        print("FI=FlashInfer groupwise, DGC=DeepGEMM contiguous, DGM=DeepGEMM masked")
        if errors:
            print()
            for e in errors:
                print(f"  {e}")


if __name__ == "__main__":
    unittest.main()
