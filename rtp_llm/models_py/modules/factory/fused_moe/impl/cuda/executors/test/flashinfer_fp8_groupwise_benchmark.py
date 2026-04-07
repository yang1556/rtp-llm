"""Performance benchmark for FlashInfer FP8 groupwise SM100 executor.

Compares FlashInfer group_gemm_fp8_nt_groupwise against DeepGEMM
on SM100 hardware for MoE FP8 per-block workloads.

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


def _bench_flashinfer_gemm(
    expert_num, num_tokens, hidden_size, moe_inter_size, top_k=2
):
    """Benchmark FlashInfer group_gemm_fp8_nt_groupwise."""
    from flashinfer.gemm import group_gemm_fp8_nt_groupwise
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul

    N = moe_inter_size * 2
    K = hidden_size
    E = expert_num
    device = "cuda"

    # Create weights
    w1_bf16 = (torch.randn(E, N, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, moe_inter_size, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    w1_fp8, _ = _per_block_quantize_fp8(w1_bf16)
    w2_fp8, _ = _per_block_quantize_fp8(w2_bf16)

    w1_scale = _recompute_float32_scales(w1_fp8)
    w2_scale = _recompute_float32_scales(w2_fp8)
    w1_scale_mn = w1_scale.permute(0, 2, 1).contiguous()
    w2_scale_mn = w2_scale.permute(0, 2, 1).contiguous()

    # Create input tokens
    hidden_states = torch.randn(num_tokens, K, device=device, dtype=torch.bfloat16) * 0.1

    # Assign tokens round-robin
    topk_ids = torch.zeros(num_tokens, top_k, device=device, dtype=torch.int32)
    for i in range(num_tokens):
        for k in range(top_k):
            topk_ids[i, k] = (i * top_k + k) % E

    # Count per expert
    num_tokens_per_expert = [0] * E
    for i in range(num_tokens):
        for k in range(top_k):
            num_tokens_per_expert[topk_ids[i, k].item()] += 1

    padded_tokens = [((t + 3) // 4) * 4 for t in num_tokens_per_expert]
    total_padded = sum(padded_tokens)

    # Build m_indptr
    m_indptr = torch.zeros(E + 1, dtype=torch.int32, device=device)
    for i in range(E):
        m_indptr[i + 1] = m_indptr[i] + padded_tokens[i]

    # Token reordering
    M = num_tokens
    flat_topk_ids = topk_ids.view(-1)
    token_indices = torch.arange(M, device=device, dtype=torch.int32).unsqueeze(1).expand(-1, top_k).reshape(-1)
    sorted_expert_ids, sort_order = flat_topk_ids.sort(stable=True)
    sorted_token_indices = token_indices[sort_order]
    grouped_bf16 = hidden_states[sorted_token_indices.long()]

    grouped_input = torch.zeros((total_padded, K), device=device, dtype=torch.bfloat16)
    offset = 0
    src_offset = 0
    for i in range(E):
        actual = num_tokens_per_expert[i]
        padded = padded_tokens[i]
        if actual > 0:
            grouped_input[offset:offset + actual] = grouped_bf16[src_offset:src_offset + actual]
        src_offset += actual
        offset += padded

    def run_once():
        # Quantize input
        input_fp8, input_scale = sgl_per_token_group_quant_fp8(
            grouped_input, group_size=BLOCK_SIZE,
            column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False,
        )
        input_scale_mn = input_scale.T.contiguous()

        # FC1
        fc1_out = group_gemm_fp8_nt_groupwise(
            a=input_fp8, b=w1_fp8,
            a_scale=input_scale_mn, b_scale=w1_scale_mn,
            m_indptr=m_indptr, scale_major_mode="MN",
            out_dtype=torch.bfloat16,
        )

        # Activation
        fc1_act = torch.empty((total_padded, N // 2), device=device, dtype=torch.bfloat16)
        silu_and_mul(fc1_act, fc1_out)

        # Quantize FC1 output
        fc2_fp8, fc2_scale = sgl_per_token_group_quant_fp8(
            fc1_act, group_size=BLOCK_SIZE,
            column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False,
        )
        fc2_scale_mn = fc2_scale.T.contiguous()

        # FC2
        fc2_out = group_gemm_fp8_nt_groupwise(
            a=fc2_fp8, b=w2_fp8,
            a_scale=fc2_scale_mn, b_scale=w2_scale_mn,
            m_indptr=m_indptr, scale_major_mode="MN",
            out_dtype=torch.bfloat16,
        )
        return fc2_out

    # Warmup
    for _ in range(WARMUP_ITERS):
        run_once()
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        run_once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / BENCH_ITERS) * 1000

    # Compute TFLOPS (FC1 + FC2)
    # FC1: total_padded * N * K * 2 (multiply-add)
    # FC2: total_padded * K * (N/2) * 2
    flops_fc1 = total_padded * N * K * 2
    flops_fc2 = total_padded * K * (N // 2) * 2
    total_flops = flops_fc1 + flops_fc2
    tflops = (total_flops / (avg_ms / 1000)) / 1e12

    return avg_ms, tflops, total_padded


class TestFlashInferFp8GroupwiseBenchmark(unittest.TestCase):
    """Performance benchmark on SM100."""

    def _print_result(self, label, expert_num, num_tokens, hidden, inter, avg_ms, tflops, total_padded):
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"  E={expert_num}, tokens={num_tokens}, hidden={hidden}, inter={inter}")
        print(f"  total_padded={total_padded}")
        print(f"  avg latency: {avg_ms:.3f} ms")
        print(f"  throughput:  {tflops:.2f} TFLOPS")
        print(f"{'='*70}")

    def test_bench_decode_small(self):
        """Decode: 8 experts, 16 tokens, DeepSeek-like dims."""
        ms, tflops, tp = _bench_flashinfer_gemm(
            expert_num=8, num_tokens=16,
            hidden_size=7168, moe_inter_size=2048,
        )
        self._print_result("Decode Small", 8, 16, 7168, 2048, ms, tflops, tp)

    def test_bench_decode_medium(self):
        """Decode: 8 experts, 64 tokens."""
        ms, tflops, tp = _bench_flashinfer_gemm(
            expert_num=8, num_tokens=64,
            hidden_size=7168, moe_inter_size=2048,
        )
        self._print_result("Decode Medium", 8, 64, 7168, 2048, ms, tflops, tp)

    def test_bench_prefill_small(self):
        """Prefill: 8 experts, 256 tokens."""
        ms, tflops, tp = _bench_flashinfer_gemm(
            expert_num=8, num_tokens=256,
            hidden_size=7168, moe_inter_size=2048,
        )
        self._print_result("Prefill Small", 8, 256, 7168, 2048, ms, tflops, tp)

    def test_bench_prefill_large(self):
        """Prefill: 8 experts, 1024 tokens."""
        ms, tflops, tp = _bench_flashinfer_gemm(
            expert_num=8, num_tokens=1024,
            hidden_size=7168, moe_inter_size=2048,
        )
        self._print_result("Prefill Large", 8, 1024, 7168, 2048, ms, tflops, tp)

    def test_bench_prefill_xlarge(self):
        """Prefill: 8 experts, 4096 tokens."""
        ms, tflops, tp = _bench_flashinfer_gemm(
            expert_num=8, num_tokens=4096,
            hidden_size=7168, moe_inter_size=2048,
        )
        self._print_result("Prefill XLarge", 8, 4096, 7168, 2048, ms, tflops, tp)

    def test_bench_many_experts(self):
        """Many experts: 64 experts, 512 tokens."""
        ms, tflops, tp = _bench_flashinfer_gemm(
            expert_num=64, num_tokens=512,
            hidden_size=7168, moe_inter_size=2048,
        )
        self._print_result("Many Experts", 64, 512, 7168, 2048, ms, tflops, tp)

    def test_bench_dsv3_256_experts(self):
        """DeepSeek-V3 style: 256 experts, 1024 tokens."""
        ms, tflops, tp = _bench_flashinfer_gemm(
            expert_num=256, num_tokens=1024,
            hidden_size=7168, moe_inter_size=2048, top_k=8,
        )
        self._print_result("DSv3 256-Expert", 256, 1024, 7168, 2048, ms, tflops, tp)


if __name__ == "__main__":
    unittest.main()
