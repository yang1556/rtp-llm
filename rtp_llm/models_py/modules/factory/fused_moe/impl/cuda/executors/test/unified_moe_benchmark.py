"""Unified FP4 vs FP8 MoE Benchmark on SM100.

All implementations run the SAME full MoE forward: FC1 + SiLU + FC2.
Same N=2048 (intermediate), K=7168 (hidden) for all.

Implementations compared:
  FP4-CD:  CuteDSL FP4 (via CutedslFp4Executor)
  FP8-FI:  FlashInfer FP8 groupwise (group_gemm_fp8_nt_groupwise)
  FP8-DGM: DeepGEMM FP8 masked (m_grouped_fp8_gemm_nt_masked)
  FP8-DGC: DeepGEMM FP8 contiguous (m_grouped_fp8_gemm_nt_contiguous)
"""
import time
import unittest
import torch

try:
    import pytest
    pytestmark = [pytest.mark.gpu(type="SM100_ARM")]
except ImportError:
    pytest = None

from rtp_llm.utils.model_weight import W

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
    """Full MoE: FC1([M,K]x[2N,K]^T) + FC2([M,N]x[K,N]^T)"""
    flops = total_tokens * (2 * N) * K * 2 + total_tokens * K * N * 2
    return (flops / (avg_ms / 1000)) / 1e12


def _make_config(E, K, N, top_k, max_batch):
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.ops import ParallelismConfig, MoeConfig
    from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter

    mc = ModelConfig()
    mc.attn_config.head_num = 2; mc.attn_config.size_per_head = 128
    mc.num_layers = 2; mc.max_seq_len = 2048; mc.vocab_size = 500000
    mc.expert_num = E; mc.hidden_size = K; mc.moe_inter_size = N; mc.moe_k = top_k
    pc = ParallelismConfig()
    pc.world_size = 1; pc.dp_size = 1; pc.tp_size = 1; pc.ep_size = 1
    pc.dp_rank = 0; pc.tp_rank = 0; pc.ep_rank = 0; pc.world_rank = 0
    pc.local_rank = 0; pc.local_world_size = 1
    moe_cfg = MoeConfig(); moe_cfg.ll_num_max_token = max_batch
    return MoEConfigAdapter(model_config=mc, parallelism_config=pc, moe_config=moe_cfg)


def _safe_run(fn):
    try:
        return fn()
    except Exception as e:
        return None, None, str(e)[:200]


# ===== 1. CuteDSL FP4 (Full MoE via Executor) =====

def _bench_cutedsl_fp4(E, M_per_expert, K, N):
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import CutedslFp4Executor
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import ExpertForwardPayload, ExpertTokensMetadata
    from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import FusedMoEQuantConfig
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

    weights = {
        W.moe_w1: w1_fp4.permute(2, 0, 1), W.moe_w2: w2_fp4.permute(2, 0, 1),
        W.moe_s1: w1_bs, W.moe_s2: w2_bs,
        W.moe_w1_s2: 1.0 / w1_gs, W.moe_w2_s2: 1.0 / w2_gs,
        W.moe_w1_i_s: torch.ones(E, dtype=torch.float32, device=device),
        W.moe_w2_i_s: torch.ones(E, dtype=torch.float32, device=device),
    }

    config = _make_config(E, K, N, 8, M_per_expert)
    executor = CutedslFp4Executor(config, FusedMoEQuantConfig(
        quant_dtype=torch.uint8, per_act_token_quant=False, per_out_ch_quant=False, block_shape=[16, 16]), weights)

    hidden = torch.randn(E, M_per_expert, K, device=device, dtype=torch.bfloat16) * 0.1
    masked_m = torch.full((E,), M_per_expert, dtype=torch.int32, device=device)
    payload = ExpertForwardPayload(
        expert_x=hidden, expert_x_origin_dtype=torch.bfloat16, expert_x_scale=None,
        expert_tokens_meta=ExpertTokensMetadata(expert_num_tokens=masked_m, expert_num_tokens_cpu=None))

    def run():
        return executor.execute(payload, "silu", None, None, False, None)

    avg_ms = _bench_time(run)
    return avg_ms, _compute_moe_tflops(E * M_per_expert, N, K, avg_ms), None


# ===== 2. FlashInfer FP8 Groupwise (Full MoE: quant+FC1+act+FC2) =====

def _per_block_quantize_fp8(tensor, block_size=128):
    has_batch = tensor.dim() == 3
    if has_batch:
        E_dim, N_dim, K_dim = tensor.shape
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
        fp8 = fp8.view(E_dim, N_dim, K_dim)
        scale = scale.view(E_dim, N_dim // block_size if N_dim % block_size == 0 else n_blocks // E_dim, k_blocks)
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

    avg_ms = _bench_time(run)
    return avg_ms, _compute_moe_tflops(total_M, N, K, avg_ms), None


# ===== 3. DeepGEMM FP8 Masked (Full MoE: quant+FC1+act+requant+FC2) =====

def _bench_deepgemm_masked(E, M_per_expert, K, N):
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import m_grouped_fp8_gemm_nt_masked
    from rtp_llm.test.utils.numeric_util import per_token_cast_to_fp8, per_block_cast_to_fp8
    from rtp_llm.models_py.utils.math import align, ceil_div
    from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul

    device = "cuda"
    max_m = align(M_per_expert, BLOCK_SIZE)

    w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    w1_fp8_data = torch.empty(E, 2 * N, K, device=device, dtype=torch.float8_e4m3fn)
    w1_fp8_scale = torch.empty(E, ceil_div(2 * N, 128), ceil_div(K, 128), device=device, dtype=torch.float32)
    w2_fp8_data = torch.empty(E, K, N, device=device, dtype=torch.float8_e4m3fn)
    w2_fp8_scale = torch.empty(E, ceil_div(K, 128), ceil_div(N, 128), device=device, dtype=torch.float32)
    for i in range(E):
        w1_fp8_data[i], w1_fp8_scale[i] = per_block_cast_to_fp8(w1_bf16[i], use_ue8m0=True)
        w2_fp8_data[i], w2_fp8_scale[i] = per_block_cast_to_fp8(w2_bf16[i], use_ue8m0=True)
    w1_fp8 = (w1_fp8_data, w1_fp8_scale)
    w2_fp8 = (w2_fp8_data, w2_fp8_scale)

    a_bf16 = torch.randn(E, max_m, K, device=device, dtype=torch.bfloat16) * 0.1
    a_fp8_data = torch.empty(E, max_m, K, device=device, dtype=torch.float8_e4m3fn)
    a_fp8_scale = torch.empty(E, max_m, ceil_div(K, 128), device=device, dtype=torch.float32)
    for i in range(E):
        a_fp8_data[i], a_fp8_scale[i] = per_token_cast_to_fp8(a_bf16[i], use_ue8m0=True)
    a_fp8 = (a_fp8_data, a_fp8_scale)

    masked_m = torch.full((E,), M_per_expert, device=device, dtype=torch.int32)
    fc1_out = torch.empty(E, max_m, 2 * N, device=device, dtype=torch.bfloat16)
    fc2_out = torch.empty(E, max_m, K, device=device, dtype=torch.bfloat16)

    def run():
        m_grouped_fp8_gemm_nt_masked(a_fp8, w1_fp8, fc1_out, masked_m, M_per_expert)
        act = torch.empty(E, max_m, N, device=device, dtype=torch.bfloat16)
        silu_and_mul(act.view(-1, N), fc1_out.view(-1, 2 * N))
        act_fp8_data = torch.empty(E, max_m, N, device=device, dtype=torch.float8_e4m3fn)
        act_fp8_scale = torch.empty(E, max_m, ceil_div(N, 128), device=device, dtype=torch.float32)
        for i in range(E):
            act_fp8_data[i], act_fp8_scale[i] = per_token_cast_to_fp8(act[i], use_ue8m0=True)
        m_grouped_fp8_gemm_nt_masked((act_fp8_data, act_fp8_scale), w2_fp8, fc2_out, masked_m, M_per_expert)
        return fc2_out

    avg_ms = _bench_time(run)
    return avg_ms, _compute_moe_tflops(E * M_per_expert, N, K, avg_ms), None


# ===== 4. DeepGEMM FP8 Contiguous (Full MoE: quant+FC1+act+requant+FC2) =====

def _bench_deepgemm_contiguous(E, M_per_expert, K, N):
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import m_grouped_fp8_gemm_nt_contiguous
    from rtp_llm.test.utils.numeric_util import per_token_cast_to_fp8, per_block_cast_to_fp8
    from rtp_llm.models_py.utils.math import align, ceil_div
    from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul

    device = "cuda"
    mk_align = 128
    aligned_m = align(M_per_expert, mk_align)
    total_M = aligned_m * E

    # Weights [E, N, K]
    w1_bf16 = (torch.randn(E, 2 * N, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w2_bf16 = (torch.randn(E, K, N, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    w1_fp8_data = torch.empty(E, 2 * N, K, device=device, dtype=torch.float8_e4m3fn)
    w1_fp8_scale = torch.empty(E, ceil_div(2 * N, 128), ceil_div(K, 128), device=device, dtype=torch.float32)
    w2_fp8_data = torch.empty(E, K, N, device=device, dtype=torch.float8_e4m3fn)
    w2_fp8_scale = torch.empty(E, ceil_div(K, 128), ceil_div(N, 128), device=device, dtype=torch.float32)
    for i in range(E):
        w1_fp8_data[i], w1_fp8_scale[i] = per_block_cast_to_fp8(w1_bf16[i], use_ue8m0=True)
        w2_fp8_data[i], w2_fp8_scale[i] = per_block_cast_to_fp8(w2_bf16[i], use_ue8m0=True)
    w1_fp8 = (w1_fp8_data, w1_fp8_scale)
    w2_fp8 = (w2_fp8_data, w2_fp8_scale)

    # Input [total_M, K] contiguous
    a_bf16 = torch.randn(total_M, K, device=device, dtype=torch.bfloat16) * 0.1
    a_fp8 = per_token_cast_to_fp8(a_bf16, use_ue8m0=True)

    # m_indices: expert id per row
    m_indices = torch.empty(total_M, device=device, dtype=torch.int32)
    for i in range(E):
        start = i * aligned_m
        m_indices[start:start + M_per_expert] = i
        m_indices[start + M_per_expert:start + aligned_m] = -1

    fc1_out = torch.empty(total_M, 2 * N, device=device, dtype=torch.bfloat16)
    fc2_out = torch.empty(total_M, K, device=device, dtype=torch.bfloat16)

    def run():
        m_grouped_fp8_gemm_nt_contiguous(a_fp8, w1_fp8, fc1_out, m_indices)
        act = torch.empty(total_M, N, device=device, dtype=torch.bfloat16)
        silu_and_mul(act, fc1_out)
        act_fp8 = per_token_cast_to_fp8(act, use_ue8m0=True)
        m_grouped_fp8_gemm_nt_contiguous(act_fp8, w2_fp8, fc2_out, m_indices)
        return fc2_out

    avg_ms = _bench_time(run)
    return avg_ms, _compute_moe_tflops(E * M_per_expert, N, K, avg_ms), None


# ===== Scenarios & Test =====

SCENARIOS = [
    # (label, E, M_per_expert, N, K, type)
    ("Decode-2tok/exp",     8,    2, 2048, 7168, "decode"),
    ("Decode-8tok/exp",     8,    8, 2048, 7168, "decode"),
    ("Decode-16tok/exp",    8,   16, 2048, 7168, "decode"),
    ("Prefill-32tok/exp",   8,   32, 2048, 7168, "prefill"),
    ("Prefill-128tok/exp",  8,  128, 2048, 7168, "prefill"),
    ("Prefill-512tok/exp",  8,  512, 2048, 7168, "prefill"),
    ("ManyExp-8tok/exp",   64,    8, 2048, 7168, "decode"),
]


class TestUnifiedMoeBenchmark(unittest.TestCase):

    def test_all_implementations(self):
        """Unified Full MoE benchmark: FP4-CD vs FP8-FI vs FP8-DGM vs FP8-DGC."""
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("SM100+ required")

        print("\n" + "=" * 140)
        print(f"  Unified MoE Benchmark — SM100 (Full MoE: FC1+SiLU+FC2)")
        print(f"  N=2048 (inter), K=7168 (hidden), Warmup={WARMUP_ITERS}, Iters={BENCH_ITERS}")
        print(f"  All implementations run identical workload for fair comparison")
        print("=" * 140)

        header = (f"{'Scenario':<22} {'Type':<8} {'E':>4} {'M/E':>5} {'TotM':>7} | "
                  f"{'FP4-CD ms':>10} {'TF':>7} | "
                  f"{'FP8-FI ms':>10} {'TF':>7} | "
                  f"{'FP8-DGM ms':>10} {'TF':>7} | "
                  f"{'FP8-DGC ms':>10} {'TF':>7} | "
                  f"{'Best':>12}")
        print(header)
        print("-" * 140)

        errors = []
        for label, E, M_per_exp, N, K, stype in SCENARIOS:
            total_tokens = E * M_per_exp

            cd_ms, cd_tf, cd_err = _safe_run(lambda: _bench_cutedsl_fp4(E, M_per_exp, K, N))
            if cd_err: errors.append(f"[FP4-CD] {label}: {cd_err}")

            fi_ms, fi_tf, fi_err = _safe_run(lambda: _bench_flashinfer_fp8(E, M_per_exp, K, N))
            if fi_err: errors.append(f"[FP8-FI] {label}: {fi_err}")

            dgm_ms, dgm_tf, dgm_err = _safe_run(lambda: _bench_deepgemm_masked(E, M_per_exp, K, N))
            if dgm_err: errors.append(f"[FP8-DGM] {label}: {dgm_err}")

            dgc_ms, dgc_tf, dgc_err = _safe_run(lambda: _bench_deepgemm_contiguous(E, M_per_exp, K, N))
            if dgc_err: errors.append(f"[FP8-DGC] {label}: {dgc_err}")

            def fmt(ms, tf):
                return (f"{ms:.3f}" if ms else "ERR", f"{tf:.1f}" if tf else "-")

            cd_s, cd_t = fmt(cd_ms, cd_tf)
            fi_s, fi_t = fmt(fi_ms, fi_tf)
            dgm_s, dgm_t = fmt(dgm_ms, dgm_tf)
            dgc_s, dgc_t = fmt(dgc_ms, dgc_tf)

            candidates = []
            if cd_ms: candidates.append(("FP4-CD", cd_ms))
            if fi_ms: candidates.append(("FP8-FI", fi_ms))
            if dgm_ms: candidates.append(("FP8-DGM", dgm_ms))
            if dgc_ms: candidates.append(("FP8-DGC", dgc_ms))
            best_name = min(candidates, key=lambda x: x[1])[0] if candidates else "N/A"

            print(f"{label:<22} {stype:<8} {E:>4} {M_per_exp:>5} {total_tokens:>7} | "
                  f"{cd_s:>10} {cd_t:>7} | "
                  f"{fi_s:>10} {fi_t:>7} | "
                  f"{dgm_s:>10} {dgm_t:>7} | "
                  f"{dgc_s:>10} {dgc_t:>7} | "
                  f"{best_name:>12}")

        print("=" * 140)
        print("FP4-CD=CuteDSL FP4, FP8-FI=FlashInfer FP8 groupwise, "
              "FP8-DGM=DeepGEMM masked, FP8-DGC=DeepGEMM contiguous")
        print(f"Workload: Full MoE (FC1[M,K]x[2N,K]^T + SiLU + FC2[M,N]x[K,N]^T)")

        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  {e}")


if __name__ == "__main__":
    unittest.main()
