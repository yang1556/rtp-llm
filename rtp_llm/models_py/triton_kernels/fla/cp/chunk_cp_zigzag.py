# -*- coding: utf-8 -*-
# Context Parallelism — Zigzag variant
#
# Each rank computes on its own zigzag tokens directly, no QKV all-gather needed.
# Each rank has 2 segments (front half + back half of zigzag).
# Total 2*cp_size segments form a causal chain.
#
# Phase 0: Conv1d with P2P exchange of tail tokens (kernel_size-1 tokens)
# Phase 1: Each rank computes (b, M) for its 2 segments locally
# Phase 2: All-gather (b, M) from all ranks, reorder to causal order
# Phase 3: cp_merge to compute h0_true for each segment
# Phase 4: Rerun Step5+Step6 with correct h0
#
# Pure-Python helpers (exchange_conv_context / prepend_conv_context /
# strip_conv_context / zigzag_causal_order / causal_positions /
# build_segment_cu_seqlens) live in `cp.utils`.

from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_o import chunk_fwd_o
from rtp_llm.models_py.triton_kernels.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.cp.chunk_cp_scan import (
    compute_br,
    compute_M_total,
)
from rtp_llm.models_py.triton_kernels.fla.cp.utils import (
    build_segment_cu_seqlens,
    causal_positions,
    zigzag_causal_order,
)
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd
from rtp_llm.models_py.triton_kernels.fla.solve_tril import solve_tril
from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd

# ---------------------------------------------------------------------------
# cp_merge_kernel: walk the causal chain of (M, b) affines applied to h0.
# Reads from `ag_hm` indirectly via `causal_order_ptr` so we never have to
# materialise a reordered copy of the gathered buffer.
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_ranks"])
def cp_merge_kernel(
    h_out,
    ag_hm,  # all-gather raw layout (NCCL order, NOT reordered)
    h0,
    causal_order_ptr,  # [2*cp_size] int tensor: causal_pos -> ag_hm row
    num_ranks,  # number of affines to apply (causal positions 0..num_ranks-1)
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    HAS_H0: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H

    stride_rank = N * H * K * (V + K)
    ag_base = (i_n * H + i_h) * K * (V + K)

    if HAS_H0:
        p_h0 = tl.make_block_ptr(
            h0 + (i_n * H + i_h) * K * V,
            (K, V),
            (V, 1),
            (0, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)

    for r in range(num_ranks):
        # Look up actual ag_hm row for causal position r — avoids materializing
        # a reordered copy of the gathered buffer (which is huge for large N).
        r_actual = tl.load(causal_order_ptr + r).to(tl.int64)
        base = r_actual * stride_rank + ag_base
        p_b = tl.make_block_ptr(
            ag_hm + base, (K, V), (V + K, 1), (0, i_v * BV), (BK, BV), (1, 0)
        )
        p_m = tl.make_block_ptr(
            ag_hm + base + V, (K, K), (V + K, 1), (0, 0), (BK, BK), (1, 0)
        )
        b_b = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)
        b_m = tl.load(p_m, boundary_check=(0, 1)).to(tl.float32)
        b_h = tl.dot(b_m, b_h) + b_b

    p_out = tl.make_block_ptr(
        h_out + (i_n * H + i_h) * K * V, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)
    )
    tl.store(p_out, b_h.to(p_out.dtype.element_ty), boundary_check=(0, 1))


def cp_merge(ag_hm, h0, num_ranks, N, H, K, V, causal_order):
    """Triton kernel merge: iterate `num_ranks` affine transforms on h0,
    reading affines from `ag_hm` in causal order via `causal_order` lookup.

    Args:
        ag_hm: all-gather raw layout, shape [2*cp_size, N, H, K, V+K].
        causal_order: [2*cp_size] int tensor; position r in causal chain
            corresponds to row `causal_order[r]` of `ag_hm`.
    """
    BK = triton.next_power_of_2(K)
    BV = 32
    h_out = torch.empty(N, H, K, V, dtype=torch.float32, device=ag_hm.device)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    cp_merge_kernel[grid](
        h_out=h_out,
        ag_hm=ag_hm,
        h0=h0,
        causal_order_ptr=causal_order,
        num_ranks=num_ranks,
        N=N,
        H=H,
        K=K,
        V=V,
        BV=BV,
        BK=BK,
        HAS_H0=h0 is not None,
        num_warps=4,
        num_stages=2,
    )
    return h_out


def _compute_both_segments(k, v, beta, g, seg_cu):
    """Run Step1-4 on both segments (seg0 + seg1 across all batches) in one pass.

    `seg_cu` treats seg0/seg1 as 2*batch independent sequences (built by
    `build_segment_cu_seqlens`). `g` is replaced by its in-segment cumsum and
    must be returned (caller's local var is shadowed).

    Returns (w, u, g_cumsum, b0, M0, b1, M1).
    """
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=seg_cu)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=seg_cu, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=seg_cu, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=seg_cu)
    b = compute_br(k=k, w=w, u=u, g=g, cu_seqlens=seg_cu)  # [2*batch, H, K, V]
    M = compute_M_total(k=k, w=w, g=g, cu_seqlens=seg_cu)  # [2*batch, H, K, K]

    b0 = b[0::2].contiguous()  # [batch, H, K, V]
    b1 = b[1::2].contiguous()
    M0 = M[0::2].contiguous()  # [batch, H, K, K]
    M1 = M[1::2].contiguous()

    return w, u, g, b0, M0, b1, M1


def chunk_gated_delta_rule_fwd_cp_zigzag(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cp_group: dist.ProcessGroup,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    seg_cu: Optional[torch.LongTensor] = None,
    causal_order: Optional[torch.Tensor] = None,
):
    """
    CP-parallel gated delta rule forward — zigzag variant.

    Each rank computes on its own zigzag tokens. No QKV all-gather needed.
    Communication is limited to SSM state affine pairs (b, M).

    Input tokens are laid out as [seg0, seg1] per sequence, where seg0 is the
    front half and seg1 is the back half of this rank's zigzag assignment.
    """
    rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)
    num_segs = 2 * cp_size

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # ---- Phase 1: compute (b, M) for both segments in one pass ----
    if seg_cu is None:
        seg_cu = build_segment_cu_seqlens(cu_seqlens)

    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=seg_cu)

    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=seg_cu, output_dtype=torch.float32
    )

    A = solve_tril(A=A, cu_seqlens=seg_cu, output_dtype=k.dtype)

    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=seg_cu)

    b = compute_br(k=k, w=w, u=u, g=g, cu_seqlens=seg_cu)  # [2*N, H, K, V]
    M = compute_M_total(k=k, w=w, g=g, cu_seqlens=seg_cu)  # [2*N, H, K, K]

    b0 = b[0::2].contiguous()  # [N, H, K, V]
    b1 = b[1::2].contiguous()
    M0 = M[0::2].contiguous()  # [N, H, K, K]
    M1 = M[1::2].contiguous()

    N = cu_seqlens.shape[0] - 1
    H = w.shape[2]
    K = k.shape[3]
    V = v.shape[-1]

    # ---- Phase 2: all-gather affine pairs ----
    packed = torch.stack(
        [
            torch.cat([b0, M0], dim=-1),
            torch.cat([b1, M1], dim=-1),
        ],
        dim=0,
    )  # [2, N, H, K, V+K]

    gathered = torch.empty(
        num_segs, *packed.shape[1:], device=packed.device, dtype=packed.dtype
    )
    dist.all_gather_into_tensor(
        gathered.view(num_segs, -1),
        packed.view(2, -1),
        group=cp_group,
    )

    # No physical reorder of `gathered` — cp_merge_kernel reads ag_hm rows
    # via `causal_order` lookup. Saves a 537MB+ transient buffer per layer
    # at high N (and avoids the PyTorch advanced-indexing grid-limit bug).
    if causal_order is None:
        causal_order = torch.tensor(
            zigzag_causal_order(cp_size), dtype=torch.long, device=packed.device
        )

    # ---- Phase 3: cp_merge to get h0 for each segment ----
    h0_global = initial_state.float() if initial_state is not None else None
    seg0_pos, seg1_pos = causal_positions(rank, cp_size)

    h0_seg0 = cp_merge(gathered, h0_global, seg0_pos, N, H, K, V, causal_order)
    h0_seg1 = cp_merge(gathered, h0_global, seg1_pos, N, H, K, V, causal_order)

    # ---- Phase 4: rerun Step5 + Step6 with correct h0 (single pass) ----
    # Interleave h0_seg0 and h0_seg1 to match seg_cu layout:
    # seg_cu sequences are [seg0_seq0, seg1_seq0, seg0_seq1, seg1_seq1, ...]
    h0_combined = torch.empty(2 * N, H, K, V, dtype=torch.float32, device=k.device)
    h0_combined[0::2] = h0_seg0
    h0_combined[1::2] = h0_seg1

    h_all, v_new_all, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0_combined,
        output_final_state=False,
        cu_seqlens=seg_cu,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new_all,
        h=h_all,
        g=g,
        scale=scale,
        cu_seqlens=seg_cu,
    )

    # final_state: result of applying every rank's affine in causal order to h0
    final_state = (
        cp_merge(gathered, h0_global, num_segs, N, H, K, V, causal_order)
        if output_final_state
        else None
    )

    return o, h_all, final_state
