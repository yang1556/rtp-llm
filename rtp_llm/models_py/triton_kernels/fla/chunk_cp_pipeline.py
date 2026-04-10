# -*- coding: utf-8 -*-
# Context Parallelism — Pipeline variant (no speculation)
#
# Each rank waits for the previous rank's h_final before running Step 5.
# No correction kernel needed. Step 1-4 run in parallel, Step 5 is pipelined.

from typing import Optional

import torch
import torch.distributed as dist
import triton

from rtp_llm.models_py.triton_kernels.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_o import chunk_fwd_o
from rtp_llm.models_py.triton_kernels.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.solve_tril import solve_tril
from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd


def chunk_gated_delta_rule_fwd_cp_pipeline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cp_group: dist.ProcessGroup,
):
    """
    CP-parallel gated delta rule forward — pipeline variant.

    Flow:
      1. All ranks: step 1-4 in parallel
      2. Rank 0: step 5 with real h0, send h_final
         Rank r>0: recv h_final from rank r-1, step 5 with it, send h_final
      3. All ranks: step 6 in parallel
    """
    rank = dist.get_rank(cp_group)
    world_size = dist.get_world_size(cp_group)

    # ---- Step 1-4: all ranks parallel ----
    g = chunk_local_cumsum(g, chunk_size=64)

    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=None, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)

    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=None,
    )

    # ---- Step 5: pipelined ----
    B = k.shape[0]
    H = w.shape[2]
    K = k.shape[3]
    V = u.shape[-1]

    if rank == 0:
        h0 = initial_state
    else:
        h0 = torch.empty(B, H, K, V, dtype=torch.float32, device=k.device)
        dist.recv(h0, src=rank - 1, group=cp_group)

    h, v_new, h_final = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0,
        output_final_state=True,
    )

    if rank < world_size - 1:
        dist.send(h_final, dst=rank + 1, group=cp_group)

    # ---- Step 6: all ranks parallel ----
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
    )

    final_state = h_final if (output_final_state and rank == world_size - 1) else None

    return o, h, final_state
