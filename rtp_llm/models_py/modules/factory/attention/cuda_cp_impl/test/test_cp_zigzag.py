# -*- coding: utf-8 -*-
# Test: Qwen3NextGatedDeltaNet CP zigzag prefill correctness + SSM state storage
#
# Verifies:
#   1. CP zigzag prefill output matches single-GPU reference (per-token)
#   2. SSM states written to kv_cache block map match reference
#
# Usage:
#   PYTHONPATH=/root/hzy/rtp-llm:$PYTHONPATH CUDA_VISIBLE_DEVICES=2,3 \
#       /opt/conda310/bin/torchrun --master_port=30200 --nproc_per_node=2 \
#       rtp_llm/models_py/triton_kernels/fla/test/test_cp_zigzag_integration.py

import logging
import math
import os
import time
from contextlib import contextmanager
from typing import List, Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

logging.basicConfig(level=logging.INFO, format="%(message)s")

PROFILE_DIR = "/root/hzy/rtp-llm/rtp_llm/models_py/triton_kernels/fla/test/profiles"


@contextmanager
def _noop_write_cache_store():
    import rtp_llm.ops.compute_ops as _co

    with patch.object(_co, "write_cache_store", lambda *a, **kw: None):
        yield


class MockLayerKVCache:
    def __init__(self, kv_cache_base: torch.Tensor, seq_size_per_block: int):
        self.kv_cache_base = kv_cache_base
        self.seq_size_per_block = seq_size_per_block


def zigzag_positions_for_rank(full_len: int, cp_size: int, rank: int) -> List[int]:
    chunk_len = full_len // cp_size
    half = chunk_len // 2
    first = list(range(rank * half, (rank + 1) * half))
    second = list(range(full_len - (rank + 1) * half, full_len - rank * half))
    return first + second


def compute_rank_positions(
    sequence_lengths: List[int], cp_size: int
) -> List[List[int]]:
    all_rank_pos = [[] for _ in range(cp_size)]
    offset = 0
    for sl in sequence_lengths:
        for r in range(cp_size):
            positions = zigzag_positions_for_rank(sl, cp_size, r)
            all_rank_pos[r].extend([p + offset for p in positions])
        offset += sl
    return all_rank_pos


def build_restore_indices(cp_chunk_lengths: List[int], cp_size: int) -> torch.Tensor:
    batch_size = len(cp_chunk_lengths)
    ag_positions = []
    for rank in range(cp_size):
        for b in range(batch_size):
            chunk_len = cp_chunk_lengths[b]
            full_len = chunk_len * cp_size
            positions = zigzag_positions_for_rank(full_len, cp_size, rank)
            seq_offset = sum(cp_chunk_lengths[:b]) * cp_size
            ag_positions.extend([p + seq_offset for p in positions])
    total = len(ag_positions)
    restore = [0] * total
    for ag_idx, orig_pos in enumerate(ag_positions):
        restore[orig_pos] = ag_idx
    return torch.tensor(restore, dtype=torch.int32)


def build_padding_mask(cp_chunk_lengths: List[int], cp_size: int) -> torch.Tensor:
    return torch.ones(sum(cp_chunk_lengths) * cp_size, dtype=torch.int32)


class _AttnInputsWrapper:
    def __init__(self, wrapped, overrides: dict):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name, value):
        try:
            setattr(object.__getattribute__(self, "_wrapped"), name, value)
        except AttributeError:
            object.__getattribute__(self, "_overrides")[name] = value


def _add_device_tensors(inputs, device):
    return _AttnInputsWrapper(
        inputs,
        {
            "prefix_lengths_d": inputs.prefix_lengths.to(device),
            "input_lengths_d": inputs.input_lengths.to(device),
        },
    )


from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    prepare_causal_conv1d_metadata,
)
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyContextParallelParams,
    get_typemeta,
)


def build_cp_attn_inputs(
    sequence_lengths,
    cp_chunk_lengths,
    cp_size,
    tokens_per_block,
    prefix_lengths=None,
    device=torch.device("cuda"),
):
    batch_size = len(cp_chunk_lengths)
    if prefix_lengths is None:
        prefix_lengths = [0] * batch_size

    inp = PyAttentionInputs()
    inp.is_prefill = True
    inp.input_lengths = torch.tensor(
        cp_chunk_lengths, dtype=torch.int32, device="cpu"
    ).pin_memory()
    inp.sequence_lengths = torch.tensor(
        sequence_lengths, dtype=torch.int32, device="cpu"
    ).pin_memory()
    inp.prefix_lengths = torch.tensor(prefix_lengths, dtype=torch.int32, device="cpu")

    cu = [0]
    for cl in cp_chunk_lengths:
        cu.append(cu[-1] + cl)
    inp.cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)

    max_blocks = max(math.ceil(sl / tokens_per_block) for sl in sequence_lengths)
    block_ids = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
    offset = 0
    for i, sl in enumerate(sequence_lengths):
        nb = math.ceil(sl / tokens_per_block)
        block_ids[i, :nb] = torch.arange(offset, offset + nb, dtype=torch.int32)
        offset += nb
    inp.kv_cache_block_id_host = block_ids
    inp.kv_cache_kernel_block_id_host = block_ids
    inp.kv_cache_block_id_device = block_ids.to(device)
    inp.kv_cache_kernel_block_id_device = block_ids.to(device)
    inp.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
    inp.cache_store_inputs = None

    new_lengths = [sl - pl for sl, pl in zip(sequence_lengths, prefix_lengths)]
    cp_info = PyContextParallelParams()
    cp_info.prefill_cp_chunk_lengths = torch.tensor(cp_chunk_lengths, dtype=torch.int32)
    cp_info.prefill_cp_padding_lengths = torch.zeros(batch_size, dtype=torch.int32)
    cp_info.prefill_qkv_padding_mask = build_padding_mask(cp_chunk_lengths, cp_size).to(
        device
    )
    cp_info.prefill_qkv_restore_indice = build_restore_indices(
        cp_chunk_lengths, cp_size
    ).to(device)
    cp_info.prefill_actual_input_lengths_cpu = torch.tensor(
        new_lengths, dtype=torch.int32
    )
    cp_info.prefill_shuffle_indices = torch.tensor([], dtype=torch.int32)
    inp.context_parallel_info = cp_info
    return inp


def build_nocp_attn_inputs(sequence_lengths, tokens_per_block, device):
    batch_size = len(sequence_lengths)
    inp = PyAttentionInputs()
    inp.is_prefill = True
    inp.input_lengths = torch.tensor(sequence_lengths, dtype=torch.int32, device="cpu")
    inp.prefix_lengths = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    inp.cache_store_inputs = None

    cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, sl in enumerate(sequence_lengths):
        cu[i + 1] = cu[i] + sl
    inp.cu_seqlens = cu

    max_blocks = max(math.ceil(sl / tokens_per_block) for sl in sequence_lengths)
    block_ids = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
    offset = 0
    for i, sl in enumerate(sequence_lengths):
        nb = math.ceil(sl / tokens_per_block)
        block_ids[i, :nb] = torch.arange(offset, offset + nb, dtype=torch.int32)
        offset += nb
    inp.kv_cache_block_id_host = block_ids
    inp.kv_cache_kernel_block_id_host = block_ids
    inp.kv_cache_block_id_device = block_ids.to(device)
    inp.kv_cache_kernel_block_id_device = block_ids.to(device)
    inp.context_parallel_info = None
    inp.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
    return inp


def allocate_kv_cache(
    sequence_lengths,
    tokens_per_block,
    num_v_heads,
    head_v_dim,
    head_k_dim,
    conv_kernel_dim,
    qkv_size,
    dtype,
    device,
):
    total_blocks = sum(math.ceil(sl / tokens_per_block) for sl in sequence_lengths)
    item_size = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}[dtype]
    ssm_bytes = num_v_heads * head_v_dim * head_k_dim * item_size
    conv_bytes = (conv_kernel_dim - 1) * qkv_size * item_size
    block_elems = (ssm_bytes + conv_bytes) // item_size
    return torch.zeros(total_blocks, block_elems, dtype=dtype, device=device)


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    from rtp_llm.models_py.distributed import collective_torch as ct
    from rtp_llm.models_py.distributed.collective_torch import Group
    from rtp_llm.ops import ParallelismConfig

    par = ParallelismConfig()
    par.world_rank = rank
    par.world_size = world_size
    par.local_rank = rank
    par.tp_size = world_size
    par.dp_size = 1

    ct._parallelism_config = par
    ct._initialized = True
    ct._group_map[Group.DP_AND_TP] = dist.group.WORLD


def run_test():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    os.makedirs(PROFILE_DIR, exist_ok=True)

    from rtp_llm.models_py.model_desc.qwen3_next import (
        CpChunkAlignInfo,
        Qwen3NextGatedDeltaNet,
        Qwen3NextMetadata,
    )
    from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
    from rtp_llm.ops import DataType, LinearAttentionConfig, ParallelismConfig
    from rtp_llm.utils.model_weight import W

    cp_size = world_size
    num_k_heads = 16
    num_v_heads = 32
    head_k_dim = 128
    head_v_dim = 128
    hidden_size = num_v_heads * head_v_dim
    conv_kernel_dim = 4
    tokens_per_block = 16
    num_layers = 2
    dtype = torch.bfloat16

    test_cases = [
        # ([512], "single_512"),
        # ([1024], "single_1024"),
        ([64 * 1024], "batch_64k"),
    ]

    torch.manual_seed(123)
    qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
    z_dim = head_v_dim * num_v_heads

    layer_weights_list = []
    for _ in range(num_layers):
        qkvz_scale = hidden_size**-0.5
        ba_scale = hidden_size**-0.5
        out_scale = (num_v_heads * head_v_dim) ** -0.5
        layer_weights_list.append(
            {
                W.linear_attn_conv1d_w: torch.randn(
                    qkv_dim, 1, conv_kernel_dim, device=device, dtype=dtype
                )
                * (conv_kernel_dim**-0.5),
                W.linear_attn_dt_b: torch.randn(num_v_heads, device=device, dtype=dtype)
                * 0.1,
                W.linear_attn_alog: torch.randn(num_v_heads, device=device, dtype=dtype)
                * 0.1,
                W.linear_attn_norm_w: torch.ones(
                    head_v_dim, device=device, dtype=dtype
                ),
                W.linear_attn_qkvz_w: torch.randn(
                    hidden_size, qkv_dim + z_dim, device=device, dtype=dtype
                )
                * qkvz_scale,
                W.linear_attn_qkvz_s: None,
                W.linear_attn_ba_w: torch.randn(
                    hidden_size, num_v_heads * 2, device=device, dtype=dtype
                )
                * ba_scale,
                W.linear_attn_out_w: torch.randn(
                    num_v_heads * head_v_dim, hidden_size, device=device, dtype=dtype
                )
                * out_scale,
                W.linear_attn_out_s: None,
            }
        )

    linear_cfg = LinearAttentionConfig()
    linear_cfg.linear_num_key_heads = num_k_heads
    linear_cfg.linear_num_value_heads = num_v_heads
    linear_cfg.linear_key_head_dim = head_k_dim
    linear_cfg.linear_value_head_dim = head_v_dim
    linear_cfg.linear_conv_kernel_dim = conv_kernel_dim
    linear_cfg.ssm_state_dtype = DataType.TYPE_BF16
    linear_cfg.conv_state_dtype = DataType.TYPE_BF16

    layers_cp = []
    layers_nocp = []
    for lw in layer_weights_list:
        m_cp = Qwen3NextGatedDeltaNet(
            linear_cfg, ParallelismConfig(), lw, layernorm_eps=1e-6
        ).to(device)
        layers_cp.append(m_cp)

        m_nocp = Qwen3NextGatedDeltaNet(
            linear_cfg, ParallelismConfig(), lw, layernorm_eps=1e-6
        ).to(device)
        layers_nocp.append(m_nocp)

    for m_cp in layers_cp:
        m_cp.parallelism_config.tp_size = cp_size
        m_cp.parallelism_config.tp_rank = rank

    cache_converter = LinearCacheConverter(
        local_num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
        head_k_dim=head_k_dim,
        ssm_state_dtype=dtype,
        linear_conv_kernel_dim=conv_kernel_dim,
        qkv_size=qkv_dim,
        conv_state_dtype=dtype,
    )

    all_passed = True

    for seq_lengths, tag in test_cases:
        torch.manual_seed(42)
        assert all(
            sl % (cp_size * 2 * 64) == 0 for sl in seq_lengths
        ), f"Seq lengths must be divisible by {cp_size * 2 * 64}"

        batch_size = len(seq_lengths)
        total_tokens = sum(seq_lengths)
        cp_chunk_lengths = [sl // cp_size for sl in seq_lengths]

        full_hidden = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)

        # ---- Non-CP reference ----
        nocp_inputs = build_nocp_attn_inputs(seq_lengths, tokens_per_block, device)
        nocp_inputs = _add_device_tensors(nocp_inputs, device)
        nocp_conv_meta = prepare_causal_conv1d_metadata(
            query_start_loc=nocp_inputs.cu_seqlens, device=device
        )
        nocp_meta = Qwen3NextMetadata(prefill_conv1d_meta=nocp_conv_meta)

        nocp_kv_list = []
        ref_hidden = full_hidden
        with torch.no_grad(), _noop_write_cache_store():
            # Warmup non-CP
            for _ in range(3):
                warmup_h = full_hidden
                for layer_idx in range(num_layers):
                    kv_tmp = MockLayerKVCache(
                        allocate_kv_cache(
                            seq_lengths,
                            tokens_per_block,
                            num_v_heads,
                            head_v_dim,
                            head_k_dim,
                            conv_kernel_dim,
                            qkv_dim,
                            dtype,
                            device,
                        ),
                        tokens_per_block,
                    )
                    warmup_h = layers_nocp[layer_idx](
                        warmup_h, None, kv_tmp, nocp_inputs, nocp_meta
                    )

            # Time non-CP
            torch.cuda.synchronize()
            nocp_times = []
            for _ in range(5):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                h = full_hidden
                for layer_idx in range(num_layers):
                    kv_tmp = MockLayerKVCache(
                        allocate_kv_cache(
                            seq_lengths,
                            tokens_per_block,
                            num_v_heads,
                            head_v_dim,
                            head_k_dim,
                            conv_kernel_dim,
                            qkv_dim,
                            dtype,
                            device,
                        ),
                        tokens_per_block,
                    )
                    h = layers_nocp[layer_idx](h, None, kv_tmp, nocp_inputs, nocp_meta)
                torch.cuda.synchronize()
                nocp_times.append((time.perf_counter() - t0) * 1000)
            nocp_avg_ms = sum(nocp_times) / len(nocp_times)

            # Profile non-CP
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            ) as nocp_prof:
                for _ in range(3):
                    prof_h = full_hidden
                    for layer_idx in range(num_layers):
                        kv_tmp = MockLayerKVCache(
                            allocate_kv_cache(
                                seq_lengths,
                                tokens_per_block,
                                num_v_heads,
                                head_v_dim,
                                head_k_dim,
                                conv_kernel_dim,
                                qkv_dim,
                                dtype,
                                device,
                            ),
                            tokens_per_block,
                        )
                        prof_h = layers_nocp[layer_idx](
                            prof_h, None, kv_tmp, nocp_inputs, nocp_meta
                        )
            nocp_prof.export_chrome_trace(
                os.path.join(PROFILE_DIR, f"nocp_zigzag_{tag}_rank{rank}.json")
            )

            # Correctness run
            for layer_idx in range(num_layers):
                nocp_kv = MockLayerKVCache(
                    allocate_kv_cache(
                        seq_lengths,
                        tokens_per_block,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        conv_kernel_dim,
                        qkv_dim,
                        dtype,
                        device,
                    ),
                    tokens_per_block,
                )
                nocp_kv_list.append(nocp_kv)
                ref_hidden = layers_nocp[layer_idx](
                    ref_hidden, None, nocp_kv, nocp_inputs, nocp_meta
                )
        ref_output = ref_hidden

        # ---- CP zigzag path ----
        all_rank_pos = compute_rank_positions(seq_lengths, cp_size)
        rank_positions = all_rank_pos[rank]
        rank_idx = torch.tensor(rank_positions, device=device)
        local_hidden = full_hidden[rank_idx].contiguous()

        cp_attn_inputs = build_cp_attn_inputs(
            seq_lengths,
            cp_chunk_lengths,
            cp_size,
            tokens_per_block,
            device=device,
        )
        cp_attn_inputs = _add_device_tensors(cp_attn_inputs, device)

        cp_info = cp_attn_inputs.context_parallel_info
        restore_indices = cp_info.prefill_qkv_restore_indice

        full_cu_from_actual = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )
        full_cu_from_actual[1:] = torch.tensor(seq_lengths, device=device).cumsum(0)

        cp_meta = Qwen3NextMetadata(
            cp_restore_indices=restore_indices,
            cp_chunk_align_info=CpChunkAlignInfo.build(
                full_cu=full_cu_from_actual,
                cp_size=cp_size,
                cp_rank=rank,
                device=device,
            ),
        )

        cp_kv_list = []
        cp_hidden = local_hidden
        with torch.no_grad(), _noop_write_cache_store():
            # Warmup CP
            for _ in range(3):
                warmup_h = local_hidden
                for layer_idx in range(num_layers):
                    kv_tmp = MockLayerKVCache(
                        allocate_kv_cache(
                            seq_lengths,
                            tokens_per_block,
                            num_v_heads,
                            head_v_dim,
                            head_k_dim,
                            conv_kernel_dim,
                            qkv_dim,
                            dtype,
                            device,
                        ),
                        tokens_per_block,
                    )
                    warmup_h = layers_cp[layer_idx](
                        warmup_h, None, kv_tmp, cp_attn_inputs, cp_meta
                    )
            dist.barrier()

            # Time CP
            cp_times = []
            for _ in range(5):
                dist.barrier()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                h = local_hidden
                for layer_idx in range(num_layers):
                    kv_tmp = MockLayerKVCache(
                        allocate_kv_cache(
                            seq_lengths,
                            tokens_per_block,
                            num_v_heads,
                            head_v_dim,
                            head_k_dim,
                            conv_kernel_dim,
                            qkv_dim,
                            dtype,
                            device,
                        ),
                        tokens_per_block,
                    )
                    h = layers_cp[layer_idx](h, None, kv_tmp, cp_attn_inputs, cp_meta)
                torch.cuda.synchronize()
                cp_times.append((time.perf_counter() - t0) * 1000)
            cp_avg_ms = sum(cp_times) / len(cp_times)

            # Profile CP
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            ) as cp_prof:
                for _ in range(3):
                    prof_h = local_hidden
                    for layer_idx in range(num_layers):
                        kv_tmp = MockLayerKVCache(
                            allocate_kv_cache(
                                seq_lengths,
                                tokens_per_block,
                                num_v_heads,
                                head_v_dim,
                                head_k_dim,
                                conv_kernel_dim,
                                qkv_dim,
                                dtype,
                                device,
                            ),
                            tokens_per_block,
                        )
                        prof_h = layers_cp[layer_idx](
                            prof_h, None, kv_tmp, cp_attn_inputs, cp_meta
                        )
            cp_prof.export_chrome_trace(
                os.path.join(PROFILE_DIR, f"cp_zigzag_{tag}_rank{rank}.json")
            )

            # Correctness run
            for layer_idx in range(num_layers):
                cp_kv = MockLayerKVCache(
                    allocate_kv_cache(
                        seq_lengths,
                        tokens_per_block,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        conv_kernel_dim,
                        qkv_dim,
                        dtype,
                        device,
                    ),
                    tokens_per_block,
                )
                cp_kv_list.append(cp_kv)
                cp_hidden = layers_cp[layer_idx](
                    cp_hidden, None, cp_kv, cp_attn_inputs, cp_meta
                )
        cp_output = cp_hidden

        # Gather CP times across ranks
        cp_time_tensor = torch.tensor([cp_avg_ms], device=device)
        cp_time_all = [torch.zeros(1, device=device) for _ in range(world_size)]
        dist.all_gather(cp_time_all, cp_time_tensor)

        if rank == 0:
            cp_max = max(t.item() for t in cp_time_all)
            cp_per_rank = [f"{t.item():.2f}" for t in cp_time_all]
            print(
                f"  [{tag}] timing: nocp={nocp_avg_ms:.2f}ms  "
                f"cp_max={cp_max:.2f}ms  speedup={nocp_avg_ms/cp_max:.2f}x  "
                f"cp_per_rank=[{', '.join(cp_per_rank)}]ms"
            )

        # ---- Verify output ----
        ref_local = ref_output[rank_idx]
        out_diff = (cp_output.float() - ref_local.float()).abs()
        out_max_diff = out_diff.max().item()
        out_mean_diff = out_diff.mean().item()
        output_ok = out_max_diff < 1e-1

        # ---- Verify SSM states ----
        ssm_ok = True
        ssm_max_diff = 0.0
        ssm_blocks_checked = 0
        block_ids_host = nocp_inputs.kv_cache_kernel_block_id_host
        chunk_size = 64

        for layer_idx in range(num_layers):
            ref_kv_flat = nocp_kv_list[layer_idx].kv_cache_base.reshape(
                nocp_kv_list[layer_idx].kv_cache_base.shape[0], -1
            )
            ref_ssm = cache_converter.get_ssm_state_tensor(ref_kv_flat)

            cp_kv_flat = cp_kv_list[layer_idx].kv_cache_base.reshape(
                cp_kv_list[layer_idx].kv_cache_base.shape[0], -1
            )
            cp_ssm = cache_converter.get_ssm_state_tensor(cp_kv_flat)

            for b in range(batch_size):
                sl = seq_lengths[b]
                num_chunks = math.ceil(sl / chunk_size)
                for c in range(num_chunks):
                    is_last = (c + 1) * chunk_size >= sl
                    is_aligned = c > 0 and (
                        (c + 1) * chunk_size % tokens_per_block == 0
                    )
                    if not (is_last or is_aligned):
                        continue
                    pos = sl - 1 if is_last else (c + 1) * chunk_size - 1
                    block_idx = pos // tokens_per_block
                    block_id = block_ids_host[b, block_idx].item()
                    ref_state = ref_ssm[block_id]
                    cp_state = cp_ssm[block_id]
                    diff = (ref_state.float() - cp_state.float()).abs().max().item()
                    ssm_max_diff = max(ssm_max_diff, diff)
                    ssm_blocks_checked += 1
                    if diff >= 1e-1:
                        ssm_ok = False

        passed = output_ok and ssm_ok

        dist.barrier()
        print(
            f"  [{tag}] rank {rank}: "
            f"out_max={out_max_diff:.6f} out_mean={out_mean_diff:.6f} "
            f"ssm_max={ssm_max_diff:.6f} ssm_blocks={ssm_blocks_checked} "
            f"{'PASS' if passed else 'FAIL'}"
        )
        dist.barrier()

        if not passed:
            all_passed = False

    if rank == 0:
        status = "ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED!"
        print(f"\n{status}")

    dist.destroy_process_group()
    if not all_passed:
        exit(1)


if __name__ == "__main__":
    run_test()
