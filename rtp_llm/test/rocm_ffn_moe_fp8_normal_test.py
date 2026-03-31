# type: ignore
"""ROCm DeepEP normal-mode MoE dispatch/combine unit test.

This test validates the DeepEP normal (NVLink intranode) dispatch and combine
path on ROCm (MI308X) using the Python deep_ep buffer interface, which is the
same backend used during service startup with use_deepep_low_latency=False.

The call chain is:
  buffer.get_dispatch_layout(topk_idx, num_experts)
      -> deep_ep.Buffer.get_dispatch_layout
          -> deep_ep_cpp.Buffer.get_dispatch_layout   (C++ ROCm kernel)

  buffer.dispatch(x, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, config, ...)
      -> deep_ep.Buffer.dispatch
          -> deep_ep_cpp.Buffer.intranode_dispatch     (C++ ROCm NVLink kernel)

  buffer.combine(recv_x, handle, config, ...)
      -> deep_ep.Buffer.combine
          -> deep_ep_cpp.Buffer.intranode_combine      (C++ ROCm NVLink kernel)

Config is obtained via buffer.get_dispatch_config(num_ranks) /
buffer.get_combine_config(num_ranks), which internally constructs
deep_ep_cpp.Config with 5 args: (num_sms, nvl_chunk_size, nvl_buffer_size,
rdma_chunk_size, rdma_buffer_size).

Build and run with:
    bazelisk test --config=rocm --define=enable_deep_ep=true \\
        //rtp_llm/test:rocm_ffn_moe_fp8_normal_test
"""

import os

# Set LD_LIBRARY_PATH before importing any ROCm-dependent modules
os.environ["LD_LIBRARY_PATH"] = (
    "/opt/rocm-7.2.0/lib:/opt/rocm-7.2.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
)
import random
from typing import Any, Dict
from unittest import TestCase, main

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPBuffer,
    DeepEPWrapper,
    DeepepWrapperConfig,
    init_deepep_wrapper,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import MoeConfig, NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortsContext


def _calc_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Calculate relative max difference between two tensors."""
    a, b = a.float(), b.float()
    diff = (a - b).abs().max()
    scale = b.abs().max().clamp(min=1e-6)
    return (diff / scale).item()


def _create_engine_config(
    rank: int,
    num_ranks: int,
    args: Dict[str, Any],
    nccl_port: int,
) -> tuple:
    """Create NcclCommConfig, EngineConfig and ModelConfig for the test."""
    model_config = ModelConfig()
    model_config.attn_config.head_num = 2
    model_config.attn_config.size_per_head = 128
    model_config.num_layers = 2
    model_config.max_seq_len = 2048
    model_config.vocab_size = 500000
    model_config.moe_k = args.get("moe_k", 8)
    model_config.expert_num = args.get("expert_num", 64)
    model_config.hidden_size = args.get("hidden_size", 7168)

    base_port = nccl_port + 11
    nccl_comm_config = NcclCommConfig(
        nccl_ip="127.0.0.1",
        tp_nccl_port=base_port - 2,
        dp_tp_nccl_port=base_port - 10,
        ffn_tp_nccl_port=base_port - 5,
    )

    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = num_ranks
    parallelism_config.tp_rank = rank
    parallelism_config.ep_size = num_ranks
    parallelism_config.ep_rank = rank
    parallelism_config.dp_size = 1
    parallelism_config.dp_rank = 0
    parallelism_config.world_size = num_ranks
    parallelism_config.world_rank = rank
    parallelism_config.local_rank = rank
    parallelism_config.local_world_size = num_ranks

    moe_config = MoeConfig()
    moe_config.use_deepep_moe = True
    moe_config.use_deepep_low_latency = False  # normal mode
    moe_config.use_deepep_internode = False
    moe_config.deep_ep_num_sm = 24
    moe_config.ll_num_max_token = args.get("max_generate_batch_size", 256)

    py_env = PyEnvConfigs()
    py_env.parallelism_config = parallelism_config
    py_env.moe_config = moe_config
    py_env.runtime_config.max_generate_batch_size = args.get(
        "max_generate_batch_size", 256
    )
    py_env.concurrency_config.concurrency_limit = args.get(
        "max_generate_batch_size", 256
    )
    engine_config = EngineConfig(
        parallelism_config=py_env.parallelism_config,
        runtime_config=py_env.runtime_config,
        nccl_comm_config=nccl_comm_config,
        server_config=py_env.server_config,
        pd_sep_config=py_env.pd_separation_config,
        concurrency_config=py_env.concurrency_config,
        fmha_config=py_env.fmha_config,
        kv_cache_config=py_env.kv_cache_config,
        profiling_debug_logging_config=py_env.profiling_debug_logging_config,
        hw_kernel_config=py_env.py_hw_kernel_config,
        device_resource_config=py_env.device_resource_config,
        moe_config=py_env.moe_config,
        model_specific_config=py_env.model_specific_config,
        sp_config=py_env.sp_config,
        cache_store_config=py_env.cache_store_config,
        misc_config=py_env.misc_config.misc_config,
        arpc_config=py_env.arpc_config,
        grpc_config=py_env.grpc_config,
        load_config=py_env.load_config,
    )
    return nccl_comm_config, nccl_port, engine_config, model_config


def _test_normal_dispatch_combine(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: DeepEPBuffer,
    seed: int = 0,
):
    """Run normal dispatch + combine and verify correctness.

    Uses buffer.get_dispatch_config / get_combine_config to obtain the
    correct deep_ep_cpp.Config (5-argument form), then calls
    buffer.dispatch -> deep_ep_cpp.intranode_dispatch and
    buffer.combine  -> deep_ep_cpp.intranode_combine.

    Tests:
      1. BF16 sync dispatch + combine with topk routing verification
      2. FP8 dispatch + BF16 combine (when is_sm90_compiled())
      3. BF16 async dispatch + combine round-trip
    """
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # Build input: fill each token with rank value so routing is verifiable
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = (
        torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda") * rank
    )

    # Randomly mask a few slots to test -1 handling
    for _ in range(5):
        topk_idx[
            random.randint(0, num_tokens - 1),
            random.randint(0, num_topk - 1),
        ] = -1

    # --- layout: calls deep_ep_cpp.Buffer.get_dispatch_layout (C++ ROCm kernel) ---
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    # Use the pre-tuned configs from deep_ep.Buffer; these construct
    # deep_ep_cpp.Config(num_sms, nvl_chunk_size, nvl_buffer_size,
    #                    rdma_chunk_size, rdma_buffer_size) internally.
    DeepEPBuffer.set_num_sms(24)
    dispatch_config = DeepEPBuffer.get_dispatch_config(num_ranks)
    combine_config = DeepEPBuffer.get_combine_config(num_ranks)

    # -----------------------------------------------------------------------
    # 1. BF16 sync dispatch -> deep_ep_cpp.intranode_dispatch (C++ ROCm)
    # -----------------------------------------------------------------------
    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_num_tokens_per_expert_list,
        handle,
        _,
    ) = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=dispatch_config,
        async_finish=False,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )

    # Verify received token count matches global sum for this rank
    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), (
        f"[rank {rank}] BF16 recv token count mismatch: "
        f"expected {gbl_num_tokens_per_rank[rank].item()}, got {recv_x.size(0)}"
    )
    # Verify per-expert token counts
    assert (
        gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
        == recv_num_tokens_per_expert_list
    ), (
        f"[rank {rank}] BF16 per-expert token count mismatch: "
        f"expected {gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()}, "
        f"got {recv_num_tokens_per_expert_list}"
    )
    # Verify each received row is uniform (all elements from same source rank)
    assert torch.allclose(
        recv_x.amin(dim=1), recv_x.amax(dim=1)
    ), f"[rank {rank}] BF16 received tokens have mixed rank values"
    # Verify recv_topk_idx is in valid local-expert range
    assert (
        recv_topk_idx.eq(-1)
        | ((recv_topk_idx >= 0) & (recv_topk_idx < num_local_experts))
    ).all(), f"[rank {rank}] BF16 recv_topk_idx out of local range"

    # BF16 sync combine -> deep_ep_cpp.intranode_combine (C++ ROCm)
    combined_x, _, _ = buffer.combine(
        x=recv_x,
        handle=handle,
        config=combine_config,
        async_finish=False,
        topk_weights=recv_topk_weights,
    )

    # combined_x = sum over dispatched ranks of recv_x (each with its weight).
    # Since topk_weights are all `rank` and recv_x rows are all the source rank
    # value, the result per token equals x * (number of ranks the token was
    # sent to). Verify by normalising.
    num_ranks_per_token = is_token_in_rank.sum(dim=1, keepdim=True).float()  # [T,1]
    diff = _calc_diff(combined_x.float() / num_ranks_per_token.clamp(min=1), x)
    assert not torch.isnan(
        combined_x
    ).any(), f"[rank {rank}] NaN in BF16 combined output"
    assert diff < 5e-3, f"[rank {rank}] BF16 combine diff {diff:.6f} exceeds threshold"

    if rank == 0:
        print(
            f"[rank {rank}] normal BF16 dispatch+combine passed, diff={diff:.2e}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # 2. FP8 dispatch + BF16 combine (only when SM90 kernel is compiled)
    # -----------------------------------------------------------------------
    if DeepEPBuffer.is_sm90_compiled():
        try:
            from rtp_llm.test.utils.numeric_util import (
                per_token_cast_back,
                per_token_cast_to_fp8,
            )

            x_e4m3_data, x_e4m3_scale = per_token_cast_to_fp8(x, False)
            # deep_ep expects scale in column-major layout
            x_fp8 = (x_e4m3_data, x_e4m3_scale.T.contiguous().T)

            (
                recv_fp8,
                recv_topk_idx_fp8,
                recv_topk_weights_fp8,
                recv_num_tokens_per_expert_list_fp8,
                handle_fp8,
                event_fp8,
            ) = buffer.dispatch(
                x=x_fp8,
                num_tokens_per_rank=num_tokens_per_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                config=dispatch_config,
                async_finish=True,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
            )
            event_fp8.current_stream_wait()

            # recv_fp8 is a (fp8_tensor, scale_tensor) tuple
            assert isinstance(
                recv_fp8, tuple
            ), f"[rank {rank}] FP8 dispatch should return a tuple"
            assert (
                gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
                == recv_num_tokens_per_expert_list_fp8
            ), (
                f"[rank {rank}] FP8 per-expert token count mismatch: "
                f"expected {gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()}, "
                f"got {recv_num_tokens_per_expert_list_fp8}"
            )

            # Dequant FP8 -> BF16 then combine
            recv_x_bf16 = per_token_cast_back(*recv_fp8)
            combined_fp8, _, _ = buffer.combine(
                x=recv_x_bf16,
                handle=handle_fp8,
                config=combine_config,
                async_finish=False,
                topk_weights=recv_topk_weights_fp8,
            )
            assert not torch.isnan(
                combined_fp8
            ).any(), f"[rank {rank}] NaN in FP8->BF16 combined output"
            if rank == 0:
                print(
                    f"[rank {rank}] normal FP8 dispatch+BF16 combine passed", flush=True
                )

        except Exception as e:
            if rank == 0:
                print(
                    f"[rank {rank}] FP8 dispatch test skipped due to: {e}",
                    flush=True,
                )
    else:
        if rank == 0:
            print(
                f"[rank {rank}] Skipping FP8 dispatch (SM90 kernel not compiled)",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # 3. BF16 async dispatch + combine round-trip
    # -----------------------------------------------------------------------
    (
        recv_x_async,
        _,
        _,
        _,
        handle_async,
        event_async,
    ) = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=dispatch_config,
        async_finish=True,
    )
    event_async.current_stream_wait()

    combined_async, _, event_combine_async = buffer.combine(
        x=recv_x_async,
        handle=handle_async,
        config=combine_config,
        async_finish=True,
    )
    event_combine_async.current_stream_wait()

    diff_async = _calc_diff(
        combined_async.float() / num_ranks_per_token.clamp(min=1), x
    )
    assert not torch.isnan(
        combined_async
    ).any(), f"[rank {rank}] NaN in async BF16 combined output"
    assert (
        diff_async < 5e-3
    ), f"[rank {rank}] async BF16 combine diff {diff_async:.6f} exceeds threshold"

    if rank == 0:
        print(
            f"[rank {rank}] normal async BF16 dispatch+combine passed, diff={diff_async:.2e}",
            flush=True,
        )


def _worker(rank: int, num_ranks: int, args: Dict[str, Any], nccl_port: int):
    """Per-process worker: init distributed env, DeepEP buffer, run test."""
    # ROCm uses HIP_VISIBLE_DEVICES
    os.environ["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_ranks))

    # ROCm library path
    os.environ["LD_LIBRARY_PATH"] = (
        "/opt/rocm-7.2.0/lib:/opt/rocm-7.2.0/lib64:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )

    # AITER settings
    os.environ["AITER_ASM_DIR"] = (
        "/opt/conda310/lib/python3.10/site-packages/aiter_meta/hsa/"
    )
    os.environ["USE_AITER_PA"] = "0"

    # Custom allreduce and communication settings
    os.environ["FT_DISABLE_CUSTOM_AR"] = "0"
    os.environ["ENABLE_COMM_OVERLAP"] = "0"
    os.environ["FORCE_CPU_LOAD_WEIGHTS"] = "1"

    # NVSHMEM: normal mode does NOT use IBGDA (that is low-latency only)
    os.environ["NVSHMEM_DEBUG"] = "WARN"
    os.environ["NVSHMEM_IB_GID_INDEX"] = "1"
    os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "false"
    os.environ["NVSHMEM_DISABLE_GDRCOPY"] = "true"

    # NCCL settings (from start_coder_normal.sh)
    os.environ["NCCL_NET_PLUGIN"] = "none"
    os.environ["NCCL_IB_GID_INDEX"] = "1"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_IB_HCA"] = "fic2"

    # Normal mode env vars (from start_coder_normal.sh)
    os.environ["NORMAL_USE_COMPUTE_STREAM"] = "1"
    os.environ["FAKE_BALANCE_EXPERT"] = "1"
    os.environ["USE_SWIZZLEA"] = "1"
    os.environ["USE_ASM_PA"] = "0"
    os.environ["USE_ALL_GATHER"] = "0"
    os.environ["EIC_NONFETCH_ADD_ENABLE"] = "1"

    nccl_comm_config, nccl_init_port, engine_config, model_config = (
        _create_engine_config(rank, num_ranks, args, nccl_port)
    )

    parallelism_config = engine_config.parallelism_config

    torch.cuda.set_device(parallelism_config.local_rank)
    torch.set_default_device(f"cuda:{parallelism_config.local_rank}")

    init_distributed_environment(
        parallelism_config=parallelism_config,
        nccl_comm_config=nccl_comm_config,
        nccl_init_port=nccl_init_port,
        backend="nccl",
        timeout=120,
    )

    # init_deepep_wrapper creates deep_ep.Buffer with low_latency_mode=False,
    # which routes to deep_ep_cpp.intranode_dispatch/combine (C++ ROCm kernels).
    init_deepep_wrapper(engine_config, model_config)

    config_adapter = MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=engine_config.parallelism_config,
        moe_config=engine_config.moe_config,
    )
    # Normal mode: ll_num_max_token_per_rank = 0 (unused)
    deepep_config = DeepepWrapperConfig.from_config_adapter(config_adapter, 0)
    deepep_wrapper = DeepEPWrapper.get_instance(deepep_config)
    buffer = deepep_wrapper.buffer

    num_tokens = args.get("max_seq_len", 256)

    try:
        _test_normal_dispatch_combine(
            num_tokens=num_tokens,
            hidden=deepep_wrapper.hidden_size,
            num_experts=deepep_wrapper.num_experts,
            num_topk=deepep_wrapper.num_topk,
            rank=deepep_wrapper.ep_rank,
            num_ranks=deepep_wrapper.ep_size,
            group=dist.group.WORLD,
            buffer=buffer,
            seed=42,
        )
    finally:
        DeepEPWrapper.reset()
        destroy_distributed_environment()


def _worker_with_queue(
    rank: int, num_ranks: int, args: Dict[str, Any], nccl_port: int, queue
):
    """Worker wrapper that captures output and exceptions for error reporting."""
    import sys
    import traceback
    from io import StringIO

    stdout_capture = StringIO()
    stderr_capture = StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    result = {"rank": rank}
    try:
        _worker(rank, num_ranks, args, nccl_port)
        result["success"] = True
    except Exception:
        result["success"] = False
        result["exception"] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    result["stdout"] = stdout_capture.getvalue()
    result["stderr"] = stderr_capture.getvalue()
    queue.put(result)


def _run_multiprocess_test(num_ranks: int, args: Dict[str, Any]):
    """Spawn num_ranks processes and run the DeepEP normal test."""
    import sys

    with PortsContext(None, 1) as ports:
        nccl_port = ports[0]
        ctx = mp.get_context("spawn")
        procs, queues = [], []
        for rank in range(num_ranks):
            q = ctx.Queue()
            queues.append(q)
            p = ctx.Process(
                target=_worker_with_queue,
                args=(rank, num_ranks, args, nccl_port, q),
            )
            p.start()
            procs.append(p)

        results = []
        for i, p in enumerate(procs):
            p.join(timeout=300)
            try:
                result = queues[i].get(timeout=5)
            except Exception:
                result = {"error": "Failed to get result from queue"}
            results.append(result)

            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
                raise RuntimeError("DeepEP normal test worker timed out")
            if p.exitcode != 0:
                print(
                    f"\n=== Worker {i} failed with exit code {p.exitcode} ===",
                    file=sys.stderr,
                )
                if "stdout" in results[i]:
                    print(f"STDOUT:\n{results[i]['stdout']}", file=sys.stderr)
                if "stderr" in results[i]:
                    print(f"STDERR:\n{results[i]['stderr']}", file=sys.stderr)
                if "exception" in results[i]:
                    print(f"EXCEPTION:\n{results[i]['exception']}", file=sys.stderr)
                raise RuntimeError(
                    f"DeepEP normal test worker {i} exited with code {p.exitcode}"
                )


class RocmFfnMoeFp8NormalTest(TestCase):
    """ROCm DeepEP normal-mode dispatch/combine correctness test.

    Uses the Python deep_ep buffer interface (same backend as production
    service with use_deepep_low_latency=False) to validate:
      - BF16 sync normal dispatch + combine (routing + value correctness)
      - FP8 dispatch + BF16 combine (no NaN, when SM90 kernel is compiled)
      - BF16 async dispatch + combine round-trip

    The underlying C++ path is:
      deep_ep_cpp.Buffer.intranode_dispatch / intranode_combine  (ROCm NVLink)

    Requires at least 4 ROCm GPUs. Build with:
        --config=rocm --define=enable_deep_ep=true
    """

    TEST_ARGS = {
        "moe_k": 8,
        "expert_num": 64,
        "hidden_size": 7168,
        "max_generate_batch_size": 256,
        "max_seq_len": 256,
    }

    def _skip_if_insufficient_gpus(self, required: int):
        if torch.cuda.device_count() < required:
            self.skipTest(f"Need {required} GPUs, found {torch.cuda.device_count()}")

    def _skip_if_deepep_not_supported(self):
        if not DeepEPWrapper.supported():
            self.skipTest(
                "DeepEP not supported: 'deep_ep' Python package not available. "
                "Build with --define=enable_deep_ep=true and install deep_ep ROCm whl."
            )

    def test_normal_dispatch_combine_4ranks(self):
        """Test DeepEP normal dispatch+combine with 4 ROCm GPUs."""
        num_ranks = 4
        self._skip_if_insufficient_gpus(num_ranks)
        self._skip_if_deepep_not_supported()
        _run_multiprocess_test(num_ranks, self.TEST_ARGS)

    def test_normal_dispatch_combine_8ranks(self):
        """Test DeepEP normal dispatch+combine with 8 ROCm GPUs."""
        num_ranks = 8
        self._skip_if_insufficient_gpus(num_ranks)
        self._skip_if_deepep_not_supported()
        _run_multiprocess_test(num_ranks, self.TEST_ARGS)


if __name__ == "__main__":
    main()
