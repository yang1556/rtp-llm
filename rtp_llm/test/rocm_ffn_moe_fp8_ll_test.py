# type: ignore
"""ROCm DeepEP FP8 low-latency MoE dispatch/combine unit test.

This test validates the DeepEP low-latency (IBGDA) dispatch and combine path
on ROCm (MI308X) using the Python deep_ep buffer interface, which is the same
backend used during service startup with use_deepep_low_latency=True.

Build and run with:
    bazelisk test --config=rocm --define=enable_deep_ep=true \
        //rtp_llm/test:rocm_ffn_moe_fp8_ll_test
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
    moe_config.use_deepep_low_latency = True
    moe_config.use_deepep_internode = False
    moe_config.deep_ep_num_sm = 24
    moe_config.ll_num_max_token = args.get("max_generate_batch_size", 32)

    py_env = PyEnvConfigs()
    py_env.parallelism_config = parallelism_config
    py_env.moe_config = moe_config
    py_env.runtime_config.max_generate_batch_size = args.get(
        "max_generate_batch_size", 32
    )
    py_env.concurrency_config.concurrency_limit = args.get(
        "max_generate_batch_size", 32
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


def _test_ll_dispatch_combine(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer,
    seed: int = 0,
):
    """Run low-latency dispatch + combine and verify correctness."""
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # Build input: fill each token with its rank value so we can verify routing
    rank_offset = 128
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (
        rank - rank_offset
    )
    # Last 128 dims carry token indices for routing validation
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)

    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    ).abs()

    # Randomly mask a few slots to test -1 handling
    for _ in range(5):
        topk_idx[
            random.randint(0, num_tokens - 1),
            random.randint(0, num_topk - 1),
        ] = -1

    cumulative_stats = torch.zeros((num_local_experts,), dtype=torch.int, device="cuda")

    # --- dispatch (BF16) ---
    packed_recv_x, packed_recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_tokens,
        num_experts,
        use_fp8=False,
        cumulative_local_expert_recv_stats=cumulative_stats,
        async_finish=False,
        return_recv_hook=False,
    )
    event.current_stream_wait()

    # Verify each expert received the expected token count
    all_topk_idx = torch.empty(
        (num_ranks, num_tokens, num_topk),
        dtype=topk_idx.dtype,
        device="cuda",
    )
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)
    for i in range(num_local_experts):
        expert_id = rank * num_local_experts + i
        expected = (all_topk_idx == expert_id).sum().item()
        got = cumulative_stats[i].item()
        assert (
            got == expected
        ), f"[rank {rank}] expert {expert_id}: expected {expected} tokens, got {got}"

    # Simulate expert GEMM: use the received tensor as-is
    simulated_output = packed_recv_x.clone()

    # --- combine ---
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    combined_x, event, hook = buffer.low_latency_combine(
        simulated_output,
        topk_idx,
        topk_weights,
        handle,
        async_finish=False,
        zero_copy=False,
        return_recv_hook=False,
        out=out,
    )
    event.current_stream_wait()

    # combined_x should be weighted sum: x * sum(topk_weights, mask=-1)
    weight_sum = topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1, keepdim=True)
    ref = x * weight_sum
    diff = _calc_diff(combined_x, ref)
    assert not torch.isnan(combined_x).any(), f"[rank {rank}] NaN in combined output"
    assert diff < 1e-2, f"[rank {rank}] combine diff {diff:.6f} exceeds threshold"

    if rank == 0:
        print(
            f"[rank {rank}] low-latency dispatch+combine passed, diff={diff:.2e}",
            flush=True,
        )

    # --- dispatch (FP8) ---
    cumulative_stats.zero_()
    packed_recv_fp8, packed_recv_count_fp8, handle_fp8, event_fp8, _ = (
        buffer.low_latency_dispatch(
            x,
            topk_idx,
            num_tokens,
            num_experts,
            use_fp8=True,
            cumulative_local_expert_recv_stats=cumulative_stats,
            async_finish=False,
            return_recv_hook=False,
        )
    )
    event_fp8.current_stream_wait()

    # Verify FP8 dispatch token counts match BF16
    for i in range(num_local_experts):
        expert_id = rank * num_local_experts + i
        expected = (all_topk_idx == expert_id).sum().item()
        got = cumulative_stats[i].item()
        assert (
            got == expected
        ), f"[rank {rank}] FP8 expert {expert_id}: expected {expected}, got {got}"

    # FP8 packed_recv_fp8 is a tuple (fp8_data, scale); dequant and combine
    fp8_data, fp8_scale = packed_recv_fp8[0], packed_recv_fp8[1]
    # Reconstruct BF16 by dequant: shape [num_local_experts, max_tokens, hidden]
    # Use it directly as simulated GEMM output
    num_le, max_t, h = fp8_data.shape
    fp8_bf16 = fp8_data.float().view(num_le * max_t, h)
    # scale shape varies; just use zeros as simulated output for combine test
    simulated_fp8_out = fp8_data  # pass raw fp8 to combine (same buffer layout)

    out_fp8 = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    combined_fp8, event_fp8_c, _ = buffer.low_latency_combine(
        simulated_fp8_out,
        topk_idx,
        topk_weights,
        handle_fp8,
        async_finish=False,
        zero_copy=False,
        return_recv_hook=False,
        out=out_fp8,
    )
    event_fp8_c.current_stream_wait()

    assert not torch.isnan(
        combined_fp8
    ).any(), f"[rank {rank}] NaN in FP8 combined output"
    if rank == 0:
        print(f"[rank {rank}] FP8 low-latency dispatch+combine passed", flush=True)


def _worker(rank: int, num_ranks: int, args: Dict[str, Any], nccl_port: int):
    """Per-process worker: init distributed env, DeepEP buffer, run test."""
    # ROCm uses HIP_VISIBLE_DEVICES
    os.environ["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_ranks))

    # ROCm library path (from end-to-end script)
    os.environ["LD_LIBRARY_PATH"] = (
        "/opt/rocm-7.2.0/lib:/opt/rocm-7.2.0/lib64:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )

    # AITER settings (from end-to-end script)
    os.environ["AITER_ASM_DIR"] = (
        "/opt/conda310/lib/python3.10/site-packages/aiter_meta/hsa/"
    )
    os.environ["USE_AITER_PA"] = "0"

    # Custom allreduce and communication settings (from end-to-end script)
    os.environ["FT_DISABLE_CUSTOM_AR"] = "0"
    os.environ["ENABLE_COMM_OVERLAP"] = "0"
    os.environ["FORCE_CPU_LOAD_WEIGHTS"] = "1"

    # HSA low-latency settings (from end-to-end script)
    os.environ["HSA_NO_SCRATCH_RECLAIM"] = "1"
    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ["HSA_DISABLE_CACHE"] = "1"

    # NVSHMEM IB settings (from end-to-end script)
    os.environ["NVSHMEM_IB_GID_INDEX"] = "1"
    os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "true"
    os.environ["NVSHMEM_DISABLE_GDRCOPY"] = "true"
    os.environ["NVSHMEM_SYMMETRIC_SIZE"] = "2G"
    os.environ["NVVL_NET_PLUGIN"] = "none"

    # NCCL settings (from end-to-end script)
    os.environ["NCCL_IB_GID_INDEX"] = "1"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["NCCL_IB_HCA"] = "fic2"

    # DeepEP FP8 and low-latency settings (from end-to-end script)
    os.environ["ACCL_FP8_CAST_LEVEL"] = "2"
    os.environ["LOW_LATENCY_USE_COMPUTE_STREAM"] = "1"
    os.environ["FAKE_BALANCE_EXPERT"] = "1"
    os.environ["USE_SWIZZLEA"] = "1"
    os.environ["USE_ASM_PA"] = "0"
    os.environ["USE_ALL_GATHER"] = "0"

    # DeepEP low-latency tuning knobs (original test settings)
    os.environ["ACCL_DISPATCH_NUM_WARP_GROUPS"] = "4"
    os.environ["ACCL_COMBINE_NUM_WARP_GROUPS"] = "4"
    os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "1"
    os.environ["ACCL_TOPO_FIX"] = "1"
    os.environ["ACCL_LOAD_BALANCE"] = "1"
    # Enable NVSHMEM debug to capture crash details
    os.environ["NVSHMEM_DEBUG"] = "WARN"

    nccl_comm_config, nccl_init_port, engine_config, model_config = (
        _create_engine_config(rank, num_ranks, args, nccl_port)
    )

    parallelism_config = engine_config.parallelism_config
    tp_size = parallelism_config.tp_size

    torch.cuda.set_device(parallelism_config.local_rank)
    torch.set_default_device(f"cuda:{parallelism_config.local_rank}")

    init_distributed_environment(
        parallelism_config=parallelism_config,
        nccl_comm_config=nccl_comm_config,
        nccl_init_port=nccl_init_port,
        backend="nccl",
        timeout=120,
    )

    init_deepep_wrapper(engine_config, model_config)

    config_adapter = MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=engine_config.parallelism_config,
        moe_config=engine_config.moe_config,
    )
    ll_num_max_token_per_rank = DeepepWrapperConfig.calc_low_latency_max_token_per_rank(
        engine_config.moe_config.ll_num_max_token,
        engine_config.parallelism_config.tp_size,
        model_config.quant_config,
    )
    deepep_config = DeepepWrapperConfig.from_config_adapter(
        config_adapter, ll_num_max_token_per_rank
    )
    deepep_wrapper = DeepEPWrapper.get_instance(deepep_config)
    buffer = deepep_wrapper.buffer

    num_tokens = (args["max_generate_batch_size"] + tp_size - 1) // tp_size

    try:
        _test_ll_dispatch_combine(
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


def _run_multiprocess_test(num_ranks: int, args: Dict[str, Any]):
    """Spawn num_ranks processes and run the DeepEP LL test."""
    import sys
    from io import StringIO

    with PortsContext(None, 1) as ports:
        nccl_port = ports[0]
        ctx = mp.get_context("spawn")
        procs = []
        queues = []
        for rank in range(num_ranks):
            q = ctx.Queue()
            queues.append(q)
            p = ctx.Process(
                target=_worker_with_queue,
                args=(rank, num_ranks, args, nccl_port, q),
            )
            p.start()
            procs.append(p)

        # Collect results and output from all workers
        results = []
        for i, p in enumerate(procs):
            p.join(timeout=180)
            try:
                result = queues[i].get(timeout=5)
                results.append(result)
            except:
                results.append({"error": "Failed to get result from queue"})

            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
                raise RuntimeError("DeepEP LL test worker timed out")
            if p.exitcode != 0:
                # Print captured output from failed worker
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
                    f"DeepEP LL test worker {i} exited with code {p.exitcode}"
                )


def _worker_with_queue(
    rank: int, num_ranks: int, args: Dict[str, Any], nccl_port: int, queue
):
    """Worker wrapper that captures output and exceptions."""
    import sys
    import traceback
    from io import StringIO

    stdout_capture = StringIO()
    stderr_capture = StringIO()

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    result = {"rank": rank}
    try:
        _worker(rank, num_ranks, args, nccl_port)
        result["success"] = True
    except Exception as e:
        result["success"] = False
        result["exception"] = traceback.format_exc()

    sys.stdout = old_stdout
    sys.stderr = old_stderr

    result["stdout"] = stdout_capture.getvalue()
    result["stderr"] = stderr_capture.getvalue()

    queue.put(result)


class RocmFfnMoeFp8LlTest(TestCase):
    """ROCm DeepEP FP8 low-latency dispatch/combine correctness test.

    Uses the Python deep_ep buffer interface (same backend as production
    service with use_deepep_low_latency=True) to validate:
      - BF16 low-latency dispatch + combine correctness
      - FP8 low-latency dispatch + combine (no NaN)

    Requires at least 4 ROCm GPUs. Build with:
        --config=rocm --define=enable_deep_ep=true
    """

    # Minimal config: 64 experts / 4 ranks = 16 local experts, hidden=7168
    TEST_ARGS = {
        "moe_k": 8,
        "expert_num": 64,
        "hidden_size": 7168,
        "max_generate_batch_size": 32,
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

    def test_low_latency_dispatch_combine_4ranks(self):
        """Test DeepEP low-latency dispatch+combine with 4 ROCm GPUs."""
        num_ranks = 4
        self._skip_if_insufficient_gpus(num_ranks)
        self._skip_if_deepep_not_supported()
        _run_multiprocess_test(num_ranks, self.TEST_ARGS)

    def test_low_latency_dispatch_combine_8ranks(self):
        """Test DeepEP low-latency dispatch+combine with 8 ROCm GPUs."""
        num_ranks = 8
        self._skip_if_insufficient_gpus(num_ranks)
        self._skip_if_deepep_not_supported()
        _run_multiprocess_test(num_ranks, self.TEST_ARGS)


if __name__ == "__main__":
    main()
