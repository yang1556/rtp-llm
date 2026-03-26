import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List

import torch
from pydantic import BaseModel
from tqdm import tqdm

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import (
    MetricState,
    TableType,
    create_metrics_table,
)
from rtp_llm.test.perf_test.test_util import create_query, write_odps
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.utils.util import check_with_info


class RunningConfig(BaseModel):
    batch_size_list: List[int]
    input_len_list: List[int]
    input_query_dict: Dict[int, str]
    env: Dict[str, Any]
    result_dir: str
    decode_test_length: int
    is_speculative: bool
    propose_step: int = 0
    generate_config: Dict[str, Any]


def write_odps_wrapper(
    device_name: str,
    model_name: str,
    model_size: float,
    prec: str,
    dp_size: int,
    tp_size: int,
    metrics_list: List[MetricState],
):
    table_name = os.environ.get("ODPS_TABLE", "perf_test_2")
    fields = [
        "model",
        "size",
        "weight_type",
        "device",
        "framework",
        "commit",
        "batch_size",
        "seq_len",
        "context_time",
        "generate_time",
        "tp_size",
        "dp_size",
    ]
    records: List[Any] = []
    for metrics_item in metrics_list:
        metrics = metrics_item.metrics
        batch_size = metrics_item.batch_size
        input_len = metrics_item.input_len
        if metrics.success_requests != metrics.total_requests:
            logging.warning(
                f"batch {batch_size} seq {input_len} not all success, {metrics.success_requests}/{metrics.total_requests}"
            )
            continue
        records.append(
            [
                model_name,
                model_size,
                prec,
                device_name,
                "",
                "",
                batch_size,
                input_len,
                metrics.avg_prefill_time,
                metrics.avg_decode_time,
                tp_size,
                dp_size,
            ]
        )
    write_odps(table_name, records, fields)


def run_single(
    port: int,
    dp_size: int,
    tp_size: int,
    batch_size_list: List[int],
    input_len_list: List[int],
    input_query_dict: Dict[int, str],
    is_decode: bool = True,
    dump_json_path: str = ".",
    decode_test_length: int = 10,
    is_speculative: bool = False,
    propose_step: int = 0,
    generate_config: Dict[str, Any] = {},
) -> List[MetricState]:
    title_prefix = f"Speculative(step={propose_step}) " if is_speculative else ""
    title = "Decode Result" if is_decode else "Prefill Result"
    title = f"{title_prefix}{title}"
    batch_size_list = [1] if not is_decode else batch_size_list
    base_port = port
    logging.info(
        f"in warmup, base_port: {base_port}, dp_size: {dp_size}, tp_size: {tp_size}, batch_size: {1 * dp_size}, input_len: {input_len_list[0]}"
    )
    _ = BatchPerfImpl(
        base_port,
        dp_size,
        tp_size,
        1 * dp_size,
        input_len_list[0],
        input_query_dict[input_len_list[0]],
        is_decode,
        1000,
        decode_test_length,
        False,
        generate_config,
    ).run()
    logging.info(f"start to run perf test")
    metrics_list: List[MetricState] = []

    total_tests = len(batch_size_list) * len(input_len_list)

    with tqdm(total=total_tests, desc=f"Running {title}", unit="test") as pbar:
        for batch_size in batch_size_list:
            for input_len in input_len_list:
                # 更新进度条描述
                pbar.set_description(
                    f"Running {title} - batch_size: {batch_size}, input_len: {input_len}"
                )

                metric = BatchPerfImpl(
                    base_port,
                    dp_size,
                    tp_size,
                    batch_size * dp_size,
                    input_len,
                    input_query_dict[input_len],
                    is_decode,
                    500,
                    decode_test_length,
                    True,
                    generate_config,
                ).run()
                metrics_list.append(MetricState(input_len, batch_size, metric))

                # 更新进度条
                pbar.update(1)

    metrics_table = create_metrics_table(
        TableType.Decode if is_decode else TableType.Prefill,
        metrics_list,
        dump_json_path,
        {"dp_size": dp_size, "tp_size": tp_size},
        title,
        generate_config,
    )
    logging.info("metrics_table: \n" + str(metrics_table))
    return metrics_list


def start_server(
    args: argparse.Namespace,
    model_env: Dict[str, Any],
    log_name: str,
):
    current_env = os.environ.copy()
    current_env.update(model_env)
    server = MagaServerManager(env_args=current_env, process_file_name=log_name)
    server.start_server(
        model_path=args.ckpt_path,
        model_type=args.model_type,
        tokenizer_path=args.tokenizer_path,
    )
    server.wait_sever_done()
    return server


def parse_args():
    parser = argparse.ArgumentParser(description="batch decode runner")
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--dp_size", type=int, required=True)
    parser.add_argument("--tp_size", type=int, required=True)
    parser.add_argument("--model_size", type=float, default=0)
    parser.add_argument("--batch_size", type=str, default="1,2,4,8,16")
    parser.add_argument("--input_len", type=str, default="128,1024,2048,4096,8192")
    parser.add_argument("--test_name", type=str, default="batch_decode_test")
    parser.add_argument("--prec", type=str, default="bf16")
    parser.add_argument("--ep_size", type=int, default=1)
    parser.add_argument("--disaggregate", type=int, default=0)
    # partial test, 0: test all, 1: test decode only, 2: test prefill only
    parser.add_argument(
        "--partial",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="partial test, 0: test all, 1: test decode only, 2: test prefill only",
    )
    parser.add_argument("--result_dir", type=str, default=".")
    parser.add_argument("--generate_config", type=str, default="{}")
    args = parser.parse_args()
    return args


def merge_state(
    decode_result: List[MetricState], prefill_result: List[MetricState]
) -> List[MetricState]:
    prefill_result_dict = {}
    for prefill_item in prefill_result:
        if prefill_item.metrics.success_requests == prefill_item.metrics.total_requests:
            prefill_result_dict[prefill_item.input_len] = (
                prefill_item.metrics.avg_prefill_time
            )
        else:
            prefill_result_dict[prefill_item.input_len] = -1
    for decode_item in decode_result:
        if decode_item.input_len in prefill_result_dict:
            decode_item.metrics.avg_prefill_time = prefill_result_dict[
                decode_item.input_len
            ]
        else:
            decode_item.metrics.avg_prefill_time = -1
    return decode_result


def run_normal_test(args: argparse.Namespace, running_config: RunningConfig):
    server = start_server(args, running_config.env, "process.log")
    decode_result = None
    prefill_result = None
    if args.partial == 0 or args.partial == 1:
        decode_result = run_single(
            server.port,
            args.dp_size,
            args.tp_size,
            running_config.batch_size_list,
            running_config.input_len_list,
            running_config.input_query_dict,
            True,
            dump_json_path=running_config.result_dir,
            decode_test_length=running_config.decode_test_length,
            is_speculative=running_config.is_speculative,
            propose_step=running_config.propose_step,
            generate_config=running_config.generate_config,
        )
    if args.partial == 0 or args.partial == 2:
        prefill_result = run_single(
            server.port,
            args.dp_size,
            args.tp_size,
            [1],
            running_config.input_len_list,
            running_config.input_query_dict,
            False,
            dump_json_path=running_config.result_dir,
            decode_test_length=running_config.decode_test_length,
            is_speculative=running_config.is_speculative,
            propose_step=running_config.propose_step,
            generate_config=running_config.generate_config,
        )
    server.stop_server()
    return decode_result, prefill_result


def run_disaggregate_test(args: argparse.Namespace, running_config: RunningConfig):
    assert args.partial == 0, "disaggregate test only support test all"
    decode_env = json.loads(os.environ.get("DECODE_CONFIG", "{}"))
    decode_env.update(running_config.env)
    decode_env["BATCH_DECODE_SCHEDULER_WARMUP_TYPE"] = "0"
    decode_server = start_server(args, decode_env, "decode.log")
    decode_result = run_single(
        decode_server.port,
        args.dp_size,
        args.tp_size,
        running_config.batch_size_list,
        running_config.input_len_list,
        running_config.input_query_dict,
        True,
        dump_json_path=running_config.result_dir,
        decode_test_length=running_config.decode_test_length,
        is_speculative=running_config.is_speculative,
        propose_step=running_config.propose_step,
        generate_config=running_config.generate_config,
    )
    decode_server.stop_server()
    prefill_env = json.loads(os.environ.get("PREFILL_CONFIG", "{}"))
    prefill_env.update(running_config.env)
    prefill_env["BATCH_DECODE_SCHEDULER_WARMUP_TYPE"] = "1"
    prefill_server = start_server(
        args,
        prefill_env,
        "prefill.log",
    )
    prefill_result = run_single(
        prefill_server.port,
        args.dp_size,
        args.tp_size,
        [1],
        running_config.input_len_list,
        running_config.input_query_dict,
        False,
        dump_json_path=running_config.result_dir,
        decode_test_length=running_config.decode_test_length,
        is_speculative=running_config.is_speculative,
        propose_step=running_config.propose_step,
        generate_config=running_config.generate_config,
    )
    prefill_server.stop_server()
    return decode_result, prefill_result


def create_test_env(
    max_len: int, max_concurrency: int, partial: int, tp_size: int, dp_size: int, ep_size: int = 1
):
    env = {
        "USE_BATCH_DECODE_SCHEDULER": "1",
        "FAKE_BALANCE_EXPERT": "1",
        "MAX_SEQ_LEN": str(max_len + 20),
        "CONCURRENCY_LIMIT": str(max_concurrency),
        "TORCH_CUDA_PROFILER_DIR": os.environ.get(
            "TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()
        ),
        "BATCH_DECODE_SCHEDULER_WARMUP_TYPE": (
            "0" if (partial == 0 or partial == 1) else "1"
        ),
        "TP_SIZE": str(tp_size),
        "DP_SIZE": str(dp_size),
        "EP_SIZE": str(ep_size),
        "WORLD_SIZE": str(tp_size * dp_size),
    }
    # 如果设置了 TORCH_PROFILE_OUTPUT_DIR，透传 profiler 相关环境变量到服务子进程
    # 优先使用用户指定的目录；如果指向 /tmp 下的路径，bazel 测试结束后会清理，
    # 建议改用 TEST_UNDECLARED_OUTPUTS_DIR（bazel 会把该目录内容保存到 testlogs/test.outputs/）
    profile_output_dir = os.environ.get("TORCH_PROFILE_OUTPUT_DIR")
    if profile_output_dir:
        # 如果用户指定的是 /tmp 路径，自动重定向到 TEST_UNDECLARED_OUTPUTS_DIR 子目录以防止被 bazel 清理
        undeclared_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if profile_output_dir.startswith("/tmp") and undeclared_outputs_dir:
            profile_output_dir = os.path.join(undeclared_outputs_dir, "torch_profile")
            logging.info(
                f"[profiler] TORCH_PROFILE_OUTPUT_DIR 指向 /tmp，已自动重定向到 "
                f"TEST_UNDECLARED_OUTPUTS_DIR: {profile_output_dir}"
            )
        env["TORCH_PROFILE_OUTPUT_DIR"] = profile_output_dir
        env["TORCH_PROFILE_WARMUP"] = os.environ.get("TORCH_PROFILE_WARMUP", "2")
        env["TORCH_PROFILE_ACTIVE"] = os.environ.get("TORCH_PROFILE_ACTIVE", "1")
        logging.info(f"[profiler] enabled, output dir: {profile_output_dir}")
    return env


def _is_gemm_kernel(name: str) -> bool:
    gemm_patterns = [r"Cijk_", r"gemm", r"cutlass", r"sgemm|dgemm|hgemm", r"ampere_.*gemm", r"volta_.*gemm"]
    name_lower = name.lower()
    return any(re.search(pattern, name_lower) for pattern in gemm_patterns)


def _is_attention_kernel(name: str) -> bool:
    attn_patterns = [r"pa_fwd", r"flash_attn", r"fmha", r"attention", r"aiter.*attn"]
    name_lower = name.lower()
    return any(re.search(pattern, name_lower) for pattern in attn_patterns)


def _analyze_trace(trace_path: str, top_n: int = 20) -> None:
    """解析 torch.profiler 生成的 Chrome Trace JSON，关联 GPU kernel 与 CPU 算子 shape 并打印报告。"""
    with open(trace_path, "r") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) and "traceEvents" in data else data

    cpu_ops_by_ext_id: Dict[int, List[Dict]] = defaultdict(list)
    kernels_by_ext_id: Dict[int, List[Dict]] = defaultdict(list)

    for event in events:
        if not isinstance(event, dict):
            continue
        args = event.get("args", {})
        ext_id = args.get("External id") or args.get("external id")
        if ext_id is None:
            continue
        cat = event.get("cat", "")
        if cat == "cpu_op":
            cpu_ops_by_ext_id[ext_id].append(event)
        elif cat == "kernel":
            kernels_by_ext_id[ext_id].append(event)

    # kernel_name -> list of (dur_us, shape_str, op_name)
    kernel_stats: Dict[str, List[tuple]] = defaultdict(list)

    for ext_id, kernels in kernels_by_ext_id.items():
        cpu_ops = cpu_ops_by_ext_id.get(ext_id, [])
        shape_str = "unknown"
        op_name = "unknown"
        if cpu_ops:
            cpu_op = cpu_ops[-1]
            op_name = cpu_op.get("name", "unknown")
            input_dims = cpu_op.get("args", {}).get("Input Dims", [])
            input_types = cpu_op.get("args", {}).get("Input type", [])
            shape_parts = []
            for i, dim in enumerate(input_dims):
                if isinstance(dim, list) and len(dim) > 0:
                    type_str = input_types[i] if i < len(input_types) else ""
                    shape_parts.append(f"arg{i}({type_str}):{dim}")
            shape_str = "  |  ".join(shape_parts) if shape_parts else "no_tensor_inputs"

        for kernel in kernels:
            kernel_stats[kernel.get("name", "unknown")].append(
                (kernel.get("dur", 0.0), shape_str, op_name)
            )

    sorted_kernels = sorted(
        kernel_stats.items(),
        key=lambda x: sum(record[0] for record in x[1]),
        reverse=True,
    )

    sep = "=" * 100
    logging.info(f"\n{sep}\n  Kernel Shape 分析报告  |  top={top_n}\n{sep}")

    for rank, (kernel_name, records) in enumerate(sorted_kernels[:top_n], 1):
        total_dur = sum(record[0] for record in records)
        avg_dur = total_dur / len(records)
        op_name = records[0][2]
        shape_groups: Dict[str, List[float]] = defaultdict(list)
        for dur, shape_str, _ in records:
            shape_groups[shape_str].append(dur)

        display_name = kernel_name[:80] + ("..." if len(kernel_name) > 80 else "")
        logging.info(
            f"[#{rank}] {display_name}\n"
            f"      PyTorch op : {op_name}\n"
            f"      调用次数   : {len(records)}\n"
            f"      总耗时     : {total_dur:.1f} μs\n"
            f"      平均耗时   : {avg_dur:.1f} μs\n"
            f"      Shape 分布 :"
        )
        for shape_str, durs in sorted(shape_groups.items(), key=lambda x: -sum(x[1])):
            logging.info(
                f"        [{len(durs)}次, avg={sum(durs)/len(durs):.1f}μs] {shape_str[:120]}"
            )

    total_kernel_time = sum(
        sum(record[0] for record in records) for records in kernel_stats.values()
    )
    logging.info(
        f"{sep}\n"
        f"  kernel 总种数: {len(kernel_stats)}  |  总耗时: {total_kernel_time:.1f} μs ({total_kernel_time/1000:.2f} ms)\n"
        f"{sep}"
    )


def _report_profile_results(profile_output_dir: str) -> None:
    """在 perf test 结束后自动查找并分析 trace 文件。"""
    trace_files = sorted(glob.glob(os.path.join(profile_output_dir, "trace_step*.json")))
    if not trace_files:
        logging.warning(f"[profiler] 未在 {profile_output_dir} 找到 trace 文件，可能 forward 次数不足")
        return
    for trace_path in trace_files:
        logging.info(f"[profiler] 分析 trace 文件: {trace_path}")
        _analyze_trace(trace_path)


def main():
    from rtp_llm.config.log_config import setup_logging

    setup_logging()
    print("current path: ", os.getcwd(), flush=True)

    args = parse_args()
    if args.partial not in [0, 1, 2]:
        raise ValueError("partial must be 0, 1, or 2")
    batch_size_list = [int(x) for x in args.batch_size.split(",")]
    input_len_list = [int(x) for x in args.input_len.split(",")]

    is_speculative = bool(os.environ.get("SP_TYPE", ""))
    decode_test_length = int(os.environ.get("DECODE_TEST_LENGTH", 10))
    propose_step = int(os.environ.get("GEN_NUM_PER_CIRCLE", 1))

    test_env = create_test_env(
        max(input_len_list) + decode_test_length,
        max(batch_size_list),
        args.partial,
        args.tp_size,
        args.dp_size,
        args.ep_size,
    )

    input_query_dict = create_query(
        args.model_type, args.tokenizer_path, input_len_list
    )

    running_config = RunningConfig(
        batch_size_list=batch_size_list,
        input_len_list=input_len_list,
        input_query_dict=input_query_dict,
        env=test_env,
        result_dir=args.result_dir,
        decode_test_length=decode_test_length,
        is_speculative=is_speculative,
        propose_step=propose_step,
        generate_config=json.loads(args.generate_config),
    )

    if args.disaggregate == 0:
        decode_result, prefill_result = run_normal_test(args, running_config)
    else:
        decode_result, prefill_result = run_disaggregate_test(args, running_config)
    profile_output_dir = os.environ.get("TORCH_PROFILE_OUTPUT_DIR")
    if profile_output_dir:
        # 计算实际写入路径（与 create_test_env 中的重定向逻辑保持一致）
        undeclared_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        actual_profile_dir = profile_output_dir
        if profile_output_dir.startswith('/tmp') and undeclared_outputs_dir:
            actual_profile_dir = os.path.join(undeclared_outputs_dir, 'torch_profile')
        _report_profile_results(actual_profile_dir)
        # 如果实际写入路径与用户指定路径不同，将文件 copy 到用户指定路径
        if actual_profile_dir != profile_output_dir:
            import shutil
            os.makedirs(profile_output_dir, exist_ok=True)
            
            # Copy all JSON files from TEST_UNDECLARED_OUTPUTS_DIR
            if undeclared_outputs_dir and os.path.exists(undeclared_outputs_dir):
                all_json_files = glob.glob(os.path.join(undeclared_outputs_dir, '**/*.json'), recursive=True)
                for json_file in all_json_files:
                    dest = os.path.join(profile_output_dir, os.path.basename(json_file))
                    shutil.copy2(json_file, dest)
                    logging.info('[profiler] JSON 文件已 copy 到用户指定路径: ' + dest)
            
            # Also copy trace files from torch_profile subdirectory
            trace_files = glob.glob(os.path.join(actual_profile_dir, 'trace_step*.json'))
            for trace_file in trace_files:
                dest = os.path.join(profile_output_dir, os.path.basename(trace_file))
                if not os.path.exists(dest):
                    shutil.copy2(trace_file, dest)
                    logging.info('[profiler] trace 文件已 copy 到用户指定路径: ' + dest)

    if args.partial != 0:
        return
    assert decode_result is not None and prefill_result is not None
    metrics_list = merge_state(decode_result, prefill_result)
    device_name = os.environ.get("DEVICE_NAME") or torch.cuda.get_device_name(0)
    # use decode parallel config as odps column for current
    write_odps_wrapper(
        device_name,
        args.model_type,
        args.model_size,
        args.prec,
        args.dp_size,
        args.tp_size,
        metrics_list,
    )
