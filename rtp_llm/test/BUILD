py_test(
    name = "generate_config_test",
    srcs = [
        "generate_config_test.py",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//rtp_llm:sdk"
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "slice_stop_word_list_test",
    srcs = [
        "slice_stop_word_list_test.py",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//rtp_llm:sdk"
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util",
    ],
    imports = ["aios/kmonitor/python_client"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "stop_words_test",
    srcs = [
        "stop_words_test.py",
    ],
    data = [],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util",
    ],
    imports = [],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "renderer_stop_words_test",
    srcs = [
        "renderer_stop_words_test.py",
    ],
    data = [],
    deps = [
        "//rtp_llm:testlib",
    ],
    imports = [],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "reasoning_tool_parsing_test",
    srcs = [
        "reasoning_tool_parsing_test.py",
    ],
    data = [],
    deps = [
        "//rtp_llm:testlib",
    ],
    imports = [],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "stop_words_pipeline_test",
    srcs = [
        "stop_words_pipeline_test.py",
    ],
    data = [],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util",
    ],
    imports = [],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "openai_response_test",
    srcs = [
        "openai_response_test.py",
    ],
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//rtp_llm/test/tokenizer_test:testdata",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "server_test",
    srcs = [
        "server_test.py",
    ],
    data = [
        "//rtp_llm:sdk",
    ],
    deps = [
        "//rtp_llm:uvicorn",
        "//rtp_llm:fastapi",
        "//rtp_llm:psutil",
        "//rtp_llm:tiktoken",
        "//rtp_llm:testlib",
    ],
    imports = ["aios/kmonitor/python_client"],
    tags = ["manual"],
    timeout = 'eternal',
)

py_test(
    name = "concurrency_limit_test",
    srcs = [
        "concurrency_limit_test.py"
    ],
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util",
    ],
    timeout = 'short',
    exec_properties = {'gpu':'A10'},
    shard_count = 1,
)

py_test(
    name = "template_test",
    srcs = [
        "template_test.py",
    ],
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
        # qwen agent
        "//rtp_llm:pydantic",
        "//rtp_llm:json5",
        "//rtp_llm:dashscope",
        "//rtp_llm:jieba",
        "//rtp_llm:openai",
        "//rtp_llm:Pillow",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//rtp_llm/test/tokenizer_test:testdata",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "pipeline_test",
    srcs = [
        "pipeline_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm:sdk",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "prefill_capture_seq_lens_test",
    srcs = [
        "prefill_capture_seq_lens_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "auto_configure_deepep_test",
    srcs = [
        "auto_configure_deepep_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm:sdk",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "rocm_ffn_moe_fp8_ll_test",
    srcs = [
        "rocm_ffn_moe_fp8_ll_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm:sdk",
    ],
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu': 'MI308X', 'gpu_count': '4'},
    env = {
        # ROCm library path
        "LD_LIBRARY_PATH": "/opt/rocm-7.2.0/lib:/opt/rocm-7.2.0/lib64:/opt/conda310/lib",
        # AITER settings
        "AITER_ASM_DIR": "/opt/conda310/lib/python3.10/site-packages/aiter_meta/hsa/",
        "USE_AITER_PA": "0",
        # Communication settings
        "FT_DISABLE_CUSTOM_AR": "0",
        "ENABLE_COMM_OVERLAP": "0",
        "FORCE_CPU_LOAD_WEIGHTS": "1",
        # HSA low-latency settings
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "HSA_ENABLE_SDMA": "0",
        "HSA_DISABLE_CACHE": "1",
        # NVSHMEM IB settings
        "NVSHMEM_IB_GID_INDEX": "1",
        "NVSHMEM_IB_ENABLE_IBGDA": "true",
        "NVSHMEM_DISABLE_GDRCOPY": "true",
        "NVSHMEM_SYMMETRIC_SIZE": "2G",
        "NVVL_NET_PLUGIN": "none",
        # NCCL settings
        "NCCL_IB_GID_INDEX": "1",
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_IB_HCA": "fic2",
        # DeepEP FP8 and low-latency settings
        "ACCL_FP8_CAST_LEVEL": "2",
        "LOW_LATENCY_USE_COMPUTE_STREAM": "1",
        "FAKE_BALANCE_EXPERT": "1",
        "USE_SWIZZLEA": "1",
        "USE_ASM_PA": "0",
        "USE_ALL_GATHER": "0",
        # DeepEP tuning knobs
        "ACCL_DISPATCH_NUM_WARP_GROUPS": "4",
        "ACCL_COMBINE_NUM_WARP_GROUPS": "4",
        "ACCL_LOW_LATENCY_OPTIMIZE": "1",
        "ACCL_TOPO_FIX": "1",
        "ACCL_LOAD_BALANCE": "1",
    },
)

py_test(
    name = "rocm_ffn_moe_fp8_normal_test",
    srcs = [
        "rocm_ffn_moe_fp8_normal_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm:sdk",
    ],
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu': 'MI308X', 'gpu_count': '4'},
    env = {
        # ROCm library path
        "LD_LIBRARY_PATH": "/opt/rocm-7.2.0/lib:/opt/rocm-7.2.0/lib64:/opt/conda310/lib",
        # AITER settings
        "AITER_ASM_DIR": "/opt/conda310/lib/python3.10/site-packages/aiter_meta/hsa/",
        "USE_AITER_PA": "0",
        # Communication settings
        "FT_DISABLE_CUSTOM_AR": "0",
        "ENABLE_COMM_OVERLAP": "0",
        "FORCE_CPU_LOAD_WEIGHTS": "1",
        # NVSHMEM settings (normal mode: IBGDA disabled)
        "NVSHMEM_DEBUG": "WARN",
        "NVSHMEM_IB_GID_INDEX": "1",
        "NVSHMEM_IB_ENABLE_IBGDA": "false",
        "NVSHMEM_DISABLE_GDRCOPY": "true",
        # NCCL settings
        "NCCL_NET_PLUGIN": "none",
        "NCCL_IB_GID_INDEX": "1",
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_HCA": "fic2",
        # Normal mode specific settings
        "NORMAL_USE_COMPUTE_STREAM": "1",
        "FAKE_BALANCE_EXPERT": "1",
        "USE_SWIZZLEA": "1",
        "USE_ASM_PA": "0",
        "USE_ALL_GATHER": "0",
        "EIC_NONFETCH_ADD_ENABLE": "1",
    },
)

py_test(
    name = "glm4_moe_detector_test",
    srcs = [
        "glm4_moe_detector_test.py",
    ],
    data = [],
    deps = [
        "//rtp_llm:testlib"
    ],
    imports = [],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name="glm47_moe_detector_test",
    srcs=[
        "glm47_moe_detector_test.py",
    ],
    data=[],
    deps=["//rtp_llm:testlib"],
    imports=[],
    exec_properties={"gpu": "A10"},
)

py_test(
    name="qwen3_coder_detector_test",
    srcs=[
        "qwen3_coder_detector_test.py",
    ],
    data=[],
    deps=["//rtp_llm:testlib"],
    imports=[],
    exec_properties={"gpu": "A10"},
)

py_test(
    name = "mtp_streaming_tool_parsing_test",
    srcs = [
        "mtp_streaming_tool_parsing_test.py",
    ],
    data = [],
    deps = [
        "//rtp_llm:testlib"
    ],
    imports = [],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name="token_normalizer_test",
    srcs=[
        "token_normalizer_test.py",
    ],
    data=[],
    deps=["//rtp_llm:testlib"],
    imports=[],
    exec_properties={"gpu": "A10"},
)

py_test(
    name="format_convert_helper_test",
    srcs=[
        "format_convert_helper_test.py",
    ],
    data=[],
    deps=["//rtp_llm:testlib"],
    imports=[],
    exec_properties={"gpu": "A10"},
)

py_test(
    name="incremental_streaming_test",
    srcs=[
        "incremental_streaming_test.py",
    ],
    data=[],
    deps=["//rtp_llm:testlib"],
    imports=[],
    exec_properties={"gpu": "A10"},
)

py_test(
    name="unknown_tool_name_test",
    srcs=[
        "unknown_tool_name_test.py",
    ],
    data=[],
    deps=["//rtp_llm:testlib"],
    imports=[],
    exec_properties={"gpu": "A10"},
)

py_test(
    name="empty_args_tool_call_test",
    srcs=[
        "empty_args_tool_call_test.py",
    ],
    data=[],
    deps=["//rtp_llm:testlib"],
    imports=[],
    exec_properties={"gpu": "A10"},
)