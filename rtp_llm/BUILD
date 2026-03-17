package(default_visibility = ["//visibility:public"])
# make different torch for different device when in compiling
load("//bazel:defs.bzl", "upload_pkg", "copy_target_to")
load("//bazel:arch_select.bzl", "requirement", "internal_deps", "jit_deps", "triton_deps")
load("//open_source/bazel:arch_select.bzl", "platform_deps")
load("//bazel:bundle.bzl", "bundle_files", "bundle_tar")

load("@release_version//:defs.bzl", "RELEASE_VERSION")

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_default_copts",
    _if_rocm = "if_rocm",
)

if_rocm = _if_rocm

tensorrt = [
    "tensorrt",
    "tensorrt-cu12-bindings",
    "tensorrt-cu12-libs",
]

flashinfer = [
    "apache-tvm-ffi",
    "flashinfer-python",
]

flashinfer_with_cache = [
    "apache-tvm-ffi",
    "flashinfer-python",
    "flashinfer-cubin",
    "nvidia-cutlass-dsl",
    "flashinfer-jit-cache"
]

flashmla = [
    "fast-hadamard-transform",
    "flash-mla",
    "tilelang",
]

deep = [
    "deep_gemm",
    "deep_ep"
]

rtp_kernel = [
    "rtp-kernel"
]

flashattn = [
    "flash_attn",
    "flash-attn-3"
]

xft_dep = select({
    "@//:using_arm": [],
    "//:xft_use_icx": [
        # "xfastertransformer_devel_icx",
    ],
    "//conditions:default": [
        # "xfastertransformer_devel",
    ],
})

arch_dep = select({
    "@//:using_arm": [],
    "@//:using_cuda12_arm": [],
    "//conditions:default": [":decord"],
})

arch_with_version_dep = select({
    "@//:using_arm": [],
    "@//:using_cuda12_arm": [],
    "//conditions:default": ["decord==0.6.0"],
})

requirement([
    "sentencepiece",
    "transformers",
    "pynvml",
    "tiktoken",
    "protobuf",
    "grpcio-tools",
    "setuptools",
    "Pillow",
    "pillow-heif",
    "pillow-avif-plugin",
    "lru-dict",
    "cpm_kernels",
    "uvicorn",
    "fastapi",
    "psutil",
    "pyarrow",
    "pyodps",
    "thrift",
    "torch",
    "torchvision",
    "numpy",
    "safetensors",
    "einops",
    "prettytable",
    "timm",
    "aiohttp",
    "onnx",
    "sentence-transformers",
    "orjson",
    "xfastertransformer_devel",
    "xfastertransformer_devel_icx",
    "decord",
    # add qwen agent package
    "pydantic",
    "json5",
    "dashscope",
    "jieba",
    "openai",
    "oss2",
    "pyOpenSSL",
    "nest_asyncio",
    "blobfile",
    "partial_json_parser",
    "librosa",
    "matplotlib", # required by qwen vl tokenizer
    "pyrsmi",
    "amdsmi",
    "fast-safetensors",
    "setproctitle",
    "bitsandbytes",
    "portalocker",
    "concurrent_log_handler",
    "aiter",
    "fastsafetensors",
    "pybind11_stubgen",
] + tensorrt + deep + flashinfer_with_cache + flashmla + rtp_kernel + flashattn)

filegroup(
    name = "cutlass_config",
    srcs = glob(["utils/gemm_utils/luts/*"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "empty_target",
    srcs = [],
    deps = [],
    visibility = ["//visibility:public"],
)

py_library(
    name = "utils",
    srcs = glob([
        "utils/**/*.py",
    ]),
    data = [":cutlass_config"],
    deps = [
        ":torch",
        ":safetensors",
        #":decord",
        ":lru-dict",
        ":cpm_kernels",
        ":prettytable",
        ":psutil",
        "//rtp_llm/aios/kmonitor:kmonitor_py",
    ] + arch_dep + internal_deps()
)

py_library(
    name = "eplb",
    srcs = glob([
        "eplb/*.py",
    ]),
    deps = [
        ":utils",
    ],
)

py_library(
    name = "gang",
    srcs = glob([
        "gang/*.py",
    ])
)

py_library(
    name = "_ft_pickler",
    srcs = ["_ft_pickler.py"],
)

py_library(
    name = "ops",
    srcs = glob([
        "ops/**/*.py",
    ]),
    deps = [
        ":torch",
        ":utils",
    ],
)

py_library(
    name = "pipeline",
    srcs = glob([
        "pipeline/**/*.py",
    ]),
)

py_library(
    name = "device",
    srcs = glob([
        "device/**/*.py",
    ]),
    deps = [
        ":torch",
        ":psutil",
        ":utils",
        ":ops",
    ],
)

py_library(
    name = "models",
    srcs = glob([
        "models/*.py",
        "models/**/*.py",
    ], exclude=["models/test/*.py"]),
    deps = [
        ":sentencepiece",
        ":sentence-transformers",
        ":transformers",
        ":prettytable",
        ":pynvml",
        ":tiktoken",
        ":protobuf",
        ":Pillow",
        ":pillow-heif",
        ":pillow-avif-plugin",
        ":torch",
        ":torchvision",
        ":pyOpenSSL",
        ":einops",
        ":utils",
        ":ops",
        ":timm",
        ":onnx",
        #":decord",
        ":nest_asyncio",
        ":matplotlib",
        "//rtp_llm/model_loader:loader",
        "//rtp_llm/models_py:models",
    ] + arch_dep + select({
        "@//:using_cuda": deep,
        "//conditions:default": []
    }) + select({
        "@//:using_cuda12_9_x86": tensorrt,
        "@//:cuda_pre_12_9": tensorrt,
        "//conditions:default": []
    }) + select({
        "@//:using_cuda12_9_x86": flashinfer_with_cache + flashmla,
        "@//:using_cuda12_arm": flashinfer_with_cache,
        "@//:cuda_pre_12_9": flashinfer,
        "//conditions:default": []
    }) + select({
        "@//:using_cuda12_9_x86": rtp_kernel,
        "@//:using_cuda12_arm": rtp_kernel,
        "@//:cuda_pre_12_9": [],
        "//conditions:default": []
    }) + select({
        "@//:using_cuda12_9_x86": flashattn,
        "@//:using_cuda12_arm": flashattn,
        "@//:cuda_pre_12_9": [],
        "//conditions:default": []
    }) + select({
        "@//:using_arm": [],
        "@//:using_cuda12_arm": [],
        # "//:xft_use_icx": [
        #     "xfastertransformer_devel_icx",
        # ],
        "//conditions:default": [
            "xfastertransformer_devel",
        ],
    }) + select({
        "@//:using_arm": [],
	    "@//:using_rocm": ["pyrsmi", "amdsmi"],
        "@//:using_cpu": [],
        "@//:using_cuda12_arm": [],
        "//conditions:default": []
    }) + triton_deps(["triton"]) + internal_deps(),
)

py_library(
    name = "vipserver",
    srcs = glob([
        "vipserver/*.py",
    ]),
)

filegroup(
    name = "alog_config",
    srcs = ["config/alog.conf"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "release_version",
    srcs = ["release_version.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "config",
    srcs = glob([
        "config/*.py",
        "config/**/*.py",
    ]),
    deps = [
        "//rtp_llm/distribute:distribute"
    ],
    data = [":alog_config"]
)

py_library(
    name = "config_ops",
    deps = [
        ":config",
        ":ops"
    ],
)

py_library(
    name = "structure",
    srcs = glob([
        "structure/*.py",
    ])
)

py_library(
    name = "cli",
    deps = [
        ":config",
        ":release_version"
    ],
    srcs = glob([
        "cli/*.py",
    ])
)

filegroup(
    name = "async_model_files",
    srcs = glob(["async_decoder_engine/**/*.py"]),
)

py_library(
    name = "async_model",
    srcs = [
        ":async_model_files"
    ],
    deps = [
        ":utils",
        ":ops",
        ":config",
        ":structure",
        "//rtp_llm/cpp/model_rpc:model_rpc_client",
    ],
)

py_library(
    name = "openai_api",
    srcs = glob([
        "openai/*.py",
        "openai/**/*.py",
    ]),
    deps = [
        ":utils",
        ":ops",
        ":config",
        ":structure",
    ],
    data = [
        "openai/renderers/qwen_agent/utils/qwen.tiktoken"
    ],
)

py_library(
    name = "frontend",
    srcs = glob([
        "frontend/*.py",
        "frontend/**/*.py",
    ]),
    deps = [
        ":openai_api"
    ]
)

py_library(
    name = "sdk",
    srcs = [
        '__init__.py',
        'model_factory.py',
        'start_server.py',
        'start_frontend_server.py',
        'start_backend_server.py',
        '_ft_pickler.py',
        'model_factory_register.py',
    ],
    deps = [
        "//rtp_llm/server:server",
        ":uvicorn",
        ":fastapi",
        ":psutil",
        ":oss2",
        ":orjson",
        # add qwen agent package
        ":pydantic",
        ":json5",
        ":dashscope",
        ":jieba",
        ":openai",
        ":librosa",
        ":setproctitle",
        ":portalocker",
        ":concurrent_log_handler",
        ":blobfile",
        ":partial_json_parser",
    ],
    data = jit_deps(),
    imports = ["."],
)

py_library(
    name = "kserve_server",
    srcs = [
        'kserve_server.py',
    ],
    deps = [
        ":sdk",
        ":models",
    ],
    imports = ["."],
)

py_library(
    name = "plugins",
    srcs = glob([
        "plugins/*.py",
    ])
)

py_library(
    name = "tokenizer",
    srcs = glob([
        "tokenizer/*.py",
    ])
)

py_library(
    name = "embedding",
    srcs = glob([
        "embedding/*.py",
    ])
)

py_library(
    name = "lora",
    srcs = glob([
        "lora/*.py",
    ])
)

py_library(
    name = "rtp_llm_frontend_lib",
    deps = [
        ":utils",
        ":eplb",
        ":ops",
        ":pipeline",
        ":device",
        ":cli",
        ":config",
        ":structure",
        "//rtp_llm/server:server",
        ":plugins",
        "//rtp_llm/cpp/model_rpc:model_rpc_client",
        ":openai_api",
        ":lora",
        ":sdk",
        ":frontend",
        "//rtp_llm/tools:model_assistant",
        "//rtp_llm/distribute:distribute",
        ":tokenizer",
        "//rtp_llm/aios/kmonitor:kmonitor_py",
        "//rtp_llm/metrics:metrics",
        "//rtp_llm/access_logger:access_logger",
    ],
    data = [
        "//rtp_llm/libs:frontend_libs"
    ]
)

py_library(
    name = "rtp_llm_lib",
    deps = [
        ":rtp_llm_frontend_lib",
    ] + select({
        "@//:using_rocm": [
            ":models",
            ":async_model",
            ":embedding",
            "//rtp_llm/tools/convert:convert",
            ":aiter",
        ],
        "//conditions:default": [
            ":models",
            ":async_model",
            ":embedding",
            "//rtp_llm/tools/convert:convert",
        ],
            }) + select({
                "//conditions:default": [],
            }),
    data = [
        "//rtp_llm/libs:libs"
    ]
)

py_library(
    name = "testlib",
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//:th_transformer",
        "//:rtp_compute_ops",
    ],
    deps = [
        ":rtp_llm_lib",
        "//rtp_llm/test/utils:test_util",
        "//deps:extension_lib",
        ":aiohttp",
    ]
)
