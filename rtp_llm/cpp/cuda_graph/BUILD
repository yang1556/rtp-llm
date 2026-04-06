load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "cuda_graph_base",
    hdrs = [
        "cuda_graph_base.h",
    ],
    deps = [
        "//rtp_llm/models_py/bindings:op_defs",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cuda_graph_srcs",
    srcs = [
        "cuda_graph_device_shims.cc",
        "cuda_graph_utils.cc",
        "cuda_graph_runner.cc",
        "cuda_graph_prefill.cc",
        "cuda_graph_decode.cc",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cuda_graph_hdrs",
    srcs = [
        "cuda_graph_base.h",
        "cuda_graph_device_shims.h",
        "cuda_graph_utils.h",
        "cuda_graph_runner.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_graph_hdrs_lib",
    hdrs = [
        "cuda_graph_base.h",
        "cuda_graph_device_shims.h",
        "cuda_graph_utils.h",
        "cuda_graph_runner.h",
    ],
    deps = [
        "//rtp_llm/models_py/bindings:op_defs",
    ],
    visibility = ["//visibility:public"],
)
