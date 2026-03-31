load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "memory_connector",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob(
        ["*.cc"],
        exclude = ["SplitKvCacheCopyCuda.cc"],
    ) + select({
        "@//:using_cuda": ["SplitKvCacheCopyCuda.cc"],
        "//conditions:default": [],
    }),
    deps = [
        "//rtp_llm/cpp/cache/connector:connector_base",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/cache:kv_cache_allocator",
        "//rtp_llm/cpp/utils:lru_cache",
        "//rtp_llm/cpp/utils:profiling_scope",
        "@havenask//aios/autil:json",
        "@havenask//aios/autil:lock_free",
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/kernels:sm_copy_kernel",
        ],
        "//conditions:default": [],
    }) + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)
