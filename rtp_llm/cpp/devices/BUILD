load("//:def.bzl", "copts")

cc_library(
    name = "cache_store_async_writer",
    hdrs = ["CacheStoreAsyncWriter.h"],
    srcs = ["CacheStoreAsyncWriter.cc"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:lock_free",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)
