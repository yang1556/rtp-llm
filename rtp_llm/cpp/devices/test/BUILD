load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

cc_test(
    name = "cache_store_async_writer_test",
    srcs = ["CacheStoreAsyncWriterTest.cpp"],
    deps = [
        "//rtp_llm/cpp/devices:cache_store_async_writer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    copts = copts(),
)
