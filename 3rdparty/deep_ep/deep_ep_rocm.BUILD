load("@//:def.bzl", "copts", "rocm_copts")
load("@//bazel:arch_select.bzl", "torch_deps")

genrule(
    name = "cpp_libraries",
    outs = [
        "libdeep_ep_rocm.so",
        "csrc/config_hip.hpp",
        "csrc/deep_ep_hip.hpp",
        "csrc/event_hip.hpp",
        "csrc/kernels/exception_hip.cuh",
        "csrc/kernels/configs_hip.cuh",
        "csrc/kernels/api_hip.cuh",
        "csrc/kernels/launch_hip.cuh"],
    cmd = """
        set -e
        DEEP_EP_PREFIX="$${DEEP_EP_ROCM_INSTALL:-/opt/conda310/lib/python3.10/site-packages}"
        if [ ! -d "$$DEEP_EP_PREFIX" ]; then
            echo "ERROR: deep_ep_rocm install dir not found: $$DEEP_EP_PREFIX"
            echo "  Install deep_ep whl (ROCm) or set DEEP_EP_ROCM_INSTALL to your site-packages dir."
            echo "  Example: export DEEP_EP_ROCM_INSTALL=$$(python -c \"import site; print(site.getsitepackages()[0])\")"
            exit 1
        fi
        INCLUDE_DIR="$$DEEP_EP_PREFIX/deep_ep/include"
        if [ ! -d "$$INCLUDE_DIR" ]; then
            echo "ERROR: deep_ep include dir not found: $$INCLUDE_DIR"
            echo "  Ensure deep_ep (ROCm) whl is installed: pip install deep_ep_rocm (or your whl path)"
            exit 1
        fi
        SO_PATH="$$(ls $$DEEP_EP_PREFIX/deep_ep_cpp.cpython-*.so 2>/dev/null | head -1)"
        if [ -z "$$SO_PATH" ] || [ ! -f "$$SO_PATH" ]; then
            echo "ERROR: deep_ep ROCm .so not found under $$DEEP_EP_PREFIX (expected deep_ep_cpp.cpython-*.so)"
            echo "  Install deep_ep (ROCm) whl for your Python version."
            exit 1
        fi
        cp -f "$$SO_PATH" $(location libdeep_ep_rocm.so)
        cp -f $$INCLUDE_DIR/config_hip.hpp $(location csrc/config_hip.hpp)
        cp -f $$INCLUDE_DIR/deep_ep_hip.hpp $(location csrc/deep_ep_hip.hpp)
        cp -f $$INCLUDE_DIR/event_hip.hpp $(location csrc/event_hip.hpp)
        cp -f $$INCLUDE_DIR/kernels/api_hip.cuh $(location csrc/kernels/api_hip.cuh)
        cp -f $$INCLUDE_DIR/kernels/configs_hip.cuh $(location csrc/kernels/configs_hip.cuh)
        cp -f $$INCLUDE_DIR/kernels/exception_hip.cuh $(location csrc/kernels/exception_hip.cuh)
        cp -f $$INCLUDE_DIR/kernels/launch_hip.cuh $(location csrc/kernels/launch_hip.cuh)
    """,
    visibility = ["//visibility:public"],
    tags = ["rocm", "local"],
)

cc_library(
    name = "deep_ep",
    srcs = ["libdeep_ep_rocm.so"],
    hdrs = [
        ":csrc/config_hip.hpp", 
        ":csrc/deep_ep_hip.hpp", 
        ":csrc/event_hip.hpp", 
        ":csrc/kernels/exception_hip.cuh",
        ":csrc/kernels/configs_hip.cuh",
        ":csrc/kernels/api_hip.cuh",
        ":csrc/kernels/launch_hip.cuh"],
    deps = [":cpp_libraries"] + torch_deps(),
    copts = [],
    linkopts = [],
    strip_include_prefix = "csrc/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)
