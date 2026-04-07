# ATREX: Pure Python + Triton MoE kernels for deterministic inference.
# No C++ compilation needed — just expose the Python package.

py_library(
    name = "atrex",
    srcs = glob(["python/atrex/**/*.py", "src/**/*.py"]),
    data = glob(["python/atrex/**/*", "src/**/*"], exclude = ["**/*.pyc", "**/__pycache__/**"]),
    imports = ["python"],
    visibility = ["//visibility:public"],
    tags = ["rocm"],
)
