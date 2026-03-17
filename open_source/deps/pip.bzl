# Stub pip dependencies - provides empty requirement() functions
# for external repos that still reference @pip_*_torch repos.
# Actual Python deps are managed by pip/pyproject.toml.

def _stub_requirement(name):
    """Returns an empty label - Python deps are managed by pip, not Bazel."""
    return name

def _stub_pip_repo_impl(ctx):
    ctx.file("BUILD", "")
    ctx.file("requirements.bzl", """
def requirement(name):
    return name

def install_deps():
    pass
""")

_stub_pip_repo = repository_rule(
    implementation = _stub_pip_repo_impl,
)

def pip_deps():
    """Create stub pip repos for backward compatibility with external deps."""
    for name in [
        "pip_cpu_torch",
        "pip_arm_torch",
        "pip_ppu_torch",
        "pip_gpu_cuda12_torch",
        "pip_gpu_cuda12_9_torch",
        "pip_cuda12_arm_torch",
        "pip_gpu_rocm_torch",
    ]:
        _stub_pip_repo(name = name)
