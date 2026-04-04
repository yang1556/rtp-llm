"""Collect and download remote worker test outputs.

Implements the Bazel TEST_UNDECLARED_OUTPUTS_DIR pattern for the REAPI
pytest remote plugin.  On the remote worker, test code writes artifacts
(server logs, smoke_actual, OOM state, coredump summaries) to the
directory pointed to by TEST_UNDECLARED_OUTPUTS_DIR.  After the test
command finishes, a shell postscript tars that directory into a single
file declared as a Command.output_file.  The local plugin then downloads
the tar from CAS and extracts it.

This module is deliberately free of pytest imports so it can be unit-tested
independently.
"""
from __future__ import annotations

import logging
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import remote_execution_pb2 as re_pb2
    from .cas_client import CASClient
    from .executor import ExecutionResult

log = logging.getLogger(__name__)

# Bazel-compatible env var name — existing code already writes to this.
OUTPUTS_ENV_VAR = "TEST_UNDECLARED_OUTPUTS_DIR"

# Directory name on the remote worker (relative to sandbox CWD).
WORKER_OUTPUTS_DIR = "_rtp_test_outputs"

# Archive name declared in Command.output_files.
OUTPUTS_ARCHIVE = "_rtp_test_outputs.tar.gz"


def make_output_collection_env() -> dict:
    """Env vars to inject into the remote Command.

    NOTE: This is intentionally empty.  REAPI Command.environment_variables
    are passed as literal strings (no shell expansion), so ``$PWD`` would
    not be resolved.  Instead, the env var is set via the shell prefix
    returned by :func:`make_mkdir_prefix`.
    """
    return {}


def make_output_files_decl() -> list:
    """output_files entries to declare in the Command proto."""
    return [OUTPUTS_ARCHIVE]


def make_tar_postscript() -> str:
    """Shell snippet appended after pytest.  Runs regardless of exit code."""
    return (
        f'if [ -d "{WORKER_OUTPUTS_DIR}" ] && [ "$(ls -A {WORKER_OUTPUTS_DIR} 2>/dev/null)" ]; then '
        f"tar -czf {OUTPUTS_ARCHIVE} -C {WORKER_OUTPUTS_DIR} . 2>/dev/null; "
        f'echo ">>>RTP_OUTPUTS_ARCHIVE size=$(stat -c%s {OUTPUTS_ARCHIVE} 2>/dev/null || echo 0)"; '
        f"fi"
    )


def make_mkdir_prefix() -> str:
    """Shell snippet prepended before pytest to create the outputs dir.

    Also exports TEST_UNDECLARED_OUTPUTS_DIR so that existing test code
    (smoke comparers, server manager, OOM hooks) writes artifacts there.
    This must happen in the shell (not REAPI env vars) because we need
    ``$PWD`` to be resolved by bash.
    """
    return (
        f"mkdir -p {WORKER_OUTPUTS_DIR}; "
        f"export {OUTPUTS_ENV_VAR}=$PWD/{WORKER_OUTPUTS_DIR}; "
    )


def download_and_extract(
    cas: "CASClient",
    result: "ExecutionResult",
    local_dest: Path,
    *,
    max_bytes: int = 200 * 1024 * 1024,
) -> Optional[Path]:
    """Download the outputs archive from CAS and extract to *local_dest*.

    Returns the local directory on success, ``None`` if no archive was
    produced or the archive exceeds *max_bytes*.
    """
    digest = result.output_files.get(OUTPUTS_ARCHIVE)
    if digest is None:
        log.debug("No %s in ActionResult — remote produced no outputs", OUTPUTS_ARCHIVE)
        return None

    if digest.size_bytes > max_bytes:
        log.warning(
            "Remote outputs archive too large (%d MiB > %d MiB limit), skipping",
            digest.size_bytes // (1024 * 1024),
            max_bytes // (1024 * 1024),
        )
        return None

    log.info(
        "Downloading remote outputs (%.1f MiB) digest=%s",
        digest.size_bytes / (1024 * 1024),
        digest.hash[:12],
    )

    try:
        data = cas.download_blob(digest)
        if not data:
            log.warning("download_blob returned empty for %s", digest.hash[:12])
            return None
    except Exception as exc:
        log.warning("Failed to download remote outputs: %s", exc)
        return None

    local_dest.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
            tf.write(data)
            tmp_path = Path(tf.name)
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(local_dest)
        log.info("Remote outputs extracted to %s", local_dest)
        return local_dest
    except Exception as exc:
        log.warning("Failed to extract remote outputs: %s", exc)
        return None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
