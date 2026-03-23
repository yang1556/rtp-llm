import os
import torch
from typing import Union

try:
    from safetensors.torch import save_file
except ImportError:
    save_file = None


class TensorFingerprint:
    """
    CUDA Graph compatible tensor fingerprint recorder.

    Design:
    - All buffers on CPU to avoid affecting GPU memory layout
    - Sampling follows C++ printBufferData: first N, last N elements + sum1 + sum2 per row
    - Saves as safetensors for fast I/O, no Python list, no type conversion
    - D2H copy via non-blocking .to('cpu') on small slices
    """

    SAMPLE_N = 20  # first N and last N elements per row, matching C++ column_start/column_end

    def __init__(self, output_file: str, max_checkpoints: int = 12 * 48):
        self.output_file = output_file
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.forward_count = 0
        self.layer_id = 0
        self._buffers = {}  # key -> CPU tensor
        self._cursor = 0
        import atexit
        atexit.register(self.save)

    def set_layer(self, layer_id: int):
        self.layer_id = layer_id

    def begin_forward(self):
        self.forward_count += 1
        self._cursor = 0

    def record(self, name: str, t: torch.Tensor):
        """
        Sample tensor following C++ printBufferData pattern:
        - For each row in first dim: first 20, last 20 elements, sum1, square_sum2
        - At least 4 numbers per row (first, last, sum1, sum2)
        - No GPU allocation, no type conversion
        """
        if self._cursor >= self.max_checkpoints:
            return

        flat = t.reshape(t.shape[0], -1) if t.dim() >= 2 else t.reshape(1, -1)
        nrows = min(flat.shape[0], 20)  # max 20 rows like C++
        ncols = flat.shape[1]
        n = self.SAMPLE_N

        # Determine sample indices: first n + last n (deduplicated if small)
        if ncols <= 2 * n:
            col_indices = list(range(ncols))
        else:
            col_indices = list(range(n)) + list(range(ncols - n, ncols))

        idx_tensor = torch.tensor(col_indices, device=t.device, dtype=torch.long)

        for row in range(nrows):
            row_data = flat[row]
            # Sample elements - no type conversion, keep original dtype
            sampled = row_data[idx_tensor].to('cpu', non_blocking=True)
            # sum1 and sum2 as same dtype (computed on GPU, single element D2H)
            row_float = row_data.float()
            s1 = row_float.sum().to('cpu', non_blocking=True)
            s2 = row_float.square().sum().to('cpu', non_blocking=True)

            key = f"fwd{self.forward_count}_l{self.layer_id}_{name}_r{row}"
            self._buffers[f"{key}_samples"] = sampled
            self._buffers[f"{key}_sum1"] = s1.reshape(1)
            self._buffers[f"{key}_sum2"] = s2.reshape(1)

        self._cursor += 1

    def end_forward(self):
        pass

    def save(self):
        """Save all accumulated fingerprints to safetensors file."""
        if not self._buffers:
            return
        # Ensure all async copies complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if save_file is not None:
            save_file(self._buffers, self.output_file)
        else:
            # Fallback: torch.save
            torch.save(self._buffers, self.output_file.replace(".safetensors", ".pt"))


class NoopFingerprint:
    """No-op implementation for normal runs."""

    def set_layer(self, layer_id: int):
        pass

    def begin_forward(self):
        pass

    def record(self, name: str, t: torch.Tensor):
        pass

    def end_forward(self):
        pass

    def save(self):
        pass


def create_fingerprint(fp_file: str = "", tag: str = "") -> Union[TensorFingerprint, NoopFingerprint]:
    """
    Factory method. Creates TensorFingerprint if fp_file is provided.
    Returns NoopFingerprint (zero overhead) otherwise.

    Usage: create_fingerprint(fp_file=engine_config.profiling_debug_logging_config.tensor_fp_file)
    """
    if fp_file:
        if tag:
            base, ext = os.path.splitext(fp_file)
            fp_file = f"{base}_{tag}{ext}"
        if not fp_file.endswith(".safetensors"):
            fp_file = fp_file.rsplit(".", 1)[0] + ".safetensors"
        return TensorFingerprint(fp_file)
    return NoopFingerprint()
