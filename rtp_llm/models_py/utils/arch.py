from typing import Tuple

import torch

from rtp_llm.ops.compute_ops import DeviceType, get_exec_ctx


def is_cuda():
    device_type = get_exec_ctx().get_device_type()
    # PPU reuses CUDA kernels via its CUDA SDK shim layer
    return device_type in (DeviceType.Cuda, DeviceType.Ppu)


def is_hip():
    device_type = get_exec_ctx().get_device_type()
    if device_type == DeviceType.ROCm:
        return True
    else:
        return False


def get_num_device_sms() -> int:
    if is_cuda():
        assert torch.cuda.is_available()
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        return props.multi_processor_count
    else:
        raise NotImplementedError("Only cuda is supported get_num_device_sms yet")


def get_sm(device_id: int = 0) -> Tuple[int, int]:
    try:
        major, minor = torch.cuda.get_device_capability(device_id)
        return major, minor
    except (RuntimeError, AssertionError):
        # No CUDA device available (e.g. PPU host during pytest collection)
        return (0, 0)
