"""Leave GPU priority and memory for the rest of the machine while training or evaluating."""

from __future__ import annotations

from collections.abc import Callable
import ctypes
import sys

import torch

# D3DKMT_SCHEDULINGPRIORITYCLASS_IDLE from d3dkmthk.h.
_GPU_PRIORITY_IDLE = 0


def _d3dkmt(name: str) -> Callable[..., int] | None:
    if sys.platform != "win32":
        return None
    try:
        return getattr(ctypes.WinDLL("gdi32"), name)  # type: ignore[no-any-return]
    except (OSError, AttributeError):
        return None


def _current_process() -> int:
    kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    return int(kernel32.GetCurrentProcess())


def gpu_priority_class() -> int | None:
    """This process's Windows GPU scheduling class, or None where unavailable."""

    getter = _d3dkmt("D3DKMTGetProcessSchedulingPriorityClass")
    if getter is None:
        return None
    priority = ctypes.c_int()
    status = getter(ctypes.c_void_p(_current_process()), ctypes.byref(priority))
    return priority.value if status == 0 else None


def lower_gpu_priority() -> bool:
    """Ask the Windows GPU scheduler to run this process at idle priority."""

    setter = _d3dkmt("D3DKMTSetProcessSchedulingPriorityClass")
    if setter is None:
        return False
    return setter(ctypes.c_void_p(_current_process()), ctypes.c_int(_GPU_PRIORITY_IDLE)) == 0


def cap_vram(device: torch.device, gigabytes: float | None) -> None:
    """Make the allocator fail rather than grow past ``gigabytes`` on this device."""

    if device.type != "cuda" or gigabytes is None:
        return
    index = torch.cuda.current_device() if device.index is None else device.index
    total = torch.cuda.get_device_properties(index).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gigabytes / total), index)


def share_gpu(device: torch.device, max_vram_gb: float | None) -> None:
    """Lower this process's GPU priority and cap its VRAM on ``device``."""

    if device.type == "cuda":
        lower_gpu_priority()
        cap_vram(device, max_vram_gb)
