from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from wombat.gpu_share import cap_vram, share_gpu


def test_cpu_devices_are_left_alone() -> None:
    cap_vram(torch.device("cpu"), 1.0)
    share_gpu(torch.device("cpu"), 1.0)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows GPU scheduler API")
def test_windows_gpu_priority_can_be_lowered_to_idle() -> None:
    # A child process, so the test runner keeps its own GPU priority.
    script = (
        "from wombat.gpu_share import gpu_priority_class, lower_gpu_priority;"
        "print(lower_gpu_priority(), gpu_priority_class())"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["True", "0"]
