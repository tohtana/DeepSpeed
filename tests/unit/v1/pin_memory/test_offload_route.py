# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Native-backend coverage for Phase 2 pin routing through offload helpers."""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_states import offload_optimizer_states
from deepspeed.utils.pin_memory import get_active_native_pinned_memory


def _require_native(monkeypatch):
    monkeypatch.setenv("DS_PIN_MEMORY_BACKEND", "native")
    try:
        if get_active_native_pinned_memory() is None:
            pytest.skip("native backend not selected")
    except Exception:
        pytest.skip("pin_memory op could not be built; native pinning unavailable")


def test_offload_helper_pin_pattern_native(monkeypatch):
    """Empty scratch pin used by offload_optimizer_states must be is_pinned under native."""
    _require_native(monkeypatch)
    accel = get_accelerator()
    # Mirrors offload_optimizer_states: allocate empty host buffer, then copy.
    src = torch.randn(32)
    pinned_buffer = accel.pin_memory(torch.empty_like(src, device="cpu"), make_copy=False)
    pinned_buffer.copy_(src)
    assert accel.is_pinned(pinned_buffer) is True
    assert accel.unpin_memory(pinned_buffer) is True


def test_offload_optimizer_states_native_is_pinned(monkeypatch):
    """offload_optimizer_states must route through accelerator pin_memory."""
    _require_native(monkeypatch)
    accel = get_accelerator()
    # The pin path only runs when the source tensor is not already on CPU.
    if accel.device_name() == "cpu":
        pytest.skip("requires non-CPU accelerator for GPU->pinned CPU offload path")

    device = accel.current_device_name()

    class _Opt:
        pass

    opt = _Opt()
    opt.state = {0: {"exp_avg": torch.randn(32, device=device)}}

    offload_optimizer_states(opt, device="cpu", pin_memory=True, non_blocking=False)
    buf = opt.state[0]["exp_avg"]
    assert buf.device.type == "cpu"
    assert accel.is_pinned(buf) is True
    assert accel.unpin_memory(buf) is True
