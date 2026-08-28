# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator.mps_accelerator import MPS_Accelerator


def test_mps_accelerator_requires_memory_api(monkeypatch):
    monkeypatch.delattr(torch.mps, "recommended_max_memory", raising=False)

    with pytest.raises(ValueError, match=r"requires torch>=2\.5"):
        MPS_Accelerator()


def test_mps_accelerator_constructs_with_memory_api(monkeypatch):
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1, raising=False)

    accelerator = MPS_Accelerator()

    assert accelerator.device_name() == "mps"
