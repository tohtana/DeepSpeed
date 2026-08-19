# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
from unit.common import DistributedTest


@pytest.mark.parametrize("stage", [1, 2])
class TestZeroEmptyParam(DistributedTest):
    world_size = 1

    def test_backward(self, stage):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 0, bias=False))
        config = {"train_batch_size": 1, "zero_optimization": {"stage": stage}}
        engine, *_ = deepspeed.initialize(model=model,
                                          model_parameters=model.parameters(),
                                          optimizer=torch.optim.AdamW(model.parameters()),
                                          config=config)
        engine.backward(engine(torch.randn(1, 4, device=engine.device)).sum())
