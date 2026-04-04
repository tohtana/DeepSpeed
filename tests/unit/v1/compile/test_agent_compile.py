# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import sys

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.utils.torch import required_torch_version

from unit.common import DistributedTest
from unit.v1.compile.util import compare_loss

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=2.6),
                                reason="DeepCompile requires PyTorch >= v2.6")


class TestDeepCompileAgent(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    def test_zero3_agent_finish(self, tmpdir):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")
        if required_torch_version(min_version=2.9):
            pytest.skip("DeepCompile ZeRO-3 is disabled on PyTorch >= 2.9 by the current runtime guard")

        log_path = str(tmpdir.join("agent_ranks.log"))
        os.environ["DEEPCOMPILE_AGENT_LOG"] = log_path

        script = "\n".join([
            "import json",
            "import os",
            "from pathlib import Path",
            "log = os.environ.get('DEEPCOMPILE_AGENT_LOG')",
            "if log:",
            "    Path(log).open('a').write(os.environ.get('LOCAL_RANK', 'na') + '\\n')",
            "print(json.dumps({'decision': 'finish', 'reason': 'done'}))",
        ])

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": 3,
            },
            "compile": {
                "deepcompile": True,
                "zero3_tuning_strategy": "agent",
                "agent_command": [sys.executable, "-c", script],
                "agent_max_iterations": 2,
                "agent_timeout_sec": 30,
            }
        }

        compare_loss(self, config_dict, torch.float32, iteration=7)

        ranks = [line.strip() for line in open(log_path, "r", encoding="utf-8").read().splitlines() if line.strip()]
        assert ranks
        assert set(ranks) == {"0"}
