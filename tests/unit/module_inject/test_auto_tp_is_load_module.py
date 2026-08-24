# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch.nn as nn

from deepspeed.module_inject.auto_tp import Loading


# Matches Loading.is_load_module() by class name only, so a lightweight stand-in
# with the right name exercises the same allowlist check real Qwen3.5 modules
# hit, without depending on a transformers version new enough to ship Qwen3.5.
def _make_module(class_name):
    return type(class_name, (nn.Module, ), {})()


class TestIsLoadModule:

    @pytest.mark.parametrize("class_name", [
        "Qwen3_5RMSNorm",
        "Qwen3_5RMSNormGated",
        "Qwen3_5MoeRMSNorm",
        "Qwen3_5MoeRMSNormGated",
    ])
    def test_qwen3_5_norm_variants_are_recognized(self, class_name):
        assert Loading.is_load_module(_make_module(class_name))

    def test_unrelated_module_is_not_recognized(self):
        assert not Loading.is_load_module(_make_module("SomeUnrelatedModule"))
