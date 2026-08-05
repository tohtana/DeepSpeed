# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.fx import GraphModule
from .passes.tp_compile import apply_autotp, defer_collectives_to_compiler


def init_autotp(model):
    """Hand the tensor-parallel collectives of an AutoTP-partitioned model over to the compiler.

    The model is expected to have been partitioned already by the regular AutoTP path, so this only
    suppresses the module-level collectives and returns a backend that emits them as graph nodes.
    """
    defer_collectives_to_compiler(model)

    def backend_fn(gm: GraphModule, real_inputs):
        apply_autotp(gm, real_inputs)
        return torch._inductor.compile(gm, real_inputs)

    return backend_fn
