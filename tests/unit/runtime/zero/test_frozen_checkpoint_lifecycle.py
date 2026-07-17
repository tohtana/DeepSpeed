# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Behavioral regressions for frozen ZeRO-3 parameters used during checkpoint replay.

Activation checkpointing and ZeRO-3 impose two different lifetimes on the same
parameter.  The original forward gathers a partition, releases it, and saves an
activation boundary.  Backward later enters a module, checkpoint recomputation
gathers that frozen partition again, and autograd consumes the recomputed
activation before the backward module can finish.  Recompute-forward therefore
cannot release the partition immediately: doing so can replace the full tensor
with its local shard while a later backward consumer still needs the full shape.

The eventual release has to belong to one checkpoint invocation, not merely to
the module object.  It can run at that invocation's last activation consumer;
when a no-grad checkpoint input provides no such local consumer, residual work
must be retired when the enclosing GraphTask completes.  A module-scoped set
plus one global LIFO deque of backward modules cannot represent that ownership:
the same module can be invoked repeatedly or nested while recompute and backward
hooks interleave, so independent invocations can add the same parameter to one
set and then pop a deque entry belonging to a different invocation.

These tests deliberately avoid any particular deferred-release API.  They
exercise a real checkpointed model and inspect the ZeRO invariants visible once
backward returns.  A stale gather appears as an AVAILABLE non-persistent
parameter, a leftover ``ds_active_sub_modules`` owner, an in-flight fetch, or a
coordinator available-numel count that disagrees with actual resident tensors.
"""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from unit.common import DistributedTest, preferred_dtype


def _zero3_config(gradient_accumulation_steps=1):
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 1000,
        "zero_optimization": {
            "stage": 3,
            # Keep even tiny test parameters non-persistent so every stale
            # gather is observable instead of being hidden by the cache.
            "stage3_param_persistence_threshold": 0,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3,
                "torch_adam": True,
            },
        },
    }
    if get_accelerator().is_bf16_supported():
        config["bf16"] = {"enabled": True}
    elif get_accelerator().is_fp16_supported():
        config["fp16"] = {"enabled": True, "initial_scale_power": 8}
    return config


class _FrozenCheckpointBlock(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.projection = torch.nn.Linear(hidden_dim, hidden_dim)
        self.norm.requires_grad_(False)

    def forward(self, value):
        return torch.nn.functional.gelu(self.projection(self.norm(value)))


class _NoGradInputModel(torch.nn.Module):
    """Checkpoint a frozen block whose input itself has no autograd edge."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.block = _FrozenCheckpointBlock(hidden_dim)

    def forward(self, value):
        return checkpoint(self.block, value, use_reentrant=False)


def _initialize_zero3(model, gradient_accumulation_steps=1):
    deepspeed.init_distributed(dist_backend=get_accelerator().communication_backend_name())
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    engine, _, _, _ = deepspeed.initialize(config=_zero3_config(gradient_accumulation_steps),
                                           model=model,
                                           model_parameters=trainable_params)
    return engine


def _assert_lifecycle_drained(engine, frozen_module):
    """Check that GraphTask completion left no gathered parameter or owner behind."""
    get_accelerator().synchronize()
    dist.barrier()

    nonpersistent = [param for param in engine.module.parameters() if not param.ds_persist]
    releasable = [param for param in nonpersistent if not param.is_external_param]
    releasable_ids = {id(param) for param in releasable}
    resident_numel = sum(param.ds_numel for param in nonpersistent if param.ds_status != ZeroParamStatus.NOT_AVAILABLE)
    coordinator = engine.optimizer.parameter_offload.get_param_coordinator()

    # These are accounting invariants rather than implementation-specific
    # checkpoint APIs.  Once backward returns, no invocation can still consume
    # a recomputed activation, so every temporary fetch owner must be gone and
    # the coordinator's count must describe exactly what is resident.
    stale_owners = {
        name: tuple(sorted(map(str, param.ds_active_sub_modules)))
        for name, param in engine.module.named_parameters()
        if id(param) in releasable_ids and param.ds_active_sub_modules
    }
    assert not stale_owners, f"checkpoint invocation owners leaked after backward: {stale_owners}"
    assert not coordinator._PartitionedParameterCoordinator__inflight_param_registry
    assert coordinator._PartitionedParameterCoordinator__n_available_params == resident_numel
    assert resident_numel == 0, "checkpoint recompute left non-persistent parameters gathered"
    assert all(param.ds_status == ZeroParamStatus.NOT_AVAILABLE for param in frozen_module.parameters())


class TestFrozenCheckpointLifecycle(DistributedTest):
    """One rank is sufficient to exercise ZeRO-3 gather/repartition lifetimes."""

    world_size = 1

    @pytest.mark.parametrize("backward_mode", ["engine", "torch"])
    def test_no_grad_checkpoint_input_eventually_repartitions(self, backward_mode):
        """A no-grad checkpoint input falls back to GraphTask cleanup.

        Non-reentrant checkpointing still recomputes because trainable weights
        inside the block need saved inputs, but ``value`` provides no input-side
        grad hook on which to retire the gather.  Each backward must therefore
        use the residual GraphTask completion path and fully balance residency
        before the next gradient-accumulation microbatch begins.
        """
        hidden_dim = 8
        engine = _initialize_zero3(_NoGradInputModel(hidden_dim), gradient_accumulation_steps=2)
        try:
            for microbatch in range(2):
                torch.manual_seed(1234 + microbatch)
                value = torch.randn(2,
                                    hidden_dim,
                                    device=get_accelerator().current_device_name(),
                                    dtype=preferred_dtype())
                assert not value.requires_grad
                loss = engine(value).float().sum()
                if backward_mode == "engine":
                    engine.backward(loss)
                else:
                    engine.scale(loss).backward()
                _assert_lifecycle_drained(engine, engine.module.block.norm)

            engine.step()
            _assert_lifecycle_drained(engine, engine.module.block.norm)
        finally:
            engine.destroy()
