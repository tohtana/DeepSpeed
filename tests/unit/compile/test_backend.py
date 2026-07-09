# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import deque

import torch

from deepspeed.compile.backend import _get_fw_real_inputs, fwd_real_inputs, set_example_values_to_symints
from deepspeed.compile import backend as backend_mod
from deepspeed.compile.inductor import patch_create_aot_dispatcher_function
from deepspeed.compile.input_storage import InputStorage
from deepspeed.compile.patch_compiled_func import (clear_backward_inputs, get_backward_inputs, patch_compiled_func,
                                                   unpatch_compiled_func)
from deepspeed.compile.profilers import ProfilingResult
from deepspeed.compile.profilers.graph_profile import _mark_profile_incomplete


def test_forward_real_inputs_prefer_closure_queue_over_global_queue():
    fwd_real_inputs.clear()
    fwd_real_inputs.append(("wrong_graph", 1))
    local_inputs = (torch.nn.Parameter(torch.ones(2, dtype=torch.float32)), )
    storage = InputStorage()
    storage.put((torch.ones(1, dtype=torch.float32), ))

    selected = _get_fw_real_inputs(deque([local_inputs]), storage, graph_id=7)

    assert selected is local_inputs
    assert fwd_real_inputs == [("wrong_graph", 1)]
    fwd_real_inputs.clear()


def test_forward_real_inputs_fall_back_to_storage_when_local_queue_is_empty():
    fwd_real_inputs.clear()
    storage = InputStorage()
    storage.put((torch.ones(3, dtype=torch.float32), ))

    selected = _get_fw_real_inputs(deque(), storage, graph_id=7)

    assert len(selected) == 1
    assert selected[0].shape == torch.Size([3])
    assert selected[0].dtype is torch.float32


def test_symint_materialization_preserves_fake_parameter_slots():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as fake_mode:
        fake_param = fake_mode.from_tensor(torch.empty((2, 3), dtype=torch.bfloat16))
    fake_param.ds_id = 123

    materialized = set_example_values_to_symints((fake_param, ), [(0, 123, torch.Size([2, 3]))])

    assert isinstance(materialized[0], torch.nn.Parameter)
    assert materialized[0].shape == torch.Size([2, 3])
    assert materialized[0].dtype is torch.bfloat16
    assert materialized[0].ds_id == 123


def test_launch_compile_passes_clears_legacy_input_queues(monkeypatch):

    class DummyDeepCompileHandle:

        def reset(self):
            pass

    fwd_real_inputs.clear()
    clear_backward_inputs()
    fwd_real_inputs.append((torch.ones(1), ))
    get_backward_inputs().append((torch.ones(1), ))
    monkeypatch.setattr(backend_mod, "log_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_mod, "get_deepcompile_handle", lambda: DummyDeepCompileHandle())

    backend_mod.init_schedule([(0, [])])
    backend_mod.launch_compile_passes(0)

    assert fwd_real_inputs == []
    assert get_backward_inputs() == []


def test_unpatch_compiled_func_clears_backward_inputs():
    clear_backward_inputs()
    patch_compiled_func()
    try:
        get_backward_inputs().append((torch.ones(1), ))
        unpatch_compiled_func()
        assert get_backward_inputs() == []
    finally:
        unpatch_compiled_func()


def test_inductor_aot_constructor_patch_is_restorable():
    from torch._dynamo.backends.common import AotAutograd

    original_init = AotAutograd.__init__
    restore = patch_create_aot_dispatcher_function(graph_id=7,
                                                   z3_partition=False,
                                                   make_fw_graph=lambda gm, sample_inputs: gm.graph,
                                                   make_bw_graph=lambda gm, sample_inputs: gm.graph,
                                                   real_inputs=(torch.ones(1), ),
                                                   param_indices=[],
                                                   param_manager={},
                                                   frame_id=0,
                                                   frames_partitioned=set())
    try:
        assert AotAutograd.__init__ is not original_init
    finally:
        restore()

    assert AotAutograd.__init__ is original_init
    assert not hasattr(AotAutograd, "__original_init")


def test_run_opt_passes_skips_memory_profile_for_incomplete_graph(monkeypatch):
    gm = torch.fx.symbolic_trace(lambda x: x + 1)
    profiling_results = {7: ProfilingResult()}

    class UnexpectedMemoryProfiler:

        def __init__(self, *args, **kwargs):
            raise AssertionError("memory profiling should be skipped for incomplete operator profiles")

    def incomplete_profile_pass(gm, *args, **kwargs):
        _mark_profile_incomplete(gm.graph)
        return gm

    monkeypatch.setattr(backend_mod, "MemoryProfilingInterpreter", UnexpectedMemoryProfiler)
    monkeypatch.setattr(backend_mod, "log_rank0", lambda *args, **kwargs: None)

    backend_mod.run_opt_passes(opt_passes=[incomplete_profile_pass],
                               gm=gm,
                               graph_id=7,
                               graph_order=[],
                               profiling_results=profiling_results,
                               create_inputs_fn=lambda: (torch.ones(1), ),
                               mem_budget=0.0,
                               param_manager={},
                               bwd=False)

    assert profiling_results[7].fwd_mem == []
    assert profiling_results[7].fwd_mem_complete is False
