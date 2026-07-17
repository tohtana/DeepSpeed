# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from deepspeed.compile import backend as backend_mod
from deepspeed.compile.backend import _get_fw_real_inputs, fwd_real_inputs
from deepspeed.compile.inductor import patch_create_aot_dispatcher_function
from deepspeed.compile.input_storage import InputStorage
from deepspeed.compile.patch_compiled_func import (clear_backward_inputs, get_backward_inputs, patch_compiled_func,
                                                   unpatch_compiled_func)


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


def _patch_aot_constructor():
    return patch_create_aot_dispatcher_function(graph_id=7,
                                                z3_partition=False,
                                                make_fw_graph=lambda gm, sample_inputs: gm.graph,
                                                make_bw_graph=lambda gm, sample_inputs: gm.graph,
                                                real_inputs=(torch.ones(1), ),
                                                param_indices=[],
                                                param_manager={},
                                                frame_id=0,
                                                frames_partitioned=set())


def test_inductor_aot_constructor_patch_is_restorable():
    from torch._dynamo.backends.common import AotAutograd

    original_init = AotAutograd.__init__
    restore = _patch_aot_constructor()
    try:
        assert AotAutograd.__init__ is not original_init
    finally:
        restore()

    assert AotAutograd.__init__ is original_init
    assert not hasattr(AotAutograd, "__original_init")


def test_older_aot_restore_does_not_clobber_newer_patch():
    from torch._dynamo.backends.common import AotAutograd

    original_init = AotAutograd.__init__
    restore_first = _patch_aot_constructor()
    restore_second = _patch_aot_constructor()
    newer_init = AotAutograd.__init__
    try:
        restore_first()
        assert AotAutograd.__init__ is newer_init
        assert hasattr(AotAutograd, "__original_init")
    finally:
        restore_second()

    assert AotAutograd.__init__ is original_init
    assert not hasattr(AotAutograd, "__original_init")


def test_backend_failure_cleanup_preserves_other_pending_frames():
    original_autograd_function = torch.autograd.Function
    backend_mod.frames_needing_bwd.clear()
    backend_mod.frames_needing_bwd.update((17, 18))
    backend_mod.patch_compiled_func()
    backend_mod.get_backward_inputs().append((torch.ones(1), ))
    gm = torch.fx.symbolic_trace(lambda x: x + 1)
    gm.meta["dynamo_compile_id"] = SimpleNamespace(frame_id=17)

    def failing_backend(gm):
        raise RuntimeError("compile failed")

    backend = backend_mod._cleanup_compiled_backward_backend_state_on_error()(failing_backend)

    try:
        with pytest.raises(RuntimeError, match="compile failed"):
            backend(gm)

        assert backend_mod.frames_needing_bwd == {18}
        assert len(backend_mod.get_backward_inputs()) == 1
        assert torch.autograd.Function is not original_autograd_function
    finally:
        backend_mod.frames_needing_bwd.clear()
        backend_mod.unpatch_compiled_func()
