# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import torch

import deepspeed.compile.passes.selective_gather as selective_gather_pass
from deepspeed.compile.profilers import ProfilingResult


class FakeAccelerator:

    def __init__(self, total_mem=1000, available_mem=800, device="cpu"):
        self._total_mem = total_mem
        self._available_mem = available_mem
        self._device = device

    def total_memory(self):
        return self._total_mem

    def available_memory(self):
        return self._available_mem

    def current_device(self):
        return self._device


class FakeDeepCompileHandle:

    def __init__(self):
        self.persistent_ds_ids = []

    def set_persistent(self, ds_id):
        self.persistent_ds_ids.append(ds_id)


def _make_param(numel, ds_persist=False):
    return SimpleNamespace(numel=numel,
                           dtype=torch.float32,
                           param=SimpleNamespace(ds_persist=ds_persist, ds_shape=(numel, )))


def _allgather_target():
    try:
        return torch.ops.dc.allgather_param.default
    except AttributeError:
        if not hasattr(_allgather_target, "_lib"):
            _allgather_target._lib = torch.library.Library("dc", "DEF")
            _allgather_target._lib.define(
                "allgather_param(Tensor a, int graph_id, int id, ScalarType? dtype = None) -> Tensor")
        return torch.ops.dc.allgather_param.default


def _allgather_node(ds_id, *, tensor_size=None, device_time=None, wall_time=None):
    metadata = {}
    if tensor_size is not None:
        metadata["tensor_size"] = tensor_size
    if device_time is not None:
        metadata["device_time"] = device_time
    if wall_time is not None:
        metadata["wall_time"] = wall_time
    return SimpleNamespace(target=_allgather_target(), args=(None, None, ds_id), meta=metadata)


def _run_selective_gather(monkeypatch,
                          *,
                          total_mem,
                          available_mem,
                          fwd_mem,
                          bwd_mem,
                          params,
                          fwd_nodes=None,
                          bwd_nodes=None):
    fake_handle = FakeDeepCompileHandle()
    messages = []

    monkeypatch.setattr(selective_gather_pass, "get_accelerator",
                        lambda: FakeAccelerator(total_mem=total_mem, available_mem=available_mem))
    monkeypatch.setattr(selective_gather_pass, "get_deepcompile_handle", lambda: fake_handle)
    monkeypatch.setattr(selective_gather_pass, "print_rank_0", messages.append)
    monkeypatch.setattr(selective_gather_pass.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(selective_gather_pass.dist, "all_reduce", lambda tensor, op: tensor)

    profiling_results = {
        0:
        ProfilingResult(fwd_graph=SimpleNamespace(nodes=list(fwd_nodes or [])),
                        bwd_graph=SimpleNamespace(nodes=list(bwd_nodes or [])),
                        fwd_mem=fwd_mem,
                        bwd_mem=bwd_mem)
    }
    param_manager = {
        0: SimpleNamespace(params=params, ds_ids={
            name: index + 1
            for index, name in enumerate(params.keys())
        })
    }

    gm = object()
    returned = selective_gather_pass.selective_gather(gm,
                                                      graph_id=0,
                                                      graph_order=[(0, True)],
                                                      profiling_results=profiling_results,
                                                      create_inputs_fn=None,
                                                      mem_budget=0.0,
                                                      param_manager=param_manager,
                                                      bwd=True)

    assert returned is gm
    return SimpleNamespace(selected=fake_handle.persistent_ds_ids, messages=messages)


def test_selective_gather_caps_selection_by_profiled_resident_headroom(monkeypatch):
    result = _run_selective_gather(monkeypatch,
                                   total_mem=1000,
                                   available_mem=800,
                                   fwd_mem=[("fwd", 850, 0, 900)],
                                   bwd_mem=[("bwd", 830, 0, 880)],
                                   params={
                                       "first": _make_param(10),
                                       "second": _make_param(5),
                                   })

    assert result.selected == [1]
    assert any("usable_available_mem=50" in message for message in result.messages)


def test_selective_gather_uses_current_free_memory_when_tighter(monkeypatch):
    result = _run_selective_gather(monkeypatch,
                                   total_mem=1000,
                                   available_mem=60,
                                   fwd_mem=[("fwd", 500, 0, 700)],
                                   bwd_mem=[("bwd", 480, 0, 680)],
                                   params={
                                       "first": _make_param(10),
                                       "second": _make_param(5),
                                   })

    assert result.selected == [1]
    assert any("current_available_budget=54" in message and "usable_available_mem=54" in message
               for message in result.messages)


def test_selective_gather_skips_empty_profile_records(monkeypatch):
    result = _run_selective_gather(monkeypatch,
                                   total_mem=1000,
                                   available_mem=800,
                                   fwd_mem=[],
                                   bwd_mem=[],
                                   params={
                                       "first": _make_param(10),
                                   })

    assert result.selected == []
    assert any("profiled_mem_lists=0" in message for message in result.messages)
    assert any("no profiling data" in message for message in result.messages)


def test_selective_gather_skips_when_profiled_budget_is_zero(monkeypatch):
    result = _run_selective_gather(monkeypatch,
                                   total_mem=1000,
                                   available_mem=800,
                                   fwd_mem=[("fwd", 930, 0, 950)],
                                   bwd_mem=[],
                                   params={
                                       "first": _make_param(10),
                                   })

    assert result.selected == []
    assert any("usable_available_mem=0" in message for message in result.messages)
    assert any("no currently available memory" in message for message in result.messages)


def test_persistence_budget_ignores_empty_profile_lists():
    budget = selective_gather_pass._compute_persistence_budget(
        [[], [("fwd", 700, 0, 900)], [], [("bwd", 650, 0, 850)]],
        total_mem=1000,
        mem_margin=0.1,
    )

    assert budget == {
        "usable_mem": 900,
        "peak_resident_alloc": 700,
        "transient_peak": 900,
        "available_mem": 200,
        "profiled_list_count": 2,
    }


def test_selective_gather_charges_profiled_tensor_size(monkeypatch):
    result = _run_selective_gather(monkeypatch,
                                   total_mem=1000,
                                   available_mem=800,
                                   fwd_mem=[("fwd", 400, 0, 500)],
                                   bwd_mem=[],
                                   fwd_nodes=[_allgather_node(1, tensor_size=700, device_time=700.0)],
                                   params={
                                       "first": _make_param(1),
                                   })

    assert result.selected == []
    assert any("candidate_bytes=700" in message for message in result.messages)
    assert any("smallest_candidate=700" in message for message in result.messages)


def test_selective_gather_fallback_size_uses_graph_param_numel(monkeypatch):
    result = _run_selective_gather(monkeypatch,
                                   total_mem=1000,
                                   available_mem=800,
                                   fwd_mem=[("fwd", 300, 0, 500)],
                                   bwd_mem=[],
                                   params={
                                       "first": _make_param(128),
                                   })

    assert result.selected == [1]
    assert any("candidate_bytes=512" in message for message in result.messages)
    assert any("selected_count=1 selected_bytes=512" in message for message in result.messages)


def test_selective_gather_excludes_existing_persistent_params_and_counts_them(monkeypatch):
    result = _run_selective_gather(monkeypatch,
                                   total_mem=2000,
                                   available_mem=1500,
                                   fwd_mem=[("fwd", 300, 0, 500)],
                                   bwd_mem=[],
                                   params={
                                       "persisted": _make_param(100, ds_persist=True),
                                       "candidate": _make_param(10),
                                   })

    assert result.selected == [2]
    assert any("persistent_count=1 persistent_bytes=400" in message for message in result.messages)
    assert any("candidate_count=1 candidate_bytes=40" in message for message in result.messages)


def test_selective_gather_uses_fallback_size_for_unprofiled_allgather_node(monkeypatch):
    result = _run_selective_gather(monkeypatch,
                                   total_mem=1000,
                                   available_mem=800,
                                   fwd_mem=[("fwd", 300, 0, 500)],
                                   bwd_mem=[],
                                   fwd_nodes=[_allgather_node(1)],
                                   params={
                                       "first": _make_param(128),
                                   })

    assert result.selected == [1]
    assert any("candidate_bytes=512" in message for message in result.messages)
    assert any("selected_count=1 selected_bytes=512" in message for message in result.messages)
