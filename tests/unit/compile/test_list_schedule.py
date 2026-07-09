# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import operator
from types import SimpleNamespace

import pytest
import torch
from torch.fx import Graph, GraphModule

import deepspeed.compile.util as compile_util
from deepspeed.compile import backend as backend_mod
from deepspeed.compile import inductor as inductor_mod
from deepspeed.compile import list_schedule as schedule_mod
from deepspeed.compile.passes import prefetch as prefetch_mod
from deepspeed.compile.passes import selective_gather as selective_gather_mod
from deepspeed.compile.passes import zero3_compile as zero3_compile_mod
from deepspeed.compile.profilers import ProfilingResult
from deepspeed.compile.profilers.graph_profile import _backfill_missing_profile_metadata, is_profile_incomplete

_DC_LIBRARIES = []


def _define_dc_ops():
    try:
        torch.ops.dc.allgather_param.default
        torch.ops.dc.wait_allgather.default
        torch.ops.dc.release_param.default
        torch.ops.dc.reduce_grad.default
        return
    except AttributeError:
        pass

    lib = torch.library.Library("dc", "DEF")
    for schema in (
            "allgather_param(Tensor a, int graph_id, int id, ScalarType? dtype = None) -> Tensor",
            "wait_allgather(Tensor(a) a, int graph_id, int id) -> Tensor(a)",
            "release_param(Tensor(a) a, int graph_id, int id, int n_users) -> Tensor(a)",
            "reduce_grad(Tensor a, int graph_id, int id) -> Tensor",
            "free_tensors(Tensor[] tensors) -> ()",
            "end_backward(Tensor[] tensors, int graph_id, bool release_reduce_buckets = True) -> ()",
    ):
        try:
            lib.define(schema)
        except RuntimeError as exc:
            if "already been registered" not in str(exc):
                raise
    _DC_LIBRARIES.append(lib)


@pytest.fixture(autouse=True)
def stub_deepcompile_ops(monkeypatch):
    _define_dc_ops()
    no_copy_ops = {torch.ops.dc.wait_allgather.default}
    monkeypatch.setattr(compile_util, "get_no_copy_ops", lambda: no_copy_ops)


def _with_meta(node, tensor_size=0, device_time=0):
    node.meta["tensor_size"] = tensor_size
    if device_time is not None:
        node.meta["device_time"] = device_time
    return node


def _placeholder(graph, name):
    return _with_meta(graph.placeholder(name))


def test_sync_memory_profile_complete_noops_without_distributed(monkeypatch):
    monkeypatch.setattr(backend_mod.dist, "is_initialized", lambda: False)

    def fail_all_reduce(*args, **kwargs):
        raise AssertionError("all_reduce should not run without distributed init")

    monkeypatch.setattr(backend_mod.dist, "all_reduce", fail_all_reduce)

    assert backend_mod._sync_memory_profile_complete(True)
    assert not backend_mod._sync_memory_profile_complete(False)


def test_sync_memory_profile_complete_reduces_asymmetric_failure(monkeypatch):
    monkeypatch.setattr(backend_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend_mod, "get_accelerator", lambda: SimpleNamespace(current_device=lambda: "cpu"))

    def mark_any_rank_failed(tensor, op):
        assert op == backend_mod.dist.ReduceOp.MIN
        tensor[0] = 0

    monkeypatch.setattr(backend_mod.dist, "all_reduce", mark_any_rank_failed)

    assert not backend_mod._sync_memory_profile_complete(True)


def test_get_last_uses_handles_dead_no_copy_node():
    graph = Graph()
    param = _placeholder(graph, "dead_wait_param")
    wait = _wait(graph, param, 1, "dead_wait")
    graph.output(())
    graph.lint()

    node_to_last_use, user_to_last_uses = compile_util.get_last_uses(graph)
    node_to_uses = compile_util.get_real_uses(graph)

    assert node_to_last_use[param] is wait
    assert user_to_last_uses[wait] == [param]
    assert node_to_uses[param] == []


def test_zero3_scheduler_budget_uses_rank_reduced_non_gathered_peak(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 2000

        def memory_allocated(self):
            return 50

    def reduce_budget_inputs(tensor, op):
        if op == zero3_compile_mod.dist.ReduceOp.MIN:
            tensor[0] = 1000
        elif op == zero3_compile_mod.dist.ReduceOp.MAX:
            tensor[0] = 850
        else:
            raise AssertionError(f"unexpected reduce op {op}")

    monkeypatch.setattr(zero3_compile_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", reduce_budget_inputs)

    graph = Graph()
    param = _placeholder(graph, "budget_builder_param")
    ag = _allgather(graph, param, 1, "budget_builder", tensor_size=200)
    wait = _wait(graph, ag, 1, "budget_builder")
    op = _neg(graph, wait, "budget_builder_op")
    op.meta["max_mem"] = 800
    release = _release(graph, op, 1, "budget_builder")
    graph.output((release, ))
    graph.lint()
    for node in graph.nodes:
        node.meta.setdefault("max_mem", 0)

    budget = zero3_compile_mod._build_scheduler_budget_from_operator_profile(graph)

    assert budget.source == "profiled_non_gathered_peak_memory"
    assert budget.total_mem == 1000
    assert budget.profiled_non_gathered_peak_mem == 850
    assert budget.safety_margin == 100
    assert budget.max_gathered_bytes == 50


def test_zero3_scheduler_budget_skips_incomplete_operator_profile_metadata():
    graph = Graph()
    graph.output(())
    graph.lint()
    _backfill_missing_profile_metadata(graph, profile_complete=False)

    budget = zero3_compile_mod._build_scheduler_budget_from_operator_profile(graph)

    assert budget is None


def test_zero3_scheduler_budget_skips_incomplete_operator_profile(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod, "get_accelerator",
                        lambda: SimpleNamespace(current_device=lambda: "cpu", available_memory=lambda: 0))
    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", lambda *args, **kwargs: None)
    graph = Graph()
    graph.output(())
    graph.lint()
    _backfill_missing_profile_metadata(graph, profile_complete=False)
    gm = GraphModule(torch.nn.Module(), graph)

    budget, disabled_reason = zero3_compile_mod._scheduler_budget_from_operator_profile(gm)

    assert budget is None
    assert disabled_reason == "incomplete_operator_profile"


def test_zero3_scheduler_budget_uses_available_memory_when_operator_profile_incomplete(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod.dist, "get_world_size", lambda: 1)

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def available_memory(self):
            return 500

    def reduce_budget_inputs(tensor, op):
        assert op == zero3_compile_mod.dist.ReduceOp.MIN
        if tensor.dtype == torch.int64:
            tensor[0] = 400

    monkeypatch.setattr(zero3_compile_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", reduce_budget_inputs)

    graph = Graph()
    param = _placeholder(graph, "incomplete_profile_param")
    ag = _allgather(graph, param, 1, "incomplete_profile", tensor_size=800)
    wait = _wait(graph, ag, 1, "incomplete_profile")
    graph.output((wait, ))
    graph.lint()
    _backfill_missing_profile_metadata(graph, profile_complete=False)
    gm = GraphModule(torch.nn.Module(), graph)

    budget, disabled_reason = zero3_compile_mod._scheduler_budget_from_operator_profile(gm)

    assert disabled_reason is None
    assert budget.source == "available_memory"
    assert budget.available_mem == 400
    assert budget.safety_margin == 40
    assert budget.max_gathered_bytes == 360
    assert ag.meta["allgather_allocation_bytes"] == 800


def test_zero3_scheduler_budget_caps_incomplete_profile_to_single_allgather(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod.dist, "get_world_size", lambda: 1)

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def available_memory(self):
            return 5000

    def reduce_budget_inputs(tensor, op):
        assert op == zero3_compile_mod.dist.ReduceOp.MIN
        if tensor.dtype == torch.int64:
            tensor[0] = 5000

    monkeypatch.setattr(zero3_compile_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", reduce_budget_inputs)

    graph = Graph()
    lhs = _placeholder(graph, "capped_incomplete_profile_lhs")
    rhs = _placeholder(graph, "capped_incomplete_profile_rhs")
    ag1 = _allgather(graph, lhs, 1, "capped_incomplete_profile_lhs", tensor_size=800)
    ag2 = _allgather(graph, rhs, 2, "capped_incomplete_profile_rhs", tensor_size=600)
    wait1 = _wait(graph, ag1, 1, "capped_incomplete_profile_lhs")
    wait2 = _wait(graph, ag2, 2, "capped_incomplete_profile_rhs")
    use = _add(graph, wait1, wait2, "capped_incomplete_profile_use")
    graph.output((use, ))
    graph.lint()
    _backfill_missing_profile_metadata(graph, profile_complete=False)
    gm = GraphModule(torch.nn.Module(), graph)

    budget, disabled_reason = zero3_compile_mod._scheduler_budget_from_operator_profile(gm)

    assert disabled_reason is None
    assert budget.source == "available_memory_single_allgather_cap"
    assert budget.max_gathered_bytes == 800
    assert zero3_compile_mod.max_possible_gathered_bytes(graph) == 1400


def test_zero3_scheduler_budget_uses_partial_operator_profile_when_incomplete(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod.dist, "get_world_size", lambda: 1)

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 1000

        def memory_allocated(self):
            return 50

        def available_memory(self):
            return 500

    def reduce_budget_inputs(tensor, op):
        if tensor.dtype == torch.int32:
            tensor[0] = 0

    monkeypatch.setattr(zero3_compile_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", reduce_budget_inputs)

    graph = Graph()
    param = _placeholder(graph, "partial_budget_param")
    ag = _allgather(graph, param, 1, "partial_budget", tensor_size=800)
    wait = _wait(graph, ag, 1, "partial_budget")
    op = _neg(graph, wait, "partial_budget_observed_op")
    op.meta["max_mem"] = 800
    release = _release(graph, op, 1, "partial_budget")
    graph.output((release, ))
    graph.lint()
    _backfill_missing_profile_metadata(graph, profile_complete=False)
    gm = GraphModule(torch.nn.Module(), graph)

    budget, disabled_reason = zero3_compile_mod._scheduler_budget_from_operator_profile(gm)

    assert disabled_reason is None
    assert budget.source == "profiled_non_gathered_peak_memory"
    assert budget.total_mem == 1000
    assert budget.profiled_non_gathered_peak_mem == 850
    assert budget.max_gathered_bytes == 50


def test_zero3_scheduler_budget_skips_non_distributed_memory_profile(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: False)
    graph = Graph()
    graph.output(())
    graph.lint()
    gm = GraphModule(torch.nn.Module(), graph)

    budget, disabled_reason = zero3_compile_mod._scheduler_budget_from_operator_profile(gm)

    assert budget is None
    assert disabled_reason == "non_distributed"


def test_zero3_scheduler_budget_skips_when_budget_cannot_constrain(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", lambda *args, **kwargs: None)

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 2000

        def memory_allocated(self):
            return 50

    monkeypatch.setattr(zero3_compile_mod, "get_accelerator", lambda: FakeAccelerator())

    graph = Graph()
    param = _placeholder(graph, "nonbinding_budget_param")
    ag = _allgather(graph, param, 1, "nonbinding_budget", tensor_size=200)
    wait = _wait(graph, ag, 1, "nonbinding_budget")
    op = _neg(graph, wait, "nonbinding_budget_op")
    op.meta["max_mem"] = 800
    release = _release(graph, op, 1, "nonbinding_budget")
    graph.output((release, ))
    graph.lint()
    for node in graph.nodes:
        node.meta.setdefault("max_mem", 0)
    gm = GraphModule(torch.nn.Module(), graph)

    budget, disabled_reason = zero3_compile_mod._scheduler_budget_from_operator_profile(gm)

    assert budget is None
    assert disabled_reason == "budget_not_constraining"


def test_profiled_non_gathered_peak_subtracts_live_gathered_residency():
    graph = Graph()
    param = _placeholder(graph, "nongathered_peak_param")
    ag = _allgather(graph, param, 2, "nongathered_peak", tensor_size=200)
    wait = _wait(graph, ag, 2, "nongathered_peak")
    release = _release(graph, wait, 2, "nongathered_peak")
    graph.output((release, ))
    graph.lint()

    assert schedule_mod.profiled_non_gathered_peak(graph, [(ag.name, 900, 0, 900), (wait.name, 950, 0, 950),
                                                           (release.name, 920, 0, 920)]) == 750


def test_zero3_stamps_padded_allgather_allocation_metadata(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod.dist, "get_world_size", lambda: 8)
    graph = Graph()

    param = _placeholder(graph, "metadata_padded_param")
    ag = _allgather(graph, param, 3, "metadata_padded", tensor_size=102)
    wait = _wait(graph, ag, 3, "metadata_padded")
    release = _release(graph, wait, 3, "metadata_padded")
    graph.output((release, ))
    graph.lint()

    zero3_compile_mod._set_allgather_allocation_metadata(graph)

    assert ag.meta["allgather_allocation_bytes"] == 112


def test_zero3_stamps_replicated_param_allgather_allocation_metadata():
    graph = Graph()

    param = _placeholder(graph, "replicated_metadata_param")
    param.meta["val"] = torch.empty((8, ), dtype=torch.float16)
    use = _neg(graph, param, "replicated_metadata_use")
    graph.output((use, ))
    graph.lint()

    param_manager = SimpleNamespace(params={param.name: SimpleNamespace(dtype=torch.bfloat16, numel=777)},
                                    ds_ids={param.name: 3})

    zero3_compile_mod.add_gather_and_release(0, graph, param_manager, [param])

    ag_nodes = [node for node in graph.nodes if node.target == torch.ops.dc.allgather_param.default]
    assert len(ag_nodes) == 1
    assert ag_nodes[0].meta["allgather_allocation_bytes"] == 1554


def test_zero3_gathers_output_only_param_for_backward_passthrough():
    graph = Graph()

    param = _placeholder(graph, "output_only_param")
    param.meta["val"] = torch.empty((8, ), dtype=torch.float16)
    graph.output((param, ))
    graph.lint()

    param_manager = SimpleNamespace(params={param.name: SimpleNamespace(dtype=torch.bfloat16, numel=777)},
                                    ds_ids={param.name: 3})

    new_graph = zero3_compile_mod.add_gather_and_release(0, graph, param_manager, [param])
    new_graph.lint()

    ag_nodes = [node for node in new_graph.nodes if node.target == torch.ops.dc.allgather_param.default]
    wait_nodes = [node for node in new_graph.nodes if node.target == torch.ops.dc.wait_allgather.default]
    release_nodes = [node for node in new_graph.nodes if node.target == torch.ops.dc.release_param.default]
    output_node = next(node for node in new_graph.nodes if node.op == "output")

    assert len(ag_nodes) == 1
    assert len(wait_nodes) == 1
    assert release_nodes == []
    assert ag_nodes[0].args[0].name == param.name
    assert wait_nodes[0].args[0] is ag_nodes[0]
    assert output_node.args == ((wait_nodes[0], ), )


def test_zero3_scheduler_debug_logs_disabled_budget(monkeypatch, capsys):
    monkeypatch.setenv(zero3_compile_mod.SCHEDULER_DEBUG_ENV, "1")
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: False)
    graph = Graph()
    graph.output(())
    graph.lint()

    zero3_compile_mod._log_scheduler_result(7,
                                            bwd=True,
                                            scheduler_budget=None,
                                            disabled_reason="missing_or_incomplete_memory_profile",
                                            graph=graph)

    captured = capsys.readouterr()
    assert "budget_enabled=False" in captured.out
    assert "disabled_reason=missing_or_incomplete_memory_profile" in captured.out


def _allgather(graph, arg, ds_id, name, tensor_size=1, device_time=1, allocation_size=None):
    node = _with_meta(
        graph.call_function(torch.ops.dc.allgather_param.default, (arg, 0, ds_id), {"dtype": torch.float16},
                            name=f"allgather_ds_param_{name}_{ds_id}"),
        tensor_size=tensor_size,
        device_time=device_time,
    )
    if allocation_size is not None:
        node.meta["allgather_allocation_bytes"] = allocation_size
    return node


def _wait(graph, arg, ds_id, name):
    return _with_meta(
        graph.call_function(torch.ops.dc.wait_allgather.default, (arg, 0, ds_id),
                            name=f"wait_allgather_ds_param_{name}_{ds_id}"))


def _neg(graph, arg, name, device_time=0):
    return _with_meta(graph.call_function(operator.neg, (arg, ), name=name), device_time=device_time)


def _add(graph, lhs, rhs, name, device_time=0):
    return _with_meta(graph.call_function(operator.add, (lhs, rhs), name=name), device_time=device_time)


def _release(graph, arg, ds_id, name, n_users=1):
    return _with_meta(
        graph.call_function(torch.ops.dc.release_param.default, (arg, 0, ds_id, n_users),
                            name=f"release_ds_param_{name}_{ds_id}"))


def _scheduled_graph(graph, scheduler_budget=None):
    return schedule_mod.fast_free_schedule(
        graph,
        0,
        0,
        debug_log=True,
        scheduler_budget=scheduler_budget,
    )


def _scheduled_names(graph, scheduler_budget=None):
    return [node.name for node in _scheduled_graph(graph, scheduler_budget=scheduler_budget).nodes]


def _scheduler_diagnostics(graph):
    return getattr(graph, schedule_mod.SCHEDULER_BUDGET_DIAGNOSTICS_ATTR)


def test_fast_free_schedule_keeps_zero_free_acc_filter():
    graph = Graph()

    safe_param = _placeholder(graph, "safe_param")
    safe_pre_param = _placeholder(graph, "safe_pre_param")
    unsafe_param = _placeholder(graph, "unsafe_param")
    unsafe_extra_param = _placeholder(graph, "unsafe_extra_param")

    safe_pre_ag = _allgather(graph, safe_pre_param, 10, "safe_pre")
    safe_pre_wait = _wait(graph, safe_pre_ag, 10, "safe_pre")
    safe_pre_use = _neg(graph, safe_pre_wait, "safe_pre_use")
    safe_ag = _allgather(graph, _add(graph, safe_param, safe_pre_use, "safe_param_dep"), 11, "safe")
    safe_wait = _wait(graph, safe_ag, 11, "safe")
    safe_use = _neg(graph, safe_wait, "safe_use", device_time=100)
    safe_release = _release(graph, safe_use, 11, "safe")

    unsafe_ag = _allgather(graph, unsafe_param, 20, "unsafe")
    unsafe_wait = _wait(graph, unsafe_ag, 20, "unsafe")
    unsafe_extra_ag = _allgather(graph, unsafe_extra_param, 21, "unsafe_extra")
    unsafe_extra_wait = _wait(graph, unsafe_extra_ag, 21, "unsafe_extra")
    unsafe_use = _add(graph, unsafe_wait, unsafe_extra_wait, "unsafe_use", device_time=1)
    unsafe_release = _release(graph, unsafe_use, 20, "unsafe")

    graph.output((safe_release, unsafe_release))
    graph.lint()

    names = _scheduled_names(graph)

    assert names.index(safe_release.name) < names.index(unsafe_ag.name)
    assert names.index(safe_release.name) < names.index(unsafe_extra_ag.name)


def test_fast_free_schedule_prefers_lower_allgather_pressure_in_zero_free_acc_bucket():
    graph = Graph()

    high_param = _placeholder(graph, "high_param")
    high_pre_param = _placeholder(graph, "high_pre_param")
    low_param = _placeholder(graph, "low_param")
    low_pre_param = _placeholder(graph, "low_pre_param")

    high_pre_ag = _allgather(graph, high_pre_param, 30, "high_pre", tensor_size=100)
    high_pre_wait = _wait(graph, high_pre_ag, 30, "high_pre")
    high_ag = _allgather(graph, _add(graph, high_param, high_pre_wait, "high_param_dep"), 31, "high")
    high_wait = _wait(graph, high_ag, 31, "high")
    high_use = _neg(graph, high_wait, "high_use", device_time=1)
    high_release = _release(graph, high_use, 31, "high")

    low_pre_ag = _allgather(graph, low_pre_param, 40, "low_pre", tensor_size=1)
    low_pre_wait = _wait(graph, low_pre_ag, 40, "low_pre")
    low_ag = _allgather(graph, _add(graph, low_param, low_pre_wait, "low_param_dep"), 41, "low")
    low_wait = _wait(graph, low_ag, 41, "low")
    low_use = _neg(graph, low_wait, "low_use", device_time=100)
    low_release = _release(graph, low_use, 41, "low")

    graph.output((high_release, low_release))
    graph.lint()

    names = _scheduled_names(graph)

    assert names.index(low_release.name) < names.index(high_ag.name)


def test_fast_free_schedule_uses_pressure_tiebreaker_in_fallback_bucket():
    graph = Graph()

    high_param = _placeholder(graph, "fallback_high_param")
    high_extra_param = _placeholder(graph, "fallback_high_extra_param")
    low_param = _placeholder(graph, "fallback_low_param")
    low_extra_param = _placeholder(graph, "fallback_low_extra_param")

    high_ag = _allgather(graph, high_param, 50, "fallback_high", tensor_size=100)
    high_wait = _wait(graph, high_ag, 50, "fallback_high")
    high_extra_ag = _allgather(graph, high_extra_param, 51, "fallback_high_extra", tensor_size=10)
    high_extra_wait = _wait(graph, high_extra_ag, 51, "fallback_high_extra")
    high_use = _add(graph, high_wait, high_extra_wait, "fallback_high_use", device_time=1)
    high_release = _release(graph, high_use, 50, "fallback_high")

    low_ag = _allgather(graph, low_param, 60, "fallback_low", tensor_size=1)
    low_wait = _wait(graph, low_ag, 60, "fallback_low")
    low_extra_ag = _allgather(graph, low_extra_param, 61, "fallback_low_extra", tensor_size=10)
    low_extra_wait = _wait(graph, low_extra_ag, 61, "fallback_low_extra")
    low_use = _add(graph, low_wait, low_extra_wait, "fallback_low_use", device_time=100)
    low_release = _release(graph, low_use, 60, "fallback_low")

    graph.output((high_release, low_release))
    graph.lint()

    names = _scheduled_names(graph)

    assert names.index(low_ag.name) < names.index(high_ag.name)


def test_fast_free_schedule_counts_live_gathered_bytes_when_filtering_candidates():
    graph = Graph()

    first_param = _placeholder(graph, "budget_first_param")
    high_param = _placeholder(graph, "budget_high_param")
    low_param = _placeholder(graph, "budget_low_param")

    first_ag = _allgather(graph, first_param, 80, "budget_first", tensor_size=70)
    first_wait = _wait(graph, first_ag, 80, "budget_first")

    high_dep = _add(graph, high_param, first_wait, "budget_high_dep")
    high_ag = _allgather(graph, high_dep, 81, "budget_high", tensor_size=40)
    high_wait = _wait(graph, high_ag, 81, "budget_high")
    high_use = _neg(graph, high_wait, "budget_high_use", device_time=1)
    high_release = _release(graph, high_use, 81, "budget_high")

    low_dep = _add(graph, low_param, first_wait, "budget_low_dep")
    low_ag = _allgather(graph, low_dep, 82, "budget_low", tensor_size=20)
    low_wait = _wait(graph, low_ag, 82, "budget_low")
    low_use = _neg(graph, low_wait, "budget_low_use", device_time=100)
    low_release = _release(graph, low_use, 82, "budget_low")

    high_low_pair = _add(graph, high_wait, low_wait, "budget_high_low_pair")
    first_use = _add(graph, first_wait, high_low_pair, "budget_first_use")
    first_release = _release(graph, first_use, 80, "budget_first")

    graph.output((first_release, high_release, low_release))
    graph.lint()

    no_budget_names = _scheduled_names(graph)
    assert no_budget_names.index(high_ag.name) < no_budget_names.index(low_ag.name)

    budget = schedule_mod.SchedulerMemoryBudget(max_gathered_bytes=100, source="test")
    scheduled_graph = _scheduled_graph(graph, scheduler_budget=budget)
    names = [node.name for node in scheduled_graph.nodes]
    diagnostics = _scheduler_diagnostics(scheduled_graph)

    assert names.index(first_ag.name) < names.index(low_ag.name)
    assert names.index(low_ag.name) < names.index(high_ag.name)
    assert diagnostics["budget_rejections"] > 0
    assert any(record["node"] == high_ag.name for record in diagnostics["budget_rejected_candidates"])


def test_fast_free_schedule_continues_to_higher_count_candidates_when_lowest_count_exceeds_budget():
    graph = Graph()

    first_param = _placeholder(graph, "budget_count_first_param")
    high_param = _placeholder(graph, "budget_count_high_param")
    low_param = _placeholder(graph, "budget_count_low_param")
    extra_param = _placeholder(graph, "budget_count_extra_param")

    first_ag = _allgather(graph, first_param, 100, "budget_count_first", tensor_size=70)
    first_wait = _wait(graph, first_ag, 100, "budget_count_first")

    high_dep = _add(graph, high_param, first_wait, "budget_count_high_dep")
    high_ag = _allgather(graph, high_dep, 101, "budget_count_high", tensor_size=60)
    high_wait = _wait(graph, high_ag, 101, "budget_count_high")
    high_use = _neg(graph, high_wait, "budget_count_high_use")
    high_release = _release(graph, high_use, 101, "budget_count_high")

    low_dep = _add(graph, low_param, first_wait, "budget_count_low_dep")
    low_ag = _allgather(graph, low_dep, 102, "budget_count_low", tensor_size=20)
    low_wait = _wait(graph, low_ag, 102, "budget_count_low")
    extra_dep = _add(graph, extra_param, low_wait, "budget_count_extra_dep")
    extra_ag = _allgather(graph, extra_dep, 103, "budget_count_extra", tensor_size=20)
    extra_wait = _wait(graph, extra_ag, 103, "budget_count_extra")
    low_use = _add(graph, low_wait, extra_wait, "budget_count_low_use")
    low_release = _release(graph, low_use, 102, "budget_count_low")
    extra_release = _release(graph, low_use, 103, "budget_count_extra")

    first_use = _add(graph, first_wait, low_wait, "budget_count_first_use")
    first_release = _release(graph, first_use, 100, "budget_count_first")

    graph.output((first_release, high_release, low_release, extra_release))
    graph.lint()

    budget = schedule_mod.SchedulerMemoryBudget(max_gathered_bytes=120, source="test")
    names = _scheduled_names(graph, scheduler_budget=budget)

    assert names.index(low_ag.name) < names.index(high_ag.name)


def test_fast_free_schedule_counts_padded_allgather_allocation_bytes():
    graph = Graph()

    param = _placeholder(graph, "budget_padded_param")
    ag = _allgather(graph, param, 110, "budget_padded", tensor_size=102, allocation_size=112)
    wait = _wait(graph, ag, 110, "budget_padded")
    use = _neg(graph, wait, "budget_padded_use")
    release = _release(graph, use, 110, "budget_padded")

    graph.output((release, ))
    graph.lint()

    budget = schedule_mod.SchedulerMemoryBudget(max_gathered_bytes=105, source="test")
    scheduled_graph = _scheduled_graph(graph, scheduler_budget=budget)
    diagnostics = _scheduler_diagnostics(scheduled_graph)

    assert diagnostics["budget_overflows"][0]["candidate_allgather_bytes"] == 112
    assert diagnostics["budget_overflows"][0]["over_budget_bytes"] == 7
    assert diagnostics["budget_overflows"][0]["path"] == "until_free"


def test_fast_free_schedule_records_diagnostic_when_no_candidate_fits_budget():
    graph = Graph()

    first_param = _placeholder(graph, "budget_fail_first_param")
    high_param = _placeholder(graph, "budget_fail_high_param")
    low_param = _placeholder(graph, "budget_fail_low_param")

    first_ag = _allgather(graph, first_param, 90, "budget_fail_first", tensor_size=80)
    first_wait = _wait(graph, first_ag, 90, "budget_fail_first")

    high_dep = _add(graph, high_param, first_wait, "budget_fail_high_dep")
    high_ag = _allgather(graph, high_dep, 91, "budget_fail_high", tensor_size=40)
    high_wait = _wait(graph, high_ag, 91, "budget_fail_high")
    high_use = _neg(graph, high_wait, "budget_fail_high_use", device_time=1)
    high_release = _release(graph, high_use, 91, "budget_fail_high")

    low_dep = _add(graph, low_param, first_wait, "budget_fail_low_dep")
    low_ag = _allgather(graph, low_dep, 92, "budget_fail_low", tensor_size=30)
    low_wait = _wait(graph, low_ag, 92, "budget_fail_low")
    low_use = _neg(graph, low_wait, "budget_fail_low_use", device_time=100)
    low_release = _release(graph, low_use, 92, "budget_fail_low", n_users=2)

    first_use = _add(graph, first_wait, low_wait, "budget_fail_first_use")
    first_release = _release(graph, first_use, 90, "budget_fail_first")
    low_last_release = _release(graph, first_use, 92, "budget_fail_low_last", n_users=2)

    graph.output((first_release, high_release, low_release, low_last_release))
    graph.lint()

    budget = schedule_mod.SchedulerMemoryBudget(max_gathered_bytes=100, source="test")
    scheduled_graph = _scheduled_graph(graph, scheduler_budget=budget)
    names = [node.name for node in scheduled_graph.nodes]
    diagnostics = _scheduler_diagnostics(scheduled_graph)

    assert names.index(first_ag.name) < names.index(low_ag.name)
    assert diagnostics["budget_rejections"] > 0
    assert diagnostics["budget_overflows"][0]["source"] == "test"
    assert diagnostics["budget_overflows"][0]["max_gathered_bytes"] == 100
    assert diagnostics["budget_overflows"][0]["over_budget_bytes"] > 0


def test_fast_free_schedule_over_budget_fallback_prefers_lower_peak_before_live_memory():
    graph = Graph()

    first_param = _placeholder(graph, "budget_debt_first_param")
    helper_param = _placeholder(graph, "budget_debt_helper_param")

    first_ag = _allgather(graph, first_param, 120, "budget_debt_first", tensor_size=80)
    first_wait = _wait(graph, first_ag, 120, "budget_debt_first")
    helper_dep = _add(graph, helper_param, first_wait, "budget_debt_helper_dep")
    helper_ag = _allgather(graph, helper_dep, 121, "budget_debt_helper", tensor_size=30)
    helper_wait = _wait(graph, helper_ag, 121, "budget_debt_helper")
    use = _add(graph, first_wait, helper_wait, "budget_debt_use")
    first_release = _release(graph, use, 120, "budget_debt_first")
    helper_release = _release(graph, use, 121, "budget_debt_helper")

    graph.output((first_release, helper_release))
    graph.lint()

    budget = schedule_mod.SchedulerMemoryBudget(max_gathered_bytes=50, source="test")
    scheduled_graph = _scheduled_graph(graph, scheduler_budget=budget)
    diagnostics = _scheduler_diagnostics(scheduled_graph)
    first_selection = diagnostics["selected"][0]

    assert first_selection["path"] == "until_ag"
    assert first_selection["schedule_until_ag_peak_mem"] < first_selection["schedule_until_free_peak_mem"]
    assert first_selection["schedule_until_ag_live_mem"] > first_selection["schedule_until_free_live_mem"]
    assert diagnostics["budget_overflows"][0]["path"] == "until_ag"


def test_over_budget_fallback_prefers_lower_peak_before_ending_residency():
    graph = Graph()
    high_peak_node = _placeholder(graph, "high_peak_zero_residency")
    low_peak_node = _placeholder(graph, "low_peak_nonzero_residency")

    high_peak_task = schedule_mod.AllgatherTask(node=high_peak_node,
                                                allgather_cost=0,
                                                free_cost=0,
                                                allgathered_mem=1000,
                                                allgather_acc_mem=1000,
                                                free_acc_mem=0,
                                                last_use=high_peak_node,
                                                n_scheduled_ags=1,
                                                schedule_until_ag=[high_peak_node],
                                                schedule_until_free=[high_peak_node],
                                                schedule_until_ag_peak_mem=1000,
                                                schedule_until_free_peak_mem=1000,
                                                schedule_until_ag_live_mem=1000,
                                                schedule_until_free_live_mem=0)
    low_peak_task = schedule_mod.AllgatherTask(node=low_peak_node,
                                               allgather_cost=0,
                                               free_cost=0,
                                               allgathered_mem=51,
                                               allgather_acc_mem=51,
                                               free_acc_mem=0,
                                               last_use=low_peak_node,
                                               n_scheduled_ags=1,
                                               schedule_until_ag=[low_peak_node],
                                               schedule_until_free=[low_peak_node],
                                               schedule_until_ag_peak_mem=51,
                                               schedule_until_free_peak_mem=51,
                                               schedule_until_ag_live_mem=51,
                                               schedule_until_free_live_mem=51)

    selected, _ = schedule_mod._select_over_budget_allgather_task([high_peak_task, low_peak_task],
                                                                  schedule_mod.SchedulerMemoryBudget(
                                                                      max_gathered_bytes=50, source="test"))

    assert selected is low_peak_task


def test_candidate_peak_resets_after_overflow_is_released():
    graph = Graph()
    first_param = _placeholder(graph, "historical_overflow_param")
    next_param = _placeholder(graph, "later_fitting_param")
    first_ag = _allgather(graph, first_param, 130, "historical_overflow", allocation_size=100)
    first_release = _release(graph, first_ag, 130, "historical_overflow")
    next_ag = _allgather(graph, next_param, 131, "later_fitting", allocation_size=40)

    tracker = schedule_mod._GatheredParamTracker({130: 1, 131: 1})
    tracker.apply(first_ag)
    tracker.apply(first_release)
    assert tracker.live_bytes == 0
    assert tracker.peak_bytes == 100

    candidate_peak, candidate_live = schedule_mod._simulate_path_stats(tracker, [next_ag])

    assert candidate_peak == 40
    assert candidate_live == 40
    assert schedule_mod._fits_budget(schedule_mod.SchedulerMemoryBudget(max_gathered_bytes=50, source="test"),
                                     candidate_peak)
    assert tracker.live_bytes == 0
    assert tracker.peak_bytes == 100


def test_fast_free_schedule_keeps_single_allgather_release_order():
    graph = Graph()

    param = _placeholder(graph, "param")
    ag = _allgather(graph, param, 70, "single")
    wait = _wait(graph, ag, 70, "single")
    use = _neg(graph, wait, "single_use")
    release = _release(graph, use, 70, "single")

    graph.output((release, ))
    graph.lint()

    names = _scheduled_names(graph)

    assert names.index(ag.name) < names.index(wait.name)
    assert names.index(wait.name) < names.index(use.name)
    assert names.index(use.name) < names.index(release.name)


def test_profile_backfill_makes_partial_profile_safe_for_profile_dependent_passes(monkeypatch):
    graph = Graph()

    param = _placeholder(graph, "partial_profile_param")
    ag = _allgather(graph, param, 90, "partial_profile", device_time=None)
    wait = _wait(graph, ag, 90, "partial_profile")
    use = _neg(graph, wait, "partial_profile_use", device_time=None)
    release = _release(graph, use, 90, "partial_profile")

    ag.meta.pop("tensor_size", None)
    for node in (ag, use):
        node.meta.pop("wall_time", None)
        node.meta.pop("alloc_mem", None)
        node.meta.pop("max_mem", None)

    graph.output((release, ))
    graph.lint()

    _backfill_missing_profile_metadata(graph)
    assert is_profile_incomplete(graph)

    for node in graph.nodes:
        if node in (ag, use):
            assert node.meta["device_time"] == 0.0
        else:
            assert "device_time" in node.meta
        assert "wall_time" in node.meta
        assert "tensor_size" in node.meta
        assert "alloc_mem" in node.meta
        assert "max_mem" in node.meta
    assert ag.meta["tensor_size"] == 0

    names = _scheduled_names(graph)
    assert names.index(ag.name) < names.index(wait.name)
    assert names.index(wait.name) < names.index(use.name)
    assert names.index(use.name) < names.index(release.name)

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 1024

        def available_memory(self):
            return 1024

    fake_ds_param = SimpleNamespace(numel=7,
                                    dtype=torch.float16,
                                    param=SimpleNamespace(ds_persist=False, ds_shape=(1, )))
    fake_param_manager = {
        0: SimpleNamespace(params={"partial_profile_param": fake_ds_param}, ds_ids={"partial_profile_param": 90})
    }
    profiling_results = {
        0: ProfilingResult(fwd_graph=graph, bwd_graph=None, fwd_mem=[("profiled_before_abort", 0, 0, 0)])
    }
    gm = GraphModule(torch.nn.Module(), graph)
    logs = []
    prefetch_logs = []
    persisted = []

    monkeypatch.setattr(prefetch_mod, "print_rank_0", lambda message: prefetch_logs.append(message))
    assert prefetch_mod.schedule_prefetch(gm,
                                          graph_id=0,
                                          graph_order=[(0, True)],
                                          profiling_results=profiling_results,
                                          create_inputs_fn=lambda: (),
                                          mem_budget=0,
                                          param_manager=fake_param_manager,
                                          bwd=False) is gm
    assert any("incomplete profiling data" in message for message in prefetch_logs)

    monkeypatch.setattr(selective_gather_mod, "print_rank_0", lambda message: logs.append(message))
    monkeypatch.setattr(selective_gather_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(selective_gather_mod, "get_deepcompile_handle",
                        lambda: SimpleNamespace(set_persistent=persisted.append))
    monkeypatch.setattr(selective_gather_mod.dist, "all_reduce", lambda *args, **kwargs: None)

    selective_gather_mod.selective_gather(gm,
                                          graph_id=0,
                                          graph_order=[(0, True)],
                                          profiling_results=profiling_results,
                                          create_inputs_fn=lambda: (),
                                          mem_budget=0,
                                          param_manager=fake_param_manager,
                                          bwd=True)
    assert persisted == []
    assert any("incomplete profiling data" in message for message in logs)


def test_schedule_prefetch_skips_when_memory_profile_incomplete(monkeypatch):
    graph = Graph()

    param = _placeholder(graph, "mem_incomplete_param")
    ag = _allgather(graph, param, 91, "mem_incomplete")
    wait = _wait(graph, ag, 91, "mem_incomplete")
    use = _neg(graph, wait, "mem_incomplete_use")
    release = _release(graph, use, 91, "mem_incomplete")

    graph.output((release, ))
    graph.lint()

    profiling_results = {
        0:
        ProfilingResult(fwd_graph=graph,
                        bwd_graph=None,
                        fwd_mem=[("profiled_before_abort", 0, 0, 0)],
                        fwd_mem_complete=False)
    }
    gm = GraphModule(torch.nn.Module(), graph)
    logs = []

    monkeypatch.setattr(prefetch_mod, "print_rank_0", lambda message: logs.append(message))

    assert prefetch_mod.schedule_prefetch(gm,
                                          graph_id=0,
                                          graph_order=[(0, False)],
                                          profiling_results=profiling_results,
                                          create_inputs_fn=lambda: (),
                                          mem_budget=0,
                                          param_manager={},
                                          bwd=False) is gm
    assert gm.graph is graph
    assert any("incomplete profiling data" in message for message in logs)


def test_graphsafe_rng_state_outputs_are_registered_no_reuse():
    graphsafe_run_with_rng_state = inductor_mod._get_graphsafe_run_with_rng_state()
    if graphsafe_run_with_rng_state is None:
        pytest.skip("graphsafe_run_with_rng_state is unavailable in this torch build")

    calls = []

    def fake_register(op_overload, **kwargs):
        calls.append((op_overload, kwargs))

    assert inductor_mod._register_graphsafe_rng_state_no_reuse(fake_register)
    assert calls == [(graphsafe_run_with_rng_state, {"never_reuse_output": True})]


def test_register_custom_ops_includes_graphsafe_rng_state_no_reuse(monkeypatch):
    graphsafe_run_with_rng_state = inductor_mod._get_graphsafe_run_with_rng_state()
    if graphsafe_run_with_rng_state is None:
        pytest.skip("graphsafe_run_with_rng_state is unavailable in this torch build")

    _define_dc_ops()
    registered_ops = []

    def fake_add_needs_realized_inputs(_op_overload):
        return None

    def fake_register_lowering(op_overload, **_kwargs):

        def record_handler(handler):
            registered_ops.append(op_overload)
            return handler

        return record_handler

    monkeypatch.setattr(inductor_mod, "add_needs_realized_inputs", fake_add_needs_realized_inputs)
    monkeypatch.setattr(inductor_mod, "register_lowering", fake_register_lowering)
    monkeypatch.setattr(inductor_mod, "fallbacks", set())
    monkeypatch.setattr(inductor_mod.Scheduler, "is_dc_patched", True, raising=False)

    inductor_mod.register_custom_ops()

    assert graphsafe_run_with_rng_state in registered_ops
