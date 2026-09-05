# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
import pytest

from deepspeed.compile.simulation.core import (CommProfileMissing, CommTable, auto_requested, enumerate_candidates,
                                               representative_sizes, simulate)


def communication():
    return {
        'header': {
            'synthetic': True
        },
        'rows': [{
            'op': 'all_gather',
            'dtype': 'torch.bfloat16',
            'bytes': 8,
            'time_ms': 2.0
        }, {
            'op': 'all_gather',
            'dtype': 'torch.bfloat16',
            'bytes': 16,
            'time_ms': 3.0
        }]
    }


@pytest.mark.parametrize('count,expected', [(0, 1), (1, 2), (2, 5), (4, 65)])
def test_partial_permutations(count, expected):
    candidates = list(enumerate_candidates([str(i) for i in range(count)]))
    assert len(candidates) == len(set(candidates)) == expected
    assert all(value[0] == 'zero3_compile' for value in candidates)


def test_communication_lookup_is_read_only_and_bounded():
    data = communication()
    original = copy.deepcopy(data)
    table = CommTable(**data)
    assert [table.lookup(size, 'torch.bfloat16') for size in (0, 1, 8, 9, 16)] == [0, 2, 2, 3, 3]
    for size, dtype in [(17, 'torch.bfloat16'), (8, 'torch.float32')]:
        with pytest.raises(CommProfileMissing):
            table.lookup(size, dtype)
    assert data == original
    assert len(representative_sizes(1024**3, 4)) == 19
    assert representative_sizes(6 * 1024**2, 4)[-1] == 8 * 1024**2


@pytest.mark.parametrize('mode,passes,schedule,expected', [('fixed', None, None, False),
                                                           ('fixed', ['z3'], None, False), ('fixed', [], None, True),
                                                           ('fixed', ['z3'], [], True),
                                                           ('fixed', ['z3'], [(0, [])], True),
                                                           ('fixed', [], [(0, ['zero3_compile'])], False),
                                                           ('auto', ['z3'], [(0, ['zero3_compile'])], True)])
def test_mode_precedence(mode, passes, schedule, expected):
    assert auto_requested(mode, passes, schedule) is expected


def test_config_preserves_literal_auto_mode():
    from deepspeed.compile.config import CompileConfig
    assert CompileConfig(pass_mode='auto').pass_mode == 'auto'
    assert CompileConfig().pass_mode == 'fixed'
    with pytest.raises(ValueError):
        CompileConfig(pass_mode='unknown')


def test_graph_signature_ignores_local_device_index():
    import torch
    from torch.fx import Graph
    from deepspeed.compile.simulation.fx_adapter import graph_signature
    graphs = []
    for rank in (0, 1):
        graph = Graph()
        value = graph.call_function(torch.ops.aten.empty.memory_format, ([2, 3], ), {
            'device': torch.device(f'cuda:{rank}'),
            'dtype': torch.bfloat16
        })
        graph.output(value)
        graphs.append(graph)
    assert graph_signature(graphs[0]) == graph_signature(graphs[1])
    next(iter(graphs[1].nodes)).kwargs = {'device': torch.device('cuda:1'), 'dtype': torch.float32}
    assert graph_signature(graphs[0]) != graph_signature(graphs[1])


def test_end_backward_dependency_order_is_not_semantic():
    import torch
    from torch.fx import Graph
    from deepspeed.compile.simulation.fx_adapter import graph_signature
    library = torch.library.Library('dc', 'FRAGMENT')
    if not hasattr(torch.ops.dc, 'end_backward'):
        library.define('end_backward(Any deps, int graph_id, bool release_reduce_buckets) -> ()')
    signatures = []
    for reverse in (False, True):
        graph = Graph()
        deps = [graph.placeholder('a'), graph.placeholder('b')]
        graph.call_function(torch.ops.dc.end_backward.default, (deps[::-1] if reverse else deps, 123, True))
        graph.output(None)
        signatures.append(graph_signature(graph))
    assert signatures[0] == signatures[1]


def run_events(graphs, storage, costs=None, limit=None):
    profile = {
        'communication': communication(),
        'storage_bytes': storage,
        'operators': {
            event['key']: {
                'time_ms': 1,
                'workspace_bytes': 0
            }
            for graph in graphs
            for event in graph['events']
        }
    }
    for key, value in (costs or {}).items():
        profile['operators'][key].update(value)
    return simulate(graphs, profile, initial_memory_bytes=100, memory_limit_bytes=limit)


def test_alias_saved_tensor_and_workspace_lifetimes():
    graphs = [{
        'events': [{
            'key': 'fw/allocate',
            'outputs': ['saved']
        }, {
            'key': 'fw/view',
            'inputs': ['saved'],
            'outputs': ['saved']
        }]
    }, {
        'events': [{
            'key': 'bw/use',
            'inputs': ['saved'],
            'outputs': ['grad']
        }, {
            'key': 'bw/output',
            'inputs': ['grad']
        }]
    }]
    result = run_events(graphs, {'saved': 10, 'grad': 5}, {'bw/use': {'workspace_bytes': 7}}, limit=121)
    assert result['estimated_time_ms'] == 4
    assert result['peak_memory_bytes'] == 122
    assert result['memory_trace'][1]['after_bytes'] == 110
    assert result['memory_trace'][2]['after_bytes'] == 105
    assert result['memory_trace'][-1]['after_bytes'] == 100
    assert result['reason'] == 'memory_limit_exceeded'


def test_prefetch_is_per_parameter_and_release_counts_are_respected():
    params = [{'id': 1, 'bytes': 8, 'dtype': 'torch.bfloat16'}, {'id': 2, 'bytes': 8, 'dtype': 'torch.bfloat16'}]
    graph = {
        'events': [{
            'key': 'prefetch',
            'kind': 'prefetch',
            'params': params
        }, {
            'key': 'gather1',
            'kind': 'gather',
            'params': params[:1]
        }, {
            'key': 'wait',
            'kind': 'wait'
        }, {
            'key': 'release1a',
            'kind': 'release',
            'param_id': 1,
            'release_count': 2
        }, {
            'key': 'release1b',
            'kind': 'release',
            'param_id': 1,
            'release_count': 2
        }, {
            'key': 'release2',
            'kind': 'release',
            'param_id': 2,
            'release_count': 1
        }]
    }
    result = run_events([graph], {})
    assert result['estimated_time_ms'] == 4  # two 8-byte gathers, not one 16-byte gather
    assert result['peak_memory_bytes'] == 116
    assert result['memory_trace'][3]['after_bytes'] == 116
    assert result['memory_trace'][4]['after_bytes'] == 108


def test_reordered_allocations_change_peak():
    events = [{
        'key': 'a',
        'outputs': ['a']
    }, {
        'key': 'use_a',
        'inputs': ['a']
    }, {
        'key': 'b',
        'outputs': ['b']
    }, {
        'key': 'use_b',
        'inputs': ['b']
    }]
    sequential = run_events([{'events': events}], {'a': 10, 'b': 20})
    reordered = run_events([{'events': [events[i] for i in [0, 2, 1, 3]]}], {'a': 10, 'b': 20})
    assert sequential['peak_memory_bytes'] == 120
    assert reordered['peak_memory_bytes'] == 130


def test_gradient_storage_and_reduce_buffers_survive_until_flush():
    events = [{
        'key': 'gradient',
        'outputs': ['grad']
    }, {
        'key': 'reduce',
        'inputs': ['grad'],
        'runtime_buffers_bytes': 40
    }, {
        'key': 'end_backward',
        'inputs': ['grad'],
        'clear_runtime_buffers': True
    }]
    result = run_events([{'events': events}], {'grad': 12})
    assert result['peak_memory_bytes'] == 152
    assert result['memory_trace'][1]['after_bytes'] == 152
    assert result['memory_trace'][2]['after_bytes'] == 100


def test_full_search_and_recipe_replay_never_measure(monkeypatch):
    import operator
    import torch
    from torch.fx import Graph
    from deepspeed.compile.passes import prefetch, zero3_compile
    from deepspeed.compile.passes.contract import register_pass_contract
    from deepspeed.compile.simulation.fx_adapter import apply_recipe, copy_module, graph_signature
    from deepspeed.compile.simulation.search import search

    library = torch.library.Library('dc', 'FRAGMENT')
    for name, schema in {
            'allgather_param': '(Tensor x, int graph_id, int ds_id, ScalarType? dtype=None) -> Tensor',
            'wait_allgather': '(Tensor x, int graph_id, int ds_id) -> Tensor',
            'release_param': '(Tensor x, int graph_id, int ds_id, int n_users) -> Tensor',
            'prefetch_params_fused': '(int graph_id, Tensor[] x, int[] ds_ids) -> ()',
            'reload_parameter': '(Tensor x) -> Tensor'
    }.items():
        if not hasattr(torch.ops.dc, name):
            library.define(name + schema)
    graph = Graph()
    p = graph.placeholder('p')
    x = graph.placeholder('x')
    ag = graph.call_function(torch.ops.dc.allgather_param.default, (p, 123, 1), {'dtype': torch.bfloat16})
    wait = graph.call_function(torch.ops.dc.wait_allgather.default, (ag, 123, 1))
    compute = graph.call_function(operator.add, (x, wait))
    graph.call_function(torch.ops.dc.release_param.default, (compute, 123, 1, 1))
    graph.output(compute)
    for node in graph.nodes:
        node.meta.update(device_time=1.0, sim_outputs=[], sim_workspace_bytes=0)
    p.meta['sim_outputs'] = [{'key': 'p:0', 'bytes': 0}]
    x.meta['sim_outputs'] = [{'key': 'x:0', 'bytes': 4}]
    ag.meta['sim_outputs'] = [{'key': 'ag:0', 'bytes': 8}]
    wait.meta['sim_outputs'] = [{'key': 'ag:0', 'bytes': 8}]
    compute.meta['sim_outputs'] = [{'key': 'compute:0', 'bytes': 4}]
    register_pass_contract('zero3_compile', zero3_compile.CONTRACT)
    register_pass_contract('prefetch', prefetch.CONTRACT)

    def forbidden(*args, **kwargs):
        raise AssertionError('Search tried to measure or execute')

    monkeypatch.setattr(prefetch, 'get_accelerator', forbidden)
    monkeypatch.setattr(prefetch, 'create_predictor', forbidden)
    monkeypatch.setattr(torch.fx.Interpreter, 'run', forbidden)
    original = copy.deepcopy([node.meta for node in graph.nodes])
    specs = {1: {'id': 1, 'bytes': 8, 'dtype': 'torch.bfloat16'}}
    result = search([graph, graph], specs, communication(), 100, 1000, ('prefetch', ), ('zero3_compile', 'prefetch'))
    assert len(result['candidates']) == 2
    assert result['candidates'][0]['result']['estimated_time_ms'] == result['candidates'][1]['result'][
        'estimated_time_ms']
    assert original == [node.meta for node in graph.nodes]
    recipe = result['candidates'][1]['recipes'][0]
    assert any('prefetch' in row for row in recipe)
    first, second = copy_module(graph), copy_module(graph)
    apply_recipe(first, 123, recipe)
    apply_recipe(second, 987, recipe)
    assert graph_signature(first.graph) == graph_signature(second.graph)
