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


@pytest.mark.parametrize('simulation_mode', ['serial', 'overlap'])
def test_full_search_and_recipe_replay_never_measure(monkeypatch, simulation_mode):
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
    result = search([graph, graph],
                    specs,
                    communication(),
                    100,
                    1000, ('prefetch', ), ('zero3_compile', 'prefetch'),
                    simulation_mode=simulation_mode)
    assert result['simulation_mode'] == simulation_mode
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

    # Two independent transfers can overlap intervening compute when grouping
    # is disabled. Serial search must still choose the lower-memory baseline.
    monkeypatch.setattr(prefetch, 'MAX_FUSE_SIZE', 1)
    second_graph = Graph()
    p1, p2, x = [second_graph.placeholder(name) for name in ('p1', 'p2', 'x')]
    a = second_graph.call_function(torch.ops.dc.allgather_param.default, (p1, 123, 1), {'dtype': torch.bfloat16})
    w1 = second_graph.call_function(torch.ops.dc.wait_allgather.default, (a, 123, 1))
    c1 = second_graph.call_function(operator.add, (x, w1))
    second_graph.call_function(torch.ops.dc.release_param.default, (c1, 123, 1, 1))
    b = second_graph.call_function(torch.ops.dc.allgather_param.default, (p2, 123, 2), {'dtype': torch.bfloat16})
    w2 = second_graph.call_function(torch.ops.dc.wait_allgather.default, (b, 123, 2))
    c2 = second_graph.call_function(operator.add, (c1, w2))
    second_graph.call_function(torch.ops.dc.release_param.default, (c2, 123, 2, 1))
    second_graph.output(c2)
    for node in second_graph.nodes:
        node.meta.update(device_time=0.0, sim_outputs=[], sim_workspace_bytes=0)
    for node, storage, size in [(x, 'x', 4), (a, 'a', 8), (w1, 'a', 8), (b, 'b', 16), (w2, 'b', 16), (c1, 'c1', 4),
                                (c2, 'c2', 4)]:
        node.meta['sim_outputs'] = [{'key': storage, 'bytes': size}]
    c1.meta.update(device_time=5.0, sim_compute_time_ms=5.0)
    c2.meta.update(device_time=7.0, sim_compute_time_ms=7.0)
    specs[2] = {'id': 2, 'bytes': 16, 'dtype': 'torch.bfloat16'}
    selected = search([second_graph, second_graph],
                      specs,
                      communication(),
                      100,
                      1000, ('prefetch', ), ('zero3_compile', 'prefetch'),
                      simulation_mode=simulation_mode)
    assert selected['selected']['passes'] == (['zero3_compile', 'prefetch']
                                              if simulation_mode == 'overlap' else ['zero3_compile'])
    assert selected['selected']['result']['estimated_time_ms'] == (28 if simulation_mode == 'overlap' else 34)


def overlap_fixture(prefetch=False, compute_ms=5):
    first = {'id': 1, 'bytes': 8, 'dtype': 'torch.bfloat16'}
    second = {'id': 2, 'bytes': 16, 'dtype': 'torch.bfloat16'}
    events = [
        {
            'key': 'g1',
            'kind': 'gather',
            'params': [first]
        },
        {
            'key': 'w1',
            'kind': 'wait',
            'param_id': 1
        },
        {
            'key': 'a',
            'outputs': ['activation']
        },
        {
            'key': 'r1',
            'kind': 'release',
            'param_id': 1,
            'release_count': 1
        },
        {
            'key': 'g2',
            'kind': 'gather',
            'params': [second]
        },
        {
            'key': 'w2',
            'kind': 'wait',
            'param_id': 2
        },
        {
            'key': 'b',
            'inputs': ['activation']
        },
        {
            'key': 'r2',
            'kind': 'release',
            'param_id': 2,
            'release_count': 1
        },
    ]
    if prefetch:
        events.insert(2, {'key': 'pf2', 'kind': 'prefetch', 'params': [second]})
    profile = {
        'communication': communication(),
        'storage_bytes': {
            'activation': 20
        },
        'operators': {
            event['key']: {
                'time_ms': 0
            }
            for event in events
        }
    }
    profile['operators']['a']['time_ms'] = compute_ms
    profile['operators']['b']['time_ms'] = 7
    return [{'events': events}], profile


@pytest.mark.parametrize('compute_ms,hidden', [(5, 3), (1, 1), (0, 0)])
def test_prefetch_hides_only_independent_compute_and_keeps_storage_live(compute_ms, hidden):
    # Missing wait dependencies would hide all communication; freeing at launch
    # would incorrectly admit the prefetch candidate under the 140-byte budget.
    baseline = simulate(*overlap_fixture(False, compute_ms),
                        initial_memory_bytes=100,
                        memory_limit_bytes=140,
                        mode='overlap')
    prefetched = simulate(*overlap_fixture(True, compute_ms),
                          initial_memory_bytes=100,
                          memory_limit_bytes=140,
                          mode='overlap')
    assert baseline['estimated_time_ms'] == 2 + compute_ms + 3 + 7
    assert prefetched['estimated_time_ms'] == baseline['estimated_time_ms'] - hidden
    assert prefetched['overlapped_ms'] == hidden
    assert prefetched['exposed_gather_wait_ms'] == 5 - hidden
    assert baseline['peak_memory_bytes'] == 136
    assert prefetched['peak_memory_bytes'] == 144
    assert baseline['feasible'] and not prefetched['feasible']
    assert prefetched['memory_timeline'][-1]['memory_bytes'] == 100
    assert [row['time_ms'] for row in prefetched['memory_timeline']
            ] == sorted(row['time_ms'] for row in prefetched['memory_timeline'])


def test_fused_group_waits_for_all_members_and_shared_stream_serializes_groups():
    # C++ records completion after ncclGroupEnd, not after each parameter call.
    graphs, profile = overlap_fixture()
    events = graphs[0]['events']
    first, second = events[0]['params'][0], events[4]['params'][0]
    events[0] = {'key': 'g1', 'kind': 'prefetch', 'params': [first, second]}
    grouped = simulate(graphs, profile, mode='overlap')
    assert grouped['timeline'][1]['end_ms'] == 5
    assert grouped['overlapped_ms'] == 0
    events[0] = {'key': 'g1', 'kind': 'gather', 'params': [first]}
    events.insert(1, {'key': 'pf2', 'kind': 'prefetch', 'params': [second]})
    profile['operators']['pf2'] = {'time_ms': 0}
    separate = simulate(graphs, profile, mode='overlap')
    assert separate['timeline'][1]['start_ms'] == 2
    assert separate['timeline'][1]['end_ms'] == 5
    assert separate['estimated_time_ms'] == grouped['estimated_time_ms'] - 3


def test_pending_gather_survives_release_and_regather_allocates_another_buffer():
    param = {'id': 1, 'bytes': 8, 'dtype': 'torch.bfloat16'}
    events = [
        {
            'key': 'g1',
            'kind': 'gather',
            'params': [param]
        },
        {
            'key': 'r1',
            'kind': 'release',
            'param_id': 1,
            'release_count': 1
        },
        {
            'key': 'g2',
            'kind': 'gather',
            'params': [param]
        },
        {
            'key': 'r2',
            'kind': 'release',
            'param_id': 1,
            'release_count': 1
        },
    ]
    profile = {
        'communication': communication(),
        'storage_bytes': {},
        'operators': {
            event['key']: {
                'time_ms': 0
            }
            for event in events
        }
    }
    result = simulate([{'events': events}], profile, mode='overlap')
    assert result['estimated_time_ms'] == 4
    assert result['peak_memory_bytes'] == 16
    assert result['memory_trace'][1]['after_bytes'] == 8
    assert result['memory_timeline'][-1]['memory_bytes'] == 0


def test_overlap_barriers_and_workspace_account_for_inflight_communication():
    graphs, profile = overlap_fixture(True)
    graphs[0]['events'].insert(3, {'key': 'reduce', 'kind': 'barrier', 'runtime_buffers_bytes': 40})
    graphs[0]['events'].append({'key': 'end', 'kind': 'barrier', 'clear_runtime_buffers': True})
    profile['operators'].update(reduce={'time_ms': 4, 'workspace_bytes': 10}, end={'time_ms': 1})
    result = simulate(graphs, profile, initial_memory_bytes=100, mode='overlap')
    reduce = next(row for row in result['timeline'] if row['node'] == 'reduce')
    assert reduce['start_ms'] == 5 and reduce['end_ms'] == 9
    assert result['estimated_time_ms'] == 22
    assert result['overlapped_ms'] == 0
    assert result['peak_memory_bytes'] == 184
    assert result['memory_timeline'][-1]['memory_bytes'] == 100


def test_overlap_rejects_missing_wait_target_and_missing_profile_without_execution():
    graphs, profile = overlap_fixture()
    graphs[0]['events'][1]['param_id'] = 99
    with pytest.raises(ValueError, match='Wait without'):
        simulate(graphs, profile, mode='overlap')
    graphs, profile = overlap_fixture()
    profile['communication']['rows'] = profile['communication']['rows'][:1]
    with pytest.raises(CommProfileMissing):
        simulate(graphs, profile, mode='overlap')
    with pytest.raises(ValueError, match='Unknown simulation mode'):
        simulate(graphs, profile, mode='unknown')


def test_empty_overlap_graph_and_zero_duration_workspace():
    empty = {'operators': {}, 'storage_bytes': {}, 'communication': communication()}
    result = simulate([], empty, initial_memory_bytes=10, mode='overlap')
    assert result['estimated_time_ms'] == 0 and result['peak_memory_bytes'] == 10
    events = [{'key': 'a'}, {'key': 'b'}]
    empty['operators'] = {key: {'time_ms': 0, 'workspace_bytes': 20} for key in ('a', 'b')}
    result = simulate([{'events': events}], empty, initial_memory_bytes=10, mode='overlap')
    assert result['peak_memory_bytes'] == 30
    assert result['memory_timeline'][-1]['memory_bytes'] == 10


def reduction_fixture(count=3, bucket_numel=8, double_buffer=True, compute_ms=2, storage_dtype='torch.bfloat16'):
    events, operators, sizes = [], {}, {}
    for i in range(count):
        storage, compute, reduce = f'grad{i}', f'compute{i}', f'reduce{i}'
        events.append({'key': compute, 'outputs': [storage]})
        sizes[storage] = 8
        operators[compute] = {'time_ms': compute_ms}
        events.append({
            'key': reduce,
            'kind': 'reduce_grad',
            'inputs': [storage],
            'gradient': {
                'id': i,
                'numel': 4,
                'bytes': 8,
                'dtype': 'torch.bfloat16',
                'itemsize': 2,
                'shard_bytes': 4,
                'storage_dtype': storage_dtype,
                'stored_shard_bytes': 8 if storage_dtype == 'torch.float32' else 4,
                'contiguous': True
            }
        })
        operators[reduce] = {'time_ms': 999}  # measured barriers must not be added a second time
    events.append({'key': 'end', 'kind': 'end_backward', 'clear_runtime_buffers': True})
    operators['end'] = {'time_ms': 999}
    table = communication()
    for op, cost in [('copy', 1), ('pre_divide', 0.5), ('reduce_scatter', 3), ('store:torch.float32', 0.75)]:
        for size in (8, 32):
            table['rows'].append({'op': op, 'dtype': 'torch.bfloat16', 'bytes': size, 'time_ms': cost})
    profile = {
        'operators': operators,
        'storage_bytes': sizes,
        'communication': table,
        'reduction': {
            'reduce_bucket_numel': bucket_numel,
            'double_buffer': double_buffer
        }
    }
    return [{'events': events}], profile


@pytest.mark.parametrize('bucket_numel,expected', [(8, 16.5), (12, 17.5)])
def test_reduce_flushes_on_overflow_not_exact_fill_and_drains_tail(bucket_numel, expected):
    # Each gradient takes compute=2, copy=1, pre-divide=.5, NCCL=3 ms.
    # With capacity 8, two gradients fit exactly: the third triggers their group.
    graphs, profile = reduction_fixture(bucket_numel=bucket_numel)
    original = copy.deepcopy((graphs, profile))
    result = simulate(graphs, profile, mode='overlap')
    groups = [row for row in result['reduction_timeline'] if row['kind'] == 'reduce_scatter']
    assert result['estimated_time_ms'] == expected
    assert result['serial_work_ms'] == 19.5
    assert result['overlapped_ms'] == 19.5 - expected
    if bucket_numel == 8:
        assert [(row['node'], row['param_ids'], row['start_ms'], row['end_ms'])
                for row in groups] == [('reduce2', [0, 1], 7, 13), ('end', [2], 13.5, 16.5)]
    else:
        assert [(row['node'], row['param_ids']) for row in groups] == [('end', [0, 1, 2])]
    serial = simulate(graphs, profile, mode='serial')
    assert serial['estimated_time_ms'] == serial['serial_work_ms'] == 19.5
    assert serial['overlapped_ms'] == 0
    assert result['memory_timeline'][-1]['memory_bytes'] == 0
    assert (graphs, profile) == original


@pytest.mark.parametrize('double_buffer,elapsed,reuse_wait', [(True, 18, 3), (False, 23, 10.5)])
def test_reduction_waits_for_buffer_reuse_and_copy_cleanup(double_buffer, elapsed, reuse_wait):
    # Four gradients cycle through the buffers, exposing reuse of an in-flight
    # slot even with double buffering. Omitting the event wait finishes too early.
    result = simulate(*reduction_fixture(count=4, bucket_numel=4, double_buffer=double_buffer), mode='overlap')
    assert result['estimated_time_ms'] == elapsed
    assert result['reduction_waits_ms']['bucket_reuse_wait'] == reuse_wait
    assert result['reduction_waits_ms']['copy_cleanup_wait'] == 1
    assert result['reduction_waits_ms']['end_backward_wait'] == elapsed - (12 if double_buffer else 19.5)
    assert result['serial_work_ms'] == 26


def test_reduction_keeps_gradients_until_flush_and_temporary_receive_until_store():
    # Three 8-byte gradients coexist before the overflow flush, along with
    # two 16-byte buckets. Freeing gradients at reduce_grad would miss this peak.
    graphs, profile = reduction_fixture(compute_ms=0)
    result = simulate(graphs, profile, initial_memory_bytes=100, memory_limit_bytes=155, mode='overlap')
    assert result['peak_memory_bytes'] == 156
    assert not result['feasible']
    assert result['reduction_waits_ms']['copy_cleanup_wait'] == 3
    assert result['memory_timeline'][-1]['memory_bytes'] == 100
    # FP32 storage needs an 8-byte receive buffer for two BF16 shards and
    # an 8-byte transient cast per shard. Both are released after the RS lane.
    graphs, profile = reduction_fixture(count=2, storage_dtype='torch.float32')
    result = simulate(graphs, profile, mode='overlap')
    assert result['estimated_time_ms'] == 13.5
    assert result['peak_memory_bytes'] == 56
    assert any(row['time_ms'] == 12 and row['memory_bytes'] == 48 for row in result['memory_timeline'])
    assert result['memory_timeline'][-1]['memory_bytes'] == 0


def test_large_gradient_grows_only_current_slot_and_contiguous_copy_has_a_cost():
    graphs, profile = reduction_fixture(count=1, bucket_numel=4)
    grad = graphs[0]['events'][1]['gradient']
    grad.update(numel=12, bytes=24, shard_bytes=12, stored_shard_bytes=12)
    profile['storage_bytes']['grad0'] = 24
    result = simulate(graphs, profile, mode='overlap')
    # During reserve: input 24 + old slot 8 + replacement 24 + shadow 8.
    assert result['peak_memory_bytes'] == 64
    assert result['memory_timeline'][-1]['memory_bytes'] == 0
    grad['contiguous'] = False
    contiguous = simulate(graphs, profile, mode='overlap')
    assert contiguous['estimated_time_ms'] == result['estimated_time_ms'] + 1
    assert contiguous['serial_work_ms'] == result['serial_work_ms'] + 1
    assert contiguous['memory_timeline'][-1]['memory_bytes'] == 0


def test_all_gather_and_reduce_scatter_share_communicator_order():
    # The first reduce group must follow a pending gather on the shared NCCL
    # communicator. Reduction pre-division can run while that gather is in flight.
    graphs, profile = reduction_fixture(count=2, bucket_numel=4, compute_ms=0)
    events = graphs[0]['events']
    events.insert(2, {
        'key': 'gather',
        'kind': 'gather',
        'params': [{
            'id': 100,
            'bytes': 16,
            'dtype': 'torch.bfloat16'
        }]
    })
    events.insert(-1, {'key': 'release', 'kind': 'release', 'param_id': 100, 'release_count': 1})
    profile['operators'].update(gather={'time_ms': 0}, release={'time_ms': 0})
    result = simulate(graphs, profile, mode='overlap')
    groups = [row for row in result['reduction_timeline'] if row['kind'] == 'reduce_scatter']
    assert groups[0]['start_ms'] == 3 and groups[0]['end_ms'] == 6
    assert result['estimated_time_ms'] == 9.5
    assert result['memory_timeline'][-1]['memory_bytes'] == 0
    events.insert(-1, {
        'key': 'gather2',
        'kind': 'gather',
        'params': [{
            'id': 101,
            'bytes': 8,
            'dtype': 'torch.bfloat16'
        }]
    })
    events.insert(-1, {'key': 'release2', 'kind': 'release', 'param_id': 101, 'release_count': 1})
    profile['operators'].update(gather2={'time_ms': 0}, release2={'time_ms': 0})
    later_gather = simulate(graphs, profile, mode='overlap')
    gather_row = next(row for row in later_gather['timeline'] if row['node'] == 'gather2')
    assert gather_row['start_ms'] == 6 and gather_row['end_ms'] == 8
    assert later_gather['estimated_time_ms'] == 11
    assert simulate(graphs, profile, mode='serial')['estimated_time_ms'] == later_gather['serial_work_ms']


def test_reduction_missing_cost_or_incomplete_flush_is_an_error():
    graphs, profile = reduction_fixture()
    profile['communication']['rows'] = [
        row for row in profile['communication']['rows'] if row['op'] != 'reduce_scatter'
    ]
    with pytest.raises(CommProfileMissing):
        simulate(graphs, profile, mode='overlap')
    graphs, profile = reduction_fixture()
    graphs[0]['events'].pop()
    with pytest.raises(ValueError, match='Unreleased runtime storage'):
        simulate(graphs, profile, mode='overlap')


def test_profile_requests_cover_copy_reduction_and_storage_conversion():
    from deepspeed.compile.simulation.gpu_profile import profile_requests
    graphs, _ = reduction_fixture(storage_dtype='torch.float32')
    gradients = {event['gradient']['id']: event['gradient'] for event in graphs[0]['events'] if 'gradient' in event}
    specs = {1: {'id': 1, 'bytes': 8192, 'dtype': 'torch.bfloat16'}}
    requests = profile_requests(specs, 2, 1, gradients)
    assert {(row['op'], row['bytes'])
            for row in requests} == {('all_gather', 4096), ('all_gather', 8192), ('copy', 4096), ('pre_divide', 4096),
                                     ('reduce_scatter', 4096), ('store:torch.float32', 4096)}
    table = CommTable({}, [dict(row, time_ms=1) for row in requests])
    for grad in gradients.values():
        for op in ('copy', 'pre_divide', 'reduce_scatter'):
            assert table.lookup(grad['bytes'], grad['dtype'], op) == 1
        assert table.lookup(grad['shard_bytes'], grad['dtype'], 'store:torch.float32') == 1


@pytest.mark.parametrize('mode,elapsed', [('overlap', 12), ('serial', 13)])
def test_fx_reduction_export_uses_frozen_costs_not_baseline_barriers(monkeypatch, mode, elapsed):
    import torch
    from torch.fx import Graph
    from deepspeed.compile.passes import zero3_compile
    from deepspeed.compile.passes.contract import register_pass_contract
    from deepspeed.compile.simulation.search import search
    from deepspeed.compile.simulation import gpu_profile
    library = torch.library.Library('dc', 'FRAGMENT')
    for name, schema in {
            'reduce_grad': '(Tensor grad, int graph_id, int ds_id) -> Tensor',
            'end_backward': '(Any deps, int graph_id, bool release_reduce_buckets) -> ()'
    }.items():
        if not hasattr(torch.ops.dc, name):
            library.define(name + schema)
    fw, bw = Graph(), Graph()
    fw.output(None)
    reductions = []
    specs = {}
    for i in range(2):
        src = bw.placeholder(f'source{i}')
        grad = bw.call_function(torch.ops.aten.clone.default, (src, ))
        grad.meta.update(val=torch.empty(4, dtype=torch.bfloat16),
                         device_time=50,
                         sim_compute_time_ms=2,
                         sim_outputs=[{
                             'key': f'grad{i}',
                             'bytes': 8
                         }])
        reductions.append(bw.call_function(torch.ops.dc.reduce_grad.default, (grad, 123, i)))
        specs[i] = {'id': i, 'dtype': 'torch.bfloat16', 'bytes': 8}
    bw.call_function(torch.ops.dc.end_backward.default, (reductions, 123, True))
    bw.output(None)
    for graph in (fw, bw):
        for node in graph.nodes:
            node.meta.setdefault('sim_outputs', [])
            node.meta.setdefault('device_time', 999 if str(node.target).startswith('dc.') else 0)
    _, profile = reduction_fixture(count=2)
    runtime = {
        'reduce_bucket_numel': 8,
        'double_buffer': True,
        'world_size': 2,
        'gradient_storage_dtypes': {
            '0': 'torch.bfloat16',
            '1': 'torch.bfloat16'
        }
    }
    register_pass_contract('zero3_compile', zero3_compile.CONTRACT)

    def forbidden(*args, **kwargs):
        raise AssertionError('Search executed a graph or benchmark')

    monkeypatch.setattr(torch.fx.Interpreter, 'run', forbidden)
    monkeypatch.setattr(gpu_profile, 'prepare_table', forbidden)
    result = search([fw, bw],
                    specs,
                    profile['communication'],
                    0,
                    1000, (), ('zero3_compile', ),
                    runtime_memory=runtime,
                    simulation_mode=mode)
    assert result['selected']['result']['estimated_time_ms'] == elapsed
    assert result['selected']['result']['serial_work_ms'] == 13
    assert result['selected']['profile']['reduction'] == profile['reduction']
    costs = result['selected']['profile']['operators']
    assert all(costs[event['key']]['time_ms'] == 0 for event in result['selected']['graphs'][1]['events']
               if event.get('kind') in ('reduce_grad', 'end_backward'))
