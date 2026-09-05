# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import math
from pathlib import Path
import time

from .core import CommTable, enumerate_candidates, simulate
from .fx_adapter import copy_module, export_graphs, graph_recipe, graph_signature


def search(graphs,
           specs,
           communication,
           resident_bytes,
           limit_bytes,
           optional_passes,
           available_passes,
           max_candidates=100,
           timeout_s=30,
           runtime_memory=None):
    """Only graph copies and table lookups are permitted inside this function."""
    from ..passes.contract import validate_schedule
    from ..passes.prefetch import plan_prefetch

    if set(optional_passes) - set(available_passes):
        raise ValueError('Search allowlist contains an unregistered pass')
    if set(optional_passes) - {'prefetch'}:
        raise ValueError('v0 only has a pure adapter for prefetch')
    table = CommTable(**communication)
    dtype_names = {value['dtype'] for value in specs.values()}
    if len(dtype_names) != 1:
        raise ValueError('v0 prefetch search requires one all-gather dtype')
    dtype = next(iter(dtype_names))
    deadline = time.monotonic() + timeout_s
    candidates = []
    original_signatures = [graph_signature(graph) for graph in graphs]
    for pass_names in enumerate_candidates(optional_passes):
        if len(candidates) >= max_candidates or time.monotonic() > deadline:
            break
        validate_schedule([(0, list(pass_names))])
        modules = [copy_module(graph) for graph in graphs]
        exported, profile = export_graphs([gm.graph for gm in modules], specs, communication, runtime_memory)
        result = simulate(exported, profile, initial_memory_bytes=resident_bytes, memory_limit_bytes=limit_bytes)
        for name in pass_names[1:]:
            if name == 'prefetch':
                for phase, gm in zip(('fw', 'bw'), modules):
                    mem = {
                        row['node'].split('/', 1)[1]: (row['after_bytes'], row['during_peak_bytes'])
                        for row in result['memory_trace'] if row['node'].startswith(phase + '/')
                    }
                    sizes = {
                        node.name: specs[node.args[2]]['bytes']
                        for node in gm.graph.nodes if str(node.target) == 'dc.allgather_param.default'
                    }
                    plan_prefetch(gm, 0, mem, sizes, limit_bytes, lambda size: table.lookup(size, dtype))
                exported, profile = export_graphs([gm.graph for gm in modules], specs, communication, runtime_memory)
                result = simulate(exported,
                                  profile,
                                  initial_memory_bytes=resident_bytes,
                                  memory_limit_bytes=limit_bytes)
        candidates.append({
            'passes': list(pass_names),
            'result': result,
            'recipes': [graph_recipe(gm.graph) for gm in modules],
            'graphs': exported,
            'profile': profile
        })
    feasible = [candidate for candidate in candidates if candidate['result']['feasible']]
    if not feasible:
        raise ValueError('No feasible candidate')
    selected = min(feasible,
                   key=lambda candidate:
                   (round(candidate['result']['estimated_time_ms'], 9), candidate['result']['peak_memory_bytes'],
                    len(candidate['passes']), candidate['passes']))
    if original_signatures != [graph_signature(graph) for graph in graphs]:
        raise RuntimeError('Search mutated the baseline graph')
    return {
        'candidates':
        candidates,
        'selected':
        selected,
        'resident_bytes':
        resident_bytes,
        'memory_limit_bytes':
        limit_bytes,
        'model':
        'serial-no-overlap-v1',
        'truncated':
        len(candidates) < sum(math.perm(len(optional_passes), count) for count in range(len(optional_passes) + 1))
    }


def save_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'search.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
