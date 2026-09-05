# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Portable, JSON-compatible inputs for the initial fixed-shape pass search."""

from bisect import bisect_left
from itertools import permutations
import math


class CommProfileMissing(ValueError):
    pass


class CommTable:
    """Frozen upper-bucket lookup; all sizes are padded gathered output bytes."""

    def __init__(self, header, rows):
        self.header = dict(header)
        self._rows = {}
        for row in rows:
            key = (row['op'], row['dtype'])
            size, duration = int(row['bytes']), float(row['time_ms'])
            if size <= 0 or not math.isfinite(duration) or duration <= 0:
                raise ValueError('Invalid communication profile row')
            if size in self._rows.setdefault(key, {}):
                raise ValueError('Duplicate communication profile row')
            self._rows[key][size] = duration

    def lookup(self, size, dtype, op='all_gather'):
        if size < 0:
            raise ValueError('Negative communication size')
        values = self._rows.get((op, dtype))
        if values is None:
            raise CommProfileMissing(f'Missing communication kind: {op}/{dtype}')
        if size == 0:
            return 0.0
        sizes = sorted(values)
        index = bisect_left(sizes, size)
        if index == len(sizes):
            raise CommProfileMissing(f'Communication size {size} exceeds {sizes[-1]} bytes')
        return values[sizes[index]]

    def to_dict(self):
        return {
            'header':
            dict(self.header),
            'rows': [{
                'op': op,
                'dtype': dtype,
                'bytes': size,
                'time_ms': duration
            } for (op, dtype), values in sorted(self._rows.items()) for size, duration in sorted(values.items())]
        }


def representative_sizes(required_bytes, quantum, minimum=4096):
    if required_bytes <= 0 or quantum <= 0:
        raise ValueError('Positive size and alignment required')
    sizes = []
    size = minimum
    while True:
        aligned = ((size + quantum - 1) // quantum) * quantum
        sizes.append(aligned)
        if aligned >= required_bytes:
            return sizes
        size *= 2


def enumerate_candidates(optional_passes):
    if len(set(optional_passes)) != len(optional_passes) or 'zero3_compile' in optional_passes:
        raise ValueError('Optional passes must be unique and exclude mandatory ZeRO-3')
    for count in range(len(optional_passes) + 1):
        for order in permutations(optional_passes, count):
            yield ('zero3_compile', ) + order


def auto_requested(pass_mode, configured_passes, schedule):
    if pass_mode == 'auto':
        return True
    if schedule is not None:
        return not schedule or any(not passes for _, passes in schedule)
    return configured_passes == []


def simulate(graphs, profile, *, initial_memory_bytes=0, memory_limit_bytes=None):
    """Sum serial operator costs and storage lifetimes without executing a tensor op.

    Graphs contain ordered events. Storage aliases use the same stable key across
    forward/backward; managed all-gather storage is freed by the final release.
    Workspace is transient, and resident state is counted only in initial memory.
    """
    events = [event for graph in graphs for event in graph['events']]
    storage_sizes = profile['storage_bytes']
    table = CommTable(**profile['communication'])
    last_use = {}
    for index, event in enumerate(events):
        for key in event.get('inputs', []) + event.get('outputs', []):
            last_use[key] = index
    live = {}
    gathered = {}
    release_counts = {}
    runtime_bytes = 0
    elapsed, peak = 0.0, initial_memory_bytes
    trace = []
    for index, event in enumerate(events):
        before = initial_memory_bytes + sum(live.values()) + sum(gathered.values()) + runtime_bytes
        op_profile = profile['operators'][event['key']]
        duration = op_profile['time_ms']
        if duration < 0 or not math.isfinite(duration):
            raise ValueError(f'Invalid operator duration: {event["key"]}')
        kind = event.get('kind', 'compute')
        if kind in ('gather', 'prefetch'):
            duration = 0.0
            for param in event['params']:
                param_id = str(param['id'])
                if param_id not in gathered:
                    gathered[param_id] = param['bytes']
                    duration += table.lookup(param['bytes'], param['dtype'])
        elif kind in ('wait', 'release'):
            duration = 0.0
        for key in event.get('outputs', []):
            if key not in live:
                live[key] = storage_sizes[key]
        runtime_bytes = event.get('runtime_buffers_bytes', runtime_bytes)
        during = initial_memory_bytes + sum(live.values()) + sum(gathered.values()) + runtime_bytes
        during += op_profile.get('workspace_bytes', 0)
        if kind == 'release':
            param_id = str(event['param_id'])
            release_counts[param_id] = release_counts.get(param_id, 0) + 1
            if release_counts[param_id] == event['release_count']:
                gathered.pop(param_id, None)
                release_counts.pop(param_id)
        if event.get('clear_runtime_buffers'):
            runtime_bytes = 0
        for key in list(live):
            if last_use[key] == index:
                del live[key]
        after = initial_memory_bytes + sum(live.values()) + sum(gathered.values()) + runtime_bytes
        elapsed += duration
        peak = max(peak, before, during, after)
        trace.append({
            'node': event['key'],
            'time_ms': elapsed,
            'duration_ms': duration,
            'before_bytes': before,
            'during_peak_bytes': during,
            'after_bytes': after
        })
    if gathered or release_counts or runtime_bytes:
        raise ValueError('Unreleased gathered parameter at end of graph pair')
    feasible = memory_limit_bytes is None or peak <= memory_limit_bytes
    return {
        'estimated_time_ms': elapsed,
        'peak_memory_bytes': peak,
        'memory_trace': trace,
        'feasible': feasible,
        'reason': None if feasible else 'memory_limit_exceeded'
    }
