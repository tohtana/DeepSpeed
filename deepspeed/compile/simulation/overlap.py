# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Compute/communication overlap with explicit waits and device storage lifetimes."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
import math

from .core import CommTable
from .reduction import ReductionScheduler


def simulate_overlap(graphs, profile, initial_memory_bytes, memory_limit_bytes, serialize=False):
    events = [event for graph in graphs for event in graph['events']]
    table = CommTable(**profile['communication'])
    reduction = ReductionScheduler(table, profile['reduction'], serialize) if 'reduction' in profile else None
    compute, communication = 0.0, 0.0
    serial_time, exposed_wait = 0.0, 0.0
    gathered, storages, intervals, timeline = {}, {}, [], []
    runtime_size, runtime_birth = 0, None

    def interval(key, size, birth, death):
        if size < 0 or death < birth:
            raise ValueError(f'Invalid storage lifetime: {key}')
        if size:
            intervals.append((key, size, birth, death))

    if initial_memory_bytes < 0:
        raise ValueError('Negative resident memory')
    for index, event in enumerate(events):
        key = event['key']
        operator = profile['operators'][key]
        duration = float(operator['time_ms'])
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f'Invalid operator duration: {key}')
        kind = event.get('kind', 'compute')
        launch = compute
        start, end, lane = compute, compute, 'compute'
        if kind in ('gather', 'prefetch'):
            pending = []
            for param in event['params']:
                param_id = str(param['id'])
                if param_id not in gathered:
                    if param_id in [str(p['id']) for p in pending]:
                        raise ValueError('Duplicate parameter in gather group')
                    pending.append(param)
            duration = sum(table.lookup(param['bytes'], param['dtype']) for param in pending)
            if pending:
                start = max(compute, communication)
                end = communication = start + duration
                # C++ records every member's completion event after ncclGroupEnd.
                # Even the first member must wait for the entire fused group.
                for param in pending:
                    gathered[str(param['id'])] = {
                        'bytes': param['bytes'],
                        'ready': end,
                        'birth': (launch, 2 * index),
                        'releases': 0,
                        'release_count': None
                    }
            lane = 'all_gather'
        elif kind == 'wait':
            param_id = str(event['param_id'])
            if param_id not in gathered:
                raise ValueError(f'Wait without a live gather: {param_id}')
            end = compute = max(compute, gathered[param_id]['ready'])
            exposed_wait += end - start
            duration, lane = 0.0, 'compute_wait'
        elif kind == 'release':
            param_id = str(event['param_id'])
            if param_id not in gathered:
                raise ValueError(f'Release without a live gather: {param_id}')
            param = gathered[param_id]
            count = event['release_count']
            if count <= 0 or param['release_count'] not in (None, count):
                raise ValueError('Inconsistent release count')
            param['release_count'] = count
            param['releases'] += 1
            if param['releases'] == count:
                # record_stream prevents allocator reuse until queued consumers
                # finish, including a gather that was released before its wait.
                death = (max(compute, param['ready']), 2 * index + 1)
                interval(f'parameter/{param_id}/{index}', param['bytes'], param['birth'], death)
                del gathered[param_id]
            duration = 0.0
        elif kind == 'barrier':
            # Baseline reduce timings include copy/flush work that is not split
            # into per-stream profiles yet. Do not invent overlap for that work.
            start = max(compute, communication)
            end = compute = communication = start + duration
            lane = 'synchronized'
        elif kind in ('reduce_grad', 'end_backward'):
            if reduction is None:
                raise ValueError('Missing reduction configuration')
            compute, communication, duration = reduction.run(event, compute, communication, index)
            end, lane = compute, 'reduction_enqueue' if kind == 'reduce_grad' else 'compute_wait'
        elif kind == 'compute':
            end = compute = start + duration
        else:
            raise ValueError(f'Unsupported overlap event: {kind}')
        if serialize:
            # New event profiles can also be evaluated with identical costs
            # and no overlap. Saved v0/v1 inputs retain their original path.
            compute = max(compute, communication, reduction.copy_ms if reduction else 0,
                          reduction.reduce_ms if reduction else 0)
            end = max(end, compute)
        serial_time += duration
        birth = (launch, 2 * index)
        death = (end, -1) if end > launch else (end, 2 * index + 1)
        for storage in event.get('outputs', []):
            if storage not in storages:
                storages[storage] = [birth, death]
        for storage in event.get('inputs', []) + event.get('outputs', []):
            if storage not in storages:
                raise ValueError(f'Storage used before allocation: {storage}')
            storages[storage][1] = max(storages[storage][1], death)
        interval(f'workspace/{key}', operator.get('workspace_bytes', 0), birth, death)
        if 'runtime_buffers_bytes' in event and event['runtime_buffers_bytes'] != runtime_size:
            if runtime_birth is not None:
                interval(f'runtime/{index}', runtime_size, runtime_birth, birth)
            runtime_size = event['runtime_buffers_bytes']
            runtime_birth = birth
        if event.get('clear_runtime_buffers'):
            if runtime_birth is not None:
                interval(f'runtime/{index}', runtime_size, runtime_birth, death)
            runtime_size, runtime_birth = 0, None
        timeline.append({
            'node': key,
            'kind': kind,
            'lane': lane,
            'launch_ms': launch,
            'start_ms': start,
            'end_ms': end,
            'compute_frontier_ms': compute,
            'duration_ms': duration
        })
    if gathered or runtime_size or (reduction and reduction.buckets):
        raise ValueError('Unreleased runtime storage at end of graph pair')
    if reduction:
        for key, death in reduction.retained_until.items():
            if key not in storages:
                raise ValueError(f'Gradient storage used before allocation: {key}')
            storages[key][1] = max(storages[key][1], death)
        for args in reduction.intervals:
            interval(*args)
    for key, (birth, death) in storages.items():
        interval(key, profile['storage_bytes'][key], birth, death)

    changes = defaultdict(lambda: {'allocate_bytes': 0, 'free_bytes': 0})
    for _, size, birth, death in intervals:
        changes[birth]['allocate_bytes'] += size
        changes[death]['free_bytes'] += size
    points, levels, memory_timeline = [], [], []
    current, peak = initial_memory_bytes, initial_memory_bytes
    for point, change in sorted(changes.items()):
        current += change['allocate_bytes'] - change['free_bytes']
        peak = max(peak, current)
        points.append(point)
        levels.append(current)
        memory_timeline.append({'time_ms': point[0], 'sequence': point[1], 'memory_bytes': current, **change})

    def level_before(point):
        index = bisect_left(points, point) - 1
        return levels[index] if index >= 0 else initial_memory_bytes

    trace = []
    for index, row in enumerate(timeline):
        left = (row['launch_ms'], 2 * index)
        right = (max(row['end_ms'], row['compute_frontier_ms']), 2 * index + 1)
        first, last = bisect_left(points, left), bisect_right(points, right)
        before = level_before(left)
        after_index = bisect_right(points, (row['compute_frontier_ms'], 2 * index + 1)) - 1
        after = levels[after_index] if after_index >= 0 else initial_memory_bytes
        trace.append({
            'node': row['node'],
            'time_ms': row['compute_frontier_ms'],
            'duration_ms': row['duration_ms'],
            'before_bytes': before,
            'during_peak_bytes': max([before, *levels[first:last]]),
            'after_bytes': after
        })
    feasible = memory_limit_bytes is None or peak <= memory_limit_bytes
    elapsed = max(compute, communication)
    result = {
        'estimated_time_ms':
        elapsed,
        'peak_memory_bytes':
        peak,
        'memory_trace':
        trace,
        'feasible':
        feasible,
        'reason':
        None if feasible else 'memory_limit_exceeded',
        'timeline':
        timeline,
        'memory_timeline':
        memory_timeline,
        'serial_work_ms':
        serial_time,
        'overlapped_ms':
        max(0.0, serial_time - elapsed),
        'exposed_gather_wait_ms':
        exposed_wait,
        'model':
        'compute-all-gather-overlap-v1',
        'assumptions': [
            'one_compute_stream', 'one_serial_all_gather_stream', 'fused_group_completion', 'synchronized_reduce_cost',
            'no_host_dispatch_or_bandwidth_contention_model'
        ]
    }
    if reduction:
        result.update(model='compute-gather-reduce-serial-v2' if serialize else 'compute-gather-reduce-overlap-v2',
                      reduction_timeline=reduction.timeline,
                      exposed_reduce_wait_ms=sum(reduction.waits.values()),
                      reduction_waits_ms=dict(reduction.waits),
                      assumptions=[
                          'one_compute_stream', 'one_copy_stream', 'one_all_gather_stream', 'one_reduce_stream',
                          'shared_nccl_communicator_serial_order', 'fused_group_completion', 'bucket_capacity_flush',
                          'one_gradient_dtype_no_accumulation', 'strided_copy_uses_contiguous_copy_cost',
                          'no_host_dispatch_or_bandwidth_contention_model'
                      ])
    return result
