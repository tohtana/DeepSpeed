# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Device events for the ZeRO-3 reduceGrad/flushReduceBucket implementation."""

from collections import defaultdict


class ReductionScheduler:
    """One copy stream, one reduction stream and explicit bucket ownership.

    Communication shares the caller's NCCL frontier with all-gather. Costs are
    frozen table lookups; no tensor or distributed runtime is imported here.
    """

    def __init__(self, table, config, serialize=False):
        self.table = table
        self.serialize = serialize
        self.capacity = int(config['reduce_bucket_numel'])
        if self.capacity <= 0:
            raise ValueError('Positive reduction bucket capacity required')
        self.buffer_count = 2 if config['double_buffer'] else 1
        self.buckets = {}
        self.copy_ms = self.reduce_ms = self.communication_ms = 0.0
        self.timeline, self.intervals = [], []
        self.retained_until = {}
        self.waits = defaultdict(float)
        self.seen = set()
        self.work_ms = 0.0

    def _point(self, time, complete=False):
        return (time, -1) if complete and time > self.launch else (time, 2 * self.index + int(complete))

    def _work(self, kind, lane, start, duration, **details):
        end = start + duration
        self.work_ms += duration
        self.timeline.append({
            'node': self.node,
            'kind': kind,
            'lane': lane,
            'start_ms': start,
            'end_ms': end,
            'duration_ms': duration,
            **details
        })
        return end

    def _wait(self, compute, until, reason):
        end = max(compute, until)
        self.waits[reason] += end - compute
        if end > compute:
            self.timeline.append({
                'node': self.node,
                'kind': reason,
                'lane': 'compute_wait',
                'start_ms': compute,
                'end_ms': end,
                'duration_ms': end - compute
            })
        return end

    def _flush(self, bucket, compute, reason):
        tasks = bucket['pending']
        if not tasks:
            return compute
        dtype = bucket['dtype']
        buffer = bucket['buffers'][bucket['index']]
        copied = max(task['copied_ms'] for task in tasks)
        start = max(compute, self.reduce_ms, copied)
        # Temporary receive storage is allocated before launching pre-division.
        temporary = sum(task['shard_bytes'] for task in tasks if task['storage_dtype'] != dtype)
        temp_birth = self._point(compute)
        for task in tasks:
            cost = self.table.lookup(task['bytes'], dtype, 'pre_divide')
            start = self._work('pre_divide', 'reduce', start, cost, param_id=task['id'])
        start = max(start, self.communication_ms)
        # The runtime groups per-parameter calls, rather than communicating one
        # concatenated bucket. All members complete at the end of the group.
        cost = sum(self.table.lookup(task['bytes'], dtype, 'reduce_scatter') for task in tasks)
        start = self._work('reduce_scatter',
                           'reduce',
                           start,
                           cost,
                           param_ids=[task['id'] for task in tasks],
                           bytes=sum(task['bytes'] for task in tasks),
                           bucket_index=bucket['index'],
                           reason=reason)
        self.communication_ms = start
        for task in tasks:
            if task['storage_dtype'] != dtype:
                cast_birth = self._point(start)
                cost = self.table.lookup(task['shard_bytes'], dtype, f'store:{task["storage_dtype"]}')
                start = self._work('gradient_store', 'reduce', start, cost, param_id=task['id'])
                self.intervals.append(
                    (f'gradient_cast/{self.node}/{task["id"]}', task['stored_shard_bytes'], cast_birth,
                     self._point(start, True)))
        self.reduce_ms = buffer['ready_ms'] = start
        self.intervals.append((f'receive/{self.node}', temporary, temp_birth, self._point(start, True)))
        # performCleanup blocks compute on copy completion before dropping the
        # references, even with double buffering. It does not wait for NCCL.
        compute = self._wait(compute, copied, 'copy_cleanup_wait')
        for task in tasks:
            death = self._point(compute, True)
            if task['contiguous']:
                for storage in task['inputs']:
                    self.retained_until[storage] = max(self.retained_until.get(storage, death), death)
            else:
                self.intervals.append((f'gradient_contiguous/{task["node"]}', task['bytes'], task['birth'], death))
        bucket['pending'] = []
        bucket['offset'] = 0
        bucket['index'] = (bucket['index'] + 1) % self.buffer_count
        return max(compute, self.reduce_ms) if self.serialize else compute

    def run(self, event, compute, communication, index):
        self.node, self.index, self.launch = event['key'], index, compute
        self.communication_ms = communication
        previous_work = self.work_ms
        if event['kind'] == 'reduce_grad':
            grad = event['gradient']
            dtype, numel = grad['dtype'], grad['numel']
            if numel <= 0 or grad['bytes'] != numel * grad['itemsize']:
                raise ValueError('Invalid gradient size')
            if grad['id'] in self.seen:
                raise ValueError('Reduction model requires one gradient per parameter and no accumulation')
            self.seen.add(grad['id'])
            if dtype not in self.buckets:
                # unordered_map iteration determines the final flush order in
                # C++; the initial model therefore requires one reduce dtype.
                if self.buckets:
                    raise ValueError('Reduction model requires one gradient dtype')
                self.buckets[dtype] = {
                    'dtype':
                    dtype,
                    'itemsize':
                    grad['itemsize'],
                    'index':
                    0,
                    'offset':
                    0,
                    'pending': [],
                    'buffers': [{
                        'numel': self.capacity,
                        'birth': self._point(compute),
                        'ready_ms': 0.0
                    } for _ in range(self.buffer_count)]
                }
            bucket = self.buckets[dtype]
            buffer = bucket['buffers'][bucket['index']]
            if bucket['offset'] > 0 and bucket['offset'] + numel > buffer['numel']:
                compute = self._flush(bucket, compute, 'capacity')
                buffer = bucket['buffers'][bucket['index']]
            if numel > buffer['numel']:
                compute = self._wait(compute, self.reduce_ms, 'bucket_resize_wait')
                # reserve allocates the replacement before dropping the old
                # tensor; include that transient overlap in the memory peak.
                self.intervals.append((f'bucket_resize/{self.node}', buffer['numel'] * bucket['itemsize'],
                                       buffer['birth'], (compute, 2 * index + 1)))
                buffer['numel'], buffer['birth'] = numel, self._point(compute)
            compute = self._wait(compute, buffer['ready_ms'], 'bucket_reuse_wait')
            birth = self._point(compute)
            if not grad['contiguous']:
                cost = self.table.lookup(grad['bytes'], dtype, 'copy')
                compute = self._work('gradient_contiguous', 'compute', compute, cost, param_id=grad['id'])
            cost = self.table.lookup(grad['bytes'], dtype, 'copy')
            self.copy_ms = self._work('gradient_copy',
                                      'copy',
                                      max(compute, self.copy_ms),
                                      cost,
                                      param_id=grad['id'],
                                      bucket_index=bucket['index'])
            bucket['offset'] += numel
            bucket['pending'].append({
                **grad, 'inputs': event.get('gradient_inputs', event.get('inputs', [])),
                'copied_ms': self.copy_ms,
                'node': self.node,
                'birth': birth
            })
        elif event['kind'] == 'end_backward':
            if not event.get('clear_runtime_buffers'):
                raise ValueError('Reduction model requires the final backward to release buckets')
            for bucket in self.buckets.values():
                compute = self._flush(bucket, compute, 'end_backward')
            compute = self._wait(compute, self.reduce_ms, 'end_backward_wait')
            for dtype, bucket in self.buckets.items():
                for slot, buffer in enumerate(bucket['buffers']):
                    self.intervals.append(
                        (f'bucket/{dtype}/{slot}', buffer['numel'] * bucket['itemsize'], buffer['birth'],
                         self._point(compute, True)))
            self.buckets.clear()
        else:
            raise ValueError(f'Unsupported reduction event: {event["kind"]}')
        return compute, self.communication_ms, self.work_ms - previous_work
