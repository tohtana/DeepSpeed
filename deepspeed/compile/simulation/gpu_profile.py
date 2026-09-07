# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import hashlib
import json
from pathlib import Path
import subprocess

import torch
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from .core import representative_sizes
from .distributed import all_rank_prepare, on_rank_zero, prepare_communication


def profile_requests(specs, world_size, max_fuse_size, gradients=None):
    requests = {}

    def add(op, dtype_name, upper, quantum):
        for size in representative_sizes(upper, quantum):
            requests[(op, dtype_name, size)] = {'op': op, 'dtype': dtype_name, 'bytes': size}

    for dtype_name in sorted({param['dtype'] for param in specs.values()}):
        dtype = getattr(torch, dtype_name.split('.')[-1])
        values = [p['bytes'] for p in specs.values() if p['dtype'] == dtype_name]
        upper = max(max(values), min(sum(values), int(max_fuse_size)))
        add('all_gather', dtype_name, upper, dtype.itemsize * world_size)
    if not requests:
        raise ValueError('No ZeRO-3 all-gathers found')
    for grad in (gradients or {}).values():
        dtype_name = grad['dtype']
        for op in ('copy', 'pre_divide', 'reduce_scatter'):
            add(op, dtype_name, grad['bytes'], grad['itemsize'] * world_size)
        if grad['storage_dtype'] != dtype_name:
            add(f'store:{grad["storage_dtype"]}', dtype_name, grad['shard_bytes'], grad['itemsize'])
    return [requests[key] for key in sorted(requests)]


def prepare_table(control, specs, output_dir, max_fuse_size, gradients=None):
    accelerator = get_accelerator()
    output_dir = Path(output_dir)
    raw_rows = []

    def environment():
        source = Path(__file__).resolve().parents[1]
        repository = source.parents[1]
        digest = hashlib.sha256()
        paths = list(
            source.rglob('*.py')) + [path for path in (repository / 'csrc/compile').rglob('*') if path.is_file()]
        paths += list((repository / 'csrc/includes').glob('*.h'))
        for path in sorted(paths):
            digest.update(str(path.relative_to(repository)).encode())
            digest.update(path.read_bytes())
        return {
            'protocol':
            'bucketed-reduction-search-v2',
            'source_sha256':
            digest.hexdigest(),
            'world_size':
            control.world_size,
            'group':
            list(range(control.world_size)),
            'backend':
            accelerator.communication_backend_name(),
            'backend_version':
            list(accelerator.communication_backend_version()),
            'torch':
            str(torch.__version__),
            'cuda':
            torch.version.cuda,
            'gpus':
            subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,pci.bus_id,driver_version', '--format=csv,noheader'], text=True),
            'topology':
            subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
            'warmup':
            5,
            'trials':
            10,
            'representative':
            'max_rank_mean_ms',
            'size_unit':
            'all_gather: padded output bytes; all other operations: source bytes',
            'async_op':
            False
        }

    header = on_rank_zero(control, environment)
    requests = all_rank_prepare(control, lambda: profile_requests(specs, control.world_size, max_fuse_size, gradients))

    def read_cache():
        path = output_dir / 'communication.json'
        return json.loads(path.read_text()) if path.exists() else None

    cached = on_rank_zero(control, read_cache)

    def allocate(request):
        dtype = getattr(torch, request['dtype'].split('.')[-1])
        numel = request['bytes'] // dtype.itemsize
        op = request['op']
        source_numel = numel // control.world_size if op == 'all_gather' else numel
        destination_numel = numel // control.world_size if op == 'reduce_scatter' else numel
        destination_dtype = getattr(torch, op.split(':')[1].split('.')[-1]) if op.startswith('store:') else dtype
        source = torch.full((source_numel, ), control.rank + 1, dtype=dtype, device=control.device)
        destination = torch.empty(destination_numel, dtype=destination_dtype, device=control.device)
        start = accelerator.Event(enable_timing=True)
        end = accelerator.Event(enable_timing=True)
        accelerator.synchronize()
        return source, destination, start, end

    def measure(buffers, request):
        source, destination, start, end = buffers

        def operation():
            op = request['op']
            if op == 'all_gather':
                dist.all_gather_into_tensor(destination, source)
            elif op == 'reduce_scatter':
                dist.reduce_scatter_tensor(destination, source)
            elif op == 'copy':
                destination.copy_(source, non_blocking=True)
            elif op == 'pre_divide':
                source.div_(control.world_size)
            elif op.startswith('store:'):
                destination.copy_(source.to(destination.dtype), non_blocking=True)
            else:
                raise ValueError(f'Unknown profile operation: {op}')

        dist.barrier()
        for _ in range(header['warmup']):
            operation()
        accelerator.synchronize()
        dist.barrier()
        start.record()
        for _ in range(header['trials']):
            operation()
        end.record()
        accelerator.synchronize()
        local_ms = start.elapsed_time(end) / header['trials']
        raw = [None] * control.world_size
        dist.all_gather_object(raw, local_ms)
        row = {key: request[key] for key in ('op', 'dtype', 'bytes')}
        row.update(time_ms=max(raw), rank_mean_ms=raw)
        raw_rows.append(row)
        return row

    table = prepare_communication(control, header, requests, cached, allocate, measure)

    def save():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'communication.json').write_text(json.dumps(table, indent=2) + '\n')
        (output_dir / 'communication-measurements.json').write_text(json.dumps(raw_rows, indent=2) + '\n')

    all_rank_prepare(control, lambda: save() if control.rank == 0 else None)
    return table
