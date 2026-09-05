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


def prepare_table(control, specs, output_dir, max_fuse_size):
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
            'serial-search-v1',
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
            'padded_gathered_output_bytes',
            'async_op':
            False
        }

    header = on_rank_zero(control, environment)
    requests = []
    for dtype_name in sorted({param['dtype'] for param in specs.values()}):
        dtype = getattr(torch, dtype_name.split('.')[-1])
        values = [p['bytes'] for p in specs.values() if p['dtype'] == dtype_name]
        # The grouping heuristic never queries a fused size >= MAX_FUSE_SIZE.
        upper = max(max(values), min(sum(values), int(max_fuse_size)))
        for size in representative_sizes(upper, dtype.itemsize * control.world_size):
            requests.append({'op': 'all_gather', 'dtype': dtype_name, 'bytes': size})
    if not requests:
        raise ValueError('No ZeRO-3 all-gathers found')

    def read_cache():
        path = output_dir / 'communication.json'
        return json.loads(path.read_text()) if path.exists() else None

    cached = on_rank_zero(control, read_cache)

    def allocate(request):
        dtype = getattr(torch, request['dtype'].split('.')[-1])
        numel = request['bytes'] // dtype.itemsize
        source = torch.full((numel // control.world_size, ), control.rank, dtype=dtype, device=control.device)
        destination = torch.empty(numel, dtype=dtype, device=control.device)
        start = accelerator.Event(enable_timing=True)
        end = accelerator.Event(enable_timing=True)
        accelerator.synchronize()
        return source, destination, start, end

    def measure(buffers, request):
        source, destination, start, end = buffers
        dist.barrier()
        for _ in range(header['warmup']):
            dist.all_gather_into_tensor(destination, source)
        accelerator.synchronize()
        dist.barrier()
        start.record()
        for _ in range(header['trials']):
            dist.all_gather_into_tensor(destination, source)
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
