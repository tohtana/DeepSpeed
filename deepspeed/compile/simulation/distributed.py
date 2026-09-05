# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""One ordered preparation phase, then a rank-0-only search with fixed broadcasts."""

import hashlib
import json
from .core import CommTable


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class CollectiveControl:

    def __init__(self, comm, device):
        import torch
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        self.torch = torch
        self.unset_fake = unset_fake_temporarily
        self.comm = comm
        self.device = device
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        # Allocate the status tensor before any possibly failing benchmark buffers.
        with self.unset_fake():
            self.status = torch.ones(1, dtype=torch.int64, device=device)

    def broadcast(self, value):
        values = [value if self.rank == 0 else None]
        with self.unset_fake():
            self.comm.broadcast_object_list(values, src=0)
        return values[0]

    def vote(self, success):
        with self.unset_fake():
            self.status.fill_(int(success))
            self.comm.all_reduce(self.status, op=self.comm.ReduceOp.MIN)
            return bool(self.status.item())

    def agree(self, value):
        expected = self.broadcast(fingerprint(value) if self.rank == 0 else None)
        if not self.vote(fingerprint(value) == expected):
            raise RuntimeError('Rank graph/plan mismatch')


def on_rank_zero(control, function):
    """Every rank enters exactly STATUS then PAYLOAD, including Python failures."""
    success, payload = True, None
    if control.rank == 0:
        try:
            payload = function()
            # Check serializability before the success status is sent.
            json.dumps(payload, allow_nan=False)
        except Exception as error:
            success = False
            payload = {'error': f'{type(error).__name__}: {error}'}
    success = control.broadcast(success)
    payload = control.broadcast(payload)
    if not success:
        raise RuntimeError(f'Pass search preparation/search failed: {payload["error"]}')
    return payload


def all_rank_prepare(control, function):
    result, error = None, None
    try:
        result = function()
    except Exception as exc:
        error = exc
    if not control.vote(error is None):
        raise RuntimeError(f'Preparation failed on a rank: {error or "peer failure"}') from error
    return result


def prepare_communication(control, header, requests, cached, allocate, measure):
    """The leader owns cache decisions; allocate votes precede data collectives.

    allocate(request) must not issue a collective. measure(buffers, request) runs
    the same pre-agreed benchmark on every rank and returns a row in milliseconds.
    """
    control.agree({'header': header, 'requests': requests})

    def manifest():
        cache = {}
        if cached and cached.get('header') == header:
            # Validate the entire file, including duplicate or invalid durations.
            CommTable(**cached)
            cache = {(r['op'], r['dtype'], r['bytes']): r for r in cached['rows']}
        return [
            dict(request, cached=cache.get((request['op'], request['dtype'], request['bytes'])))
            for request in requests
        ]

    manifest_rows = on_rank_zero(control, manifest)
    rows = []
    for request in manifest_rows:
        if request['cached'] is not None:
            rows.append(request['cached'])
            continue
        buffers = all_rank_prepare(control, lambda: allocate(request))
        try:
            row = measure(buffers, request)
            rows.append(row)
        finally:
            del buffers
    table = on_rank_zero(control, lambda: CommTable(header, rows).to_dict())
    return table
