# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression tests for more than one Ulysses model living in the same process.

The head count the uneven-head all-to-all splits against used to be read from the
process-wide ``tp_shard`` slot, which AutoTP writes too (#8291). Whoever wrote it first
decided how every later all-to-all was sharded. ``DistributedAttention`` carries the count
per instance now; these tests hold that line.
"""

import pytest
import torch

import deepspeed.comm as dist
from deepspeed.module_inject.tp_shard import set_num_kv_heads
from deepspeed.sequence.layer import DistributedAttention
from deepspeed.utils import groups

from unit.common import DistributedTest

HEAD_DIM = 8
LOCAL_SEQ = 4


class _RecordingAttention(torch.nn.Module):
    """Stands in for the local attention and records the head shard it was handed."""

    def __init__(self):
        super().__init__()
        self.head_ids = None

    def forward(self, query, key, value, *args, **kwargs):
        # Each head carries its own global index, so the shard is readable off the tensor.
        self.head_ids = sorted({int(v) for v in query[0, 0, :, 0].tolist()})
        return value


def _run_attention(attn, num_heads):
    """One forward + backward through ``attn``; returns the head ids this rank received."""
    query = torch.zeros(1, LOCAL_SEQ, num_heads, HEAD_DIM, requires_grad=True)
    with torch.no_grad():
        query[:] = torch.arange(num_heads).view(1, 1, -1, 1).float()
    output = attn(query, query.clone(), query.clone(), 0)
    output.sum().backward()
    return attn.local_attn.head_ids


class TestUlyssesMultipleModels(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    @pytest.fixture(autouse=True)
    def clean_shard_globals(self):
        set_num_kv_heads(None)
        yield
        set_num_kv_heads(None)

    def _sequence_parallel_group(self):
        groups.mesh_device = dist.initialize_mesh_device((1, self.world_size), ("data_parallel", "sequence_parallel"))
        return groups.mesh_device.get_group(mesh_dim="sequence_parallel")

    def test_second_model_does_not_reshard_the_first(self):
        # 3 heads over 2 ranks is uneven ([2, 1]) and so is 5 ([3, 2]). The counts are
        # distinguishable, so a shared value shows up as one model taking the other's split.
        sp_group = self._sequence_parallel_group()
        rank = dist.get_rank(group=sp_group)

        teacher = DistributedAttention(_RecordingAttention(), sp_group, scatter_idx=2, gather_idx=1)
        student = DistributedAttention(_RecordingAttention(), sp_group, scatter_idx=2, gather_idx=1)

        expected_teacher = [[0, 1], [2]][rank]
        expected_student = [[0, 1, 2], [3, 4]][rank]

        assert _run_attention(teacher, 3) == expected_teacher
        assert _run_attention(student, 5) == expected_student
        # The teacher still splits against its own head count after the student has run.
        assert _run_attention(teacher, 3) == expected_teacher

    def test_autotp_head_count_does_not_reach_ulysses(self):
        # AutoTP publishes the model's kv-head count into the same slot on every injection
        # (replace_module.py, engine.py). 6 heads over 2 ranks divides evenly, so this model
        # never needs the uneven path at all - but 6 % 3 == 0, so a leftover 3 makes the
        # shard helper hand rank 0 four heads and rank 1 two.
        sp_group = self._sequence_parallel_group()
        rank = dist.get_rank(group=sp_group)

        attn = DistributedAttention(_RecordingAttention(), sp_group, scatter_idx=2, gather_idx=1)
        set_num_kv_heads(3)

        assert _run_attention(attn, 6) == [[0, 1, 2], [3, 4, 5]][rank]

    def test_explicit_head_count_is_honoured(self):
        # Callers that know the count up front can pass it instead of having it inferred.
        sp_group = self._sequence_parallel_group()
        rank = dist.get_rank(group=sp_group)

        attn = DistributedAttention(_RecordingAttention(), sp_group, scatter_idx=2, gather_idx=1, num_total_heads=3)

        assert _run_attention(attn, 3) == [[0, 1], [2]][rank]
