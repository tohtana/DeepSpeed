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
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import get_num_kv_heads, set_num_kv_heads
from deepspeed.sequence.layer import DistributedAttention, _SeqAllToAll
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


def _tagged(num_heads):
    tensor = torch.zeros(1,
                         LOCAL_SEQ,
                         num_heads,
                         HEAD_DIM,
                         device=get_accelerator().current_device_name(),
                         requires_grad=True)
    with torch.no_grad():
        tensor[:] = torch.arange(num_heads, device=tensor.device).view(1, 1, -1, 1).float()
    return tensor


def _run_attention(attn, num_heads, num_kv_heads=None):
    """One forward + backward through ``attn``; returns the query head ids this rank received."""
    query = _tagged(num_heads)
    key = _tagged(num_kv_heads if num_kv_heads is not None else num_heads)
    output = attn(query, key, key.clone(), 0)
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

    def test_gqa_partitions_by_kv_groups(self):
        # Q=6 / KV=3 over 2 ranks. 6 divides evenly, but a query head has to stay on the rank
        # holding its KV head, so the split follows the KV groups [2, 1] and Q becomes [4, 2].
        # Reading the count off the query tensor instead would tear group 1 apart.
        sp_group = self._sequence_parallel_group()
        rank = dist.get_rank(group=sp_group)

        attn = DistributedAttention(_RecordingAttention(), sp_group, scatter_idx=2, gather_idx=1)

        assert _run_attention(attn, 6, num_kv_heads=3) == [[0, 1, 2, 3], [4, 5]][rank]

    def test_direct_all_to_all_replays_the_inferred_count_in_backward(self):
        # Megatron-DeepSpeed reaches _SeqAllToAll directly and threads nothing. Only the scatter
        # direction can see that 3 heads do not divide by 2; backward runs with scatter and gather
        # swapped, so the count it infers here has to survive to that point.
        sp_group = self._sequence_parallel_group()

        query = torch.randn(1,
                            LOCAL_SEQ,
                            3,
                            HEAD_DIM,
                            device=get_accelerator().current_device_name(),
                            requires_grad=True)
        output = _SeqAllToAll.apply(sp_group, query, 2, 1, 0)
        output.sum().backward()

        assert query.grad is not None
        assert get_num_kv_heads() == 3

    def test_fewer_kv_heads_than_ranks_is_rejected_before_the_collective(self):
        # 1 KV head over 2 ranks leaves rank 1 with nothing to attend over. Both ranks have to
        # reject it together: one of them raising inside the all-to-all hangs the other.
        sp_group = self._sequence_parallel_group()

        attn = DistributedAttention(_RecordingAttention(), sp_group, scatter_idx=2, gather_idx=1)
        device = get_accelerator().current_device_name()
        query = torch.zeros(1, LOCAL_SEQ, 2, HEAD_DIM, device=device, requires_grad=True)
        key = torch.zeros(1, LOCAL_SEQ, 1, HEAD_DIM, device=device, requires_grad=True)

        with pytest.raises(AssertionError, match="at least the sequence parallel size"):
            attn(query, key, key.clone(), 0)

    def test_explicit_head_count_is_honoured(self):
        # Callers that know the count up front can pass it instead of having it inferred.
        sp_group = self._sequence_parallel_group()
        rank = dist.get_rank(group=sp_group)

        attn = DistributedAttention(_RecordingAttention(), sp_group, scatter_idx=2, gather_idx=1, num_kv_heads=3)

        assert _run_attention(attn, 3) == [[0, 1], [2]][rank]
