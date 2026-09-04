# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.compile.executor_arena import (ArenaOccurrence, DEFAULT_FUSE_BUDGET, DEFAULT_LIVE_BUDGET,
                                              EXECUTOR_ARENA_ALIGNMENT, GraphArenaPlan, admit_executor_arena,
                                              executor_plan_signature, freeze_persistence, pack_executor_arena,
                                              register_executor_arena)


class FakeNativeZ3:

    def __init__(self):
        self.arena_config = None

    def configure_z3_gather_arena(self, *args):
        self.arena_config = args


def test_executor_arena_interval_pack_aligns_and_reuses_released_storage():
    plan = pack_executor_arena([
        ArenaOccurrence(ds_id=1, occurrence=0, first_use=0, release=2, nbytes=257),
        ArenaOccurrence(ds_id=2, occurrence=0, first_use=1, release=4, nbytes=128),
        ArenaOccurrence(ds_id=3, occurrence=0, first_use=3, release=5, nbytes=400),
    ])

    assert EXECUTOR_ARENA_ALIGNMENT == 256
    assert [entry.offset for entry in plan.entries] == [0, 512, 0]
    assert [entry.aligned_nbytes for entry in plan.entries] == [512, 256, 512]
    assert plan.capacity == 768
    assert plan.max_live_bytes == 768


def test_executor_arena_tracks_repeated_ds_id_occurrences_independently():
    plan = pack_executor_arena([
        ArenaOccurrence(ds_id=7, occurrence=0, first_use=0, release=1, nbytes=256),
        ArenaOccurrence(ds_id=7, occurrence=1, first_use=2, release=3, nbytes=256),
    ])

    assert [(entry.ds_id, entry.occurrence, entry.offset) for entry in plan.entries] == [(7, 0, 0), (7, 1, 0)]
    assert plan.capacity == 256


def test_executor_arena_keeps_escape_as_independent_fallback():
    escaping = ArenaOccurrence(ds_id=9,
                               occurrence=0,
                               first_use=0,
                               release=1,
                               nbytes=256,
                               eligible=False,
                               fallback_reason="graph_output_escape")
    plan = pack_executor_arena([escaping])

    assert plan.entries == ()
    assert plan.fallbacks == (escaping, )
    assert plan.capacity == 0


def test_executor_arena_admission_charges_alignment_above_demand_profile():
    plan = pack_executor_arena([ArenaOccurrence(ds_id=1, occurrence=0, first_use=0, release=1, nbytes=257)])

    rejected = admit_executor_arena(plan, demand_profile_bytes=256, live_budget=255)
    accepted = admit_executor_arena(plan, demand_profile_bytes=256, live_budget=256)

    assert rejected.incremental_bytes == 256
    assert not rejected.accepted
    assert rejected.reason == "live_budget_exceeded"
    assert accepted.accepted


def test_persistence_freeze_reserves_live_budget_and_does_not_reclaim_slack():
    frozen = freeze_persistence([(1, 300), (2, 250), (3, 100)],
                                headroom_bytes=1000,
                                live_budget=400,
                                safety_reserve_bytes=100)

    assert frozen.available_bytes == 500
    assert frozen.selected_ds_ids == (1, 3)
    assert frozen.selected_bytes == 400
    assert frozen.unused_bytes == 100
    assert frozen.reserved_live_bytes == 400


def test_executor_arena_defaults_and_consensus_signatures_are_stable():
    plan = pack_executor_arena([ArenaOccurrence(ds_id=1, occurrence=0, first_use=0, release=1, nbytes=1)])

    assert DEFAULT_LIVE_BUDGET == 4_000_000_000
    assert DEFAULT_FUSE_BUDGET == 1_000_000_000
    assert executor_plan_signature(plan) == executor_plan_signature(plan)
    assert executor_plan_signature(None, "disabled") != executor_plan_signature(None, "no_plan")


def test_executor_arena_registration_keeps_fallback_occurrences_in_runtime_sequence():
    occurrences = (
        ArenaOccurrence(ds_id=7, occurrence=0, first_use=0, release=1, nbytes=256, dtype=torch.float16),
        ArenaOccurrence(ds_id=7,
                        occurrence=1,
                        first_use=2,
                        release=3,
                        nbytes=256,
                        dtype=torch.float16,
                        eligible=False,
                        fallback_reason="saved_tensor_escape"),
        ArenaOccurrence(ds_id=7, occurrence=2, first_use=4, release=5, nbytes=256, dtype=torch.float16),
    )
    graph_plan = GraphArenaPlan(occurrences=occurrences, packed=pack_executor_arena(occurrences))
    native = FakeNativeZ3()

    registration = register_executor_arena(native, graph_id=11, graph_plan=graph_plan)

    assert registration.enabled
    graph_id, capacity, alignment, ds_ids, occurrence_ids, offsets, nbytes, dtypes, signature = native.arena_config
    assert graph_id == 11
    assert capacity == 256
    assert alignment == 256
    assert ds_ids == [7, 7, 7]
    assert occurrence_ids == [0, 1, 2]
    assert offsets == [0, -1, 0]
    assert nbytes == [256, 256, 256]
    assert dtypes == [torch.float16, torch.float16, torch.float16]
    assert signature == registration.signature


def test_executor_arena_disabled_registration_has_no_runtime_backing_plan():
    native = FakeNativeZ3()

    registration = register_executor_arena(native, graph_id=12, graph_plan=None, disabled_reason="incomplete_profile")

    assert not registration.enabled
    assert registration.reason == "incomplete_profile"
    assert native.arena_config[0:3] == (12, 0, 256)
    assert native.arena_config[3:8] == ([], [], [], [], [])
