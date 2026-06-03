# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed.compile.backend as backend


def test_dummy_backward_compile_keeps_frame_pending(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "unpatch_compiled_func", lambda: calls.append("unpatch"))
    backend.frames_needing_bwd.clear()
    backend.frames_needing_bwd.add(34)

    try:
        backend._finalize_backward_frame(34, consumed_runtime_inputs=False)

        assert backend.frames_needing_bwd == {34}
        assert calls == []
    finally:
        backend.frames_needing_bwd.clear()


def test_runtime_backward_compile_removes_final_frame(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "unpatch_compiled_func", lambda: calls.append("unpatch"))
    backend.frames_needing_bwd.clear()
    backend.frames_needing_bwd.add(34)

    try:
        backend._finalize_backward_frame(34, consumed_runtime_inputs=True)

        assert backend.frames_needing_bwd == set()
        assert calls == ["unpatch"]
    finally:
        backend.frames_needing_bwd.clear()


def test_runtime_backward_compile_keeps_patch_for_remaining_frames(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "unpatch_compiled_func", lambda: calls.append("unpatch"))
    backend.frames_needing_bwd.clear()
    backend.frames_needing_bwd.update({34, 35})

    try:
        backend._finalize_backward_frame(34, consumed_runtime_inputs=True)

        assert backend.frames_needing_bwd == {35}
        assert calls == []
    finally:
        backend.frames_needing_bwd.clear()


def test_runtime_backward_compile_tolerates_untracked_frame(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "unpatch_compiled_func", lambda: calls.append("unpatch"))
    backend.frames_needing_bwd.clear()

    try:
        backend._finalize_backward_frame(34, consumed_runtime_inputs=True)

        assert backend.frames_needing_bwd == set()
        assert calls == ["unpatch"]
    finally:
        backend.frames_needing_bwd.clear()
