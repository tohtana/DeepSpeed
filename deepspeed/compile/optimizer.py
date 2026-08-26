# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import copy
import gc
import inspect
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten, tree_leaves, treespec_dumps

try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except ImportError:
    pass

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from .agent_runner import AgentRunner
from .config import CompileConfig
from .evaluation_context import (AgentResponseError, EvaluationDecision, GraphSlotRef, GraphVersionTracker,
                                 candidate_evaluation_payload, parse_evaluation_decision,
                                 serialize_evaluation_context)
from .graph_edit import (GraphEditPayload, apply_graph_edit, candidate_fingerprint, finalize_graph_edit,
                         structural_fingerprint)
from .profilers import ProfilingResult
from .profilers.graph_profile import MemoryProfilingInterpreter, ProfilingInterpreter, is_profile_incomplete
from .transform_context import parse_optimizer_edit, serialize_transform_context


_logger = logging.getLogger(__name__)
_INSPECTION_SESSION_ROOT = None


@dataclass
class OptimizationContext:
    gm: GraphModule
    graph_id: int
    graph_slot: Tuple[int, str]
    graph_order: List[Tuple[int, bool]]
    profiling_results: Dict[int, ProfilingResult]
    create_inputs_fn: Callable
    bwd: bool
    debug_log: bool
    compile_config: Optional[CompileConfig]
    warmup_trace: List[Dict[str, Any]] = field(default_factory=list)
    runtime_abi: Optional[_RuntimeABIDescriptor] = None


@dataclass
class OptimizationTraceEntry:
    iteration: int
    action: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    trace: List[OptimizationTraceEntry] = field(default_factory=list)


@dataclass
class _CapturedProfileCall:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _RuntimeABIDescriptor:
    positional_arg_counts: Tuple[int, ...]
    keyword_names: Tuple[Tuple[str, ...], ...]
    output_treespec: str
    output_leaf_kinds: Tuple[str, ...]


class _RuntimeABIError(RuntimeError):
    pass


@dataclass
class _TensorRegistration:
    owner: torch.nn.Module
    name: str
    registry_name: str


@dataclass
class _TensorStateSnapshot:
    tensor: torch.Tensor
    value: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    registrations: List[_TensorRegistration] = field(default_factory=list)


def _distributed() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def broadcast_json_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Broadcast one JSON string so nonzero ranks never deserialize executable Python objects."""
    if not _distributed():
        if payload is None:
            raise ValueError("Rank zero must provide a JSON payload")
        return json.loads(json.dumps(payload, allow_nan=False))

    if _rank() == 0:
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    else:
        encoded = None
    object_list = [encoded]
    dist.broadcast_object_list(object_list, src=0)
    if not isinstance(object_list[0], str):
        raise RuntimeError("Rank-zero JSON broadcast did not produce a string payload")
    decoded = json.loads(object_list[0])
    if not isinstance(decoded, dict):
        raise RuntimeError("Rank-zero JSON broadcast must decode to an object")
    return decoded


def gather_rank_records(local_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    local_record = json.loads(json.dumps(local_record, allow_nan=False))
    if not _distributed():
        return [local_record]
    records = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(records, local_record)
    return records


def broadcast_edit_payload(payload: Optional[GraphEditPayload]) -> Optional[GraphEditPayload]:
    envelope = {"edit": None if payload is None else payload.to_dict()} if _rank() == 0 else None
    received = broadcast_json_payload(envelope)
    if received.get("edit") is None:
        return None
    return GraphEditPayload.from_dict(received["edit"], require_result_fingerprint=True)


def _set_profile_from_graph(profile: ProfilingResult,
                            graph_id: int,
                            gm: GraphModule,
                            memory,
                            bwd: bool,
                            memory_complete: bool) -> None:
    node_time = []
    tensor_sizes = []
    for node in gm.graph.nodes:
        node_time.append((node.name, node.meta.get("device_time", 0.0), node.meta.get("wall_time", 0.0)))
        tensor_sizes.append((node.name, node.meta.get("tensor_size", 0)))
    if bwd:
        profile.bwd_graph = gm.graph
        profile.bwd_time = node_time
        profile.bwd_tensor_sizes = tensor_sizes
        profile.bwd_mem = memory
        profile.bwd_mem_complete = memory_complete
    else:
        profile.fwd_graph = gm.graph
        profile.fwd_time = node_time
        profile.fwd_tensor_sizes = tensor_sizes
        profile.fwd_mem = memory
        profile.fwd_mem_complete = memory_complete


def _capture_profile_calls(ctx: OptimizationContext) -> List[_CapturedProfileCall]:
    return [_CapturedProfileCall(args=tuple(ctx.create_inputs_fn())),
            _CapturedProfileCall(args=tuple(ctx.create_inputs_fn()))]


def _output_leaf_kind(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return "tensor"
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _validate_runtime_call_contract(gm: GraphModule, profile_calls: List[_CapturedProfileCall]) -> None:
    try:
        signature = inspect.signature(gm.forward)
        for call in profile_calls:
            signature.bind(*call.args, **call.kwargs)
    except (TypeError, ValueError) as exc:
        raise _RuntimeABIError(f"Graph cannot accept the captured positional caller ABI: {exc}") from exc


def _runtime_abi_descriptor(gm: GraphModule,
                            profile_calls: List[_CapturedProfileCall],
                            output: Any) -> _RuntimeABIDescriptor:
    _validate_runtime_call_contract(gm, profile_calls)

    leaves, treespec = tree_flatten(output)
    try:
        serialized_treespec = treespec_dumps(treespec)
    except Exception as exc:
        raise _RuntimeABIError(f"Graph output pytree cannot be described for the AOT caller ABI: {exc}") from exc
    return _RuntimeABIDescriptor(
        positional_arg_counts=tuple(len(call.args) for call in profile_calls),
        keyword_names=tuple(tuple(sorted(call.kwargs)) for call in profile_calls),
        output_treespec=serialized_treespec,
        output_leaf_kinds=tuple(_output_leaf_kind(leaf) for leaf in leaves),
    )


def _profile_phase_consensus(phase: str,
                             error: Optional[Exception] = None,
                             runtime_abi: Optional[_RuntimeABIDescriptor] = None) -> None:
    local = {
        "rank": _rank(),
        "phase": phase,
        "success": error is None,
        "error": None if error is None else str(error),
        "abi_incompatible": isinstance(error, _RuntimeABIError),
        "runtime_abi": None if runtime_abi is None else asdict(runtime_abi),
    }
    records = gather_rank_records(local)
    failures = [record for record in records if not record["success"]]
    abi_descriptors = {
        json.dumps(record["runtime_abi"], sort_keys=True)
        for record in records if record["success"] and record["runtime_abi"] is not None
    }
    if not failures and (runtime_abi is None or len(abi_descriptors) == 1):
        return

    details = "; ".join(f"rank {record['rank']}: {record['error']}" for record in failures)
    if not failures:
        details = "runtime output ABI descriptors differ across ranks"
    message = f"Distributed {phase} phase failed: {details}"
    if any(record["abi_incompatible"] for record in failures) or len(abi_descriptors) > 1:
        raise _RuntimeABIError(message)
    raise RuntimeError(message)


def _profile_graph(gm: GraphModule,
                   ctx: OptimizationContext,
                   profile_calls: Optional[List[_CapturedProfileCall]] = None,
                   expected_abi: Optional[_RuntimeABIDescriptor] = None) -> Tuple[ProfilingResult,
                                                                                _RuntimeABIDescriptor]:
    if profile_calls is None:
        profile_calls = _capture_profile_calls(ctx)

    timing_profiler = None
    call_contract_error = None
    try:
        _validate_runtime_call_contract(gm, profile_calls)
        timing_profiler = ProfilingInterpreter(gm, debug_log=ctx.debug_log)
    except Exception as exc:
        call_contract_error = exc
    _profile_phase_consensus("captured-call validation", call_contract_error)

    runtime_abi = None
    timing_error = None
    try:
        timing_output = timing_profiler.run(*profile_calls[0].args)
        if is_profile_incomplete(gm.graph):
            raise RuntimeError("Timing profiling was incomplete")
        runtime_abi = _runtime_abi_descriptor(gm, profile_calls, timing_output)
        if expected_abi is not None and runtime_abi != expected_abi:
            raise _RuntimeABIError("Candidate output pytree/container or leaf-kind ABI differs from the accepted graph")
    except Exception as exc:
        timing_error = exc
    _profile_phase_consensus("timing and runtime-ABI validation", timing_error, runtime_abi)

    memory_profiler = MemoryProfilingInterpreter(gm, debug_log=ctx.debug_log)
    memory_output = memory_profiler.run(*profile_calls[1].args)
    if not memory_profiler.profile_complete:
        raise RuntimeError("Memory profiling was incomplete")
    memory_abi = _runtime_abi_descriptor(gm, profile_calls, memory_output)
    if memory_abi != runtime_abi:
        raise _RuntimeABIError("Timing and memory profiling produced different runtime output ABI descriptors")
    memory = [(name, current, delta, peak) for name, current, delta, peak in memory_profiler.mem_record]
    profile = copy.deepcopy(ctx.profiling_results[ctx.graph_id])
    _set_profile_from_graph(profile, ctx.graph_id, gm, memory, ctx.bwd, True)
    return profile, runtime_abi


def _profile_metrics(profile: ProfilingResult, bwd: bool) -> Dict[str, float]:
    times = profile.bwd_time if bwd else profile.fwd_time
    memory = profile.bwd_mem if bwd else profile.fwd_mem
    return {
        "device_time": float(sum(row[1] for row in times)),
        "peak_memory": float(max([row[3] for row in memory], default=0)),
    }


def _cleanup_after_candidate() -> None:
    with unset_fake_temporarily():
        get_accelerator().synchronize()
        gc.collect()
        get_accelerator().empty_cache()


def _snapshot_candidate_state(gm: GraphModule,
                              profile_calls: List[_CapturedProfileCall]) -> List[_TensorStateSnapshot]:
    snapshots_by_identity = {}

    def snapshot_tensor(tensor: torch.Tensor) -> _TensorStateSnapshot:
        identity = id(tensor)
        if identity not in snapshots_by_identity:
            snapshots_by_identity[identity] = _TensorStateSnapshot(
                tensor=tensor,
                value=tensor.detach().to(device="cpu", copy=True),
                device=tensor.device,
                dtype=tensor.dtype,
            )
        return snapshots_by_identity[identity]

    with unset_fake_temporarily(), torch.no_grad():
        for module in gm.modules():
            for registry_name in ("_parameters", "_buffers"):
                registry = getattr(module, registry_name)
                for name, tensor in registry.items():
                    if tensor is None:
                        continue
                    snapshot = snapshot_tensor(tensor)
                    snapshot.registrations.append(
                        _TensorRegistration(owner=module, name=name, registry_name=registry_name))
        for call in profile_calls:
            for value in tree_leaves((call.args, call.kwargs)):
                if isinstance(value, torch.Tensor):
                    snapshot_tensor(value)
    return list(snapshots_by_identity.values())


def _restore_candidate_state(snapshots: List[_TensorStateSnapshot]) -> None:
    errors = []
    with unset_fake_temporarily(), torch.no_grad():
        for index, snapshot in enumerate(snapshots):
            try:
                for registration in snapshot.registrations:
                    registry = getattr(registration.owner, registration.registry_name)
                    registry[registration.name] = snapshot.tensor
                tensor = snapshot.tensor
                identity_changed = (tensor.device != snapshot.device or tensor.dtype != snapshot.dtype
                                    or tensor.shape != snapshot.value.shape)
                if identity_changed:
                    tensor.data = snapshot.value.to(device=snapshot.device, dtype=snapshot.dtype)
                else:
                    tensor.copy_(snapshot.value)
            except Exception as exc:
                errors.append(f"tensor snapshot {index}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def _safe_write_text(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        _logger.warning("Unable to write DeepCompile agent artifact %s: %s", path, exc)


def _safe_write_json(path: Path, payload: Any) -> None:
    _safe_write_text(path, json.dumps(payload, indent=2, default=str))


def _resolve_inspection_session_root(parent_dir: str) -> Optional[Path]:
    global _INSPECTION_SESSION_ROOT
    try:
        parent = Path(parent_dir).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        if _INSPECTION_SESSION_ROOT is not None and _INSPECTION_SESSION_ROOT.parent == parent:
            return _INSPECTION_SESSION_ROOT
        base = f"session_{int(time.time() * 1000)}_{os.getpid()}"
        for suffix in [""] + [f"_{index}" for index in range(1, 10)]:
            candidate = parent / f"{base}{suffix}"
            try:
                candidate.mkdir()
                _INSPECTION_SESSION_ROOT = candidate
                return candidate
            except FileExistsError:
                continue
    except OSError as exc:
        _logger.warning("Unable to create DeepCompile inspection session under %s: %s", parent_dir, exc)
    return None


def _reset_inspection_session_root() -> None:
    global _INSPECTION_SESSION_ROOT
    _INSPECTION_SESSION_ROOT = None


class TwoAgentLoopOptimizer:

    def __init__(self, evaluator_runner: AgentRunner, optimizer_runner: AgentRunner, compile_config: CompileConfig):
        self.evaluator_runner = evaluator_runner
        self.optimizer_runner = optimizer_runner
        self.max_iterations = compile_config.agent_max_iterations
        self.max_retries = compile_config.agent_max_retries_per_iteration
        self.debug_log = compile_config.debug_log

    @staticmethod
    def _checked_run(runner: AgentRunner,
                     prompt: str,
                     iteration_dir: Path,
                     role: str,
                     artifact_prefix: Optional[str] = None):
        result = runner.run(prompt, iteration_dir, role=role, artifact_prefix=artifact_prefix)
        if result.timed_out:
            raise AgentResponseError(f"{role} command timed out")
        if result.returncode != 0:
            raise AgentResponseError(f"{role} command exited with code {result.returncode}: {result.stderr.strip()}")
        return result

    def _iteration_root(self, ctx: OptimizationContext, is_rank_zero: bool):
        if not is_rank_zero:
            return None, None, False
        configured = os.environ.get("DEEPCOMPILE_AGENT_ARTIFACT_ROOT", "").strip()
        if configured:
            session = _resolve_inspection_session_root(configured)
            if session is not None:
                graph_root = session / f"graph_{ctx.graph_slot[0]}_{ctx.graph_slot[1]}"
                graph_root.mkdir(parents=True, exist_ok=True)
                return graph_root, session, True
        return Path(tempfile.mkdtemp(prefix="deepcompile_two_agent_")), None, False

    def _write_session_metadata(self, session_root: Path, ctx: OptimizationContext) -> None:
        path = session_root / "session.json"
        if path.exists():
            return
        if ctx.compile_config is None:
            compile_config = None
        elif hasattr(ctx.compile_config, "model_dump"):
            compile_config = ctx.compile_config.model_dump(mode="json")
        else:
            compile_config = ctx.compile_config.dict()
        _safe_write_json(path, {
            "compile_config": compile_config,
            "evaluator_command": self.evaluator_runner.config.command,
            "optimizer_command": self.optimizer_runner.config.command,
            "host_argv": sys.argv,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "capture_rank": 0,
        })

    @staticmethod
    def _snapshot_consensus(tracker: GraphVersionTracker) -> Tuple[bool, List[Dict[str, Any]]]:
        try:
            snapshot = asdict(tracker.current_ref())
            local = {"rank": _rank(), "success": True, "snapshot": snapshot, "error": None}
        except Exception as exc:
            local = {"rank": _rank(), "success": False, "snapshot": None, "error": str(exc)}
        records = gather_rank_records(local)
        snapshots = [record["snapshot"] for record in records if record["success"]]
        return len(snapshots) == len(records) and all(snapshot == snapshots[0] for snapshot in snapshots), records

    @staticmethod
    def _profile_accepted_graph(ctx: OptimizationContext) -> Tuple[bool, List[Dict[str, Any]]]:
        local_profile = None
        local_abi = None
        profile_calls = None
        snapshots = None
        try:
            profile_calls = _capture_profile_calls(ctx)
            snapshots = _snapshot_candidate_state(ctx.gm, profile_calls)
            local_snapshot = {"rank": _rank(), "success": True, "phase": "state_snapshot", "error": None}
        except Exception as exc:
            local_snapshot = {
                "rank": _rank(),
                "success": False,
                "phase": "state_snapshot",
                "error": str(exc),
            }
        snapshot_records = gather_rank_records(local_snapshot)
        if not all(record["success"] for record in snapshot_records):
            return False, snapshot_records

        profile_error = None
        restore_error = None
        try:
            local_profile, local_abi = _profile_graph(ctx.gm, ctx, profile_calls)
        except Exception as exc:
            profile_error = str(exc)
        finally:
            try:
                _restore_candidate_state(snapshots)
            except Exception as exc:
                restore_error = str(exc)

        if profile_error is None and restore_error is None:
            local = {
                "rank": _rank(),
                "success": True,
                "metrics": _profile_metrics(local_profile, ctx.bwd),
                "error": None,
                "abi": asdict(local_abi),
                "state_restore_failed": False,
            }
        else:
            errors = []
            if profile_error is not None:
                errors.append(f"profiling failed: {profile_error}")
            if restore_error is not None:
                errors.append(f"accepted state restore failed: {restore_error}")
            local = {
                "rank": _rank(),
                "success": False,
                "metrics": None,
                "error": "; ".join(errors),
                "abi": None,
                "state_restore_failed": restore_error is not None,
            }
        records = gather_rank_records(local)
        if any(record.get("state_restore_failed", False) for record in records):
            raise RuntimeError("Accepted graph state restoration failed; live tensor state may be corrupted")
        abi_descriptors = {json.dumps(record["abi"], sort_keys=True) for record in records if record["success"]}
        success = all(record["success"] for record in records) and len(abi_descriptors) == 1
        if success:
            ctx.profiling_results[ctx.graph_id] = local_profile
            ctx.runtime_abi = local_abi
        return success, records

    @staticmethod
    def _apply_and_profile_candidate(ctx: OptimizationContext, payload: GraphEditPayload):
        candidate = None
        candidate_profile = None
        try:
            candidate = apply_graph_edit(ctx.gm, payload, ctx.graph_id)
            fingerprint = candidate_fingerprint(candidate, payload, ctx.graph_id)
            local_apply = {
                "rank": _rank(),
                "success": fingerprint == payload.expected_result_fingerprint,
                "fingerprint": fingerprint,
                "error": None,
            }
        except Exception as exc:
            local_apply = {
                "rank": _rank(),
                "success": False,
                "fingerprint": None,
                "error": str(exc),
            }
        apply_records = gather_rank_records(local_apply)
        apply_success = all(record["success"] for record in apply_records)
        fingerprints = {record["fingerprint"] for record in apply_records if record["success"]}
        apply_success = apply_success and fingerprints == {payload.expected_result_fingerprint}
        if not apply_success:
            return candidate, candidate_profile, apply_records

        profile_calls = None
        snapshots = None
        try:
            profile_calls = _capture_profile_calls(ctx)
            snapshots = _snapshot_candidate_state(ctx.gm, profile_calls)
            local_snapshot = {"rank": _rank(), "success": True, "phase": "state_snapshot", "error": None}
        except Exception as exc:
            local_snapshot = {
                "rank": _rank(),
                "success": False,
                "phase": "state_snapshot",
                "error": str(exc),
            }
        snapshot_records = gather_rank_records(local_snapshot)
        if not all(record["success"] for record in snapshot_records):
            return candidate, candidate_profile, snapshot_records

        profile_error = None
        restore_error = None
        abi_error = None
        candidate_abi = None
        try:
            candidate_profile, candidate_abi = _profile_graph(candidate,
                                                              ctx,
                                                              profile_calls,
                                                              expected_abi=ctx.runtime_abi)
        except _RuntimeABIError as exc:
            abi_error = str(exc)
            profile_error = str(exc)
        except Exception as exc:
            profile_error = str(exc)
        finally:
            try:
                _restore_candidate_state(snapshots)
            except Exception as exc:
                restore_error = str(exc)

        if profile_error is None and restore_error is None:
            local_profile = {
                "rank": _rank(),
                "success": True,
                "fingerprint": payload.expected_result_fingerprint,
                "metrics": _profile_metrics(candidate_profile, ctx.bwd),
                "error": None,
                "abi": asdict(candidate_abi),
                "abi_compatible": True,
                "state_restore_failed": False,
            }
        else:
            candidate_profile = None
            errors = []
            if profile_error is not None:
                errors.append(f"profiling failed: {profile_error}")
            if restore_error is not None:
                errors.append(f"candidate state restore failed: {restore_error}")
            local_profile = {
                "rank": _rank(),
                "success": False,
                "fingerprint": payload.expected_result_fingerprint,
                "metrics": None,
                "error": "; ".join(errors),
                "abi": None,
                "abi_compatible": abi_error is None,
                "state_restore_failed": restore_error is not None,
            }
        profile_records = gather_rank_records(local_profile)
        return candidate, candidate_profile, profile_records

    @staticmethod
    def _commit_candidate(ctx: OptimizationContext,
                          tracker: GraphVersionTracker,
                          candidate: GraphModule,
                          candidate_profile: ProfilingResult) -> Tuple[bool, List[Dict[str, Any]]]:
        previous_graph = ctx.gm.graph
        try:
            ctx.gm.graph = candidate.graph
            ctx.gm.recompile()
            local = {
                "rank": _rank(),
                "success": True,
                "fingerprint": structural_fingerprint(ctx.gm, ctx.graph_id),
                "error": None,
            }
        except Exception as exc:
            local = {"rank": _rank(), "success": False, "fingerprint": None, "error": str(exc)}
        records = gather_rank_records(local)
        fingerprints = {record["fingerprint"] for record in records if record["success"]}
        success = all(record["success"] for record in records) and len(fingerprints) == 1
        if not success:
            ctx.gm.graph = previous_graph
            ctx.gm.recompile()
            return False, records
        ctx.profiling_results[ctx.graph_id] = candidate_profile
        tracker.accept(ctx.gm)
        return True, records

    def optimize(self, gm: GraphModule, ctx: OptimizationContext) -> OptimizationResult:
        trace = []
        history = list(ctx.warmup_trace)
        is_rank_zero = _rank() == 0
        slot = GraphSlotRef(index=ctx.graph_slot[0], direction=ctx.graph_slot[1])
        tracker = GraphVersionTracker(slot, gm, ctx.graph_id)
        iteration_root, session_root, inspection_enabled = self._iteration_root(ctx, is_rank_zero)
        retain_temporary_root = self.debug_log
        if is_rank_zero and inspection_enabled:
            self._write_session_metadata(session_root, ctx)

        consensus, consensus_records = self._snapshot_consensus(tracker)
        if not consensus:
            if is_rank_zero:
                trace.append(OptimizationTraceEntry(iteration=0,
                                                    action="abort",
                                                    summary="Post-Z3 graph topology differs across ranks",
                                                    details={"rank_results": consensus_records}))
            retain_temporary_root = True
            return OptimizationResult(trace=trace)

        profile_success, profile_records = self._profile_accepted_graph(ctx)
        if not profile_success:
            if is_rank_zero:
                trace.append(OptimizationTraceEntry(iteration=0,
                                                    action="abort",
                                                    summary="Post-Z3 accepted graph profiling failed",
                                                    details={"rank_results": profile_records}))
            retain_temporary_root = True
            return OptimizationResult(trace=trace)

        for iteration in range(self.max_iterations):
            iteration_dir = iteration_root / f"iter_{iteration}" if is_rank_zero else None
            if is_rank_zero and inspection_enabled:
                _safe_write_json(iteration_dir / "accepted_snapshot.json", asdict(tracker.current_ref()))
                _safe_write_text(iteration_dir / "accepted_graph.py", ctx.gm.code)

            evaluation = None
            if is_rank_zero:
                try:
                    prompt = serialize_evaluation_context(ctx, tracker, history)
                    result = self._checked_run(self.evaluator_runner, prompt, iteration_dir, "evaluator")
                    evaluation = parse_evaluation_decision(result.stdout, tracker.current_ref(), "accepted_graph")
                    _safe_write_json(iteration_dir / "evaluation.json", evaluation.to_dict())
                    trace.append(OptimizationTraceEntry(iteration=iteration,
                                                        action="evaluate",
                                                        summary=evaluation.summary,
                                                        details={"decision": evaluation.decision}))
                    evaluation_envelope = {"continue": evaluation.decision == "continue", "error": None}
                except Exception as exc:
                    evaluation_envelope = {"continue": False, "error": str(exc)}
                    retain_temporary_root = True
                    trace.append(OptimizationTraceEntry(iteration=iteration,
                                                        action="abort",
                                                        summary=f"Evaluator failed: {exc}"))
            else:
                evaluation_envelope = None
            evaluation_envelope = broadcast_json_payload(evaluation_envelope)
            if not evaluation_envelope["continue"]:
                break

            edit = None
            rank_zero_candidate = None
            mechanical_feedback = []
            if is_rank_zero:
                for attempt in range(self.max_retries + 1):
                    try:
                        prompt = serialize_transform_context(ctx, evaluation, tracker, history, mechanical_feedback)
                        prefix = "optimizer" if attempt == 0 else f"optimizer_retry_{attempt}"
                        result = self._checked_run(self.optimizer_runner,
                                                   prompt,
                                                   iteration_dir,
                                                   "optimizer",
                                                   artifact_prefix=prefix)
                        raw_edit = parse_optimizer_edit(result.stdout, tracker)
                        edit, rank_zero_candidate = finalize_graph_edit(ctx.gm, raw_edit, ctx.graph_id)
                        _safe_write_json(iteration_dir / "graph_edit.json", edit.to_dict())
                        break
                    except Exception as exc:
                        mechanical_feedback.append(str(exc))
                        trace.append(OptimizationTraceEntry(iteration=iteration,
                                                            action="optimizer_retry",
                                                            summary=f"Mechanical edit replay failed: {exc}",
                                                            details={"attempt": attempt + 1}))
                if edit is None:
                    retain_temporary_root = True
            edit = broadcast_edit_payload(edit)
            if edit is None:
                break

            candidate, candidate_profile, rank_results = self._apply_and_profile_candidate(ctx, edit)
            candidate_success = all(record["success"] for record in rank_results)
            candidate_context = None
            if is_rank_zero:
                display_candidate = candidate if candidate is not None else rank_zero_candidate
                candidate_context = candidate_evaluation_payload(edit.to_dict(), display_candidate, rank_results,
                                                                 ctx.graph_id)
                _safe_write_json(iteration_dir / "candidate_result.json", candidate_context)
                if display_candidate is not None and inspection_enabled:
                    _safe_write_text(iteration_dir / "candidate_graph.py", display_candidate.code)

            state_restore_failed = any(record.get("state_restore_failed", False) for record in rank_results)
            if state_restore_failed:
                retain_temporary_root = True
                if is_rank_zero:
                    trace.append(OptimizationTraceEntry(iteration=iteration,
                                                        action="abort",
                                                        summary="Candidate state restoration failed",
                                                        details={"rank_results": rank_results}))
                raise RuntimeError("Candidate state restoration failed; accepted tensor state may be corrupted")

            abi_incompatible = any(record.get("abi_compatible") is False for record in rank_results)
            stop_after_candidate = False
            if is_rank_zero:
                candidate_decision = None
                if abi_incompatible:
                    accept = False
                    candidate_summary = "Candidate runtime ABI is incompatible with the accepted AOT caller"
                else:
                    try:
                        prompt = serialize_evaluation_context(ctx, tracker, history, candidate=candidate_context)
                        result = self._checked_run(self.evaluator_runner,
                                                   prompt,
                                                   iteration_dir,
                                                   "evaluator",
                                                   artifact_prefix="candidate_evaluator")
                        candidate_result_fingerprint = edit.expected_result_fingerprint
                        candidate_decision = parse_evaluation_decision(
                            result.stdout,
                            tracker.current_ref(),
                            "candidate",
                            candidate_generation=edit.generation,
                            candidate_fingerprint=candidate_result_fingerprint)
                        accept = candidate_decision.decision == "accept" and candidate_success
                        if candidate_decision.decision == "accept" and not candidate_success:
                            candidate_decision.summary += " (mechanical apply/profile failure forced rejection)"
                        _safe_write_json(iteration_dir / "candidate_evaluation.json", candidate_decision.to_dict())
                        candidate_summary = candidate_decision.summary
                    except Exception as exc:
                        accept = False
                        stop_after_candidate = True
                        retain_temporary_root = True
                        candidate_summary = None
                        trace.append(OptimizationTraceEntry(iteration=iteration,
                                                            action="abort",
                                                            summary=f"Candidate evaluator failed: {exc}"))
                decision_envelope = {
                    "accept": accept,
                    "stop": stop_after_candidate,
                    "summary": candidate_summary,
                }
            else:
                decision_envelope = None
            decision_envelope = broadcast_json_payload(decision_envelope)

            accepted = False
            commit_records = []
            if decision_envelope["accept"]:
                if candidate is not None and candidate_profile is not None:
                    accepted, commit_records = self._commit_candidate(ctx, tracker, candidate, candidate_profile)
                if not accepted:
                    retain_temporary_root = True
            outcome = "accepted" if accepted else "rejected"
            history_entry = {
                "generation": edit.generation,
                "outcome": outcome,
                "reason": edit.reason,
                "edit": edit.to_dict(),
                "rank_results": rank_results,
                "commit_results": commit_records,
                "evaluator_summary": decision_envelope["summary"],
            }
            history.append(history_entry)
            if is_rank_zero:
                _safe_write_json(iteration_dir / "outcome.json", history_entry)
                trace.append(OptimizationTraceEntry(iteration=iteration,
                                                    action=outcome,
                                                    summary=decision_envelope["summary"] or outcome,
                                                    details={
                                                        "generation": edit.generation,
                                                        "rank_results": rank_results,
                                                    }))
            if decision_envelope["stop"]:
                break

            cleanup_error = None
            try:
                _cleanup_after_candidate()
            except Exception as exc:
                cleanup_error = str(exc)
            cleanup_records = gather_rank_records({
                "rank": _rank(),
                "success": cleanup_error is None,
                "error": cleanup_error,
            })
            if not all(record["success"] for record in cleanup_records):
                retain_temporary_root = True
                if is_rank_zero:
                    trace.append(OptimizationTraceEntry(iteration=iteration,
                                                        action="abort",
                                                        summary="Post-candidate cleanup failed",
                                                        details={"rank_results": cleanup_records}))
                break

        if is_rank_zero and not inspection_enabled and not retain_temporary_root and iteration_root.exists():
            shutil.rmtree(iteration_root, ignore_errors=True)
        return OptimizationResult(trace=trace)
