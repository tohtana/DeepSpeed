# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import copy
import gc
import hashlib
import importlib.util
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
from .evaluation_context import (AgentResponseError, GeneratedPassProposal, GeneratedPassSelection,
                                 aggregate_rank_metrics, build_evaluation_packet, build_reference_pass_inventory,
                                 parse_search_response, sanitize_json_value, serialize_search_context,
                                 validate_selection, verify_proposal_identity)
from .graph_edit import (clone_graph_module, generated_graph_fingerprint_details, structural_fingerprint)
from .profilers import ProfilingResult
from .profilers.graph_profile import MemoryProfilingInterpreter, ProfilingInterpreter, is_profile_incomplete

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
    mem_budget: float
    param_manager: Any
    bwd: bool
    debug_log: bool
    compile_config: Optional[CompileConfig]
    warmup_trace: List[Dict[str, Any]] = field(default_factory=list)
    runtime_abi: Optional[_RuntimeABIDescriptor] = None


@dataclass
class FrozenGraphContext:
    graph_module: GraphModule
    graph_fingerprint: str
    graph_slot: Tuple[int, str]
    graph_order: List[Tuple[int, bool]]
    baseline_rank_results: List[Dict[str, Any]]
    baseline_aggregate: Dict[str, Any]
    mem_budget: float
    param_manager: Any


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
    with unset_fake_temporarily():
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
    with unset_fake_temporarily():
        dist.all_gather_object(records, local_record)
    return records


def _set_profile_from_graph(profile: ProfilingResult, graph_id: int, gm: GraphModule, memory, bwd: bool,
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
    return [
        _CapturedProfileCall(args=tuple(ctx.create_inputs_fn())),
        _CapturedProfileCall(args=tuple(ctx.create_inputs_fn()))
    ]


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


def _runtime_abi_descriptor(gm: GraphModule, profile_calls: List[_CapturedProfileCall],
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


def _profile_graph(
        gm: GraphModule,
        ctx: OptimizationContext,
        profile_calls: Optional[List[_CapturedProfileCall]] = None,
        expected_abi: Optional[_RuntimeABIDescriptor] = None) -> Tuple[ProfilingResult, _RuntimeABIDescriptor]:
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
            raise _RuntimeABIError(
                "Candidate output pytree/container or leaf-kind ABI differs from the accepted graph")
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
    local_mem_record = getattr(memory_profiler, "local_mem_record", memory_profiler.mem_record)
    local_memory = [(name, current, delta, peak) for name, current, delta, peak in local_mem_record]
    profile = copy.deepcopy(ctx.profiling_results[ctx.graph_id])
    _set_profile_from_graph(profile, ctx.graph_id, gm, memory, ctx.bwd, True)
    local_memory_field = "_deepcompile_local_bwd_mem" if ctx.bwd else "_deepcompile_local_fwd_mem"
    setattr(profile, local_memory_field, local_memory)
    return profile, runtime_abi


def _profile_metrics(profile: ProfilingResult, bwd: bool, local: bool = False) -> Dict[str, float]:
    times = profile.bwd_time if bwd else profile.fwd_time
    memory = profile.bwd_mem if bwd else profile.fwd_mem
    if local:
        graph = profile.bwd_graph if bwd else profile.fwd_graph
        times = [(node.name, node.meta.get("local_device_time", 0.0), node.meta.get("local_wall_time", 0.0))
                 for node in graph.nodes]
        local_memory_field = "_deepcompile_local_bwd_mem" if bwd else "_deepcompile_local_fwd_mem"
        memory = getattr(profile, local_memory_field, memory)
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


class GeneratedPassValidationError(RuntimeError):

    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def _hard_write_source(path: Path, source: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.encode("utf-8"))
        written_source = path.read_bytes()
    except OSError as exc:
        raise GeneratedPassValidationError("write", f"Unable to write exact generated source: {exc}") from exc
    if written_source != source.encode("utf-8"):
        raise GeneratedPassValidationError("write", "Generated source bytes changed while writing the local file")


def load_generated_pass(proposal: GeneratedPassProposal, source_path: Path) -> Callable:
    if hashlib.sha256(proposal.source.encode("utf-8")).hexdigest() != proposal.source_sha256:
        raise GeneratedPassValidationError("source_identity", "Generated source SHA-256 does not match its envelope")
    _hard_write_source(source_path, proposal.source)
    try:
        compile(proposal.source, str(source_path), "exec")
    except SyntaxError as exc:
        raise GeneratedPassValidationError("syntax", str(exc)) from exc

    spec = importlib.util.spec_from_file_location(proposal.module_name, source_path)
    if spec is None or spec.loader is None:
        raise GeneratedPassValidationError("import", "Unable to construct a generated-pass module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[proposal.module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(proposal.module_name, None)
        raise GeneratedPassValidationError("import", str(exc)) from exc

    entrypoint = getattr(module, proposal.entrypoint, None)
    if entrypoint is None:
        raise GeneratedPassValidationError("callable", f"Generated module has no {proposal.entrypoint} entrypoint")
    if not callable(entrypoint):
        raise GeneratedPassValidationError("callable",
                                           f"Generated module entrypoint {proposal.entrypoint} is not callable")
    return entrypoint


def _generated_pass_arguments(gm: GraphModule, ctx: OptimizationContext) -> Tuple[Any, ...]:
    return (gm, ctx.graph_id, ctx.graph_order, ctx.profiling_results, ctx.create_inputs_fn, ctx.mem_budget,
            ctx.param_manager, ctx.bwd)


def apply_generated_pass(proposal: GeneratedPassProposal,
                         gm: GraphModule,
                         ctx: OptimizationContext,
                         source_path: Path,
                         entrypoint: Optional[Callable] = None) -> Tuple[GraphModule, Callable]:
    if entrypoint is None:
        entrypoint = load_generated_pass(proposal, source_path)
    arguments = _generated_pass_arguments(gm, ctx)
    try:
        inspect.signature(entrypoint).bind(*arguments)
    except (TypeError, ValueError) as exc:
        raise GeneratedPassValidationError("signature", str(exc)) from exc
    try:
        result = entrypoint(*arguments)
    except Exception as exc:
        raise GeneratedPassValidationError("call", str(exc)) from exc
    if result is not None and result is not gm:
        raise GeneratedPassValidationError("return_contract", "deepcompile_pass must return None or the identical gm")
    try:
        gm.graph.lint()
    except Exception as exc:
        raise GeneratedPassValidationError("lint", str(exc)) from exc
    try:
        gm.recompile()
    except Exception as exc:
        raise GeneratedPassValidationError("recompile", str(exc)) from exc
    return gm, entrypoint


def restore_frozen_base(gm: GraphModule, frozen: FrozenGraphContext) -> None:
    restored = clone_graph_module(frozen.graph_module)
    gm.graph = restored.graph
    gm.recompile()


class GraphAgentLoopOptimizer:

    def __init__(self, graph_agent_runner: AgentRunner, compile_config: CompileConfig):
        self.graph_agent_runner = graph_agent_runner
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
        return Path(tempfile.mkdtemp(prefix="deepcompile_graph_agent_")), None, False

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
        _safe_write_json(
            path, {
                "compile_config": compile_config,
                "graph_agent_command": self.graph_agent_runner.config.command,
                "host_argv": sys.argv,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "world_size": dist.get_world_size() if dist.is_initialized() else 1,
                "capture_rank": 0,
            })

    @staticmethod
    def _snapshot_consensus(ctx: OptimizationContext) -> Tuple[bool, List[Dict[str, Any]]]:
        try:
            fingerprint = structural_fingerprint(ctx.gm, ctx.graph_id)
            local = {"rank": _rank(), "success": True, "fingerprint": fingerprint, "error": None}
        except Exception as exc:
            local = {"rank": _rank(), "success": False, "fingerprint": None, "error": str(exc)}
        records = gather_rank_records(local)
        fingerprints = {record["fingerprint"] for record in records if record["success"]}
        return all(record["success"] for record in records) and len(fingerprints) == 1, records

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
            local_metrics = _profile_metrics(local_profile, ctx.bwd, local=True)
            reduced_metrics = _profile_metrics(local_profile, ctx.bwd)
            local = {
                "rank": _rank(),
                "success": True,
                "local_device_time": local_metrics["device_time"],
                "local_peak_memory": local_metrics["peak_memory"],
                "reduced_device_time": reduced_metrics["device_time"],
                "reduced_peak_memory": reduced_metrics["peak_memory"],
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
                "local_device_time": None,
                "local_peak_memory": None,
                "reduced_device_time": None,
                "reduced_peak_memory": None,
                "error": "; ".join(errors),
                "abi": None,
                "state_restore_failed": restore_error is not None,
            }
        records = gather_rank_records(sanitize_json_value(local))
        if any(record.get("state_restore_failed", False) for record in records):
            raise RuntimeError("Accepted graph state restoration failed; live tensor state may be corrupted")
        abi_descriptors = {json.dumps(record["abi"], sort_keys=True) for record in records if record["success"]}
        success = all(record["success"] for record in records) and len(abi_descriptors) == 1
        if success:
            ctx.profiling_results[ctx.graph_id] = local_profile
            ctx.runtime_abi = local_abi
        return success, records

    @staticmethod
    def _failed_rank_results(records: List[Dict[str, Any]], fallback_error: str) -> List[Dict[str, Any]]:
        failed = []
        for record in records:
            failed.append({
                "rank": record["rank"],
                "success": False,
                "fingerprint": record.get("fingerprint"),
                "local_device_time": None,
                "local_peak_memory": None,
                "reduced_device_time": None,
                "reduced_peak_memory": None,
                "error": record.get("error") or fallback_error,
                "abi": None,
                "abi_compatible": None,
                "state_restore_failed": False,
            })
        return failed

    @staticmethod
    def _apply_and_profile_candidate(ctx: OptimizationContext, frozen: FrozenGraphContext,
                                     proposal: GeneratedPassProposal, proposal_index: int, source_path: Path):
        validation = {}
        candidate = None
        candidate_profile = None
        entrypoint = None

        try:
            verify_proposal_identity(proposal, ctx.graph_slot, proposal_index)
            live_fingerprint = structural_fingerprint(ctx.gm, ctx.graph_id)
            frozen_fingerprint = structural_fingerprint(frozen.graph_module, ctx.graph_id)
            identity_success = live_fingerprint == frozen.graph_fingerprint == frozen_fingerprint
            identity_error = None if identity_success else "Local live/frozen graph identity changed"
        except Exception as exc:
            live_fingerprint = None
            frozen_fingerprint = None
            identity_success = False
            identity_error = str(exc)
        local_identity = {
            "rank": _rank(),
            "success": identity_success,
            "candidate_id": proposal.candidate_id,
            "entrypoint": proposal.entrypoint,
            "source_sha256": proposal.source_sha256,
            "proposal_hash": proposal.proposal_hash,
            "live_frozen_base_fingerprint": live_fingerprint,
            "clone_frozen_base_fingerprint": frozen_fingerprint,
            "error": identity_error,
        }
        identity_records = gather_rank_records(local_identity)
        identity_values = {(record["candidate_id"], record["entrypoint"], record["source_sha256"],
                            record["proposal_hash"])
                           for record in identity_records if record["success"]}
        identity_success = all(record["success"] for record in identity_records) and len(identity_values) == 1
        validation["source_consensus"] = {"success": identity_success, "rank_results": identity_records}
        validation["frozen_base_identity"] = {
            "success": identity_success,
            "fingerprint": frozen.graph_fingerprint,
        }
        if not identity_success:
            rank_results = GraphAgentLoopOptimizer._failed_rank_results(identity_records,
                                                                        "All-rank source/base identity failed")
            packet = build_evaluation_packet(proposal, rank_results, frozen.baseline_aggregate,
                                             frozen.graph_fingerprint, validation)
            return candidate, candidate_profile, entrypoint, packet

        phase_order = [
            "clone", "write", "syntax", "import", "callable", "signature", "call", "return_contract", "lint",
            "recompile", "fingerprint"
        ]
        local_validation = {phase: {"success": False, "skipped": True, "error": None} for phase in phase_order}
        local_error = None
        fingerprint_details = None
        try:
            candidate = clone_graph_module(frozen.graph_module)
            local_validation["clone"] = {"success": True, "skipped": False, "error": None}
            entrypoint = load_generated_pass(proposal, source_path)
            for phase in ("write", "syntax", "import", "callable"):
                local_validation[phase] = {"success": True, "skipped": False, "error": None}
            candidate, entrypoint = apply_generated_pass(proposal, candidate, ctx, source_path, entrypoint)
            for phase in ("signature", "call", "return_contract", "lint", "recompile"):
                local_validation[phase] = {"success": True, "skipped": False, "error": None}
            fingerprint_details = generated_graph_fingerprint_details(candidate, ctx.graph_id)
            local_validation["fingerprint"] = {"success": True, "skipped": False, "error": None}
        except GeneratedPassValidationError as exc:
            local_error = f"{exc.phase}: {exc}"
            if exc.phase in local_validation:
                failed_index = phase_order.index(exc.phase)
                for phase in phase_order[:failed_index]:
                    if local_validation[phase]["skipped"]:
                        local_validation[phase] = {"success": True, "skipped": False, "error": None}
                local_validation[exc.phase] = {"success": False, "skipped": False, "error": str(exc)}
        except Exception as exc:
            local_error = f"clone_or_fingerprint: {exc}"
            phase = "fingerprint" if candidate is not None else "clone"
            local_validation[phase] = {"success": False, "skipped": False, "error": str(exc)}

        local_apply = {
            "rank": _rank(),
            "success": local_error is None,
            "fingerprint": None if fingerprint_details is None else fingerprint_details["fingerprint"],
            "fingerprint_details": fingerprint_details,
            "validation": local_validation,
            "error": local_error,
        }
        apply_records = gather_rank_records(local_apply)
        fingerprints = {record["fingerprint"] for record in apply_records if record["success"]}
        fingerprint_success = all(record["success"] for record in apply_records) and len(fingerprints) == 1
        validation["rank_mechanical"] = apply_records
        for phase in phase_order:
            phase_records = [{"rank": record["rank"], **record["validation"][phase]} for record in apply_records]
            validation[phase] = {
                "success": all(record["success"] for record in phase_records),
                "rank_results": phase_records,
            }
        validation["candidate_fingerprint_consensus"] = {
            "success": fingerprint_success,
            "fingerprint": next(iter(fingerprints)) if fingerprint_success else None,
            "opaque_fallbacks": [record.get("fingerprint_details") for record in apply_records],
        }
        if not fingerprint_success:
            rank_results = GraphAgentLoopOptimizer._failed_rank_results(
                apply_records, "Candidate fingerprint differed across ranks or another rank failed validation")
            packet = build_evaluation_packet(proposal, rank_results, frozen.baseline_aggregate,
                                             frozen.graph_fingerprint, validation)
            return candidate, candidate_profile, entrypoint, packet

        profile_calls = None
        snapshots = None
        try:
            profile_calls = _capture_profile_calls(ctx)
            snapshots = _snapshot_candidate_state(candidate, profile_calls)
            local_snapshot = {"rank": _rank(), "success": True, "phase": "state_snapshot", "error": None}
        except Exception as exc:
            local_snapshot = {
                "rank": _rank(),
                "success": False,
                "phase": "state_snapshot",
                "error": str(exc),
            }
        snapshot_records = gather_rank_records(local_snapshot)
        validation["state_snapshot"] = {
            "success": all(record["success"] for record in snapshot_records),
            "rank_results": snapshot_records
        }
        if not all(record["success"] for record in snapshot_records):
            rank_results = GraphAgentLoopOptimizer._failed_rank_results(snapshot_records,
                                                                        "Candidate state snapshot failed")
            packet = build_evaluation_packet(proposal, rank_results, frozen.baseline_aggregate,
                                             frozen.graph_fingerprint, validation)
            return candidate, candidate_profile, entrypoint, packet

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
            local_metrics = _profile_metrics(candidate_profile, ctx.bwd, local=True)
            reduced_metrics = _profile_metrics(candidate_profile, ctx.bwd)
            local_profile = {
                "rank": _rank(),
                "success": True,
                "fingerprint": next(iter(fingerprints)),
                "local_device_time": local_metrics["device_time"],
                "local_peak_memory": local_metrics["peak_memory"],
                "reduced_device_time": reduced_metrics["device_time"],
                "reduced_peak_memory": reduced_metrics["peak_memory"],
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
                "fingerprint": next(iter(fingerprints)),
                "local_device_time": None,
                "local_peak_memory": None,
                "reduced_device_time": None,
                "reduced_peak_memory": None,
                "error": "; ".join(errors),
                "abi": None,
                "abi_compatible": abi_error is None,
                "state_restore_failed": restore_error is not None,
            }
        profile_records = gather_rank_records(sanitize_json_value(local_profile))
        if any(record.get("state_restore_failed", False) for record in profile_records):
            raise RuntimeError("Candidate state restoration failed; frozen graph tensor state may be corrupted")
        validation["runtime_abi"] = {
            "success": all(record["success"] and record["abi_compatible"] for record in profile_records),
            "rank_results": profile_records,
        }
        packet = build_evaluation_packet(proposal, profile_records, frozen.baseline_aggregate,
                                         frozen.graph_fingerprint, validation)
        return candidate, candidate_profile, entrypoint, packet

    @staticmethod
    def _restore_after_final_failure(ctx: OptimizationContext, frozen: FrozenGraphContext) -> List[Dict[str, Any]]:
        try:
            restore_frozen_base(ctx.gm, frozen)
            local = {"rank": _rank(), "success": True, "error": None}
        except Exception as exc:
            local = {"rank": _rank(), "success": False, "error": str(exc)}
        return gather_rank_records(local)

    @staticmethod
    def _finalize_candidate(ctx: OptimizationContext, frozen: FrozenGraphContext, proposal: GeneratedPassProposal,
                            evaluation: Dict[str, Any], entrypoint: Callable, source_path: Path) -> Dict[str, Any]:
        try:
            proposal_index = int(proposal.candidate_id.split("_")[1])
            verify_proposal_identity(proposal, ctx.graph_slot, proposal_index)
            live_fingerprint = structural_fingerprint(ctx.gm, ctx.graph_id)
            _hard_write_source(source_path, proposal.source)
            source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            local_preflight = {
                "rank": _rank(),
                "success": live_fingerprint == frozen.graph_fingerprint and source_hash == proposal.source_sha256,
                "candidate_id": proposal.candidate_id,
                "entrypoint": proposal.entrypoint,
                "proposal_hash": proposal.proposal_hash,
                "live_frozen_base_fingerprint": live_fingerprint,
                "source_sha256": source_hash,
                "error": None,
            }
        except Exception as exc:
            local_preflight = {
                "rank": _rank(),
                "success": False,
                "candidate_id": proposal.candidate_id,
                "entrypoint": proposal.entrypoint,
                "proposal_hash": proposal.proposal_hash,
                "live_frozen_base_fingerprint": None,
                "source_sha256": None,
                "error": str(exc),
            }
        preflight_records = gather_rank_records(local_preflight)
        if not all(record["success"] for record in preflight_records):
            return {"success": False, "live_source_executed": False, "preflight": preflight_records}

        local_error = None
        fingerprint_details = None
        try:
            apply_generated_pass(proposal, ctx.gm, ctx, source_path, entrypoint)
            fingerprint_details = generated_graph_fingerprint_details(ctx.gm, ctx.graph_id)
        except Exception as exc:
            local_error = str(exc)
        apply_records = gather_rank_records({
            "rank":
            _rank(),
            "success":
            local_error is None,
            "fingerprint":
            None if fingerprint_details is None else fingerprint_details["fingerprint"],
            "fingerprint_details":
            fingerprint_details,
            "error":
            local_error,
        })
        expected_fingerprint = evaluation["validation"]["candidate_fingerprint_consensus"]["fingerprint"]
        final_fingerprints = {record["fingerprint"] for record in apply_records if record["success"]}
        apply_success = (all(record["success"] for record in apply_records)
                         and final_fingerprints == {expected_fingerprint})
        if not apply_success:
            restore_records = GraphAgentLoopOptimizer._restore_after_final_failure(ctx, frozen)
            raise RuntimeError("Final generated-pass application failed after live source execution; "
                               f"apply={apply_records}, graph_restore={restore_records}")

        profile_calls = None
        snapshots = None
        try:
            profile_calls = _capture_profile_calls(ctx)
            snapshots = _snapshot_candidate_state(ctx.gm, profile_calls)
            local_snapshot = {"rank": _rank(), "success": True, "error": None}
        except Exception as exc:
            local_snapshot = {"rank": _rank(), "success": False, "error": str(exc)}
        snapshot_records = gather_rank_records(local_snapshot)
        if not all(record["success"] for record in snapshot_records):
            restore_records = GraphAgentLoopOptimizer._restore_after_final_failure(ctx, frozen)
            raise RuntimeError("Final live state snapshot failed after source execution; "
                               f"snapshot={snapshot_records}, graph_restore={restore_records}")

        final_profile = None
        final_abi = None
        profile_error = None
        tensor_restore_error = None
        try:
            final_profile, final_abi = _profile_graph(ctx.gm, ctx, profile_calls, expected_abi=ctx.runtime_abi)
        except Exception as exc:
            profile_error = str(exc)
        finally:
            try:
                _restore_candidate_state(snapshots)
            except Exception as exc:
                tensor_restore_error = str(exc)
        local_profile = {
            "rank": _rank(),
            "success": profile_error is None and tensor_restore_error is None,
            "error": profile_error or tensor_restore_error,
            "state_restore_failed": tensor_restore_error is not None,
            "abi": None if final_abi is None else asdict(final_abi),
            "local_metrics": None if final_profile is None else _profile_metrics(final_profile, ctx.bwd, local=True),
            "reduced_metrics": None if final_profile is None else _profile_metrics(final_profile, ctx.bwd),
        }
        profile_records = gather_rank_records(sanitize_json_value(local_profile))
        if not all(record["success"] for record in profile_records):
            restore_records = GraphAgentLoopOptimizer._restore_after_final_failure(ctx, frozen)
            raise RuntimeError("Final live profiling/restoration failed after source execution; "
                               f"profile={profile_records}, graph_restore={restore_records}")

        ctx.profiling_results[ctx.graph_id] = final_profile
        ctx.runtime_abi = final_abi
        return {
            "success": True,
            "live_source_executed": True,
            "candidate_id": proposal.candidate_id,
            "source_sha256": proposal.source_sha256,
            "fingerprint": expected_fingerprint,
            "preflight": preflight_records,
            "apply": apply_records,
            "profile": profile_records,
        }

    def optimize(self, gm: GraphModule, ctx: OptimizationContext) -> OptimizationResult:
        trace = []
        history = []
        is_rank_zero = _rank() == 0
        iteration_root, session_root, inspection_enabled = self._iteration_root(ctx, is_rank_zero)
        retain_temporary_root = self.debug_log
        if is_rank_zero and inspection_enabled:
            self._write_session_metadata(session_root, ctx)

        consensus, consensus_records = self._snapshot_consensus(ctx)
        if not consensus:
            if is_rank_zero:
                trace.append(
                    OptimizationTraceEntry(iteration=0,
                                           action="abort",
                                           summary="Post-Z3 graph topology differs across ranks",
                                           details={"rank_results": consensus_records}))
            retain_temporary_root = True
            return OptimizationResult(trace=trace)

        profile_success, profile_records = self._profile_accepted_graph(ctx)
        if not profile_success:
            if is_rank_zero:
                trace.append(
                    OptimizationTraceEntry(iteration=0,
                                           action="abort",
                                           summary="Post-Z3 accepted graph profiling failed",
                                           details={"rank_results": profile_records}))
            retain_temporary_root = True
            return OptimizationResult(trace=trace)

        frozen_graph = None
        frozen_error = None
        try:
            frozen_graph = clone_graph_module(ctx.gm)
            frozen_fingerprint = structural_fingerprint(frozen_graph, ctx.graph_id)
            if frozen_fingerprint != structural_fingerprint(ctx.gm, ctx.graph_id):
                raise RuntimeError("Frozen graph clone changed the post-required-pass topology")
        except Exception as exc:
            frozen_fingerprint = None
            frozen_error = str(exc)
        frozen_records = gather_rank_records({
            "rank": _rank(),
            "success": frozen_error is None,
            "fingerprint": frozen_fingerprint,
            "error": frozen_error,
        })
        frozen_fingerprints = {record["fingerprint"] for record in frozen_records if record["success"]}
        if not all(record["success"] for record in frozen_records) or len(frozen_fingerprints) != 1:
            if is_rank_zero:
                trace.append(
                    OptimizationTraceEntry(iteration=0,
                                           action="abort",
                                           summary="Unable to freeze one equivalent post-required-pass graph",
                                           details={"rank_results": frozen_records}))
            return OptimizationResult(trace=trace)

        frozen = FrozenGraphContext(graph_module=frozen_graph,
                                    graph_fingerprint=frozen_fingerprint,
                                    graph_slot=ctx.graph_slot,
                                    graph_order=list(ctx.graph_order),
                                    baseline_rank_results=profile_records,
                                    baseline_aggregate=aggregate_rank_metrics(profile_records),
                                    mem_budget=ctx.mem_budget,
                                    param_manager=ctx.param_manager)
        reference_inventory = build_reference_pass_inventory() if is_rank_zero else None
        if is_rank_zero:
            inventory_path = (session_root if inspection_enabled else iteration_root) / "reference_pass_inventory.json"
            _safe_write_json(inventory_path, reference_inventory)
            _safe_write_json(
                iteration_root / "frozen_base.json", {
                    "graph_slot": list(ctx.graph_slot),
                    "graph_fingerprint": frozen.graph_fingerprint,
                    "baseline_rank_results": profile_records,
                    "baseline_aggregate": frozen.baseline_aggregate,
                    "required_pass_trace": ctx.warmup_trace,
                })

        rank_source_root = Path(tempfile.mkdtemp(prefix=f"deepcompile_generated_rank_{_rank()}_"))
        loaded_entrypoints = {}
        source_paths = {}
        turn_index = 0
        while True:
            selection_only = len(history) >= self.max_iterations
            turn_dir = iteration_root / f"turn_{turn_index:03d}" if is_rank_zero else None
            mechanical_feedback = []
            if is_rank_zero:
                response = None
                response_error = None
                selection_proposal = None
                for attempt in range(self.max_retries + 1):
                    try:
                        prompt = serialize_search_context(ctx,
                                                          frozen,
                                                          reference_inventory,
                                                          history,
                                                          mechanical_feedback=mechanical_feedback,
                                                          selection_only=selection_only)
                        prefix = "search" if attempt == 0 else f"search_retry_{attempt}"
                        result = self._checked_run(self.graph_agent_runner,
                                                   prompt,
                                                   turn_dir,
                                                   "coding_agent",
                                                   artifact_prefix=prefix)
                        response = parse_search_response(result.stdout, len(history), history, ctx.graph_slot)
                        if selection_only and isinstance(response, GeneratedPassProposal):
                            raise AgentResponseError("Candidate budget is exhausted; this turn must finish and select")
                        if isinstance(response, GeneratedPassSelection):
                            selection_proposal = validate_selection(response, history, frozen.graph_fingerprint)
                        break
                    except Exception as exc:
                        response = None
                        response_error = str(exc)
                        mechanical_feedback.append(response_error)
                        trace.append(
                            OptimizationTraceEntry(iteration=turn_index,
                                                   action="coding_agent_retry",
                                                   summary=f"Mechanical coding-agent response failed: {exc}",
                                                   details={"attempt": attempt + 1}))
                if isinstance(response, GeneratedPassProposal):
                    control_envelope = {"action": "evaluate", "proposal": response.to_dict(), "error": None}
                elif isinstance(response, GeneratedPassSelection):
                    control_envelope = {
                        "action": "finish",
                        "selection": response.to_dict(),
                        "proposal": None if selection_proposal is None else selection_proposal.to_dict(),
                        "error": None,
                    }
                else:
                    control_envelope = {"action": "abort", "error": response_error}
                    retain_temporary_root = True
            else:
                control_envelope = None
            control_envelope = broadcast_json_payload(control_envelope)

            if control_envelope["action"] == "abort":
                if is_rank_zero:
                    trace.append(
                        OptimizationTraceEntry(iteration=turn_index,
                                               action="abort",
                                               summary="Coding-agent response retries exhausted; no selection made",
                                               details={"error": control_envelope.get("error")}))
                break

            if control_envelope["action"] == "finish":
                selection = GeneratedPassSelection(**control_envelope["selection"])
                selected_proposal = None
                selection_error = None
                finalization_error = None
                try:
                    selected_proposal = validate_selection(selection, history, frozen.graph_fingerprint)
                    rebroadcast_proposal = control_envelope.get("proposal")
                    expected_proposal = None if selected_proposal is None else selected_proposal.to_dict()
                    if rebroadcast_proposal != expected_proposal:
                        raise AgentResponseError("Final selected source envelope does not match stored exact bytes")
                    local_selection = {"rank": _rank(), "success": True, "error": None}
                except Exception as exc:
                    selection_error = str(exc)
                    local_selection = {"rank": _rank(), "success": False, "error": selection_error}
                selection_records = gather_rank_records(local_selection)
                if not all(record["success"] for record in selection_records):
                    if is_rank_zero:
                        trace.append(
                            OptimizationTraceEntry(iteration=turn_index,
                                                   action="abort",
                                                   summary="Final selection identity was invalid",
                                                   details={"rank_results": selection_records}))
                    break

                if selection.kind == "baseline":
                    live_fingerprint = None
                    baseline_error = None
                    try:
                        live_fingerprint = structural_fingerprint(ctx.gm, ctx.graph_id)
                        if live_fingerprint != frozen.graph_fingerprint:
                            raise RuntimeError("Live graph no longer matches the frozen baseline")
                    except Exception as exc:
                        baseline_error = str(exc)
                    final_record = {
                        "rank": _rank(),
                        "success": baseline_error is None,
                        "fingerprint": live_fingerprint,
                        "error": baseline_error,
                    }
                    final_records = gather_rank_records(final_record)
                    final_application = {
                        "selection": selection.to_dict(),
                        "success": all(record["success"] for record in final_records),
                        "live_source_executed": False,
                        "rank_results": final_records,
                    }
                else:
                    entrypoint = loaded_entrypoints.get(selected_proposal.candidate_id)
                    selected_history = next(record for record in history
                                            if record["proposal"]["candidate_id"] == selected_proposal.candidate_id)
                    loaded_records = gather_rank_records({
                        "rank":
                        _rank(),
                        "success":
                        entrypoint is not None,
                        "error":
                        None if entrypoint is not None else "Selected generated-pass module is not loaded",
                    })
                    if not all(record["success"] for record in loaded_records):
                        final_application = {
                            "success": False,
                            "live_source_executed": False,
                            "error": "Selected generated-pass module is not loaded on every rank",
                            "rank_results": loaded_records,
                        }
                    else:
                        try:
                            final_application = self._finalize_candidate(ctx, frozen, selected_proposal,
                                                                         selected_history["evaluation"], entrypoint,
                                                                         source_paths[selected_proposal.candidate_id])
                        except Exception as exc:
                            finalization_error = exc
                            final_application = {
                                "success": False,
                                "live_source_executed": True,
                                "candidate_id": selected_proposal.candidate_id,
                                "source_sha256": selected_proposal.source_sha256,
                                "error": str(exc),
                            }

                if is_rank_zero:
                    _safe_write_json(iteration_root / "selection.json", selection.to_dict())
                    _safe_write_json(iteration_root / "final_application.json", final_application)
                    action = "selected" if final_application["success"] else "abort"
                    summary = selection.summary if final_application["success"] else "Final selection failed"
                    trace.append(
                        OptimizationTraceEntry(iteration=turn_index,
                                               action=action,
                                               summary=summary,
                                               details=final_application))
                if not final_application["success"]:
                    retain_temporary_root = True
                if finalization_error is not None:
                    raise finalization_error
                break

            proposal = GeneratedPassProposal(**control_envelope["proposal"])
            proposal_index = len(history)
            source_path = rank_source_root / proposal.candidate_id / "generated_pass.py"
            source_paths[proposal.candidate_id] = source_path
            candidate, candidate_profile, entrypoint, evaluation = self._apply_and_profile_candidate(
                ctx, frozen, proposal, proposal_index, source_path)
            if entrypoint is not None:
                loaded_entrypoints[proposal.candidate_id] = entrypoint
            history_entry = {
                "turn_index": turn_index,
                "summary": proposal.summary,
                "proposal": proposal.to_dict(),
                "evaluation": evaluation,
            }
            history.append(history_entry)
            if is_rank_zero:
                candidate_dir = iteration_root / f"candidate_{proposal_index:03d}"
                _safe_write_text(candidate_dir / "generated_pass.py", proposal.source)
                _safe_write_json(
                    candidate_dir / "proposal.json", {
                        **proposal.to_dict(),
                        "source": "stored in generated_pass.py",
                        "turn_index": turn_index,
                        "frozen_base_fingerprint": frozen.graph_fingerprint,
                        "inventory_sha256": reference_inventory["inventory_sha256"],
                    })
                _safe_write_json(candidate_dir / "validation.json", evaluation["validation"])
                _safe_write_json(candidate_dir / "evaluation.json", evaluation)
                if candidate is not None and inspection_enabled:
                    _safe_write_text(candidate_dir / "candidate_graph.py", candidate.code)
                trace.append(
                    OptimizationTraceEntry(iteration=proposal_index,
                                           action="evaluated",
                                           summary=proposal.summary,
                                           details={
                                               "candidate_id": proposal.candidate_id,
                                               "valid": evaluation["valid"],
                                               "result_fingerprint": evaluation["result_fingerprint"],
                                           }))

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
                    trace.append(
                        OptimizationTraceEntry(iteration=proposal_index,
                                               action="abort",
                                               summary="Post-candidate cleanup failed",
                                               details={"rank_results": cleanup_records}))
                break
            turn_index += 1

        if is_rank_zero and not inspection_enabled and not retain_temporary_root and iteration_root.exists():
            shutil.rmtree(iteration_root, ignore_errors=True)
        return OptimizationResult(trace=trace)
