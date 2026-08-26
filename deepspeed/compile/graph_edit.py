# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import importlib
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.fx import GraphModule, Node


SCHEMA_VERSION = 1
FX_NODE_OPS = {"placeholder", "get_attr", "call_function", "call_method", "call_module", "output"}
SCALAR_TYPES = (str, int, float, bool, type(None))
RUNTIME_GRAPH_ID = {"runtime_local": "graph_id"}


class GraphEditError(ValueError):
    pass


def _callable_path(target: Any) -> str:
    target_text = str(target)
    if target_text.startswith(("aten.", "prims.", "dc.", "_c10d_functional.")):
        return f"torch.ops.{target_text}"

    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if module == "_operator":
        module = "operator"
    if not module or not qualname or "<locals>" in qualname or "<lambda>" in qualname:
        raise GraphEditError(f"Callable target '{target}' has no import-resolvable symbolic path")
    return f"{module}.{qualname}"


def _resolve_path(path: str) -> Any:
    if not isinstance(path, str) or not path:
        raise GraphEditError("Symbolic targets must be non-empty strings")

    components = path.split(".")
    imported = None
    imported_count = 0
    for count in range(len(components), 0, -1):
        try:
            imported = importlib.import_module(".".join(components[:count]))
            imported_count = count
            break
        except ImportError:
            continue
    if imported is None:
        raise GraphEditError(f"Unable to import symbolic target '{path}'")

    target = imported
    try:
        for component in components[imported_count:]:
            target = getattr(target, component)
    except AttributeError as exc:
        raise GraphEditError(f"Unable to resolve symbolic target '{path}'") from exc
    return target


def resolve_callable_target(target_spec: Any) -> Any:
    if isinstance(target_spec, str):
        path = target_spec
    elif isinstance(target_spec, dict):
        module = target_spec.get("module")
        qualname = target_spec.get("qualname")
        if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
            raise GraphEditError("Callable target objects require non-empty module and qualname strings")
        path = f"{module}.{qualname}"
    else:
        raise GraphEditError("call_function targets must be symbolic strings or module/qualname objects")

    target = _resolve_path(path)
    if not callable(target):
        raise GraphEditError(f"Symbolic target '{path}' does not resolve to a callable")
    return target


def _is_current_cuda_device(device: torch.device) -> bool:
    if device.type != "cuda" or device.index is None:
        return True
    try:
        return device.index == torch.cuda.current_device()
    except Exception:
        return False


def encode_argument(value: Any, node_ids: Dict[Node, str], runtime_graph_id: Optional[int] = None) -> Any:
    if isinstance(value, Node):
        if value not in node_ids:
            raise GraphEditError(f"Node '{value.name}' has no stable ID in this graph")
        return {"node": node_ids[value]}
    if runtime_graph_id is not None and type(value) is int and value == runtime_graph_id:
        return dict(RUNTIME_GRAPH_ID)
    if isinstance(value, SCALAR_TYPES):
        return value
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, slice):
        return {"slice": [
            encode_argument(value.start, node_ids, runtime_graph_id),
            encode_argument(value.stop, node_ids, runtime_graph_id),
            encode_argument(value.step, node_ids, runtime_graph_id),
        ]}
    if isinstance(value, tuple):
        return {"tuple": [encode_argument(item, node_ids, runtime_graph_id) for item in value]}
    if isinstance(value, list):
        return [encode_argument(item, node_ids, runtime_graph_id) for item in value]
    if isinstance(value, dict):
        return {
            "dict": [[
                encode_argument(key, node_ids, runtime_graph_id),
                encode_argument(item, node_ids, runtime_graph_id)
            ]
                     for key, item in value.items()]
        }
    if isinstance(value, torch.dtype):
        dtype_name = str(value)
        if dtype_name.startswith("torch."):
            dtype_name = dtype_name[len("torch."):]
        return {"torch_dtype": dtype_name}
    if isinstance(value, torch.device):
        index = "current" if _is_current_cuda_device(value) else value.index
        return {"torch_device": {"type": value.type, "index": index}}
    torch_symbol_types = tuple(
        symbol_type for symbol_type in (getattr(torch, "layout", None), getattr(torch, "memory_format", None),
                                        getattr(torch, "qscheme", None)) if isinstance(symbol_type, type))
    if isinstance(value, torch_symbol_types):
        path = str(value)
        if _resolve_path(path) is value:
            return {"python_symbol": path}

    try:
        path = _callable_path(value)
    except GraphEditError as exc:
        raise GraphEditError(f"Argument value of type '{type(value).__name__}' is not data-serializable") from exc
    if _resolve_path(path) is not value:
        raise GraphEditError(f"Argument symbol '{path}' does not round-trip to the local value")
    return {"python_symbol": path}


def decode_argument(value: Any, nodes: Dict[str, Node], runtime_graph_id: Optional[int] = None) -> Any:
    if isinstance(value, SCALAR_TYPES):
        return value
    if isinstance(value, list):
        return [decode_argument(item, nodes, runtime_graph_id) for item in value]
    if not isinstance(value, dict):
        raise GraphEditError(f"Encoded argument has unsupported type '{type(value).__name__}'")

    if set(value) == {"node"}:
        node_id = value["node"]
        if not isinstance(node_id, str) or node_id not in nodes:
            raise GraphEditError(f"Argument references unavailable node ID '{node_id}'")
        return nodes[node_id]
    if value == RUNTIME_GRAPH_ID:
        if runtime_graph_id is None:
            raise GraphEditError("runtime-local graph_id requires the local graph ID during replay")
        return runtime_graph_id
    if set(value) == {"tuple"}:
        items = value["tuple"]
        if not isinstance(items, list):
            raise GraphEditError("tuple encodings require an array")
        return tuple(decode_argument(item, nodes, runtime_graph_id) for item in items)
    if set(value) == {"dict"}:
        items = value["dict"]
        if not isinstance(items, list):
            raise GraphEditError("dict encodings require an array of key/value pairs")
        decoded = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                raise GraphEditError("dict entries must be two-element arrays")
            decoded[decode_argument(pair[0], nodes, runtime_graph_id)] = decode_argument(pair[1], nodes,
                                                                                         runtime_graph_id)
        return decoded
    if set(value) == {"slice"}:
        items = value["slice"]
        if not isinstance(items, list) or len(items) != 3:
            raise GraphEditError("slice encodings require [start, stop, step]")
        return slice(*(decode_argument(item, nodes, runtime_graph_id) for item in items))
    if set(value) == {"ellipsis"} and value["ellipsis"] is True:
        return Ellipsis
    if set(value) == {"torch_dtype"}:
        dtype = getattr(torch, value["torch_dtype"], None)
        if not isinstance(dtype, torch.dtype):
            raise GraphEditError(f"Unknown torch dtype '{value['torch_dtype']}'")
        return dtype
    if set(value) == {"torch_device"}:
        spec = value["torch_device"]
        if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
            raise GraphEditError("torch_device requires type and optional index")
        index = spec.get("index")
        if index == "current" and spec["type"] == "cuda":
            index = torch.cuda.current_device()
        return torch.device(spec["type"], index) if index is not None else torch.device(spec["type"])
    if set(value) == {"python_symbol"}:
        return _resolve_path(value["python_symbol"])
    raise GraphEditError(f"Unknown encoded argument object keys {sorted(value)}")


def base_node_ids(graph) -> Tuple[Dict[Node, str], Dict[str, Node]]:
    by_node = {}
    by_id = {}
    for position, node in enumerate(graph.nodes):
        node_id = f"base:{position}"
        by_node[node] = node_id
        by_id[node_id] = node
    return by_node, by_id


def _target_descriptor(node: Node) -> Any:
    if node.op == "call_function":
        try:
            return _callable_path(node.target)
        except GraphEditError:
            target_type = type(node.target)
            module = getattr(node.target, "__module__", None)
            qualname = getattr(node.target, "__qualname__", None)
            name = getattr(node.target, "__name__", None)
            return {
                "opaque_callable": {
                    "module": module if isinstance(module, str) else None,
                    "qualname": qualname if isinstance(qualname, str) else None,
                    "name": name if isinstance(name, str) else None,
                    "type": f"{target_type.__module__}.{target_type.__qualname__}",
                }
            }
    return str(node.target)


def normalized_graph(gm: GraphModule,
                     include_hints: bool = False,
                     runtime_graph_id: Optional[int] = None) -> List[Dict[str, Any]]:
    node_ids, _ = base_node_ids(gm.graph)
    normalized = []
    for node in gm.graph.nodes:
        item = {
            "id": node_ids[node],
            "op": node.op,
            "target": _target_descriptor(node),
            "args": [encode_argument(arg, node_ids, runtime_graph_id) for arg in node.args],
            "kwargs": {
                key: encode_argument(value, node_ids, runtime_graph_id) for key, value in node.kwargs.items()
            },
        }
        if include_hints:
            item["name_hint"] = node.name
            item["profile"] = {
                key: value for key, value in node.meta.items()
                if key in {"device_time", "wall_time", "tensor_size", "alloc_mem", "max_mem"}
                and isinstance(value, SCALAR_TYPES)
            }
        normalized.append(item)
    return normalized


def structural_fingerprint(gm: GraphModule, runtime_graph_id: Optional[int] = None) -> str:
    encoded = json.dumps(normalized_graph(gm, runtime_graph_id=runtime_graph_id),
                         sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def normalized_graph_order(graph_order: List[Tuple[int, bool]], runtime_graph_id: int) -> List[Dict[str, Any]]:
    return [{
        "position": position,
        "direction": "bwd" if bwd else "fwd",
        "current_graph": graph_id == runtime_graph_id,
    } for position, (graph_id, bwd) in enumerate(graph_order)]


def clone_graph_module(gm: GraphModule) -> GraphModule:
    graph = copy.deepcopy(gm.graph)
    clone = GraphModule(gm, graph)
    clone.meta = dict(getattr(gm, "meta", {}))
    clone.recompile()
    return clone


def _resolve_module_attr(root: Any, target: str) -> Any:
    value = root
    try:
        for atom in target.split("."):
            value = getattr(value, atom)
    except AttributeError as exc:
        raise GraphEditError(f"GraphModule target '{target}' is unavailable locally") from exc
    return value


def _data_only_meta_patch(patch: Any) -> Dict[str, Any]:
    if patch is None:
        return {}
    if not isinstance(patch, dict) or any(not isinstance(key, str) for key in patch):
        raise GraphEditError("meta patches must be objects with string keys")

    def validate(value: Any, path: str) -> None:
        if type(value) in SCALAR_TYPES:
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                validate(item, f"{path}[{index}]")
            return
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            for key, item in value.items():
                validate(item, f"{path}.{key}")
            return
        raise GraphEditError(f"meta patch value at '{path}' is not recursively JSON/data-only")

    validate(patch, "meta")
    try:
        json.dumps(patch, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise GraphEditError(f"meta patch is not JSON/data-only: {exc}") from exc
    return copy.deepcopy(patch)


class GraphEditor:

    def __init__(self, gm: GraphModule, runtime_graph_id: Optional[int] = None):
        self.gm = gm
        self.runtime_graph_id = runtime_graph_id
        _, self.nodes = base_node_ids(gm.graph)
        self.operations = []

    def _node(self, node_id: Any) -> Node:
        if not isinstance(node_id, str) or node_id not in self.nodes:
            raise GraphEditError(f"Operation references unavailable node ID '{node_id}'")
        return self.nodes[node_id]

    def _args(self, payload: Any) -> Tuple[Any, ...]:
        if not isinstance(payload, list):
            raise GraphEditError("Node args must be an array")
        return tuple(decode_argument(value, self.nodes, self.runtime_graph_id) for value in payload)

    def _kwargs(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
            raise GraphEditError("Node kwargs must be an object with string keys")
        return {
            key: decode_argument(value, self.nodes, self.runtime_graph_id) for key, value in payload.items()
        }

    def create_node(self, operation: Dict[str, Any]) -> None:
        node_id = operation.get("id")
        if not isinstance(node_id, str) or not node_id or node_id in self.nodes:
            raise GraphEditError(f"create_node requires a unique non-empty id, received '{node_id}'")
        node_op = operation.get("node_op")
        if node_op not in FX_NODE_OPS:
            raise GraphEditError(f"Unsupported FX node op '{node_op}'")
        target_spec = operation.get("target", "output" if node_op == "output" else None)
        if node_op == "call_function":
            target = resolve_callable_target(target_spec)
        elif not isinstance(target_spec, str) or not target_spec:
            raise GraphEditError(f"{node_op} targets must be non-empty strings")
        else:
            target = target_spec
            if node_op == "call_module":
                module = _resolve_module_attr(self.gm, target)
                if not isinstance(module, torch.nn.Module):
                    raise GraphEditError(f"call_module target '{target}' is not a module")
            elif node_op == "get_attr":
                _resolve_module_attr(self.gm, target)

        args = self._args(operation.get("args", []))
        kwargs = self._kwargs(operation.get("kwargs", {}))
        output_nodes = [node for node in self.gm.graph.nodes if node.op == "output"]
        context = self.gm.graph.inserting_before(output_nodes[0]) if output_nodes else self.gm.graph.inserting_after()
        with context:
            node = self.gm.graph.create_node(node_op,
                                             target,
                                             args=args,
                                             kwargs=kwargs,
                                             name=operation.get("name_hint"))

        copy_meta_from = operation.get("copy_meta_from")
        if copy_meta_from is not None:
            node.meta = dict(self._node(copy_meta_from).meta)
        node.meta.update(_data_only_meta_patch(operation.get("meta")))
        self.nodes[node_id] = node

    def set_args_kwargs(self, operation: Dict[str, Any]) -> None:
        node = self._node(operation.get("id"))
        if "args" in operation:
            node.args = self._args(operation["args"])
        if "kwargs" in operation:
            node.kwargs = self._kwargs(operation["kwargs"])
        if "args" not in operation and "kwargs" not in operation:
            raise GraphEditError("set_args_kwargs/rewire must provide args or kwargs")

    def delete_node(self, operation: Dict[str, Any]) -> None:
        node_id = operation.get("id")
        node = self._node(node_id)
        self.gm.graph.erase_node(node)
        del self.nodes[node_id]

    def reorder(self, operation: Dict[str, Any]) -> None:
        order = operation.get("order")
        if not isinstance(order, list) or any(not isinstance(node_id, str) for node_id in order):
            raise GraphEditError("reorder.order must be an array of node IDs")
        if len(order) != len(set(order)):
            raise GraphEditError("reorder.order contains duplicate node IDs")
        if set(order) != set(self.nodes):
            missing = sorted(set(self.nodes) - set(order))
            extra = sorted(set(order) - set(self.nodes))
            raise GraphEditError("reorder.order must name every final node exactly once; "
                                 f"missing={missing}, extra={extra}")
        ordered_nodes = [self.nodes[node_id] for node_id in order]
        for index in range(len(ordered_nodes) - 2, -1, -1):
            ordered_nodes[index + 1].prepend(ordered_nodes[index])

    def patch_meta(self, operation: Dict[str, Any]) -> None:
        self._node(operation.get("id")).meta.update(_data_only_meta_patch(operation.get("meta")))

    def apply(self, operations: List[Dict[str, Any]]) -> None:
        if not isinstance(operations, list):
            raise GraphEditError("operations must be an array")
        reorder_count = 0
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                raise GraphEditError(f"Operation {index} must be an object")
            op = operation.get("op")
            if op == "create_node":
                self.create_node(operation)
            elif op in {"set_args_kwargs", "rewire"}:
                self.set_args_kwargs(operation)
            elif op == "delete_node":
                self.delete_node(operation)
            elif op == "reorder":
                self.reorder(operation)
                reorder_count += 1
                if index != len(operations) - 1:
                    raise GraphEditError("The complete reorder operation must be the final operation")
            elif op == "patch_meta":
                self.patch_meta(operation)
            else:
                raise GraphEditError(f"Unsupported graph edit operation '{op}'")
            self.operations.append(copy.deepcopy(operation))
        if reorder_count != 1:
            raise GraphEditError("A graph edit log must end with exactly one complete reorder operation")
        self.gm.graph.lint()
        self.gm.recompile()


@dataclass
class GraphEditPayload:
    generation: int
    graph_slot: Tuple[int, str]
    base_fingerprint: str
    expected_result_fingerprint: Optional[str]
    operations: List[Dict[str, Any]]
    reason: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "graph_slot": list(self.graph_slot),
            "base_fingerprint": self.base_fingerprint,
            "expected_result_fingerprint": self.expected_result_fingerprint,
            "reason": self.reason,
            "operations": self.operations,
        }

    @classmethod
    def from_dict(cls, payload: Any, require_result_fingerprint: bool = False) -> "GraphEditPayload":
        if not isinstance(payload, dict):
            raise GraphEditError("Graph edit payload must be an object")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise GraphEditError(f"Unsupported graph edit schema_version {payload.get('schema_version')}")
        generation = payload.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise GraphEditError("generation must be a positive integer")
        slot = payload.get("graph_slot")
        if (not isinstance(slot, list) or len(slot) != 2 or isinstance(slot[0], bool)
                or not isinstance(slot[0], int) or slot[0] < 0 or slot[1] not in {"fwd", "bwd"}):
            raise GraphEditError("graph_slot must be [non-negative index, 'fwd'|'bwd']")
        base = payload.get("base_fingerprint")
        if not isinstance(base, str) or not base:
            raise GraphEditError("base_fingerprint must be a non-empty string")
        result = payload.get("expected_result_fingerprint")
        if result is not None and (not isinstance(result, str) or not result):
            raise GraphEditError("expected_result_fingerprint must be a non-empty string or null")
        if require_result_fingerprint and result is None:
            raise GraphEditError("Synchronized graph edits require expected_result_fingerprint")
        operations = payload.get("operations")
        if not isinstance(operations, list):
            raise GraphEditError("operations must be an array")
        reason = payload.get("reason", "")
        if not isinstance(reason, str):
            raise GraphEditError("reason must be a string")
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise GraphEditError(f"Graph edit payload is not JSON/data-only: {exc}") from exc
        return cls(schema_version=SCHEMA_VERSION,
                   generation=generation,
                   graph_slot=(slot[0], slot[1]),
                   base_fingerprint=base,
                   expected_result_fingerprint=result,
                   reason=reason,
                   operations=copy.deepcopy(operations))


def candidate_fingerprint(gm: GraphModule,
                          payload: GraphEditPayload,
                          runtime_graph_id: Optional[int] = None) -> str:
    canonical_payload = copy.deepcopy(payload.to_dict())
    canonical_payload["expected_result_fingerprint"] = None
    encoded = json.dumps({
        "result_structure": structural_fingerprint(gm, runtime_graph_id),
        "finalized_edit": canonical_payload,
    },
                         sort_keys=True,
                         separators=(",", ":"),
                         allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def apply_graph_edit(gm: GraphModule,
                     payload: GraphEditPayload,
                     runtime_graph_id: Optional[int] = None) -> GraphModule:
    local_base = structural_fingerprint(gm, runtime_graph_id)
    if local_base != payload.base_fingerprint:
        raise GraphEditError(f"Graph edit base fingerprint {payload.base_fingerprint} does not match local graph "
                             f"{local_base}")
    candidate = clone_graph_module(gm)
    editor = GraphEditor(candidate, runtime_graph_id)
    editor.apply(payload.operations)
    result = candidate_fingerprint(candidate, payload, runtime_graph_id)
    if payload.expected_result_fingerprint is not None and result != payload.expected_result_fingerprint:
        raise GraphEditError(f"Graph edit result fingerprint {result} does not match expected "
                             f"{payload.expected_result_fingerprint}")
    return candidate


def finalize_graph_edit(gm: GraphModule,
                        payload: GraphEditPayload,
                        runtime_graph_id: Optional[int] = None) -> Tuple[GraphEditPayload, GraphModule]:
    if payload.expected_result_fingerprint is not None:
        raise GraphEditError("Rank-zero optimizer payload must leave expected_result_fingerprint null")
    candidate = apply_graph_edit(gm, payload, runtime_graph_id)
    finalized = GraphEditPayload(schema_version=payload.schema_version,
                                 generation=payload.generation,
                                 graph_slot=payload.graph_slot,
                                 base_fingerprint=payload.base_fingerprint,
                                 expected_result_fingerprint=None,
                                 reason=payload.reason,
                                 operations=copy.deepcopy(payload.operations))
    finalized.expected_result_fingerprint = candidate_fingerprint(candidate, finalized, runtime_graph_id)
    return finalized, candidate
