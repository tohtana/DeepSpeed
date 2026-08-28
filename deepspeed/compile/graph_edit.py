# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import types
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.fx import GraphModule, Node

from deepspeed.accelerator import get_accelerator

SCALAR_TYPES = (str, int, float, bool, type(None))
RUNTIME_GRAPH_ID = {"runtime_local": "graph_id"}


class GraphDescriptionError(ValueError):
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
        raise GraphDescriptionError(f"Callable target '{target}' has no import-resolvable symbolic path")
    return f"{module}.{qualname}"


def _resolve_path(path: str) -> Any:
    if not isinstance(path, str) or not path:
        raise GraphDescriptionError("Symbolic targets must be non-empty strings")

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
        raise GraphDescriptionError(f"Unable to import symbolic target '{path}'")

    target = imported
    try:
        for component in components[imported_count:]:
            target = getattr(target, component)
    except AttributeError as exc:
        raise GraphDescriptionError(f"Unable to resolve symbolic target '{path}'") from exc
    return target


def _is_current_cuda_device(device: torch.device) -> bool:
    if device.type != "cuda" or device.index is None:
        return True
    try:
        return device.index == get_accelerator().current_device()
    except Exception:
        return False


def _torch_constant_name(value: Any) -> str:
    name = str(value).removeprefix("torch.")
    if not name or getattr(torch, name, None) is not value:
        raise GraphDescriptionError(f"PyTorch constant '{value}' has no stable torch attribute name")
    return name


def encode_argument(value: Any, node_ids: Dict[Node, str], runtime_graph_id: Optional[int] = None) -> Any:
    """Encode the known post-structural-pass graph used in the coding-agent prompt."""
    if isinstance(value, Node):
        if value not in node_ids:
            raise GraphDescriptionError(f"Node '{value.name}' has no stable ID in this graph")
        return {"node": node_ids[value]}
    if runtime_graph_id is not None and type(value) is int and value == runtime_graph_id:
        return dict(RUNTIME_GRAPH_ID)
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_float": str(value)}
    if isinstance(value, SCALAR_TYPES):
        return value
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, slice):
        return {
            "slice": [
                encode_argument(value.start, node_ids, runtime_graph_id),
                encode_argument(value.stop, node_ids, runtime_graph_id),
                encode_argument(value.step, node_ids, runtime_graph_id),
            ]
        }
    if isinstance(value, tuple):
        return {"tuple": [encode_argument(item, node_ids, runtime_graph_id) for item in value]}
    if isinstance(value, list):
        return [encode_argument(item, node_ids, runtime_graph_id) for item in value]
    if isinstance(value, dict):
        return {
            "dict": [[
                encode_argument(key, node_ids, runtime_graph_id),
                encode_argument(item, node_ids, runtime_graph_id),
            ] for key, item in value.items()]
        }
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        index = "current" if _is_current_cuda_device(value) else value.index
        return {"torch_device": {"type": value.type, "index": index}}
    if isinstance(value, torch.memory_format):
        return {"torch_memory_format": _torch_constant_name(value)}
    if isinstance(value, torch.layout):
        return {"torch_layout": _torch_constant_name(value)}

    try:
        path = _callable_path(value)
    except GraphDescriptionError as exc:
        raise GraphDescriptionError(f"Argument value of type '{type(value).__name__}' is not describable") from exc
    if _resolve_path(path) is not value:
        raise GraphDescriptionError(f"Argument symbol '{path}' does not round-trip to the local value")
    return {"python_symbol": path}


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
        except GraphDescriptionError:
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
                key: encode_argument(value, node_ids, runtime_graph_id)
                for key, value in node.kwargs.items()
            },
        }
        if include_hints:
            item["name_hint"] = node.name
            item["profile"] = {
                key: value
                for key, value in node.meta.items()
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


def _mutable_container_ids(value: Any, seen: Optional[Set[int]] = None) -> Set[int]:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return set()
    seen.add(identity)

    identities = set()
    if isinstance(value, dict):
        identities.add(identity)
        children = list(value.keys()) + list(value.values())
    elif isinstance(value, (list, set)):
        identities.add(identity)
        children = list(value)
    elif isinstance(value, tuple):
        children = list(value)
    else:
        children = []
    for child in children:
        identities.update(_mutable_container_ids(child, seen))
    return identities


def _tensor_identity_memo(value: Any,
                          memo: Optional[Dict[int, Any]] = None,
                          seen: Optional[Set[int]] = None) -> Dict[int, Any]:
    if memo is None:
        memo = {}
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return memo
    seen.add(identity)

    if isinstance(value, torch.Tensor):
        memo[identity] = value
        return memo
    if isinstance(value, dict):
        children = list(value.keys()) + list(value.values())
    elif isinstance(value, (list, set, tuple)):
        children = list(value)
    else:
        children = []
    for child in children:
        _tensor_identity_memo(child, memo, seen)
    return memo


def clone_graph_module(gm: GraphModule) -> GraphModule:
    """Clone graph topology and make all per-node metadata containers independent."""
    graph = copy.deepcopy(gm.graph)
    clone = GraphModule(gm, graph)
    source_nodes = list(gm.graph.nodes)
    cloned_nodes = list(clone.graph.nodes)
    if len(source_nodes) != len(cloned_nodes):
        raise RuntimeError("Cloned FX graph has a different node count")
    for source_node, cloned_node in zip(source_nodes, cloned_nodes):
        cloned_node.meta = copy.deepcopy(source_node.meta, memo=_tensor_identity_memo(source_node.meta))
        source_containers = _mutable_container_ids(source_node.meta)
        cloned_containers = _mutable_container_ids(cloned_node.meta)
        if source_containers.intersection(cloned_containers):
            raise RuntimeError(f"Cloned metadata for node '{source_node.name}' aliases the frozen graph")
    module_meta = getattr(gm, "meta", {})
    clone.meta = copy.deepcopy(module_meta, memo=_tensor_identity_memo(module_meta))
    clone.recompile()
    return clone


def _opaque_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _code_constant(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_float": str(value)}
    if isinstance(value, SCALAR_TYPES):
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if isinstance(value, tuple):
        return {"tuple": [_code_constant(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_code_constant(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"frozenset": items}
    if isinstance(value, types.CodeType):
        return {"code": _code_descriptor(value)}
    return {"type": _opaque_type(value)}


def _code_descriptor(code: types.CodeType) -> Dict[str, Any]:
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
        "bytecode": code.co_code.hex(),
        "constants": [_code_constant(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _stable_binding_value(value: Any, opaque_types: List[str], seen: Optional[Set[int]] = None) -> Any:
    if seen is None:
        seen = set()
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_float": str(value)}
    if isinstance(value, SCALAR_TYPES):
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        return {"torch_device": {"type": value.type}}
    if isinstance(value, torch.memory_format):
        return {"torch_memory_format": _torch_constant_name(value)}
    if isinstance(value, torch.layout):
        return {"torch_layout": _torch_constant_name(value)}
    if isinstance(value, torch.Tensor):
        opaque_types.append("tensor_binding")
        shape = [dimension if isinstance(dimension, int) else str(dimension) for dimension in value.shape]
        return {
            "tensor": {
                "dtype": str(value.dtype),
                "shape": shape,
                "device_type": value.device.type,
                "requires_grad": value.requires_grad,
            }
        }

    identity = id(value)
    if identity in seen:
        return {"recursive_binding": _opaque_type(value)}
    seen.add(identity)
    try:
        if isinstance(value, tuple):
            return {"tuple": [_stable_binding_value(item, opaque_types, seen) for item in value]}
        if isinstance(value, list):
            return [_stable_binding_value(item, opaque_types, seen) for item in value]
        if isinstance(value, (set, frozenset)):
            items = [_stable_binding_value(item, opaque_types, seen) for item in value]
            items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
            return {"set": items}
        if isinstance(value, dict):
            items = [[
                _stable_binding_value(key, opaque_types, seen),
                _stable_binding_value(item, opaque_types, seen),
            ] for key, item in value.items()]
            items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
            return {"dict": items}
        if callable(value):
            return {"callable": _callable_binding_descriptor(value, opaque_types, seen)}
    finally:
        seen.remove(identity)

    opaque_type = _opaque_type(value)
    opaque_types.append(opaque_type)
    return {"opaque_binding": {"type": opaque_type}}


def _callable_binding_descriptor(target: Any, opaque_types: List[str], seen: Optional[Set[int]] = None) -> Any:
    descriptor = {"type": _opaque_type(target)}
    try:
        descriptor["path"] = _callable_path(target)
    except Exception:
        module = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", None)
        name = getattr(target, "__name__", None)
        opaque_types.append("opaque_callable")
        descriptor["opaque_path"] = {
            "module": module if isinstance(module, str) else None,
            "qualname": qualname if isinstance(qualname, str) else None,
            "name": name if isinstance(name, str) else None,
        }

    implementation = getattr(target, "__code__", None)
    if implementation is None:
        implementation = getattr(getattr(target, "__func__", None), "__code__", None)
    if implementation is None:
        implementation = getattr(getattr(type(target), "__call__", None), "__code__", None)
    if isinstance(implementation, types.CodeType):
        encoded = json.dumps(_code_descriptor(implementation), sort_keys=True, separators=(",", ":"))
        descriptor["implementation_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    defaults = getattr(target, "__defaults__", None)
    if defaults:
        descriptor["defaults"] = _stable_binding_value(defaults, opaque_types, seen)
    keyword_defaults = getattr(target, "__kwdefaults__", None)
    if keyword_defaults:
        descriptor["keyword_defaults"] = _stable_binding_value(keyword_defaults, opaque_types, seen)
    closure = getattr(target, "__closure__", None)
    if closure:
        closure_values = []
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                closure_values.append({"empty_cell": True})
            else:
                closure_values.append(_stable_binding_value(value, opaque_types, seen))
        descriptor["closure"] = closure_values
    return descriptor


def _module_tensor_descriptor(tensor: Optional[torch.Tensor]) -> Any:
    if tensor is None:
        return None
    shape = [dimension if isinstance(dimension, int) else str(dimension) for dimension in tensor.shape]
    return {
        "dtype": str(tensor.dtype),
        "shape": shape,
        "device_type": tensor.device.type,
        "requires_grad": tensor.requires_grad,
    }


def _module_binding_descriptor(module: torch.nn.Module, opaque_types: List[str], runtime_graph_id: Optional[int],
                               module_stack: Set[int]) -> Dict[str, Any]:
    descriptor = {
        "type": _opaque_type(module),
        "training": module.training,
        "parameters": {
            name: _module_tensor_descriptor(parameter)
            for name, parameter in module.named_parameters(recurse=False)
        },
        "buffers": {
            name: _module_tensor_descriptor(buffer)
            for name, buffer in module.named_buffers(recurse=False)
        },
    }
    identity = id(module)
    if identity in module_stack:
        descriptor["recursive"] = True
        return descriptor

    module_stack.add(identity)
    try:
        attributes = {}
        for name, value in sorted(vars(module).items()):
            if name.startswith("_") or name in {"graph", "meta", "training"}:
                continue
            attributes[name] = _stable_binding_value(value, opaque_types)
        if attributes:
            descriptor["attributes"] = attributes
        children = {
            name: _module_binding_descriptor(child, opaque_types, runtime_graph_id, module_stack)
            for name, child in module.named_children()
        }
        if children:
            descriptor["children"] = children
        if isinstance(module, GraphModule):
            descriptor["graph"] = _generated_graph_nodes(module, opaque_types, runtime_graph_id, module_stack)
    finally:
        module_stack.remove(identity)
    return descriptor


def _total_argument(value: Any, node_ids: Dict[Node, str], opaque_types: List[str],
                    runtime_graph_id: Optional[int]) -> Any:
    if isinstance(value, Node):
        return {"node": node_ids.get(value, "unavailable")}
    if runtime_graph_id is not None and type(value) is int and value == runtime_graph_id:
        return dict(RUNTIME_GRAPH_ID)
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_float": str(value)}
    if isinstance(value, SCALAR_TYPES):
        return value
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, slice):
        return {
            "slice": [
                _total_argument(value.start, node_ids, opaque_types, runtime_graph_id),
                _total_argument(value.stop, node_ids, opaque_types, runtime_graph_id),
                _total_argument(value.step, node_ids, opaque_types, runtime_graph_id),
            ]
        }
    if isinstance(value, tuple):
        return {"tuple": [_total_argument(item, node_ids, opaque_types, runtime_graph_id) for item in value]}
    if isinstance(value, list):
        return [_total_argument(item, node_ids, opaque_types, runtime_graph_id) for item in value]
    if isinstance(value, dict):
        return {
            "dict": [[
                _total_argument(key, node_ids, opaque_types, runtime_graph_id),
                _total_argument(item, node_ids, opaque_types, runtime_graph_id),
            ] for key, item in value.items()]
        }
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        index = "current" if _is_current_cuda_device(value) else value.index
        return {"torch_device": {"type": value.type, "index": index}}
    if isinstance(value, torch.memory_format):
        return {"torch_memory_format": _torch_constant_name(value)}
    if isinstance(value, torch.layout):
        return {"torch_layout": _torch_constant_name(value)}
    if isinstance(value, torch.Tensor):
        opaque_types.append("tensor_constant")
        shape = [dimension if isinstance(dimension, int) else str(dimension) for dimension in value.shape]
        return {
            "opaque_tensor": {
                "dtype": str(value.dtype),
                "shape": shape,
                "device_type": value.device.type,
            }
        }
    try:
        return {"python_symbol": _callable_path(value)}
    except Exception:
        opaque_type = _opaque_type(value)
        opaque_types.append(opaque_type)
        return {"opaque_argument": {"type": opaque_type}}


def _total_target_descriptor(gm: GraphModule, node: Node, opaque_types: List[str], runtime_graph_id: Optional[int],
                             module_stack: Set[int]) -> Any:
    if node.op == "call_module":
        target = str(node.target)
        try:
            module = gm.get_submodule(target)
            return {
                "path": target,
                "binding": _module_binding_descriptor(module, opaque_types, runtime_graph_id, module_stack),
            }
        except Exception as exc:
            opaque_type = _opaque_type(exc)
            opaque_types.append(f"call_module_binding_error:{opaque_type}")
            return {"path": target, "binding_error": opaque_type}
    if node.op != "call_function":
        try:
            return str(node.target)
        except Exception:
            opaque_type = _opaque_type(node.target)
            opaque_types.append(opaque_type)
            return {"opaque_target": {"type": opaque_type}}
    return _callable_binding_descriptor(node.target, opaque_types)


def _generated_graph_nodes(gm: GraphModule, opaque_types: List[str], runtime_graph_id: Optional[int],
                           module_stack: Set[int]) -> List[Dict[str, Any]]:
    node_ids, _ = base_node_ids(gm.graph)
    nodes = []
    for node in gm.graph.nodes:
        try:
            nodes.append({
                "id": node_ids[node],
                "op": node.op,
                "target": _total_target_descriptor(gm, node, opaque_types, runtime_graph_id, module_stack),
                "args": _total_argument(node.args, node_ids, opaque_types, runtime_graph_id),
                "kwargs": _total_argument(node.kwargs, node_ids, opaque_types, runtime_graph_id),
            })
        except Exception as exc:
            fallback_type = f"fingerprint_error:{_opaque_type(exc)}"
            opaque_types.append(fallback_type)
            nodes.append({
                "id": node_ids[node],
                "op": node.op,
                "target": {
                    "opaque_target": {
                        "type": _opaque_type(node.target)
                    }
                },
                "args": {
                    "opaque_argument": {
                        "type": _opaque_type(node.args)
                    }
                },
                "kwargs": {
                    "opaque_argument": {
                        "type": _opaque_type(node.kwargs)
                    }
                },
            })
    return nodes


def generated_graph_fingerprint_details(gm: GraphModule, runtime_graph_id: Optional[int] = None) -> Dict[str, Any]:
    """Return a total structural fingerprint and disclose every opaque fallback used."""
    opaque_types = []
    nodes = _generated_graph_nodes(gm, opaque_types, runtime_graph_id, set())
    encoded = json.dumps(nodes, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "fingerprint": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "opaque_fallback_count": len(opaque_types),
        "opaque_fallback_types": sorted(set(opaque_types)),
    }


def generated_graph_fingerprint(gm: GraphModule, runtime_graph_id: Optional[int] = None) -> str:
    return generated_graph_fingerprint_details(gm, runtime_graph_id)["fingerprint"]
