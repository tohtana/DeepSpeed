# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.fx import GraphModule, Node

from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer

from ..custom_ops import tp_collectives  # noqa: F401

COLUMN_PARALLEL_OP = torch.ops.autotp.copy_to_tp_region.default
ROW_PARALLEL_OP = torch.ops.autotp.reduce_from_tp_region.default

# AutoTP replaces nn.Linear with these layers and shards their weights, so the injected layer type
# already records the partitioning decision the pass needs. Reading it back is more robust than
# re-deriving column/row from parameter-name patterns.
COLUMN_PARALLEL_LAYER = LinearLayer
ROW_PARALLEL_LAYER = LinearAllreduce

# The injected layers compute their matmul with torch.matmul; the plain nn.Linear spelling is
# accepted too so the pass keeps working if a layer is lowered differently.
_MATMUL_TARGETS = {
    torch.matmul,
    torch.ops.aten.matmul.default,
    torch.ops.aten.linear.default,
    torch._C._nn.linear,
}


def defer_collectives_to_compiler(model) -> int:
    """Suppress the module-level TP collectives on layers this pass will handle in the graph.

    Returns the number of layers handed over to the pass. Layers the pass does not rewrite (a
    column-parallel layer that gathers its output, the fused sub-param variants, conv and
    embedding layers) keep their module-level collectives and stay correct as-is.
    """
    deferred = 0
    for module in model.modules():
        is_row_parallel = type(module) is ROW_PARALLEL_LAYER
        # gather_output adds a further collective that this pass does not emit yet, so leave those
        # layers to the module-level path.
        is_column_parallel = type(module) is COLUMN_PARALLEL_LAYER and not module.gather_output
        if not (is_row_parallel or is_column_parallel):
            continue
        if module.mp_group is None:
            continue
        if type(module).tp_overlap_comm:
            raise NotImplementedError("AutoTP compile pass does not support tp_overlap_comm. Set "
                                      "'tp_overlap_comm': false to emit the collectives into the graph.")
        module.defer_collectives_to_compiler = True
        deferred += 1
    return deferred


def _originating_layer_type(node: Node):
    """Return the innermost nn.Module type a node was traced from, or None."""
    module_stack = node.meta.get("nn_module_stack")
    if not module_stack:
        return None
    _, module_type = list(module_stack.values())[-1]
    return module_type


def _insert_after(gm: GraphModule, node: Node, op) -> Node:
    """Insert ``op(node)`` and re-point every consumer of ``node`` at the new node."""
    with gm.graph.inserting_after(node):
        collective_node = gm.graph.call_function(op, args=(node, ))
    collective_node.meta["val"] = node.meta.get("val")
    # Steal every consumer first, then hand the original back as this node's own input; doing it in
    # the other order would leave the new node feeding itself.
    node.replace_all_uses_with(collective_node)
    collective_node.update_arg(0, node)
    return collective_node


def pass_insert_tp_collectives(gm: GraphModule, real_inputs):
    """Insert the tensor-parallel collectives around the matmuls of the injected AutoTP layers."""
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in _MATMUL_TARGETS:
            continue

        layer_type = _originating_layer_type(node)
        if layer_type is ROW_PARALLEL_LAYER:
            _insert_after(gm, node, ROW_PARALLEL_OP)
        elif layer_type is COLUMN_PARALLEL_LAYER:
            activation = node.args[0]
            # Column-parallel layers that share an activation (q/k/v, gate/up) need only one
            # collective. Inserting it already re-pointed the sibling matmuls at the new node, so
            # finding one here means this activation has been handled.
            if activation.op == "call_function" and activation.target is COLUMN_PARALLEL_OP:
                continue
            _insert_after(gm, activation, COLUMN_PARALLEL_OP)


def pass_canonicalize(gm: GraphModule, real_inputs):
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


AUTOTP_PASSES = [
    pass_insert_tp_collectives,
    pass_canonicalize,
]


def apply_autotp(gm: GraphModule, real_inputs, passes=None):
    """Apply the AutoTP transformation passes to the graph.

    The collectives are shape-preserving, so unlike AutoSP this needs no shape re-propagation.
    """
    for opt_pass in passes or AUTOTP_PASSES:
        opt_pass(gm, real_inputs)
    return gm
