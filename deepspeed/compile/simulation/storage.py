# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.utils._pytree import tree_leaves


def tensor_leaves(value):
    return [value for value in tree_leaves(value) if torch.is_tensor(value)]


def record_storage(node, output, environment):
    """Record aliases against currently live inputs, without retaining tensor storage.

    A pointer reused after its previous tensor dies therefore becomes a new key.
    Storage identities are never exported; portable keys use graph node names.
    """
    aliases = {}
    for source, value in environment.items():
        for tensor, description in zip(tensor_leaves(value), source.meta.get('sim_outputs', [])):
            aliases[tensor.untyped_storage()._cdata] = description
    descriptions = []
    for index, tensor in enumerate(tensor_leaves(output)):
        storage = tensor.untyped_storage()
        identity = storage._cdata
        if identity not in aliases:
            aliases[identity] = {
                'key': f'{node.name}:{index}',
                'bytes': storage.nbytes() if tensor.device.type != 'cpu' else 0
            }
        descriptions.append(dict(aliases[identity]))
    node.meta['sim_outputs'] = descriptions
