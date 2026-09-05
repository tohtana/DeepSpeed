# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Copy graph structure and export profiles without copying real parameters."""

import copy
import math
import torch
from torch.fx import Graph, GraphModule, Node
from torch.utils._pytree import tree_leaves


def clone_graph(graph):
    result, env = Graph(), {}
    for node in graph.nodes:
        if node.op == 'get_attr':
            raise ValueError('v0 search requires lifted inputs, without get_attr')
        new = result.node_copy(node, lambda source: env[source.name])
        new.meta = dict(node.meta)
        for key in ('sim_outputs', ):
            if key in new.meta:
                new.meta[key] = copy.deepcopy(new.meta[key])
        env[node.name] = new
    return result


def graph_signature(graph):

    def encode(value):
        if isinstance(value, Node):
            return {'node': value.name}
        if isinstance(value, torch.device):
            return {'device_type': value.type}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        return str(value)

    rows = []
    for node in graph.nodes:
        args = list(node.args)
        target = str(node.target)
        if target == 'dc.end_backward.default':
            # The C++ op only uses this list as dependencies; scheduling can
            # change its order without changing the set of reductions to await.
            args[0] = sorted(args[0], key=lambda dependency: dependency.name)
        if target.startswith('dc.'):
            # Runtime graph IDs are process-local, not part of graph identity.
            position = 0 if 'prefetch_params_fused' in target else 1
            if len(args) > position:
                args[position] = 'GRAPH_ID'
        value = node.meta.get('val')
        tensors = [v for v in tree_leaves(value) if isinstance(v, torch.Tensor)]
        rows.append({
            'name': node.name,
            'op': node.op,
            'target': target,
            'args': encode(args),
            'kwargs': encode(node.kwargs),
            'tensors':
            [[str(v.dtype), list(map(str, v.shape)), list(map(str, v.stride()))] for v in tensors]
        })
    return rows


def graph_recipe(graph):
    result = []
    for node in graph.nodes:
        if str(node.target) == 'dc.prefetch_params_fused.default':
            result.append({'prefetch': [n.name for n in node.args[1]], 'ids': list(node.args[2]), 'name': node.name})
        else:
            result.append({'name': node.name})
    return result


def apply_recipe(gm, graph_id, recipe):
    original = {n.name: n for n in gm.graph.nodes}
    required = [row['name'] for row in recipe if 'prefetch' not in row]
    if set(required) != set(original) or len(required) != len(original):
        raise ValueError('Recaptured ZeRO-3 graph differs from planned graph')
    result, env = Graph(), {}
    for row in recipe:
        if 'prefetch' in row:
            node = result.create_node('call_function',
                                      torch.ops.dc.prefetch_params_fused.default,
                                      (graph_id, [env[name] for name in row['prefetch']], row['ids']),
                                      name=row['name'])
        else:
            node = result.node_copy(original[row['name']], lambda source: env[source.name])
        env[row['name']] = node
    result.lint()
    gm.graph = result
    gm.recompile()


def parameter_specs(graphs, manager, world_size):
    shapes = {manager.ds_ids[name]: value.shape for name, value in manager.params.items()}
    specs = {}
    for graph in graphs:
        for node in graph.nodes:
            if str(node.target) == 'dc.allgather_param.default':
                ds_id = int(node.args[2])
                dtype = node.kwargs['dtype']
                nbytes = math.ceil(math.prod(shapes[ds_id]) / world_size) * world_size * dtype.itemsize
                spec = {'id': ds_id, 'bytes': nbytes, 'dtype': str(dtype)}
                if ds_id in specs and specs[ds_id] != spec:
                    raise ValueError('v0 requires one gather dtype per parameter')
                specs[ds_id] = spec
    return specs


def export_graphs(graphs, specs, communication, runtime_memory=None):
    """Use actual storage aliases; connect AOT saved values by placeholder name."""
    storage_bytes, operators, exported = {}, {}, []
    saved = {}
    for phase, graph in zip(('fw', 'bw'), graphs):
        remap = {}
        nodes = list(graph.nodes)
        for node in nodes:
            descriptions = node.meta.get('sim_outputs', [])
            if node.op == 'placeholder' and phase == 'bw' and node.name in saved:
                if len(descriptions) != len(saved[node.name]):
                    raise ValueError(f'Saved tensor structure mismatch: {node.name}')
                for desc, previous in zip(descriptions, saved[node.name]):
                    remap[desc['key']] = previous
        managed = set()
        for node in nodes:
            if str(node.target) == 'dc.allgather_param.default':
                managed.update(v['key'] for v in node.meta.get('sim_outputs', []))

        def keys(node):
            result = []
            for desc in node.meta.get('sim_outputs', []):
                key = remap.get(desc['key'], f'{phase}/{desc["key"]}')
                nbytes = 0 if desc['key'] in managed else desc['bytes']
                if key in storage_bytes and storage_bytes[key] != nbytes:
                    raise ValueError(f'Storage size mismatch for {key}')
                storage_bytes[key] = nbytes
                result.append(key)
            return list(dict.fromkeys(result))

        events = []
        buckets = {}
        for node in nodes:
            target = str(node.target)
            key = f'{phase}/{node.name}'
            inputs = list(dict.fromkeys(k for source in node.all_input_nodes for k in keys(source)))
            outputs = keys(node)
            event = {'key': key, 'inputs': inputs, 'outputs': outputs}
            if node.op not in ('placeholder', 'output') and 'sim_outputs' not in node.meta:
                if target != 'dc.prefetch_params_fused.default':
                    raise ValueError(f'Missing storage profile for {key}')
            if node.op not in ('placeholder', 'output') and 'device_time' not in node.meta:
                if target not in ('dc.prefetch_params_fused.default', 'dc.end_backward.default'):
                    raise ValueError(f'Missing operator duration for {key}')
            duration = float(node.meta.get('device_time', 0))
            workspace = int(node.meta.get('sim_workspace_bytes', 0))
            if target == 'dc.allgather_param.default':
                event.update(kind='gather', params=[specs[node.args[2]]])
                workspace = 0
            elif target == 'dc.prefetch_params_fused.default':
                event.update(kind='prefetch', params=[specs[ds_id] for ds_id in node.args[2]])
                workspace = 0
            elif target == 'dc.wait_allgather.default':
                event['kind'] = 'wait'
                workspace = 0
            elif target == 'dc.release_param.default':
                event.update(kind='release', param_id=node.args[2], release_count=node.args[3])
                workspace = 0
            elif target == 'dc.reduce_grad.default' and runtime_memory:
                value = node.args[0].meta['val']
                dtype, numel = value.dtype, value.numel()
                # The first adapter supports the normal BF16 leaf-grad contract.
                if str(dtype) != specs[node.args[2]]['dtype']:
                    raise ValueError('v0 requires reduce input dtype to match the parameter dtype')
                bucket = buckets.setdefault(
                    str(dtype), {
                        'sizes': [runtime_memory['reduce_bucket_numel']] *
                        (2 if runtime_memory['double_buffer'] else 1),
                        'index': 0,
                        'offset': 0,
                        'pending': [],
                        'itemsize': dtype.itemsize
                    })
                index = bucket['index']
                if bucket['offset'] > 0 and bucket['offset'] + numel > bucket['sizes'][index]:
                    event['inputs'].extend(bucket['pending'])
                    bucket['pending'] = []
                    index = (index + 1) % len(bucket['sizes'])
                    bucket['index'], bucket['offset'] = index, 0
                bucket['sizes'][index] = max(bucket['sizes'][index], numel)
                bucket['offset'] += numel
                bucket['pending'].extend(keys(node.args[0]))
                event['runtime_buffers_bytes'] = sum(sum(b['sizes']) * b['itemsize'] for b in buckets.values())
            elif target == 'dc.end_backward.default' and runtime_memory:
                for bucket in buckets.values():
                    event['inputs'].extend(bucket['pending'])
                event['clear_runtime_buffers'] = True
            operators[key] = {'time_ms': duration, 'workspace_bytes': workspace}
            events.append(event)
            if phase == 'fw':
                saved[node.name] = outputs
                if 'original_output_name' in node.meta:
                    saved[node.meta['original_output_name']] = outputs
        exported.append({'phase': phase, 'events': events})
    return exported, {'operators': operators, 'storage_bytes': storage_bytes, 'communication': communication}


def copy_module(graph):
    return GraphModule({}, clone_graph(graph))
