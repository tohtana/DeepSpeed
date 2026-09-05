# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
from pathlib import Path
import torch
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from .distributed import CollectiveControl, all_rank_prepare, on_rank_zero
from .fx_adapter import apply_recipe, graph_signature, parameter_specs
from .search import search, save_result


class SearchSession:
    """Engine-owned v0 state; prepare before backend globals are cleared."""

    def __init__(self, config, optional_passes, available_passes, capture_step, reduce_bucket_size):
        self.config = config
        self.optional_passes = optional_passes
        self.available_passes = available_passes
        self.selected = None
        self.control = None
        self.baseline_signatures = None
        self.capture_step = capture_step
        self.current_step = -1
        self.recorded = set()
        self.runtime_memory = {'reduce_bucket_numel': reduce_bucket_size, 'double_buffer': config.double_buffer}

    def should_record(self):
        return self.current_step == self.capture_step and self.selected is None

    def record(self, gm, graph_id, bwd, inputs, profiles):
        from ..profilers.graph_profile import BaselineRecordingInterpreter
        from ..backend import set_time_and_tensor_size
        if bwd in self.recorded:
            raise ValueError('v0 supports one forward/backward execution per baseline step')
        recorder = BaselineRecordingInterpreter(gm)
        result = recorder.run(*inputs)
        set_time_and_tensor_size(graph_id, gm.graph, recorder.mem_record, bwd, profiles)
        self.recorded.add(bwd)
        return result

    def prepare(self, profiles, managers, graph_order):
        from ..profilers.graph_profile import is_profile_incomplete, _get_mem_usage_out_of_torch
        from ..passes.prefetch import MAX_FUSE_SIZE, MARGIN
        from .gpu_profile import prepare_table

        accelerator = get_accelerator()
        self.control = CollectiveControl(dist, torch.device(accelerator.current_device()))

        def snapshot():
            if self.recorded != {False, True}:
                raise ValueError('Actual ZeRO-3 forward/backward baseline was not recorded')
            if len(graph_order) != 1 or not graph_order[0][1]:
                raise ValueError('v0 search requires one forward/backward graph pair')
            graph_id = graph_order[0][0]
            profile = profiles[graph_id]
            graphs = [profile.fwd_graph, profile.bwd_graph]
            if not profile.fwd_mem_complete or not profile.bwd_mem_complete or any(
                    graph is None or is_profile_incomplete(graph) for graph in graphs):
                raise ValueError('Mandatory ZeRO-3 baseline profile is incomplete')
            specs = parameter_specs(graphs, managers[graph_id], dist.get_world_size())
            return graphs, specs

        graphs, specs = all_rank_prepare(self.control, snapshot)
        self.baseline_signatures = [graph_signature(graph) for graph in graphs]

        def save_signature():
            directory = Path(self.config.search_output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f'baseline-signature-rank-{self.control.rank}.json'
             ).write_text(json.dumps(self.baseline_signatures, indent=2) + '\n')

        all_rank_prepare(self.control, save_signature)
        self.control.agree(self.baseline_signatures)
        communication = prepare_table(self.control, specs, self.config.search_output_dir, MAX_FUSE_SIZE)
        input_storages = {}
        for node in graphs[0].nodes:
            if node.op == 'placeholder':
                for desc in node.meta.get('sim_outputs', []):
                    input_storages[desc['key']] = desc['bytes']
        # The auto path retains its real inputs. Subtract those input storages,
        # which the simulator allocates at placeholders; keep mature optimizer
        # state, parameter shards and external device usage in the resident floor.
        resident = max(0,
                       accelerator.memory_allocated() + _get_mem_usage_out_of_torch() - sum(input_storages.values()))
        values = torch.tensor([resident, int(accelerator.total_memory() * (1 - MARGIN))],
                              dtype=torch.int64,
                              device=self.control.device)
        dist.all_reduce(values[:1], op=dist.ReduceOp.MAX)
        dist.all_reduce(values[1:], op=dist.ReduceOp.MIN)

        def run_search():
            result = search(graphs,
                            specs,
                            communication,
                            int(values[0]),
                            int(values[1]),
                            self.optional_passes,
                            self.available_passes,
                            runtime_memory=self.runtime_memory)
            result['baseline_measured_memory'] = {
                'fw': profiles[graph_order[0][0]].fwd_mem,
                'bw': profiles[graph_order[0][0]].bwd_mem
            }
            save_result(result, self.config.search_output_dir)
            return {'passes': result['selected']['passes'], 'recipes': result['selected']['recipes']}

        self.selected = on_rank_zero(self.control, run_search)
        self.control.agree(self.selected)

    def apply(self, gm, graph_id, graph_order, profiles, managers, bwd):
        from ..passes.zero3_compile import restore_z3_structure

        def local_apply():
            restore_z3_structure(gm, graph_id, graph_order, profiles, managers, bwd)
            # Compare structure independently of fast_free_schedule's saved order.
            actual = sorted(graph_signature(gm.graph), key=lambda row: row['name'])
            expected = sorted(self.baseline_signatures[int(bwd)], key=lambda row: row['name'])
            if actual != expected:
                differences = [(a, b) for a, b in zip(actual, expected) if a != b]
                detail = json.dumps(differences[:1])
                raise ValueError(f'Recaptured graph differs from profiled graph: bwd={bwd}; {detail}')
            apply_recipe(gm, graph_id, self.selected['recipes'][int(bwd)])
            return graph_signature(gm.graph)

        signature = all_rank_prepare(self.control, local_apply)
        self.control.agree(signature)
