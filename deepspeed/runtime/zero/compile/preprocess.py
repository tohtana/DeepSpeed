# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from .stage3_backend import param_manager, profiling_results
from .prefetch import enable_prefetch

WARMUP_STEPS: int = 5
MEM_MARGIN: int = 10_000_000_000

persistent_optimized = False
max_alloc_mem = 0
last_optimize_step = 0
nz3 = None


def sort_params_by_time_per_size():
    ds_id_to_size = {}
    ds_id_to_time = defaultdict(float)
    ds_id_to_prof_dtime = defaultdict(float)
    ds_id_to_prof_wtime = defaultdict(float)

    for graph_id, pm in param_manager.items():
        params = pm.params
        for param_name, param in params.items():
            ds_id = pm.ds_ids[param_name]
            ds_id_to_size[ds_id] = param.numel * param.dtype.itemsize

        profile = profiling_results[graph_id]
        for n in profile.fwd_graph.nodes:
            if n.target == torch.ops.native_z3.allgather_param:
                assert "tensor_size" in n.meta
                ds_id_to_size[n.args[2]] = n.meta["tensor_size"]
                assert "device_time" in n.meta
                ds_id_to_time[n.args[2]] += n.meta["device_time"]

                ds_id_to_prof_dtime[n.args[2]] = n.meta["device_time"]
                ds_id_to_prof_wtime[n.args[2]] = n.meta["wall_time"]

        if profile.bwd_graph is not None:
            for n in profile.bwd_graph.nodes:
                if n.target == torch.ops.native_z3.allgather_param:
                    assert "tensor_size" in n.meta
                    ds_id_to_size[n.args[2]] = n.meta["tensor_size"]
                    assert "device_time" in n.meta
                    ds_id_to_time[n.args[2]] += n.meta["device_time"]

    ds_ids = list(ds_id_to_size.keys())
    ds_ids.sort(key=lambda ds_id: ds_id_to_time[ds_id] / ds_id_to_size[ds_id], reverse=True)

    # print(f"ds_id_to_size={ds_id_to_size}")
    # print(f"ds_id_to_time={ds_id_to_time}")

    if dist.get_rank() == 0:
        for ds_id in ds_ids:
            dtime_in_sec = ds_id_to_prof_dtime[ds_id]
            wtime_in_sec = ds_id_to_prof_wtime[ds_id]
            size_in_mb = ds_id_to_size[ds_id] / 1024 / 1024
            print(
                f"ds_id={ds_id} time_per_size={ds_id_to_time[ds_id] / ds_id_to_size[ds_id]:.5f} dtime={dtime_in_sec:.3f} wtime={wtime_in_sec:.3f} size={size_in_mb:.2f}MB bw={size_in_mb/dtime_in_sec:.2f}MB/s"
            )

    sorted_ds_ids = {ds_id: ds_id_to_size[ds_id] for ds_id in ds_ids}

    accelerator = get_accelerator()
    max_alloc_mem = accelerator.max_memory_allocated()
    total_mem = accelerator.total_memory()
    available_mem = (total_mem - max_alloc_mem) - MEM_MARGIN

    persistent_mem = 0
    for ds_id, size in sorted_ds_ids.items():
        if persistent_mem + size > available_mem:
            break
        persistent_mem += size
        nz3.set_persistent(ds_id, True)
        if dist.get_rank() == 0:
            print(f"Set persistent: {ds_id} size: {size} persistent_mem: {persistent_mem}")


def reset_graph():
    print(f"reset_graph")

    enable_prefetch()
    torch._dynamo.reset()


optimize_schedule = [
    (WARMUP_STEPS, reset_graph),
    # (WARMUP_STEPS * 2, sort_params_by_time_per_size),
]


def start_forward(nz3_handle, micro_steps: int, global_steps: int, update: bool):
    global nz3
    nz3 = nz3_handle

    if len(optimize_schedule) > 0 and global_steps == optimize_schedule[0][0]:
        _, optimize_fn = optimize_schedule.pop(0)
        optimize_fn()

    nz3.start_forward()