# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from functools import partial
from threading import Lock

import torch

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import InsertPostInitMethodToModuleSubClasses
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload

from .passes import zero3_compile, prefetch, selective_gather, offload_parameters, offload_activation
from .backend import agent_optimization_loop, make_backend, launch_compile_passes, init_schedule
from .patch_fake_tensor import patch_fake_tensor
from .util import get_deepcompile_handle, add_pre_backward_hook, add_post_backward_hook
from .z3_eager_fallback import DeepCompileZ3EagerFallback

WARMUP = 5

_MISSING = object()
_DYNAMO_CONFIG_NAMES = ("force_parameter_static_shapes", "force_nn_module_property_static_shapes")
_DYNAMO_CONFIG_OWNERS = {}
_DYNAMO_CONFIG_LOCK = Lock()


def _allow_dynamo_dynamic_parameter_shapes_for_z3(compile_kwargs):
    """Acquire process-wide ZeRO-3 Dynamo config ownership and return its release callback."""
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is None:
        try:
            import torch._dynamo as dynamo
        except ImportError:
            return None

    dynamo_config = getattr(dynamo, "config", None)
    if dynamo_config is None:
        return None

    owner_token = object()
    config_key = id(dynamo_config)
    with _DYNAMO_CONFIG_LOCK:
        state = _DYNAMO_CONFIG_OWNERS.get(config_key)
        if state is None or state["config"] is not dynamo_config:
            previous_values = {
                config_name: getattr(dynamo_config, config_name)
                for config_name in _DYNAMO_CONFIG_NAMES if hasattr(dynamo_config, config_name)
            }
            if not previous_values:
                return None
            state = {"config": dynamo_config, "previous_values": previous_values, "owner_tokens": set()}
            _DYNAMO_CONFIG_OWNERS[config_key] = state
        state["owner_tokens"].add(owner_token)
        for config_name in state["previous_values"]:
            setattr(dynamo_config, config_name, False)

    def restore():
        with _DYNAMO_CONFIG_LOCK:
            state = _DYNAMO_CONFIG_OWNERS.get(config_key)
            if state is None or state["config"] is not dynamo_config or owner_token not in state["owner_tokens"]:
                return
            state["owner_tokens"].remove(owner_token)
            if state["owner_tokens"]:
                return
            for config_name, previous_value in state["previous_values"].items():
                setattr(dynamo_config, config_name, previous_value)
            del _DYNAMO_CONFIG_OWNERS[config_key]

    return restore


def _resolve_expected_grad_dtype(param):
    # Match PyTorch's leaf grad accumulation contract. grad_dtype can be a
    # dtype, or None to allow any incoming gradient dtype:
    # https://docs.pytorch.org/docs/main/generated/torch.sparse.semi_structured.SparseSemiStructuredTensorCUSPARSELT.html#torch.sparse.semi_structured.SparseSemiStructuredTensorCUSPARSELT.grad_dtype
    grad_dtype = getattr(param, "grad_dtype", _MISSING)
    if grad_dtype is None:
        return None
    if grad_dtype is not _MISSING:
        return grad_dtype
    return param.dtype


def _default_z3_schedule(compile_config):
    use_agent = compile_config.zero3_tuning_strategy == "agent"
    schedule = []
    if compile_config.offload_parameters:
        parameter_passes = [zero3_compile.add_z3_gather_release, offload_parameters.offload_parameter_fwd]
        schedule.append((0, parameter_passes))
        if use_agent:
            schedule.append((WARMUP, parameter_passes + [agent_optimization_loop]))
    elif compile_config.offload_opt_states:
        from .passes.offload_adam_states import move_opt_states, offload_adam_states_for_init
        schedule.append((0, [zero3_compile.add_z3_gather_release]))
        schedule.append((1, [offload_adam_states_for_init, zero3_compile.add_z3_gather_release, move_opt_states]))
        if use_agent:
            schedule.append((WARMUP, [
                offload_adam_states_for_init, zero3_compile.add_z3_gather_release, move_opt_states,
                agent_optimization_loop
            ]))
    elif compile_config.offload_activation:
        offload_activation.register_activation_offload_ops()
        schedule.append((0, [zero3_compile.add_z3_gather_release, offload_activation.offload_activation_floor]))
        warmup_passes = [
            zero3_compile.add_z3_gather_release, offload_activation.offload_activation_floor,
            offload_activation.offload_activation
        ]
        if use_agent:
            warmup_passes.append(agent_optimization_loop)
        schedule.append((WARMUP, warmup_passes))
    elif use_agent:
        schedule.append((0, [zero3_compile.add_z3_gather_release]))
        schedule.append((WARMUP, [zero3_compile.add_z3_gather_release, agent_optimization_loop]))
    else:
        schedule.append((0, [zero3_compile.add_z3_gather_release]))
        schedule.append(
            (WARMUP,
             [zero3_compile.add_z3_gather_release, prefetch.schedule_prefetch, selective_gather.selective_gather]))
    return schedule


def _compose_agent_schedule(schedule, compile_config):
    if compile_config.zero3_tuning_strategy != "agent":
        return schedule

    composed = [(step, list(passes)) for step, passes in schedule]
    for index, (step, passes) in enumerate(composed):
        if step == WARMUP:
            structural_passes = [opt_pass for opt_pass in passes if opt_pass is not agent_optimization_loop]
            composed[index] = (step, structural_passes + [agent_optimization_loop])
            return composed

    # Experimental explicit schedules without a warmup entry reuse the latest pre-warmup
    # structural pass set. This preserves the user's capture setup when the warm graph is recaptured.
    prior_entries = [(step, passes) for step, passes in composed if step < WARMUP]
    if prior_entries:
        warmup_passes = list(max(prior_entries, key=lambda entry: entry[0])[1])
    else:
        warmup_passes = [zero3_compile.add_z3_gather_release]
    warmup_entry = (WARMUP, warmup_passes + [agent_optimization_loop])
    insert_at = next((index for index, (step, _) in enumerate(composed) if step > WARMUP), len(composed))
    composed.insert(insert_at, warmup_entry)
    return composed


def init_z3(engine, backend, compile_config, compile_kwargs, schedule=None):

    # Validate before touching the engine: everything below removes hooks and unpatches modules,
    # so raising later would leave a half-converted engine behind.
    # zero_use_cpu_optimizer(), not zero_offload_optimizer(): the latter returns the config
    # object, which is present but inert for `offload_optimizer: {}` or `device: none`.
    if compile_config.offload_opt_states and engine.zero_use_cpu_optimizer():
        raise ValueError("compile.offload_opt_states cannot be combined with ZeRO's "
                         "zero_optimization.offload_optimizer set to cpu or nvme: both manage the "
                         "same optimizer state. ZeRO keeps it off the accelerator for the whole step "
                         "and runs the optimizer there, while this pass keeps it resident when memory "
                         "allows and moves it around the compiled graph. Enable one of them.")

    if compile_config.offload_activation and (compile_config.offload_parameters or compile_config.offload_opt_states):
        raise ValueError("compile.offload_activation cannot be combined with offload_parameters or "
                         "offload_opt_states; choose one offloading target per run. Each of them plans "
                         "against the whole memory budget on its own, so two of them together move far "
                         "more data than the run needs.")

    optimizer = engine.optimizer
    use_opt = not isinstance(optimizer, DeepSpeedZeRoOffload)

    if use_opt and hasattr(optimizer, "ipg_buckets"):
        optimizer.ipg_buckets.clear()
        get_accelerator().empty_cache()

    dc = get_deepcompile_handle()
    dc.init(engine.data_parallel_group, compile_config, engine.zero_reduce_bucket_size())

    engine._deepcompile_z3_eager_fallback = DeepCompileZ3EagerFallback(engine)
    add_post_backward_hook(engine._deepcompile_z3_eager_fallback.complete_backward)

    if use_opt:
        optimizer.parameter_offload._remove_module_hooks()

        for hook in optimizer._grad_acc_hooks:
            hook.remove()
        optimizer._grad_acc_hooks.clear()

    # Unpatch linear
    if hasattr(InsertPostInitMethodToModuleSubClasses, "linear_bk"):
        torch.nn.functional.linear = InsertPostInitMethodToModuleSubClasses.linear_bk

    if compile_config.symmetric_memory:
        group_name = engine.data_parallel_group.group_name
        dist.enable_symm_mem_for_group(group_name)

    for p in engine.module.parameters():
        grad_buffer = torch.Tensor()
        # Frozen params (e.g. the base weights of a LoRA setup) are absent from the optimizer's
        # grad partition map, which is built from the trainable groups only. They keep the empty
        # buffer: no reduce op is scheduled for a param without a grad node, so it is never read.
        if use_opt and p.requires_grad:
            grad_buffer = optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[p.ds_id]

        # Disable persistent param
        p.ds_persist = False
        dc.register_z3_param(p.ds_id, p.ds_shape, p.ds_tensor, grad_buffer, p.ds_persist,
                             _resolve_expected_grad_dtype(p))

    if schedule is None:
        if compile_config.offload_parameters and compile_config.offload_opt_states:
            raise ValueError("offload_parameters and offload_opt_states cannot be enabled together; "
                             "choose one offloading target per run. Note that offload_parameters may have "
                             "been enabled implicitly: the engine turns it on when the ZeRO config "
                             "offloads both optimizer and parameters to CPU.")
        schedule = _default_z3_schedule(compile_config)
    schedule = _compose_agent_schedule(schedule, compile_config)

    init_schedule(schedule)

    if use_opt:

        def set_grad_buffer(_is_gradient_accumulation_boundary):
            for i, sub_group in enumerate(optimizer.fp16_groups):
                optimizer.averaged_gradients[i] = [
                    optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[param.ds_id]
                    if param.requires_grad else torch.zeros_like(param.ds_tensor) for param in sub_group
                ]

        add_pre_backward_hook(set_grad_buffer)

        # offloading opt states need additional setup
        from .passes.offload_adam_states import (move_opt_states, move_opt_states_sync, offload_adam_states_for_init,
                                                 init_offload_opt_states)
        for _, passes in schedule:
            if move_opt_states in passes or move_opt_states_sync in passes or offload_adam_states_for_init in passes:
                init_offload_opt_states(optimizer, dc)
                break

    engine._deepcompile_owned_frames = set()
    engine.launch_compile_passes = partial(launch_compile_passes, owned_frames=engine._deepcompile_owned_frames)

    patch_fake_tensor()
    torch._inductor.config.size_asserts = False

    previous_restore = getattr(engine, "_deepcompile_dynamo_config_restore", None)
    if previous_restore is not None:
        previous_restore()
        del engine._deepcompile_dynamo_config_restore
    restore_dynamo_config = _allow_dynamo_dynamic_parameter_shapes_for_z3(compile_kwargs)
    if restore_dynamo_config is not None:
        engine._deepcompile_dynamo_config_restore = restore_dynamo_config

    return make_backend(backend,
                        compile_config,
                        compile_kwargs=compile_kwargs,
                        owned_frames=engine._deepcompile_owned_frames)
