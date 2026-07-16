# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
import torch
from collections import OrderedDict
import threading
from torch.autograd.graph import _get_grad_fn_or_grad_acc, register_multi_grad_hook
from torch.utils._pytree import tree_flatten
from deepspeed.utils import z3_leaf_module, set_z3_leaf_module
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.runtime.zero.utils import apply_to_tensors_only, is_zero_param
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import _init_external_params
from deepspeed.runtime.zero.partition_parameters import *
from deepspeed.runtime.zero.partitioned_param_coordinator import (PartitionedParameterCoordinator,
                                                                  InflightParamRegistry, current_graph_task_id,
                                                                  iter_params)
from deepspeed.accelerator import get_accelerator
from deepspeed import utils

FWD_MODULE_STACK = list()


class _FrozenForwardBoundary:
    """One frozen-module activation boundary awaiting its last consumer."""

    __slots__ = ("active", "callback", "deferred_releases", "grad_nodes", "handle", "pending_boundaries")

    def __init__(self, pending_boundaries, grad_nodes=()):
        self.active = True
        self.callback = None
        self.deferred_releases = []
        self.grad_nodes = grad_nodes
        self.handle = None
        self.pending_boundaries = pending_boundaries


#for each tensor in outputs run the forward_function and register backward_function as hook
def _apply_forward_and_backward_to_tensors_only(module, forward_function, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_forward_and_backward_to_tensors_only(module, forward_function, backward_function,
                                                                         output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        forward_function(outputs)
        if outputs.requires_grad:
            outputs.register_hook(backward_function)
        return outputs
    else:
        return outputs


class ZeROOrderedDict(OrderedDict):

    def __init__(self, parent_module, *args, **kwargs):
        """A replacement for ``collections.OrderedDict`` to detect external ZeRO params.

        Args:
            parent_module (``collections.OrderedDict``): the collection to replace
        """

        super().__init__(*args, **kwargs)
        self._parent_module = parent_module
        self._in_forward = False

    def __reduce__(self):
        r0, _, *r2 = super().__reduce__()
        return (r0, (self._parent_module, )) + tuple(r2)

    def __getitem__(self, key):
        param = super().__getitem__(key)

        # Params can be registered as None (e.g., bias)
        if param is None:
            return param

        if hasattr(param, "ds_status") and param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if self._parent_module._parameters._in_forward and not torch.compiler.is_compiling():
                from deepspeed.compile.z3_eager_fallback import get_active_z3_eager_fallback
                fallback = get_active_z3_eager_fallback()
                if fallback is None:
                    register_external_parameter(FWD_MODULE_STACK[-1], param)
                    param.all_gather()
                else:
                    param.all_gather()
                    fallback.record_gathered_param(param)
                print_rank_0(f'Registering external parameter from getter {key} ds_id = {param.ds_id}', force=False)

        return param


def _inject_parameters(module, cls):
    for module in module.modules():
        module._original_parameters = module._parameters

        if cls == ZeROOrderedDict:
            new_param = cls(parent_module=module)
        else:
            new_param = cls()

        for key, param in module._parameters.items():
            new_param[key] = param

        module._parameters = new_param


def ensure_zero_ordered_dict(module):
    """Wrap ``module._parameters`` in :class:`ZeROOrderedDict` if not already.

    PyTorch 2.5+ defaults ``nn.Module._parameters`` to a plain ``dict``
    (pytorch/pytorch#129164), which rejects the ``_in_forward`` attribute
    the forward prologue sets. Modules not converted by ``_inject_parameters``
    at engine init (e.g. submodules attached after ``deepspeed.initialize``,
    or restored by ``deepspeed/compile/init_z3.py``) hit issue #6961.
    Idempotent; no-op if already wrapped, missing, or a non-dict container.
    """
    params = getattr(module, "_parameters", None)
    if isinstance(params, ZeROOrderedDict) or not isinstance(params, dict):
        return
    # Preserve the original container only on first wrap so the un-injection
    # path in ``deepspeed/compile/init_z3.py`` can restore it.
    if not hasattr(module, "_original_parameters"):
        module._original_parameters = params
    new_param = ZeROOrderedDict(parent_module=module)
    for key, param in params.items():
        new_param[key] = param
    module._parameters = new_param


class DeepSpeedZeRoOffload(object):

    def __init__(
        self,
        module,
        timers,
        ds_config,
        zenflow=False,
        overlap_comm=True,
        prefetch_bucket_size=50000000,
        max_reuse_distance=1000000000,
        max_live_parameters=1000000000,
        param_persistence_threshold=100000,
        model_persistence_threshold=sys.maxsize,
        dp_process_group=None,
        offload_param_config=None,
        mpu=None,
        zero_param_parallel_group=None,
        zero_quantized_weights=False,
        zero_quantized_nontrainable_weights=False,
        zero_module_granularity_threshold=0,
        log_trace_cache_warnings=False,
    ):

        see_memory_usage("DeepSpeedZeRoOffload initialize [begin]", force=False)

        print_rank_0(f"initialized {__class__.__name__} with args: {locals()}", force=False)

        self.module = module
        self._has_frozen_params = any(not param.requires_grad for param in module.parameters())
        if self._has_frozen_params:
            self._frozen_boundary_lock = threading.Lock()
            self._frozen_boundaries = set()
        self.timers = timers
        self.zenflow = zenflow
        self.dtype = list(module.parameters())[0].dtype
        self.dp_process_group = dp_process_group
        self.offload_device = None
        self.offload_param_pin_memory = False
        self.zero_param_parallel_group = zero_param_parallel_group
        self.zero_quantized_weights = zero_quantized_weights
        self.zero_quantized_nontrainable_weights = zero_quantized_nontrainable_weights
        self.log_trace_cache_warnings = log_trace_cache_warnings

        if offload_param_config is not None and offload_param_config.device != OffloadDeviceEnum.none:
            self.offload_device = offload_param_config.device
            self.offload_param_pin_memory = offload_param_config.pin_memory

        self._convert_to_zero_parameters(ds_config, module, mpu)

        for m in module.modules():
            _init_external_params(m)

        _inject_parameters(module, ZeROOrderedDict)

        self.param_numel_persistence_threshold = int(param_persistence_threshold)
        self.model_persistence_threshold = int(model_persistence_threshold)
        self.persistent_parameters = self.mark_persistent_parameters(self.param_numel_persistence_threshold,
                                                                     self.model_persistence_threshold)

        self._prefetch_bucket_sz = int(prefetch_bucket_size)
        self._max_reuse_distance_in_numel = int(max_reuse_distance)
        self._max_available_parameters_in_numel = int(max_live_parameters)
        self.__allgather_stream = None if get_accelerator().is_synchronized_device() else get_accelerator().Stream(
        ) if overlap_comm else get_accelerator().default_stream()

        if not hasattr(module, "ds_inflight_param_registry"):
            module.ds_inflight_param_registry = InflightParamRegistry()
        self.__inflight_param_registry = module.ds_inflight_param_registry

        self.fast_sharding_for_leaf_module = False

        if zero_module_granularity_threshold > 0:
            self.min_granularity_value = sys.maxsize
            self.min_granularity_layer = None
            self.granularity_info = set()
            self.z3_leaf_layers = []
            self._set_z3_leaf_modules_by_threshold(module, zero_module_granularity_threshold)
            self.fast_sharding_for_leaf_module = True

        self.param_coordinator = PartitionedParameterCoordinator(
            prefetch_bucket_sz=self._prefetch_bucket_sz,
            max_reuse_distance_in_numel=self._max_reuse_distance_in_numel,
            max_available_parameters_in_numel=self._max_available_parameters_in_numel,
            allgather_stream=self.__allgather_stream,
            inflight_param_registry=self.__inflight_param_registry,
            prefetch_nvme=self.offload_device == OffloadDeviceEnum.nvme,
            timers=self.timers,
            zero_quantized_weights=self.zero_quantized_weights,
            zero_quantized_nontrainable_weights=self.zero_quantized_nontrainable_weights,
            fast_sharding_for_leaf_module=self.fast_sharding_for_leaf_module,
            log_trace_cache_warnings=self.log_trace_cache_warnings,
        )
        self._backward_hook_state_manager = None

        self.forward_hooks = []
        self.backward_hooks = []
        self.fwd_pre_hook = None

        self.setup_zero_stage3_hooks()
        print_rank_0(
            f'Created module hooks: forward = {len(self.forward_hooks)}, backward = {len(self.backward_hooks)}',
            force=False)

        see_memory_usage("DeepSpeedZeRoOffload initialize [end]", force=False)

    @instrument_w_nvtx
    def partition_all_parameters(self):
        """Partitioning Parameters that were not partitioned usually if parameters
        of modules whose input parameters do not require grad computation do not
        trigger post call and will therefore will remain unpartitioned"""
        if self._has_frozen_params:
            self._clear_frozen_boundaries()
        self.get_param_coordinator().release_and_reset_all(self.module)
        for param in iter_params(self.module, recurse=True):
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(f"{param.ds_summary()} expected to be released")

    def get_param_coordinator(self):
        return self.param_coordinator

    def set_backward_hook_state_manager(self, manager):
        """Reuse the optimizer's one outer-GraphTask completion callback."""
        self._backward_hook_state_manager = manager

    @property
    def has_frozen_params(self):
        return self._has_frozen_params

    def _begin_outer_backward(self):
        manager = self._backward_hook_state_manager
        if manager is None:
            return

        graph_task_id = current_graph_task_id()
        if (not manager.post_backward_callback_queued
                or manager.post_backward_callback_graph_task_id != graph_task_id):
            raise RuntimeError("ZeRO-3 root backward did not register the outer GraphTask callback")
        self.get_param_coordinator().begin_outer_backward(graph_task_id)

    def release_outer_backward(self):
        """Run from BackwardHookStateManager's existing completion callback."""
        manager = self._backward_hook_state_manager
        if manager is None or manager.post_backward_callback_graph_task_id is None:
            return
        self.get_param_coordinator().release_outer_backward(manager.post_backward_callback_graph_task_id)
        self._clear_frozen_boundaries()

    def _new_frozen_boundary(self, input_tensors, pending_boundaries):
        """Register last-consumer timing on one frozen invocation's activation inputs."""
        with torch.enable_grad():
            grad_nodes = tuple(_get_grad_fn_or_grad_acc(tensor) for tensor in input_tensors)
        boundary = _FrozenForwardBoundary(pending_boundaries, grad_nodes)

        def finish_boundary(unused_grads):
            deferred_releases = None
            hook_handle = None
            with self._frozen_boundary_lock:
                if not boundary.active:
                    return
                boundary.active = False
                self._frozen_boundaries.discard(boundary)
                if pending_boundaries is not None and boundary in pending_boundaries:
                    pending_boundaries.remove(boundary)
                deferred_releases = boundary.deferred_releases
                boundary.deferred_releases = []
                hook_handle = boundary.handle
                boundary.handle = None
            try:
                for deferred_release in deferred_releases:
                    self.get_param_coordinator().finish_deferred_release(deferred_release)
            finally:
                if hook_handle is not None:
                    hook_handle.remove()

        boundary.callback = finish_boundary
        hook_handle = None
        try:
            if input_tensors:
                hook_handle = register_multi_grad_hook(input_tensors, finish_boundary)
                boundary.handle = hook_handle
            with self._frozen_boundary_lock:
                if pending_boundaries is not None:
                    pending_boundaries.append(boundary)
                self._frozen_boundaries.add(boundary)
        except Exception:
            if hook_handle is not None:
                hook_handle.remove()
            raise
        return boundary

    def _register_frozen_boundary(self, input_tensors, pending_boundaries):
        """Record an ordinary-forward activation boundary for checkpoint replay."""
        self._new_frozen_boundary(input_tensors, pending_boundaries)

    def _bind_frozen_boundary(self,
                              input_tensors,
                              pending_boundaries,
                              fallback_boundary_pools=(),
                              protect_all_params=False):
        """Bind replay to its activation last-consumer in the active GraphTask."""
        coordinator = self.get_param_coordinator()
        deferred_release = (coordinator.begin_deferred_release(
            protect_all_params=True) if protect_all_params else coordinator.begin_deferred_release())
        boundaries = []
        try:
            # Select every live invocation whose activation node will execute
            # in the current GraphTask. This covers ordinary
            # non-reentrant boundaries as well as a non-reentrant checkpoint
            # created by a parent reentrant replay, while skipping later unused
            # calls without an autograd graph scan. Repeated uses share the same
            # parameters, so the last executable callback is their true local
            # parameter-consumer boundary.
            boundary_pools = (() if pending_boundaries is None else
                              (pending_boundaries, )) + tuple(fallback_boundary_pools)
            with self._frozen_boundary_lock:
                for boundary_pool in boundary_pools:
                    for candidate in reversed(boundary_pool):
                        if (candidate.active and candidate.grad_nodes
                                and any(torch._C._will_engine_execute_node(node) for node in candidate.grad_nodes)):
                            boundaries.append(candidate)
                    if boundaries:
                        break

            if not boundaries:
                # Reentrant replay forward still runs in the outer GraphTask;
                # checkpoint starts its nested GraphTask only after recompute.
                # With no executable ordinary boundary, the replay inputs are
                # therefore the valid local boundary. Empty inputs fall back to
                # the outer completion callback.
                # Keep this boundary in the module-local invocation list: a
                # nested non-reentrant recompute may execute later in the
                # boundary's child GraphTask and must bind to the same hook.
                boundaries = [self._new_frozen_boundary(input_tensors, pending_boundaries)]

            coordinator.set_deferred_release_boundary_count(deferred_release, len(boundaries))
            with self._frozen_boundary_lock:
                if any(not boundary.active for boundary in boundaries):
                    raise RuntimeError("ZeRO-3 frozen replay boundary completed before binding")
                for boundary in boundaries:
                    boundary.deferred_releases.append(deferred_release)
        except Exception:
            with self._frozen_boundary_lock:
                for boundary in boundaries:
                    if deferred_release in boundary.deferred_releases:
                        boundary.deferred_releases.remove(deferred_release)
            coordinator.cancel_deferred_release(deferred_release)
            raise
        return deferred_release

    def _clear_frozen_boundaries(self):
        hook_handles = []
        with self._frozen_boundary_lock:
            boundaries = self._frozen_boundaries
            self._frozen_boundaries = set()
            for boundary in boundaries:
                if not boundary.active:
                    continue
                boundary.active = False
                boundary.deferred_releases = []
                if boundary.handle is not None:
                    hook_handles.append(boundary.handle)
                    boundary.handle = None
                if boundary.pending_boundaries is not None:
                    boundary.pending_boundaries.clear()
        for hook_handle in hook_handles:
            hook_handle.remove()

    def empty_partition_cache(self):
        self.partition_all_parameters()

    def _convert_to_zero_parameters(self, ds_config, module, mpu):
        non_zero_params = [p for p in module.parameters() if not is_zero_param(p)]
        if non_zero_params:
            zero_params = [p for p in module.parameters() if is_zero_param(p)]
            if zero_params:
                zero_params[0].convert_to_zero_parameters(param_list=non_zero_params)
            else:
                group = None
                # parallel_state_sp doesn't have get_data_parallel_group
                if mpu and hasattr(mpu, "get_data_parallel_group"):
                    group = mpu.get_data_parallel_group()

                Init(module=module,
                     data_parallel_group=group,
                     dtype=self.dtype,
                     config_dict_or_path=ds_config,
                     remote_device=self.offload_device,
                     pin_memory=self.offload_param_pin_memory,
                     mpu=mpu,
                     zero_param_parallel_group=self.zero_param_parallel_group,
                     zero_quantized_weights=self.zero_quantized_weights,
                     zero_quantized_nontrainable_weights=self.zero_quantized_nontrainable_weights)

    def destroy(self):
        self._remove_module_hooks()

    def _remove_module_hooks(self):
        num_forward_hooks = len(self.forward_hooks)
        num_backward_hooks = len(self.backward_hooks)

        for hook in self.forward_hooks:
            hook.remove()

        for hook in self.backward_hooks:
            hook.remove()

        if self._has_frozen_params:
            self._clear_frozen_boundaries()

        if self.fwd_pre_hook is not None:
            self.fwd_pre_hook.remove()
            self.fwd_pre_hook = None

        print_rank_0(f'Deleted module hooks: forward = {num_forward_hooks}, backward = {num_backward_hooks}',
                     force=False)

    @instrument_w_nvtx
    def _start_of_forward_hook(self, module, *args):
        self.get_param_coordinator().reset_step()

    @instrument_w_nvtx
    def _start_of_frozen_forward_hook(self, module, *args):
        self._clear_frozen_boundaries()
        self.get_param_coordinator().reset_step()

    def _register_deepspeed_module_hooks(self):
        """Register ZeRO hooks with the root reset ordered before root fetch."""

        if self.fwd_pre_hook is not None:
            self.fwd_pre_hook.remove()
        self._register_deepspeed_module(self.module)
        # Both hooks prepend, so registering reset last makes it run first.
        start_of_forward_hook = self._start_of_frozen_forward_hook if self._has_frozen_params else self._start_of_forward_hook
        self.fwd_pre_hook = self.module.register_forward_pre_hook(start_of_forward_hook, prepend=True)

    def setup_zero_stage3_hooks(self):
        self.hierarchy = 0
        self._register_deepspeed_module_hooks()

        # Add top module to stack trace
        global FWD_MODULE_STACK
        FWD_MODULE_STACK.append(self.module)

    def mark_persistent_parameters(self, param_threshold, model_threshold):
        persistent_params = []
        total_persistent_parameters = 0
        params_count = 0
        for name, param in self.module.named_parameters(recurse=True):
            if param.ds_numel + total_persistent_parameters > model_threshold:
                continue

            if param.ds_numel <= param_threshold:
                params_count += 1
                param.ds_persist = True
                persistent_params.append(param)
                total_persistent_parameters += param.ds_numel

        print_rank_0(
            f"Parameter Offload - Persistent parameters statistics: param_count = {params_count}, numel = {total_persistent_parameters}",
            force=False)

        return persistent_params

    def _register_deepspeed_module(self, module, count=[0]):
        # re-registering hooks on the root module leaves the coordinator trace stale;
        # invalidate so it re-records on the next forward.
        if module is self.module:
            coordinator = self.get_param_coordinator()
            if coordinator is not None and not coordinator.is_invalid_trace():
                coordinator._invalidate_trace()
        my_count = count[0]
        module.ds_id = my_count

        #print(f"{module.__class__} : {module.ds_id}")

        if z3_leaf_module(module):
            for param in module.parameters():
                param.ds_z3_leaf_module = module
        else:
            for child in module.children():
                count[0] = count[0] + 1
                self._register_deepspeed_module(child, count=count)

        active_forward_invocations = threading.local()
        module_has_frozen_params = any(not param.requires_grad
                                       for param in iter_params(module, recurse=z3_leaf_module(module)))
        module_contains_frozen_params = self._has_frozen_params and any(not param.requires_grad
                                                                        for param in module.parameters())
        pending_forward_boundaries = [] if module_contains_frozen_params else None
        if module_contains_frozen_params:
            module._ds_frozen_forward_boundaries = pending_forward_boundaries

        def _invocation_depth():
            return getattr(active_forward_invocations, "depth", 0)

        def _root_forward_is_active():
            global FWD_MODULE_STACK
            return len(FWD_MODULE_STACK) > 1 and FWD_MODULE_STACK[1] is self.module

        def _push_deferred_release(deferred_release):
            stack = getattr(active_forward_invocations, "deferred_releases", None)
            if stack is None:
                stack = []
                active_forward_invocations.deferred_releases = stack
            stack.append(deferred_release)

        def _set_deferred_release(deferred_release):
            stack = getattr(active_forward_invocations, "deferred_releases", None)
            if not stack or stack[-1] is not None:
                raise RuntimeError("ZeRO-3 frozen forward invocation has no deferred-release slot")
            stack[-1] = deferred_release

        def _pop_deferred_release():
            stack = getattr(active_forward_invocations, "deferred_releases", None)
            if not stack:
                return None
            return stack.pop()

        if module_has_frozen_params:

            def release_after_forward(module):
                self.post_sub_module_forward_function(module, deferred_release=_pop_deferred_release())

        else:
            release_after_forward = self.post_sub_module_forward_function

        if module_has_frozen_params:

            @torch.compiler.disable
            def _pre_forward_module_hook(module, *args):
                see_memory_usage(f"Before sub module function {module.__class__.__name__}", force=False)

                global FWD_MODULE_STACK
                FWD_MODULE_STACK.append(module)
                active_forward_invocations.depth = _invocation_depth() + 1
                _push_deferred_release(None)

                self.pre_sub_module_forward_function(module)

        else:

            @torch.compiler.disable
            def _pre_forward_module_hook(module, *args):
                see_memory_usage(f"Before sub module function {module.__class__.__name__}", force=False)

                global FWD_MODULE_STACK
                FWD_MODULE_STACK.append(module)
                active_forward_invocations.depth = _invocation_depth() + 1

                self.pre_sub_module_forward_function(module)

        @instrument_w_nvtx
        def _post_forward_module_hook(module, input, output):

            # A global or earlier pre-hook can fail before ZeRO's pre-hook runs,
            # while PyTorch still runs local always-call post-hooks.
            depth = _invocation_depth()
            if depth == 0:
                return
            active_forward_invocations.depth = depth - 1

            global FWD_MODULE_STACK
            if not FWD_MODULE_STACK or FWD_MODULE_STACK[-1] is not module:
                raise RuntimeError(f"ZeRO-3 forward module stack is unbalanced at {module.__class__.__name__}")
            early_stop_deferred_release = None
            if (output is None and self._has_frozen_params and not module_has_frozen_params
                    and self.get_param_coordinator().has_active_outer_backward()):
                boundary_pools = []
                seen_boundary_pools = set()
                for enclosing_module in reversed(FWD_MODULE_STACK[:-1]):
                    boundary_pool = getattr(enclosing_module, "_ds_frozen_forward_boundaries", None)
                    if boundary_pool is not None and id(boundary_pool) not in seen_boundary_pools:
                        boundary_pools.append(boundary_pool)
                        seen_boundary_pools.add(id(boundary_pool))
                input_tensors = tuple(obj for obj in tree_flatten(input)[0]
                                      if torch.is_tensor(obj) and obj.requires_grad)
                try:
                    early_stop_deferred_release = self._bind_frozen_boundary(input_tensors,
                                                                             None,
                                                                             boundary_pools,
                                                                             protect_all_params=True)
                except Exception:
                    # PyTorch suppresses an always-call hook exception while
                    # propagating checkpoint's internal early-stop exception.
                    # Preserve correctness by retaining this exact invocation
                    # for the already-armed root callback instead.
                    early_stop_deferred_release = self.get_param_coordinator().begin_deferred_release(
                        protect_all_params=True)
            FWD_MODULE_STACK.pop()
            if output is None:
                output = []
            elif not isinstance(output, (list, tuple)):
                if torch.is_tensor(output):
                    output = [output]
                else:
                    #print(f'got UNKNOWN type {type(output)}')
                    outputs = []
                    output = output if isinstance(output, dict) else vars(output)
                    for name, val in output.items():
                        if not name.startswith('__') and torch.is_tensor(val):
                            outputs.append(val)
                    output = outputs

            for item in filter(lambda item: is_zero_param(item) or hasattr(item, 'ds_param_alias'), output):
                key = id(item) if hasattr(item, 'ds_id') else id(item.ds_param_alias)
                actual_external_param = item if hasattr(item, 'ds_id') else item.ds_param_alias

                if not any(key in m._external_params for m in FWD_MODULE_STACK):
                    actual_external_param.is_external_param = True
                    module_to_register = FWD_MODULE_STACK[-1]
                    register_external_parameter(module_to_register, actual_external_param)
                    print_rank_0(
                        f'Registering dangling parameter for module {module_to_register.__class__.__name__}, ds_id = {actual_external_param.ds_id}.',
                        force=False)

                    # It's possible that the parameter was already external to the completed module. If so, remove it the
                    # registration as it will be covered by the outer module instead.
                    if key in module._external_params:
                        print_rank_0(
                            f'  Unregistering nested dangling parameter from module {module.__class__.__name__}, ds_id = {actual_external_param.ds_id}',
                            force=False)
                        unregister_external_parameter(module, actual_external_param)

                    actual_external_param.all_gather()

            if early_stop_deferred_release is None:
                release_after_forward(module)
            else:
                self.post_sub_module_forward_function(module, deferred_release=early_stop_deferred_release)

        def _bwd_hook_unexpected_inputs_msg(value):
            return f"A module has unknown inputs or outputs type ({type(value)}) and the tensors embedded in it cannot be detected. " \
                "The ZeRO-3 hooks designed to trigger before or after backward pass of the module relies on knowing the input and " \
                "output tensors and therefore may not get triggered properly."

        if self._has_frozen_params:

            def _pre_backward_module_hook(module, inputs, output):
                needs_root_fallback = not any(
                    torch.is_tensor(obj) and obj.requires_grad for obj in tree_flatten(inputs)[0])

                def apply_frozen_pre_backward(output):
                    return module.frozen_pre_bwd_fn.apply(output, needs_root_fallback)

                return apply_to_tensors_only(apply_frozen_pre_backward,
                                             output,
                                             warning_msg_fn=_bwd_hook_unexpected_inputs_msg)

        else:

            def _pre_backward_module_hook(module, inputs, output):

                return apply_to_tensors_only(module.pre_bwd_fn.apply,
                                             output,
                                             warning_msg_fn=_bwd_hook_unexpected_inputs_msg)

        #This is an alternate to doing _post_backward_module_hook
        #it uses tensor.register_hook instead of using torch.autograd.Function
        def _alternate_post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0

            #print(f"Before Forward {module.__class__.__name__}")

            def _run_after_backward_hook(*unused):
                module.ds_grads_remaining = module.ds_grads_remaining - 1
                if module.ds_grads_remaining == 0:
                    #print(f"After backward {module.__class__.__name__}")
                    self.post_sub_module_backward_function(module)

            def _run_before_forward_function(input):
                if input.requires_grad:
                    module.ds_grads_remaining += 1

            return _apply_forward_and_backward_to_tensors_only(module, _run_before_forward_function,
                                                               _run_after_backward_hook, inputs)

        if module_has_frozen_params:

            @torch.compiler.disable
            def _post_backward_module_hook(module, inputs, kwargs):
                module.ds_grads_remaining = 0
                deferred_release = None
                coordinator = self.get_param_coordinator()
                input_tensors = tuple(obj for obj in tree_flatten((inputs, kwargs))[0]
                                      if torch.is_tensor(obj) and obj.requires_grad)
                if coordinator.has_active_outer_backward():
                    deferred_release = self._bind_frozen_boundary(input_tensors, pending_forward_boundaries)
                elif not _root_forward_is_active():
                    raise RuntimeError("ZeRO-3 frozen checkpoint replay requires backward from the engine output")
                else:
                    self._register_frozen_boundary(input_tensors, pending_forward_boundaries)
                _set_deferred_release(deferred_release)

                inputs = apply_to_tensors_only(module.post_bwd_fn.apply,
                                               inputs,
                                               warning_msg_fn=_bwd_hook_unexpected_inputs_msg)
                return inputs, kwargs

        elif module_contains_frozen_params:

            @torch.compiler.disable
            def _post_backward_module_hook(module, inputs):
                module.ds_grads_remaining = 0
                coordinator = self.get_param_coordinator()
                if not coordinator.has_active_outer_backward():
                    if not _root_forward_is_active():
                        raise RuntimeError("ZeRO-3 frozen checkpoint replay requires backward from the engine output")
                    input_tensors = tuple(obj for obj in tree_flatten(inputs)[0]
                                          if torch.is_tensor(obj) and obj.requires_grad)
                    self._register_frozen_boundary(input_tensors, pending_forward_boundaries)

                return apply_to_tensors_only(module.post_bwd_fn.apply,
                                             inputs,
                                             warning_msg_fn=_bwd_hook_unexpected_inputs_msg)

        else:

            @torch.compiler.disable
            def _post_backward_module_hook(module, inputs):
                module.ds_grads_remaining = 0

                return apply_to_tensors_only(module.post_bwd_fn.apply,
                                             inputs,
                                             warning_msg_fn=_bwd_hook_unexpected_inputs_msg)

        # Pre forward hook
        self.forward_hooks.append(module.register_forward_pre_hook(_pre_forward_module_hook))

        # Non-reentrant checkpoint early-stop raises an internal exception. Native
        # always-call hooks unwind the module stack immediately, before a later
        # reentrant recompute in the same outer backward observes its parent.
        self.forward_hooks.append(module.register_forward_hook(_post_forward_module_hook, always_call=True))

        # Pre backward hook
        if self._has_frozen_params and not hasattr(module, "frozen_pre_bwd_fn"):

            if module is self.module:

                @instrument_w_nvtx
                def _run_before_backward_function(sub_module, needs_root_fallback):
                    self._begin_outer_backward()
                    if sub_module.applied_pre_backward_ref_cnt > 0:
                        self.pre_sub_module_backward_function(sub_module)
                        if needs_root_fallback:
                            self.get_param_coordinator().defer_missing_post_backward(sub_module)
                        sub_module.applied_pre_backward_ref_cnt -= 1

            else:

                @instrument_w_nvtx
                def _run_before_backward_function(sub_module, needs_root_fallback):
                    if sub_module.applied_pre_backward_ref_cnt > 0:
                        self.pre_sub_module_backward_function(sub_module)
                        if needs_root_fallback:
                            self.get_param_coordinator().defer_missing_post_backward(sub_module)
                        sub_module.applied_pre_backward_ref_cnt -= 1

            class FrozenPreBackwardFunctionForModule(torch.autograd.Function):

                @staticmethod
                def forward(outputs, needs_root_fallback):
                    return outputs.detach()

                @staticmethod
                def setup_context(ctx, inputs, output):
                    _, needs_root_fallback = inputs
                    ctx.module = module
                    ctx.needs_root_fallback = needs_root_fallback
                    ctx.pre_backward_function = _run_before_backward_function
                    if not hasattr(ctx.module, "applied_pre_backward_ref_cnt"):
                        ctx.module.applied_pre_backward_ref_cnt = 0
                    ctx.module.applied_pre_backward_ref_cnt += 1

                @staticmethod
                def backward(ctx, *args):
                    ctx.pre_backward_function(ctx.module, ctx.needs_root_fallback)
                    return args + (None, )

            module.frozen_pre_bwd_fn = FrozenPreBackwardFunctionForModule

        elif not self._has_frozen_params and not hasattr(module, "pre_bwd_fn"):

            if module is self.module and self._has_frozen_params:

                @instrument_w_nvtx
                def _run_before_backward_function(sub_module):
                    self._begin_outer_backward()
                    if sub_module.applied_pre_backward_ref_cnt > 0:
                        self.pre_sub_module_backward_function(sub_module)
                        sub_module.applied_pre_backward_ref_cnt -= 1

            else:

                @instrument_w_nvtx
                def _run_before_backward_function(sub_module):
                    # some models (e.g. Albert) may run multiple forwards on the same layer in a loop
                    # before doing backwards, so each backward will need a pre-fetch - using reference
                    # counting to support this scenario
                    #print(f"COUNTER before: {sub_module.applied_pre_backward_ref_cnt}")
                    if sub_module.applied_pre_backward_ref_cnt > 0:
                        self.pre_sub_module_backward_function(sub_module)
                        sub_module.applied_pre_backward_ref_cnt -= 1
                    #print(f"COUNTER after: {sub_module.applied_pre_backward_ref_cnt}")

            class PreBackwardFunctionForModule(torch.autograd.Function):

                @staticmethod
                def forward(outputs):
                    return outputs.detach()

                @staticmethod
                def setup_context(ctx, inputs, output):
                    ctx.module = module
                    ctx.pre_backward_function = _run_before_backward_function
                    if not hasattr(ctx.module, "applied_pre_backward_ref_cnt"):
                        ctx.module.applied_pre_backward_ref_cnt = 0
                    ctx.module.applied_pre_backward_ref_cnt += 1

                @staticmethod
                def backward(ctx, *args):
                    ctx.pre_backward_function(ctx.module)
                    return args

            module.pre_bwd_fn = PreBackwardFunctionForModule

        self.backward_hooks.append(module.register_forward_hook(_pre_backward_module_hook))

        # post backward hook
        if not hasattr(module, "post_bwd_fn"):

            @instrument_w_nvtx
            def _run_after_backward_function(sub_module):
                if sub_module.ds_grads_remaining == 0:
                    self.post_sub_module_backward_function(sub_module)

            class PostBackwardFunctionModule(torch.autograd.Function):

                @staticmethod
                def forward(output):
                    return output.detach()

                @staticmethod
                def setup_context(ctx, inputs, output):
                    (output_in, ) = inputs
                    ctx.module = module
                    if output_in.requires_grad:
                        #TODO SOME TIMES post backward does not seem to be triggered debug in detail
                        #Should only cause increase in memory not correctness issue
                        #if output.grad_fn.__class__.__name__ == 'ViewBackward':
                        #    ctx.view=True
                        #    print(f"Warning view tensor for input to module : {module.__class__.__name__}. Backward hooks may not trigger properly")
                        #assert len(module.parameters(recurse=False)), "The input tensor to the module is a view, and autograd Function or register_hook is not triggered with view tensors."
                        #if module.ds_grads_remaining == 0:
                        #    print(f"Before Forward: {ctx.module.__class__.__name__}")
                        module.ds_grads_remaining += 1
                        ctx.post_backward_function = _run_after_backward_function

                @staticmethod
                def backward(ctx, *args):
                    ctx.module.ds_grads_remaining = ctx.module.ds_grads_remaining - 1
                    if ctx.module.ds_grads_remaining == 0:
                        ctx.post_backward_function(ctx.module)
                    return args

            module.post_bwd_fn = PostBackwardFunctionModule

        if module_has_frozen_params:
            self.backward_hooks.append(module.register_forward_pre_hook(_post_backward_module_hook, with_kwargs=True))
        else:
            self.backward_hooks.append(module.register_forward_pre_hook(_post_backward_module_hook))

    @torch.no_grad()
    def pre_sub_module_forward_function(self, sub_module):
        param_coordinator = self.get_param_coordinator()
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            param_coordinator.record_module(sub_module)
        param_coordinator.fetch_sub_module(sub_module, forward=True)

        if self.zenflow:
            params_to_fetch = set(iter_params(sub_module, recurse=z3_leaf_module(sub_module)))
            for param in params_to_fetch:
                param.data = param.data.t() if len(param.ds_shape) != 1 else param.data

        see_memory_usage(f"Before sub module function {sub_module.__class__.__name__} after fetch", force=False)

    @torch.no_grad()
    def post_sub_module_forward_function(self, sub_module, deferred_release=None):
        see_memory_usage(
            f"After sub module function {sub_module.__class__.__name__} {sub_module.ds_id} before release",
            force=False)

        if self.zenflow:
            params_to_fetch = set(iter_params(sub_module, recurse=z3_leaf_module(sub_module)))
            for param in params_to_fetch:
                param.data = param.data.t() if len(param.ds_shape) != 1 else param.data

        param_coordinator = self.get_param_coordinator()
        param_coordinator.release_sub_module(sub_module, forward=True, deferred_release=deferred_release)

        see_memory_usage(
            f"After sub module function {sub_module.__class__.__name__}  {sub_module.ds_id} after release",
            force=False)

    @torch.no_grad()
    def pre_sub_module_backward_function(self, sub_module):
        # assert sub_module.training, "backward pass is invalid for module in evaluation mode"
        param_coordinator = self.get_param_coordinator()
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            param_coordinator.record_module(sub_module)
        param_coordinator.fetch_sub_module(sub_module, forward=False)

        if self.zenflow:
            params_to_fetch = set(iter_params(sub_module, recurse=z3_leaf_module(sub_module)))
            for param in params_to_fetch:
                param.data = param.data.t() if len(param.ds_shape) != 1 else param.data

    @torch.no_grad()
    def post_sub_module_backward_function(self, sub_module):
        # assert sub_module.training, "backward pass is invalid for module in evaluation mode"
        see_memory_usage(
            f"After sub module backward function {sub_module.__class__.__name__} {sub_module.ds_id} before release",
            force=False)

        if self.zenflow:
            params_to_fetch = set(iter_params(sub_module, recurse=z3_leaf_module(sub_module)))
            for param in params_to_fetch:
                param.data = param.data.t() if len(param.ds_shape) != 1 else param.data

        self.get_param_coordinator().release_sub_module(sub_module, forward=False)

        see_memory_usage(
            f"After sub module backward function {sub_module.__class__.__name__} {sub_module.ds_id} after release",
            force=False)

    def _set_z3_leaf_modules_by_threshold(self, module, zero_module_granularity_threshold):

        self._get_granularity_recursively(module)
        print_rank_0(f"{'MODULE NAME'.ljust(30)}|{'GRANULARITY VALUE'.rjust(20)}", force=False)
        for granularity in self.granularity_info:
            print_rank_0(granularity, force=False)

        if self.min_granularity_value <= zero_module_granularity_threshold:
            self._set_leaf_by_threshold_preorder(module, zero_module_granularity_threshold)
            utils.logger.info(
                f"z3_leaf_module was set by stage3_module_granularity_threshold:{zero_module_granularity_threshold}")
            for layer in self.z3_leaf_layers:
                print_rank_0(f"{layer.__class__.__name__}:{layer.ds_model_granularity}", force=False)
        else:
            utils.logger.warning(
                f"The smallest module granularity is [{self.min_granularity_layer}:{self.min_granularity_value}]. "\
                f"To make stage3_module_granularity_threshold effective, you need to set stage3_module_granularity_threshold >= {self.min_granularity_value}. "\
                f"Current Value:{zero_module_granularity_threshold}"
            )

    def _get_granularity_recursively(self, module):
        """This function is used to recursively obtain the granularity of each module."""

        # avoid setting as leaf for particularly large models, even if the granularity is very small
        # an oversized leaf module increases the number of live parameters, introducing memory overhead
        Z3_MAX_LEAF_SIZE = 1e9

        if not list(module.parameters()):
            # skip Modules without parameters, such as GELU, etc.
            module.ds_model_granularity = sys.maxsize
            return 0, 0

        num_layers = 0
        num_params = 0
        num_params += sum(p.ds_numel for p in module.parameters(recurse=False))
        if not any(module.children()):
            # torch leaf module
            module.ds_model_granularity = sys.maxsize
            return 1, num_params

        for child in module.children():
            layers_in_child, params_in_child = self._get_granularity_recursively(child)
            num_layers += layers_in_child
            num_params += params_in_child

        if module.__class__.__name__ in torch.nn.modules.container.__all__:
            # Do not set container modules like ModuleList as leaf modules
            # as this will prevent hooks from being set on their children
            # and they may do not invoke the forward method
            module.ds_model_granularity = sys.maxsize
            return num_layers, num_params

        num_layers += 1
        ds_model_granularity = (num_params // num_layers) if num_params <= Z3_MAX_LEAF_SIZE else sys.maxsize
        module.ds_model_granularity = ds_model_granularity
        # module.ds_model_num_layers = num_layers
        # module.ds_model_num_params = num_params
        if self.min_granularity_value > ds_model_granularity:
            self.min_granularity_value = ds_model_granularity
            self.min_granularity_layer = module.__class__.__name__
        self.granularity_info.add(f"{module.__class__.__name__.ljust(30)}|{str(ds_model_granularity).rjust(20)}")

        return num_layers, num_params

    def _set_leaf_by_threshold_preorder(self, module, granularity_treshhold):
        '''Set modules as leaf modules based on the threshold, prioritizing parent nodes.'''

        num_params = sum(p.ds_numel for p in module.parameters())
        if num_params == 0:
            # skip Modules without parameters, such as GELU, etc.
            return
        if module.ds_model_granularity <= granularity_treshhold:
            set_z3_leaf_module(module, True)
            self.z3_leaf_layers.append(module)
            return

        for sub_module in module.children():
            self._set_leaf_by_threshold_preorder(sub_module, granularity_treshhold)
