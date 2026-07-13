# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import inspect
import sys
import torch
from collections import defaultdict, OrderedDict
import threading
from deepspeed.utils import z3_leaf_module, set_z3_leaf_module
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.runtime.zero.utils import apply_to_tensors_only, is_zero_param
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import _init_external_params
from deepspeed.runtime.zero.partition_parameters import *
from deepspeed.runtime.zero.partitioned_param_coordinator import (PartitionedParameterCoordinator,
                                                                  InflightParamRegistry, current_graph_task_id,
                                                                  iter_params)
import deepspeed.runtime.compiler as compiler
from deepspeed.accelerator import get_accelerator
from deepspeed import utils

FWD_MODULE_STACK = list()
FORWARD_HOOK_ALWAYS_CALL_SUPPORTED = "always_call" in inspect.signature(
    torch.nn.Module.register_forward_hook).parameters
_MODULE_CALL_IMPL_CODE = inspect.unwrap(torch.nn.Module._call_impl).__code__
_MISSING_CLASS_CALL = object()
_MODULE_CALL_GUARD_LOCK = threading.RLock()
_MODULE_CALL_GUARD_CLASSES = {}


def _module_forward_call_token(module):
    """Return the active ``Module._call_impl`` frame id for one invocation."""

    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code is _MODULE_CALL_IMPL_CODE and frame.f_locals.get("self") is module:
            return id(frame)
        frame = frame.f_back
    raise RuntimeError(f"ZeRO-3 could not identify the active forward invocation for {module.__class__.__name__}")


def _install_module_call_guard(module, guard):
    """Intercept ``module(...)`` on versions where ``__call__`` bypasses instance ``_call_impl``.

    PyTorch 2.0 binds ``Module.__call__`` directly to the class implementation, so
    replacing ``module._call_impl`` does not cover exceptions raised by hooks. Patch
    each concrete module class once and dispatch only registered instances through
    their ZeRO guard. This preserves the concrete class identity and the class's
    original ``__call__`` implementation for guarded and unguarded instances.
    """

    module_class = type(module)
    with _MODULE_CALL_GUARD_LOCK:
        class_state = _MODULE_CALL_GUARD_CLASSES.get(module_class)
        if class_state is None:
            original_owned_call = module_class.__dict__.get("__call__", _MISSING_CLASS_CALL)
            original_call = getattr(module_class, "__call__")
            class_state = {
                "original_owned_call": original_owned_call,
                "original_call": original_call,
                "guards": {},
            }

            @functools.wraps(original_call)
            def guarded_module_call(called_module, *args, **kwargs):
                registered_guard = class_state["guards"].get(id(called_module))
                if registered_guard is None or registered_guard[0] is not called_module:
                    return original_call(called_module, *args, **kwargs)
                return registered_guard[1](original_call, called_module, *args, **kwargs)

            class_state["wrapper"] = guarded_module_call
            try:
                module_class.__call__ = guarded_module_call
            except (AttributeError, TypeError) as error:
                raise RuntimeError(
                    f"ZeRO-3 cannot install its PyTorch 2.0 module-call guard for {module_class.__name__}") from error
            _MODULE_CALL_GUARD_CLASSES[module_class] = class_state

        if id(module) in class_state["guards"]:
            raise RuntimeError(f"ZeRO-3 module-call guard is already installed for {module_class.__name__}")
        class_state["guards"][id(module)] = (module, guard)
        return class_state


def _module_call_guard_is_current(module, guard, class_state):
    module_class = type(module)
    registered_guard = class_state["guards"].get(id(module))
    return (module_class.__dict__.get("__call__") is class_state["wrapper"] and registered_guard is not None
            and registered_guard[0] is module and registered_guard[1] is guard)


def _remove_module_call_guard(module, guard, class_state):
    """Remove one instance guard and restore its class when the last guard leaves."""

    module_class = type(module)
    with _MODULE_CALL_GUARD_LOCK:
        if not _module_call_guard_is_current(module, guard, class_state):
            raise RuntimeError(
                f"ZeRO-3 cannot safely remove its module-call guard after rebinding: {module_class.__name__}")
        del class_state["guards"][id(module)]
        if class_state["guards"]:
            return

        if class_state["original_owned_call"] is _MISSING_CLASS_CALL:
            delattr(module_class, "__call__")
        else:
            module_class.__call__ = class_state["original_owned_call"]
        del _MODULE_CALL_GUARD_CLASSES[module_class]


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
            if self._parent_module._parameters._in_forward and not compiler.is_compiling():
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

        self.forward_hooks = []
        self.backward_hooks = []
        self.forward_wrappers = []
        self.forward_call_wrappers = []
        self.forward_wrapper_states = {}
        self.fwd_pre_hook = None
        self.__fwd_modules_by_graph = defaultdict(list)
        self.__fwd_module_cleanup_callbacks = set()
        self.__fwd_module_stack_lock = threading.Lock()

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
        self.get_param_coordinator().release_and_reset_all(self.module)
        for param in iter_params(self.module, recurse=True):
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(f"{param.ds_summary()} expected to be released")

    def get_param_coordinator(self):
        return self.param_coordinator

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

        self._check_forward_wrappers_removable()
        for hook in self.forward_hooks:
            hook.remove()

        for hook in self.backward_hooks:
            hook.remove()

        if self.fwd_pre_hook is not None:
            self.fwd_pre_hook.remove()
            self.fwd_pre_hook = None

        self._remove_forward_wrappers()

        print_rank_0(f'Deleted module hooks: forward = {num_forward_hooks}, backward = {num_backward_hooks}',
                     force=False)

    def _remove_forward_wrappers(self):
        """Restore forwards and module calls wrapped for PyTorch 2.0 hook pairing."""

        self._check_forward_wrappers_removable()
        for module, _, state in reversed(self.forward_wrappers):
            module.forward = state["forward"]
        for module, guard, state in reversed(self.forward_call_wrappers):
            _remove_module_call_guard(module, guard, state)
        self.forward_wrappers.clear()
        self.forward_call_wrappers.clear()
        self.forward_wrapper_states.clear()

    def _check_forward_wrappers_removable(self):
        """Fail before hook mutation if another framework rebound a wrapped forward."""

        rebound_modules = [module for module, wrapper, _ in self.forward_wrappers if module.forward is not wrapper]
        rebound_modules.extend(module for module, guard, state in self.forward_call_wrappers
                               if not _module_call_guard_is_current(module, guard, state))
        if rebound_modules:
            names = ", ".join(module.__class__.__name__ for module in rebound_modules)
            raise RuntimeError("ZeRO-3 cannot safely remove its forward exception wrappers because a wrapped forward "
                               f"or module call was rebound after initialization: {names}")

    def _get_forward_delegate(self, module):
        """Return the logical forward owned by an optional ZeRO exception wrapper."""

        wrapper_state = self.forward_wrapper_states.get(module)
        if wrapper_state is None:
            return module.forward

        wrapper, state = wrapper_state
        if module.forward is not wrapper:
            raise RuntimeError("ZeRO-3 cannot update a forward exception wrapper because module.forward was rebound "
                               f"after initialization: {module.__class__.__name__}")
        return state["forward"]

    def _set_forward_delegate(self, module, forward):
        """Update the logical forward without replacing ZeRO's outer exception wrapper."""

        wrapper_state = self.forward_wrapper_states.get(module)
        if wrapper_state is None:
            module.forward = forward
            return

        wrapper, state = wrapper_state
        if module.forward is not wrapper:
            raise RuntimeError("ZeRO-3 cannot update a forward exception wrapper because module.forward was rebound "
                               f"after initialization: {module.__class__.__name__}")
        state["forward"] = forward

    @instrument_w_nvtx
    def _start_of_forward_hook(self, module, *args):
        if current_graph_task_id() == -1:
            self._release_unfinished_forward_modules()
        self.get_param_coordinator().reset_step()

    def _register_deepspeed_module_hooks(self):
        """Register ZeRO hooks with the root reset ordered before root fetch."""

        if self.fwd_pre_hook is not None:
            self.fwd_pre_hook.remove()
        self._register_deepspeed_module(self.module)
        # Both hooks prepend on PyTorch 2.1+, so registering reset last makes it
        # run first. The PyTorch 2.0 fallback appends fetch and has the same order.
        self.fwd_pre_hook = self.module.register_forward_pre_hook(self._start_of_forward_hook, prepend=True)

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

        always_call_supported = FORWARD_HOOK_ALWAYS_CALL_SUPPORTED
        active_forward_invocations = threading.local()
        active_fallback_calls = threading.local()

        def _invocation_entries():
            if not hasattr(active_forward_invocations, "entries"):
                active_forward_invocations.entries = []
            return active_forward_invocations.entries

        def _fallback_call_tokens():
            if not hasattr(active_fallback_calls, "tokens"):
                active_fallback_calls.tokens = []
            return active_fallback_calls.tokens

        def _forward_call_token():
            if always_call_supported:
                return _module_forward_call_token(module)
            call_states = _fallback_call_tokens()
            if not call_states:
                raise RuntimeError(
                    f"ZeRO-3 could not identify the active fallback forward invocation for {module.__class__.__name__}"
                )
            return call_states[-1]["token"]

        @compiler.disable
        def _pre_forward_module_hook(module, *args):
            entry = {"token": _forward_call_token(), "acquired": False, "exception_state": sys.exc_info()}
            _invocation_entries().append(entry)
            see_memory_usage(f"Before sub module function {module.__class__.__name__}", force=False)

            global FWD_MODULE_STACK
            FWD_MODULE_STACK.append(module)
            entry["acquired"] = True
            if not always_call_supported:
                _fallback_call_tokens()[-1]["zero_pre_acquired"] = True
            self._track_forward_module(module)

            self.pre_sub_module_forward_function(module)

        @instrument_w_nvtx
        def _post_forward_module_hook(module, input, output):

            exception_state = sys.exc_info()
            token = _forward_call_token()
            entries = _invocation_entries()
            entry = None
            for index in range(len(entries) - 1, -1, -1):
                if entries[index]["token"] == token:
                    entry = entries.pop(index)
                    break

            # A global or earlier pre-hook can fail before ZeRO's pre-hook runs,
            # while PyTorch still runs local always_call post-hooks. Only unwind
            # invocations for which ZeRO actually pushed the module stack entry.
            if entry is None or not entry["acquired"]:
                return

            # ``sys.exc_info`` can already be populated when a model is invoked
            # from an outer except block. Only classify this call as failed when
            # PyTorch is handling a different exception state than ZeRO observed
            # on entry.
            forward_failed = any(current is not previous
                                 for current, previous in zip(exception_state, entry["exception_state"]))

            defer_release = output is None and current_graph_task_id() != -1
            global FWD_MODULE_STACK
            FWD_MODULE_STACK.pop()
            self._finish_forward_module(module)
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

            self.post_sub_module_forward_function(module, defer_release=defer_release)

            # A partial root forward can prefetch parameters owned by modules whose
            # hooks never ran. The next reset assumes no such residency and resets
            # its counter to zero, so fully reconcile an ordinary failed forward
            # before a caller can catch the exception and retry. Recompute failures
            # stay scoped to their graph-task cleanup instead.
            if forward_failed and module is self.module and current_graph_task_id() == -1:
                self.partition_all_parameters()
                if not always_call_supported:
                    _fallback_call_tokens()[-1]["root_reconciled"] = True

        def _bwd_hook_unexpected_inputs_msg(value):
            return f"A module has unknown inputs or outputs type ({type(value)}) and the tensors embedded in it cannot be detected. " \
                "The ZeRO-3 hooks designed to trigger before or after backward pass of the module relies on knowing the input and " \
                "output tensors and therefore may not get triggered properly."

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

        @compiler.disable
        def _post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0

            return apply_to_tensors_only(module.post_bwd_fn.apply,
                                         inputs,
                                         warning_msg_fn=_bwd_hook_unexpected_inputs_msg)

        # Pre forward hook
        self.forward_hooks.append(
            module.register_forward_pre_hook(_pre_forward_module_hook, prepend=FORWARD_HOOK_ALWAYS_CALL_SUPPORTED))

        # Non-reentrant checkpoint recompute stops by raising an internal exception
        # once all needed tensors have been recreated. PyTorch 2.1+ can run the paired
        # post hook on that path. On the supported PyTorch 2.0 floor, wrap forward so
        # the same hook runs before another recomputation can observe a stale parent.
        if always_call_supported:
            self.forward_hooks.append(module.register_forward_hook(_post_forward_module_hook, always_call=True))
        else:
            self.forward_hooks.append(module.register_forward_hook(_post_forward_module_hook))
            forward_state = {"forward": module.forward}

            @functools.wraps(forward_state["forward"])
            def exception_safe_forward(*args, **kwargs):
                try:
                    return forward_state["forward"](*args, **kwargs)
                except Exception:
                    try:
                        _post_forward_module_hook(module, args, None)
                    except Exception as hook_error:
                        utils.logger.warning(f"ZeRO-3 failed to unwind a forward exception: {hook_error}")
                    raise

            module.forward = exception_safe_forward
            self.forward_wrappers.append((module, exception_safe_forward, forward_state))
            self.forward_wrapper_states[module] = (exception_safe_forward, forward_state)

            # ``module.forward`` begins only after every pre-hook has run. Guard the
            # whole class-dispatched module call as well, so this also works on
            # PyTorch 2.0 where ``Module.__call__`` bypasses instance ``_call_impl``.
            def exception_safe_module_call(original_call, called_module, *args, **kwargs):
                call_state = {
                    "token": object(),
                    "zero_pre_acquired": False,
                    "root_reconciled": False,
                }
                call_states = _fallback_call_tokens()
                call_states.append(call_state)
                try:
                    return original_call(called_module, *args, **kwargs)
                except Exception:
                    try:
                        _post_forward_module_hook(module, args, None)
                    except Exception as hook_error:
                        utils.logger.warning(f"ZeRO-3 failed to unwind a module-call exception: {hook_error}")
                    if (call_state["zero_pre_acquired"] and not call_state["root_reconciled"] and module is self.module
                            and current_graph_task_id() == -1):
                        try:
                            self.partition_all_parameters()
                            call_state["root_reconciled"] = True
                        except Exception as hook_error:
                            utils.logger.warning(f"ZeRO-3 failed to reconcile a root forward exception: {hook_error}")
                    raise
                finally:
                    if call_states[-1] is not call_state:
                        raise RuntimeError("ZeRO-3 fallback module-call tokens were unbalanced")
                    call_states.pop()

            call_guard_state = _install_module_call_guard(module, exception_safe_module_call)
            self.forward_call_wrappers.append((module, exception_safe_module_call, call_guard_state))

        # Pre backward hook
        if not hasattr(module, "pre_bwd_fn"):

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

    def _track_forward_module(self, sub_module):
        """Arrange to unwind forward-stack entries skipped by checkpoint early-stop."""
        graph_task_id = current_graph_task_id()
        if graph_task_id == -1:
            return

        with self.__fwd_module_stack_lock:
            self.__fwd_modules_by_graph[graph_task_id].append(sub_module)
            if graph_task_id in self.__fwd_module_cleanup_callbacks:
                return

            engine = getattr(torch.autograd.Variable, "_execution_engine", None)
            if engine is None or not hasattr(engine, "queue_callback"):
                return
            self.__fwd_module_cleanup_callbacks.add(graph_task_id)

            def release_forward_modules():
                self._release_unfinished_forward_modules(graph_task_id)

            try:
                engine.queue_callback(release_forward_modules)
            except Exception:
                self.__fwd_module_cleanup_callbacks.discard(graph_task_id)
                raise

    def _finish_forward_module(self, sub_module):
        """Remove a normally completed module from graph-task fallback tracking."""
        graph_task_id = current_graph_task_id()
        if graph_task_id == -1:
            return

        with self.__fwd_module_stack_lock:
            graph_modules = self.__fwd_modules_by_graph.get(graph_task_id, [])
            for index in range(len(graph_modules) - 1, -1, -1):
                if graph_modules[index] is sub_module:
                    del graph_modules[index]
                    break

    def _release_unfinished_forward_modules(self, graph_task_id=None):
        """Remove only stack entries left by forward exceptions in completed graphs."""
        with self.__fwd_module_stack_lock:
            graph_task_ids = list(self.__fwd_modules_by_graph) if graph_task_id is None else [graph_task_id]
            unfinished_modules = []
            for completed_graph_task_id in graph_task_ids:
                unfinished_modules.extend(self.__fwd_modules_by_graph.pop(completed_graph_task_id, []))
                self.__fwd_module_cleanup_callbacks.discard(completed_graph_task_id)

        global FWD_MODULE_STACK
        for sub_module in reversed(unfinished_modules):
            # The first entry is the engine root sentinel. Remove the rightmost
            # matching invocation so nested/reentrant uses of one module stay paired.
            for index in range(len(FWD_MODULE_STACK) - 1, 0, -1):
                if FWD_MODULE_STACK[index] is sub_module:
                    del FWD_MODULE_STACK[index]
                    break

    @torch.no_grad()
    def post_sub_module_forward_function(self, sub_module, defer_release=False):
        see_memory_usage(
            f"After sub module function {sub_module.__class__.__name__} {sub_module.ds_id} before release",
            force=False)

        if self.zenflow:
            params_to_fetch = set(iter_params(sub_module, recurse=z3_leaf_module(sub_module)))
            for param in params_to_fetch:
                param.data = param.data.t() if len(param.ds_shape) != 1 else param.data

        param_coordinator = self.get_param_coordinator()
        param_coordinator.release_sub_module(sub_module, forward=True, defer_release=defer_release)

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
