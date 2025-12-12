# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.utils.torch import required_torch_version

# Use a mutable container for the flag to ensure it's shared correctly
_deepcompile_state = {'z3_active': False}

try:
    from torch._subclasses import FakeTensorMode
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch._dynamo.variables.builder import wrap_to_fake_tensor_and_record
except ImportError:
    # Unsupported torch version
    pass


def set_deepcompile_z3_active(active: bool):
    _deepcompile_state['z3_active'] = active


def wrap_if_ds_param(t):
    if hasattr(t, 'ds_id'):
        data = torch.rand(t.ds_shape,
                          dtype=t.dtype,
                          layout=t.layout,
                          device=t.device,
                          pin_memory=t.is_pinned(),
                          requires_grad=t.requires_grad)
        if isinstance(t, torch.nn.Parameter):
            t = torch.nn.Parameter(data, requires_grad=t.requires_grad)
        else:
            t = data
    return t


def _patch_for_pytorch_2_9_plus():
    """
    Apply patches specific to PyTorch 2.9+.

    PyTorch 2.9 introduced stricter guard validation that checks parameter shapes
    immediately after guards are created. This causes issues with ZeRO3 where
    parameters are sharded (size 0) but traced with full shapes.

    These patches:
    1. Patch tensor_always_has_static_shape to allow dynamic shapes for ZeRO3 parameters
    2. Patch GuardBuilder.TENSOR_MATCH to skip tensor match guards for parameters
    3. Patch CheckFunctionManager to suppress guard check failures for parameter shapes
    """
    # Patch tensor_always_has_static_shape to allow dynamic shapes for ZeRO3 parameters
    try:
        from torch._dynamo import utils as dynamo_utils
        from torch._dynamo import guards as dynamo_guards
        original_tensor_always_has_static_shape = dynamo_utils.tensor_always_has_static_shape

        def tensor_always_has_static_shape_wrapper(tensor, is_tensor, tensor_source):
            # For ZeRO3 parameters (with ds_id), allow dynamic shapes
            if hasattr(tensor, 'ds_id'):
                return False, None
            return original_tensor_always_has_static_shape(tensor, is_tensor, tensor_source)

        # Patch in utils module
        dynamo_utils.tensor_always_has_static_shape = tensor_always_has_static_shape_wrapper
        # Also patch in guards module since it imports the function directly
        dynamo_guards.tensor_always_has_static_shape = tensor_always_has_static_shape_wrapper
    except (ImportError, AttributeError):
        pass

    # Patch TENSOR_MATCH to skip ZeRO3 parameters (similar to FSDP skip mechanism)
    try:
        from torch._dynamo.guards import GuardBuilder
        original_TENSOR_MATCH = GuardBuilder.TENSOR_MATCH

        def TENSOR_MATCH_wrapper(self, guard, value=None):
            # When DeepCompile ZeRO3 is active, skip tensor match for parameters
            # because their shapes are dynamic (sharded)
            if _deepcompile_state['z3_active']:
                # Get the actual value if not provided
                actual_value = value
                if actual_value is None:
                    try:
                        actual_value = self.get(guard.name)
                    except Exception:
                        pass

                # Skip tensor match guard for parameters when ZeRO3 is active
                if actual_value is not None and isinstance(actual_value, torch.nn.Parameter):
                    # Use ID_MATCH instead for ZeRO3 parameters
                    self.ID_MATCH(guard)
                    return

            return original_TENSOR_MATCH(self, guard, value)

        GuardBuilder.TENSOR_MATCH = TENSOR_MATCH_wrapper
    except (ImportError, AttributeError):
        pass

    # Patch CheckFunctionManager to suppress guard check failures for ZeRO3 parameter shapes
    try:
        from torch._dynamo.guards import CheckFunctionManager
        original_init = CheckFunctionManager.__init__

        def patched_init(self, *args, **kwargs):
            if _deepcompile_state['z3_active']:
                # Temporarily set output_graph.skip_guards_check if available
                if len(args) >= 1:
                    output_graph = args[0]
                    if hasattr(output_graph, 'skip_guards_check'):
                        old_value = output_graph.skip_guards_check
                        output_graph.skip_guards_check = True
                        try:
                            return original_init(self, *args, **kwargs)
                        finally:
                            output_graph.skip_guards_check = old_value
            # Try to call original_init and suppress parameter shape guard failures
            try:
                return original_init(self, *args, **kwargs)
            except AssertionError as e:
                err_msg = str(e)
                # Only suppress guard check failures related to parameter shapes in ZeRO3 mode
                if _deepcompile_state['z3_active'] and 'size mismatch' in err_msg and '_parameters' in err_msg:
                    # Skip the assertion and continue - the guard will just not be used
                    pass
                else:
                    raise

        CheckFunctionManager.__init__ = patched_init
    except (ImportError, AttributeError):
        pass


def _patch_wrap_to_fake_tensor_and_record():
    """
    Patch wrap_to_fake_tensor_and_record to handle ZeRO3 parameters.

    This patch replaces sharded ZeRO3 parameters with dummy full-sized tensors
    during tracing so that the traced graph has the correct shapes.

    The function signature changed in PyTorch 2.9:
    - PyTorch 2.9+: (e, tx, *, source, is_tensor, parent_context=None)
    - PyTorch 2.7-2.8: (t, *args, **kwargs)
    """
    original_wrap_to_fake_tensor_and_record = wrap_to_fake_tensor_and_record

    # Check the signature to handle different PyTorch versions
    import inspect
    sig = inspect.signature(original_wrap_to_fake_tensor_and_record)
    params = list(sig.parameters.keys())

    if params[0] == 'e' and len(params) >= 2 and params[1] == 'tx':
        # PyTorch 2.9+ signature: (e, tx, *, source, is_tensor, parent_context=None)
        def wrap_to_fake_tensor_and_record_wrapper(e, tx, **kwargs):
            dummy_tensor = wrap_if_ds_param(e)
            ret = original_wrap_to_fake_tensor_and_record(dummy_tensor, tx, **kwargs)
            if tracing_context := torch._guards.TracingContext.try_get():
                if e in tracing_context.tensor_to_context:
                    pass  # Already exists, no need to swap
                elif dummy_tensor in tracing_context.tensor_to_context:
                    tracing_context.tensor_to_context[e] = tracing_context.tensor_to_context.pop(dummy_tensor)
            return ret
    else:
        # PyTorch 2.7-2.8 signature: (t, *args, **kwargs)
        def wrap_to_fake_tensor_and_record_wrapper(t, *args, **kwargs):
            dummy_tensor = wrap_if_ds_param(t)
            ret = original_wrap_to_fake_tensor_and_record(dummy_tensor, *args, **kwargs)
            if tracing_context := torch._guards.TracingContext.try_get():
                tracing_context.tensor_to_context[t] = tracing_context.tensor_to_context.pop(dummy_tensor)
            return ret

    torch._dynamo.variables.builder.wrap_to_fake_tensor_and_record = wrap_to_fake_tensor_and_record_wrapper


def _patch_fake_tensor_mode():
    """
    Patch FakeTensorMode.from_tensor to handle ZeRO3 parameters.

    This is used by aot_module_simplified to process inputs.
    """
    original_from_tensor = FakeTensorMode.from_tensor

    def from_tensor_wrapper(self, t, *args, **kwargs):
        with unset_fake_temporarily():
            return original_from_tensor(self, wrap_if_ds_param(t), *args, **kwargs)

    FakeTensorMode.from_tensor = from_tensor_wrapper


def patch_fake_tensor():
    """
    Apply all necessary patches for DeepCompile ZeRO3 support.

    This function applies version-specific patches to handle ZeRO3's
    sharded parameters during torch.compile tracing.
    """
    # PyTorch 2.9+ requires additional patches for stricter guard validation
    if required_torch_version(min_version=2.9):
        _patch_for_pytorch_2_9_plus()

    # These patches are needed for all supported PyTorch versions
    _patch_wrap_to_fake_tensor_and_record()
    _patch_fake_tensor_mode()
