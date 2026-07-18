# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from unit.common import DistributedTest
from unit.v1.zero.test_zero_user_backward import get_config_dict, initialize_distributed


class _StatusProbe:
    """A Python-only holder used by an autograd boundary probe."""

    def __init__(self, parameter, observations):
        self.parameter = parameter
        self.observations = observations


class _RecordStatusAfterConsumer(torch.autograd.Function):
    """Record ZeRO state after the following module has consumed its input in backward.

    A probe is inserted immediately before a parameterized module. Autograd reaches this
    backward only after that module has finished its own backward and after DeepSpeed's
    post-backward wrapper on the module input has run. The observation therefore separates
    release at the real activation consumer from eventual cleanup in an outer module or the
    engine epilogue.
    """

    @staticmethod
    def forward(ctx, value, probe):
        ctx.probe = probe
        return value

    @staticmethod
    def backward(ctx, grad_output):
        parameter = ctx.probe.parameter
        ctx.probe.observations.append({
            "status": parameter.ds_status,
            "active_sub_modules": set(parameter.ds_active_sub_modules),
        })
        return grad_output, None


class _RaiseOnceInBackward(torch.autograd.Function):
    """Abort one backward after checkpoint recomputation has built real ZeRO state."""

    @staticmethod
    def forward(ctx, value, control):
        ctx.control = control
        return value

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.control["raise"]:
            ctx.control["raise"] = False
            raise RuntimeError("injected incomplete checkpoint backward")
        return grad_output, None


def _zero3_config(*, gradient_accumulation_steps=1, fp16=False):
    config = get_config_dict(3, gradient_accumulation_steps=gradient_accumulation_steps, force_fp32=True)
    # A zero reuse window and no prefetch make event-time residency assertions about the
    # current consumer, rather than about a parameter intentionally retained for a future use.
    config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0
    config["zero_optimization"]["stage3_max_reuse_distance"] = 0
    if fp16:
        config["fp16"] = {"enabled": True, "initial_scale_power": 8}
    return config


def _initialize_zero3(model, *, gradient_accumulation_steps=1, fp16=False):
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    engine, _, _, _ = deepspeed.initialize(config=_zero3_config(
        gradient_accumulation_steps=gradient_accumulation_steps, fp16=fp16),
                                           model=model,
                                           model_parameters=trainable_parameters)
    return engine


def _synchronize():
    get_accelerator().synchronize()
    dist.barrier()


def _assert_checkpoint_state_clean(engine, *, require_partitioned=True):
    """Check supported ZeRO state after a complete backward or reset boundary."""
    for module_name, module in engine.module.named_modules():
        recompute_parameters = getattr(module, "ds_recompute_parameters", set())
        assert not recompute_parameters, (
            f"module {module_name or '<root>'} kept recompute parameters after the lifecycle boundary: "
            f"{[parameter.ds_id for parameter in recompute_parameters]}")

    for parameter_name, parameter in engine.module.named_parameters():
        assert not parameter.ds_active_sub_modules, (
            f"parameter {parameter_name} kept active ZeRO consumers after the lifecycle boundary: "
            f"{sorted(parameter.ds_active_sub_modules)}")
        if (require_partitioned and not parameter.ds_persist and not parameter.is_external_param):
            assert parameter.ds_status == ZeroParamStatus.NOT_AVAILABLE, (
                f"parameter {parameter_name} remained gathered after the lifecycle boundary: "
                f"status={parameter.ds_status}")


class _RecursiveFrozenBlock(torch.nn.Module):
    """Recursively invoke one module instance so its ZeRO ds_id overlaps with itself."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.empty(hidden_dim), requires_grad=False)
        torch.nn.init.orthogonal_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, value, remaining_depth):
        value = torch.tanh(F.linear(value, self.weight, self.bias))
        if remaining_depth:
            # This is intentionally self(...) rather than a second module object. DeepSpeed's
            # forward/backward hooks therefore see independent nested invocations with one ds_id.
            value = self(value, remaining_depth - 1)
        return value


class _RecursiveCheckpointModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.shared = _RecursiveFrozenBlock(hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, value):
        value = checkpoint(self.shared, value, 2, use_reentrant=False)
        return self.head(value)


class _ReleaseTimingModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.frozen = torch.nn.Linear(hidden_dim, hidden_dim)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)
        self.trainable = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.observations = []

    def _checkpointed(self, value):
        probe = _StatusProbe(self.frozen.weight, self.observations)
        # The probe's backward executes immediately after self.frozen's real backward consumer.
        value = _RecordStatusAfterConsumer.apply(value, probe)
        value = torch.tanh(self.frozen(value))
        return torch.tanh(self.trainable(value))

    def forward(self, value):
        return self.head(checkpoint(self._checkpointed, value, use_reentrant=False))


class _NoGradInputModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.frozen = torch.nn.Linear(hidden_dim, hidden_dim)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)
        self.adapter = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def _checkpointed(self, value):
        return self.adapter(torch.tanh(self.frozen(value)))

    def forward(self, value):
        value = checkpoint(self._checkpointed, value, use_reentrant=False)
        return self.head(torch.tanh(value))


class _EarlyStopCheckpointModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.frozen = torch.nn.Linear(hidden_dim, hidden_dim)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)
        self.trainable = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.recompute_started = 0
        self.recompute_reached_tail = 0

    def _checkpointed(self, value):
        in_recompute = torch._C._current_graph_task_id() != -1
        if in_recompute:
            self.recompute_started += 1
        value = torch.sin(self.frozen(value))
        value = self.trainable(value)
        # Non-reentrant early-stop should terminate replay after the saved tensors needed by
        # backward have been rebuilt; this discarded tail is deliberately after that point.
        torch.cos(value.detach())
        if in_recompute:
            self.recompute_reached_tail += 1
        return value

    def forward(self, value):
        with set_checkpoint_early_stop(True):
            value = checkpoint(self._checkpointed, value, use_reentrant=False)
        return self.head(value)


class _RecomputeExceptionModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.frozen = torch.nn.Linear(hidden_dim, hidden_dim)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)
        self.trainable = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.raise_during_recompute = True

    def _checkpointed(self, value):
        value = self.frozen(value)
        if self.raise_during_recompute and torch._C._current_graph_task_id() != -1:
            raise RuntimeError("injected checkpoint recompute forward")
        return self.trainable(torch.tanh(value))

    def forward(self, value):
        return self.head(checkpoint(self._checkpointed, value, use_reentrant=False))


class _IncompleteBackwardModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.frozen = torch.nn.Linear(hidden_dim, hidden_dim)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)
        self.trainable = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.backward_control = {"raise": True}

    def _checkpointed(self, value):
        value = self.trainable(torch.tanh(self.frozen(value)))
        return _RaiseOnceInBackward.apply(value, self.backward_control)

    def forward(self, value):
        return self.head(checkpoint(self._checkpointed, value, use_reentrant=False))


class _FrozenNoConsumerModel(torch.nn.Module):

    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.embedding.weight.requires_grad_(False)
        self.grad_token = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.projection = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.observations = []

    def _checkpointed(self, indices):
        value = self.embedding(indices) + self.grad_token
        probe = _StatusProbe(self.embedding.weight, self.observations)
        # The token makes this boundary differentiable without giving the frozen embedding a
        # backward consumer. The probe runs after the projection has consumed the activation.
        value = _RecordStatusAfterConsumer.apply(value, probe)
        return self.projection(value)

    def forward(self, indices):
        value = checkpoint(self._checkpointed, indices, use_reentrant=False)
        return self.head(torch.tanh(value)).sum()


class _ExternalBiasLinear(torch.nn.Linear):

    def forward(self, value):
        output = F.linear(value, self.weight, self.bias)
        return output, self.bias


class _ExternalCheckpointModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.producer = _ExternalBiasLinear(hidden_dim, hidden_dim)
        self.producer.bias.requires_grad_(False)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.external_consumer_statuses = []

    def _checkpointed(self, value):
        value, external_bias = self.producer(value)
        parameter = external_bias if hasattr(external_bias, "ds_status") else external_bias.ds_param_alias
        self.external_consumer_statuses.append(parameter.ds_status)
        # The parameter is returned from its defining child and consumed in the enclosing
        # checkpointed function, which is the supported ZeRO external-parameter topology.
        return torch.tanh(value + external_bias)

    def forward(self, value):
        return self.head(checkpoint(self._checkpointed, value, use_reentrant=False)).sum()


class TestZero3ActivationCheckpointLifecycle(DistributedTest):
    """Independent black-box lifecycle checks for the current PR #8148 implementation."""

    world_size = 1

    def test_reused_checkpointed_module_invocations_release_independently(self):
        """Nested calls to one module must retain and retire independent hook invocations.

        The checkpoint replay happens inside the outer autograd GraphTask. Each recursive call
        to the same module produces a distinct pre/post-backward pair even though every pair
        has the same ZeRO module id. Failure means module-id-keyed bookkeeping collapsed two
        live invocations, typically as an assertion during release or residual active state.
        """
        device, _, _ = initialize_distributed()
        engine = _initialize_zero3(_RecursiveCheckpointModel(hidden_dim=8))
        value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)

        engine.backward(engine(value).sum())
        _synchronize()

        assert value.grad is not None and torch.isfinite(value.grad).all()
        _assert_checkpoint_state_clean(engine)
        engine.destroy()

    def test_recomputed_parameter_releases_at_last_activation_consumer(self):
        """A gathered frozen parameter must release at its actual backward consumer.

        Eventual partitioning after engine.backward() is too weak: an outer module or the
        ZeRO epilogue can hide excess residency. The autograd probe runs after the frozen
        Linear's backward and its DeepSpeed input wrapper but before outer cleanup. AVAILABLE
        here means the invocation retained the parameter beyond its true last consumer.
        """
        device, _, _ = initialize_distributed()
        model = _ReleaseTimingModel(hidden_dim=8)
        engine = _initialize_zero3(model)
        value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)

        engine.backward(engine(value).sum())
        _synchronize()

        assert model.observations, "the last-consumer autograd boundary did not execute"
        for observation in model.observations:
            assert observation["status"] == ZeroParamStatus.NOT_AVAILABLE, (
                "frozen parameter stayed gathered after its last activation consumer: "
                f"status={observation['status']}, active={sorted(observation['active_sub_modules'])}")
        _assert_checkpoint_state_clean(engine)
        engine.destroy()

    def test_no_grad_checkpoint_input_direct_backward_releases(self):
        """Direct Tensor.backward must run the no-grad-input checkpoint cleanup path.

        With a no-grad checkpoint input, a module post-backward hook can be absent. This uses
        PyTorch's direct backward entrypoint, so success requires the engine output hook to run
        the same ZeRO epilogue cleanup as DeepSpeedEngine.backward after every microbatch.
        """
        device, _, _ = initialize_distributed()
        engine = _initialize_zero3(_NoGradInputModel(hidden_dim=8), gradient_accumulation_steps=2)

        for microbatch in range(2):
            value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=False)
            engine(value).sum().backward()
            _synchronize()
            _assert_checkpoint_state_clean(engine)
            engine.step()

        engine.destroy()

    def test_no_grad_checkpoint_input_scaled_backward_releases(self):
        """A scaled PyTorch backward must also clean no-grad checkpoint leftovers.

        This is deliberately engine.scale(loss).backward(), not engine.backward(loss). The
        manual-backward hook must validate the scaler, finish ZeRO reduction, and drain any
        checkpoint owner whose input supplied no autograd post-hook.
        """
        device, _, _ = initialize_distributed()
        engine = _initialize_zero3(_NoGradInputModel(hidden_dim=8), fp16=True)
        dtype = next(engine.module.parameters()).dtype
        value = torch.randn(2, 8, device=device, dtype=dtype, requires_grad=False)

        engine.scale(engine(value).sum()).backward()
        _synchronize()

        _assert_checkpoint_state_clean(engine)
        engine.step()
        engine.destroy()

    def test_nonreentrant_checkpoint_early_stop_unwinds_for_clean_retry(self):
        """Non-reentrant early-stop must leave hooks and ZeRO ownership reusable.

        PyTorch can terminate replay before the Python checkpoint function reaches its tail
        once all tensors needed by backward have been reconstructed. DeepSpeed must still
        balance every pre/post hook and release the replay parameters so a second iteration
        starts from clean state. Reaching the discarded replay tail means this topology did
        not exercise early-stop and must not be used as lifecycle evidence.
        """
        device, _, _ = initialize_distributed()
        model = _EarlyStopCheckpointModel(hidden_dim=8)
        engine = _initialize_zero3(model)

        for _ in range(2):
            value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
            engine.backward(engine(value).sum())
            _synchronize()
            _assert_checkpoint_state_clean(engine)
            engine.step()

        assert model.recompute_started == 2, "checkpoint replay did not execute once per backward"
        assert model.recompute_reached_tail == 0, "the test topology did not trigger non-reentrant early-stop"
        engine.destroy()

    def test_recompute_forward_exception_unwinds_for_clean_retry(self):
        """An exception during replay must unwind the GraphTask and permit a clean retry.

        The first backward gathers frozen parameters and then raises from the checkpoint
        recompute forward. A supported retry cannot inherit the failed GraphTask's hook stack,
        active consumers, loss-scaling state, or recompute owners. Failure on the second pass
        means exception cleanup was deferred to a success-only epilogue.
        """
        device, _, _ = initialize_distributed()
        model = _RecomputeExceptionModel(hidden_dim=8)
        engine = _initialize_zero3(model)
        value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)

        try:
            engine.backward(engine(value).sum())
        except RuntimeError as error:
            assert "injected checkpoint recompute forward" in str(error)
        else:
            raise AssertionError("the injected recompute exception did not execute")

        model.raise_during_recompute = False
        retry = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        engine(retry).sum().backward()
        _synchronize()

        assert retry.grad is not None and torch.isfinite(retry.grad).all()
        _assert_checkpoint_state_clean(engine)
        engine.destroy()

    def test_reset_step_drains_incomplete_backward_state(self):
        """The next root forward reset must drain state left by an aborted backward.

        The custom autograd node raises after checkpoint replay has created genuine ZeRO
        active/recompute bookkeeping but before normal post-backward and engine epilogue
        cleanup. A user pre-hook registered after DeepSpeed initialization observes the next
        root forward immediately after DeepSpeed's reset hook. Any residual owner at that
        point proves reset_step did not make retry state independent of the failed GraphTask.
        """
        device, _, _ = initialize_distributed()
        model = _IncompleteBackwardModel(hidden_dim=8)
        engine = _initialize_zero3(model)
        value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)

        try:
            engine.backward(engine(value).sum())
        except RuntimeError as error:
            assert "injected incomplete checkpoint backward" in str(error)
        else:
            raise AssertionError("the injected incomplete backward did not execute")

        observe_reset = {"enabled": True}

        def _observe_after_deepspeed_reset(module, unused_inputs):
            if observe_reset["enabled"]:
                _assert_checkpoint_state_clean(engine)
                observe_reset["enabled"] = False

        handle = engine.module.register_forward_pre_hook(_observe_after_deepspeed_reset)
        retry = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        engine(retry).sum().backward()
        _synchronize()
        handle.remove()

        assert not observe_reset["enabled"], "the post-reset observer did not execute"
        _assert_checkpoint_state_clean(engine)
        engine.destroy()

    def test_frozen_parameter_without_backward_consumer_releases_at_last_use(self):
        """A frozen parameter with no backward consumer must not be retained to epilogue.

        Integer indices and a frozen Embedding produce no parameter or input gradient for the
        Embedding, but non-reentrant replay still gathers its weight to reconstruct a later
        trainable activation. A separate trainable token makes an observable downstream
        autograd boundary. AVAILABLE at that boundary means recompute ownership selected a
        parameter more broadly than the real activation lifetime requires.
        """
        device, _, _ = initialize_distributed()
        model = _FrozenNoConsumerModel(vocab_size=16, hidden_dim=8)
        engine = _initialize_zero3(model)
        indices = torch.randint(0, 16, (2, 4), device=device)

        engine.backward(engine(indices))
        _synchronize()

        assert model.observations, "the frozen-parameter last-use boundary did not execute"
        for observation in model.observations:
            assert observation["status"] == ZeroParamStatus.NOT_AVAILABLE, (
                "frozen parameter with no backward consumer stayed gathered after its last use: "
                f"status={observation['status']}, active={sorted(observation['active_sub_modules'])}")
        _assert_checkpoint_state_clean(engine)
        engine.destroy()

    def test_checkpoint_external_parameter_lifecycle(self):
        """A checkpoint-replayed external parameter must remain valid for its parent consumer.

        The child returns its frozen bias as a ZeRO external parameter, and the enclosing
        checkpointed function consumes it in both the original forward and replay. Two full
        iterations verify the bias is AVAILABLE at every real consumer and that ownership is
        clean afterward. Passing confirms the current external-parameter exclusion works for
        this checkpoint topology; it is not evidence of an external-parameter defect.
        """
        device, _, _ = initialize_distributed()
        model = _ExternalCheckpointModel(hidden_dim=8)
        engine = _initialize_zero3(model)

        for _ in range(2):
            value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
            engine.backward(engine(value))
            _synchronize()
            assert value.grad is not None and torch.isfinite(value.grad).all()
            _assert_checkpoint_state_clean(engine, require_partitioned=False)
            engine.step()

        assert model.external_consumer_statuses
        assert all(status == ZeroParamStatus.AVAILABLE for status in model.external_consumer_statuses)
        assert model.producer.bias.is_external_param
        engine.destroy()
