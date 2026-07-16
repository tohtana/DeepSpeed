# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
import collections
from collections import UserDict
import threading
from typing import Deque, Optional, Set

from deepspeed import comm as dist
from deepspeed.utils import z3_leaf_module
from deepspeed.utils.logging import logger
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import *
from deepspeed.runtime.zero.partitioned_param_profiler import PartitionedParameterProfiler
from deepspeed.runtime.swap_tensor.partitioned_param_swapper import PartitionedParamStatus
from deepspeed.utils.debug import debug_param2name_id_shape
from deepspeed.accelerator import get_accelerator
import deepspeed.runtime.compiler as compiler
from deepspeed.runtime.compiler import is_compiling

import logging

ENABLE_PROFILER = False


def current_graph_task_id() -> int:
    """Return the active autograd graph task, or -1 outside backward."""
    return torch._C._current_graph_task_id()


def debug_rank0(message: str) -> None:
    if dist.get_rank() == 0:
        logger.debug(message)


@instrument_w_nvtx
def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())


@compiler.enable(min_version="2.7.0")
def iter_params(module: Module, recurse=False) -> Iterable[Parameter]:
    return map(lambda pair: pair[1], get_all_parameters(module, recurse))


class ZeRoTraceMode(Enum):
    # Record trace of the network during a single forward+backward (for training) or forward (for inference)
    RECORD = 1
    # Use recorded network trace to optimize current forward+backward or forward
    COMPLETE = 2
    # Recorded trace does not match current forward+backward or forward pass.
    INVALID = 3


class InflightParamRegistry(UserDict):
    """registry for parameters in flight"""

    def __setitem__(self, param: Parameter, handle: AllGatherCoalescedHandle) -> None:
        if param in self.data:
            raise RuntimeError(f"{param.ds_summary()} already in registry")
        if param.ds_status != ZeroParamStatus.INFLIGHT:
            raise RuntimeError(f"attempted to add non-inflight parameter to registry {param.ds_summary()}")
        self.data[param] = handle


class PartitionedParameterCoordinator:
    FORWARD_FETCH_SUBMIT = 'forward_fetch_submit'
    FORWARD_FETCH_WAIT = 'forward_fetch_wait'
    FORWARD_PREFETCH_SUBMIT = 'forward_prefetch_submit'
    BACKWARD_FETCH_SUBMIT = 'backward_fetch_submit'
    BACKWARD_FETCH_WAIT = 'backward_fetch_wait'
    BACKWARD_PREFETCH_SUBMIT = 'backward_prefetch_submit'
    FORWARD_ALL_GATHER = 'forward_all_gather'
    BACKWARD_ALL_GATHER = 'backward_all_gather'
    """Handles partitioning and gathering of parameters."""

    @dataclass
    class __ParamInTrace:
        param: Parameter
        step_id_last_used_at: int

    class __DeferredRelease:
        """One frozen checkpoint-recompute invocation and its exact forward boundary."""

        __slots__ = ("epoch_id", "params", "free_data", "active", "pending_boundaries", "submodule_id",
                     "protect_all_params")

        def __init__(self, epoch_id: int, protect_all_params: bool = False):
            self.epoch_id = epoch_id
            self.params = None
            self.free_data = True
            self.active = True
            self.pending_boundaries = None
            self.submodule_id = None
            self.protect_all_params = protect_all_params

    def __init__(
        self,
        prefetch_bucket_sz: int,
        max_reuse_distance_in_numel: int,
        max_available_parameters_in_numel: int,
        allgather_stream: get_accelerator().Stream,
        inflight_param_registry: InflightParamRegistry,
        prefetch_nvme: bool = False,
        timers=None,
        zero_quantized_weights=False,
        zero_quantized_nontrainable_weights=False,
        fast_sharding_for_leaf_module=False,
        log_trace_cache_warnings=False,
    ) -> None:
        # mapping of param -> handle for each param that is currently in flight
        self.__inflight_param_registry = inflight_param_registry
        # keeps track of the number of submodules invoked so far.
        self.__step_id: int = 0
        # network tracing mode
        self.__trace_mode: ZeRoTraceMode = ZeRoTraceMode.INVALID
        # sequence of submodules/parameters in forward pass + backward pass
        self.__submodule_order: Iterable[Module] = []
        self.__param_order: Iterable[__class__.__ParamInTrace] = []
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(lambda: int(-1e10))
        self.__step_id_module_fetched_for = collections.defaultdict(lambda: collections.deque())
        # number of available params, and max number of available params
        self.__n_available_params: int = 0
        self.__max_n_available_params: int = max_available_parameters_in_numel
        # max distance between two use of the module beyond which module is released
        self.__max_reuse_dist_in_numel: int = max_reuse_distance_in_numel
        # queue for parameters to fetch. parameters will be popped off the left
        # side of the dequeue as they are fetched
        self.__param_queue: Deque[__class__.__ParamInTrace] = None
        self.__prefetch_bucket_sz: int = prefetch_bucket_sz
        self.__prefetch_nvme: bool = prefetch_nvme
        self.hierarchy: int = 0
        self.zero_quantized_weights = zero_quantized_weights
        self.zero_quantized_nontrainable_weights = zero_quantized_nontrainable_weights

        # stream that will be used for allgather operations
        self.__allgather_stream: get_accelerator().Stream = allgather_stream

        # limit the number of fetch events that can be queued at once
        # otherwise, what happens is memory is allocated by the host thread at the
        # time of the call, but not used until later by the asynchronous cuda stream.
        # allowing an infinite number of these to queue up causes a lot of memory
        # pressure that then becomes detrimental to performance.
        # this is a much less elegant way of fixing this vs something like using
        # cudaMallocAsync/cudaFreeAsync. Choosing to not expose this to the user now
        # because ideally in the future its replaced by an async allocation
        # mechanism which doesn't require any configuration by the user.
        self.__ongoing_fetch_events: Deque[get_accelerator().Event] = collections.deque()
        # TODO. make this configurable via JSON
        self.__max_ongoing_fetch_events: int = 2
        self.__profiler = PartitionedParameterProfiler(timers if ENABLE_PROFILER else None)

        # Whether to log trace cache warnings, e.g. invalidation events
        self.__log_trace_cache_warnings = log_trace_cache_warnings

        # whether to enable fast fetch for the z3 leaf module.
        # this will improve fetch speed but will not break down leaf module parameters to alleviate memory pressure.
        self.fast_sharding_for_leaf_module = fast_sharding_for_leaf_module

        # Thread synchronization for leaf module fetches during backward pass.
        # When autograd executes hooks in multiple threads (e.g., for modules returning multiple tensors),
        # we need to ensure only one thread fetches parameters for a given leaf module at a time.
        # This is only needed during backward pass; forward pass is single-threaded.
        self.__ongoing_fetch_leaf_module_events = collections.defaultdict(threading.Event)
        self.__leaf_module_lock = threading.Lock()

        # Frozen checkpoint invocations receive unique owners only after an actual
        # release is deferred. The set and lock remain lazy on the ordinary path.
        self.__outer_backward_graph_task_id: Optional[int] = None
        self.__deferred_releases: Optional[Set[__class__.__DeferredRelease]] = None
        self.__deferred_release_lock = None

    """Tracing and Tracking
    TODO. consider performing trace before initializing PartitionedParameterCoordinator
    and passing trace results into constructor. This way all the code in here can
    just assume that the trace is complete and the results can be entirely
    immutable.

    Bookkeeping operations used to track where we are in the forward/backward pass
    """

    def _clear_trace_structures(self) -> None:
        self.__submodule_order = []
        self.__param_order = []
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(lambda: int(-1e10))
        # clear the fetch-step deque too; a stale entry here causes record_parameters() to
        # pop an empty deque (IndexError) after trace invalidation.
        self.__step_id_module_fetched_for = collections.defaultdict(lambda: collections.deque())
        self.__param_queue = None

    def is_complete_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.COMPLETE

    def is_invalid_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.INVALID

    def is_record_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.RECORD

    def _clean_inflight_param_registry(self) -> None:
        for param, handle in self.__inflight_param_registry.items():
            handle.wait()
            self.__release_param(param)
        self.__inflight_param_registry.clear()

    def _invalidate_trace(self) -> None:
        if self.is_invalid_trace():
            raise RuntimeError("attempted to invalidate already invalid trace")
        self.__trace_mode = ZeRoTraceMode.INVALID
        self._clear_trace_structures()
        self._clean_inflight_param_registry()

    def trace_prologue(self, sub_module: Module) -> None:
        if self.is_complete_trace():
            # sub_module must match expectation else invalidate trace cache
            if len(self.__submodule_order) <= self.__step_id:
                print_rank_0(
                    f"Invalidate trace cache @ step {self.__step_id} and module {sub_module.ds_id}: "
                    f"cache has only {len(self.__submodule_order)} modules",
                    force=self.__log_trace_cache_warnings)
                self._invalidate_trace()
                return

            if sub_module != self.__submodule_order[self.__step_id]:
                expected_module_id = self.__submodule_order[self.__step_id].ds_id
                print_rank_0(
                    f"Invalidate trace cache @ step {self.__step_id}: "
                    f"expected module {expected_module_id}, but got module {sub_module.ds_id}",
                    force=self.__log_trace_cache_warnings)
                self._invalidate_trace()

    @compiler.enable(min_version="2.7.0")
    def record_module(self, sub_module: Module) -> None:
        """adds sub module to trace"""
        if is_compiling():
            return

        if not self.is_record_trace():
            raise RuntimeError(f"attempted to record trace when status = {self.__trace_mode}")

        self.__submodule_order.append(sub_module)
        self.__step_id_module_fetched_for[sub_module.ds_id].append(self.__step_id)

    def record_parameters(self, sub_module: Module) -> None:
        if is_compiling():
            return
        """adds sub module to trace"""
        if not self.is_record_trace():
            raise RuntimeError(f"attempted to record trace when status = {self.__trace_mode}")

        step_id = self.__step_id_module_fetched_for[sub_module.ds_id].popleft()
        for param in sorted(set(iter_params(sub_module, recurse=z3_leaf_module(sub_module))), key=lambda p: p.ds_id):
            self.__param_order.append(__class__.__ParamInTrace(param=param, step_id_last_used_at=step_id))

    def construct_parameter_trace_from_module_trace(self):
        """use module trace to construct parameter trace"""
        self.__param_order = []
        for sub_module in self.__submodule_order:
            self.record_parameters(sub_module)

    @compiler.disable
    def reset_step(self) -> None:
        """indicate that we have completed one fwd+bwd for the model"""
        if is_compiling():
            return

        # A GraphTask callback may be skipped when backward terminates with an
        # exception. A new root forward is the existing reset safety boundary.
        if self.__outer_backward_graph_task_id is not None:
            self.release_outer_backward(self.__outer_backward_graph_task_id)
        self._clean_inflight_param_registry()

        if not self.is_complete_trace():  # not self.trace_complete:
            # Make sure that recorded submodule orders are identical across ranks
            assert_ints_same_as_other_ranks([m.ds_id for m in self.__submodule_order])

            if self.is_record_trace():
                # Successfully recorded a trace
                self.construct_parameter_trace_from_module_trace()
                # Make sure that recorded parameter orders are identical across ranks
                assert_ints_same_as_other_ranks([p.param.ds_id for p in self.__param_order])
                assert_ints_same_as_other_ranks([p.step_id_last_used_at for p in self.__param_order])

                self.__submodule_order = tuple(self.__submodule_order)  # freeze
                self.__param_order = tuple(self.__param_order)  # freeze
                self.__trace_mode = ZeRoTraceMode.COMPLETE
                print_rank_0(
                    f"completed record trace of {len(self.__submodule_order)} sub modules: {[m.ds_id for m in self.__submodule_order]}",
                    force=False)
            else:
                # Enable trace recording for next forward/backward pass
                self.__trace_mode = ZeRoTraceMode.RECORD

        else:
            if self.__profiler is not None:
                self.__profiler.log_events()

        self.__param_queue = collections.deque(self.__param_order)  # reset fetch queue
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(lambda: int(-1e10))
        self.__step_id_module_fetched_for = collections.defaultdict(lambda: collections.deque())
        self.__step_id = 0
        self.__n_available_params = 0
        self.__profiler.reset_events()
        # Clear leaf module fetch events for clean state
        self.__ongoing_fetch_leaf_module_events.clear()

    def _dump_params(self, tag, sub_module, params, step_id=None):
        if step_id is None:
            step_id = self.__step_id
        param_names = [debug_param2name_id_shape(p) for p in params]
        print_rank_0(f'{tag} step = {step_id} p_names = {param_names}', force=False)

    def _dump_param_ids(self, tag, mod_id, p_ids, step_id=None):
        if step_id is None:
            step_id = self.__step_id
        print_rank_0(f'{tag} mod = {mod_id}, step = {step_id}, p_ids = {p_ids}', force=False)

    """Fetch and Release
    Fetching, prefetching, and releasing parameters
    """

    @compiler.disable
    @instrument_w_nvtx
    @torch.no_grad()
    def fetch_sub_module(self, current_submodule: Module, forward: bool) -> None:
        """This method does the following (in order):
        1. kick off fetch for parameters in immediately required sub module
        2. kick off fetch for next few parameters we will need later (prefetch)
        3. block on parameters in immediately required sub module
        """
        # For leaf modules during backward pass, autograd may trigger hooks from multiple
        # threads concurrently (e.g., when a module returns multiple tensors). We need to
        # serialize access to prevent race conditions in parameter state management.
        # Forward pass is single-threaded, so no synchronization is needed there.
        is_leaf = z3_leaf_module(current_submodule)
        needs_sync = is_leaf and not forward
        if needs_sync:
            event_to_wait = None
            with self.__leaf_module_lock:
                event = self.__ongoing_fetch_leaf_module_events.get(current_submodule.ds_id)
                if event is not None:
                    # Another thread is already fetching this leaf module, wait for it
                    event_to_wait = event
                else:
                    # Mark that we're starting a fetch for this leaf module
                    new_event = threading.Event()
                    self.__ongoing_fetch_leaf_module_events[current_submodule.ds_id] = new_event

            if event_to_wait is not None:
                # Wait outside the lock to avoid deadlock
                event_to_wait.wait()
                return

        try:
            self._fetch_sub_module_impl(current_submodule, forward, is_leaf)
        finally:
            if needs_sync:
                # Signal that we're done fetching this leaf module and remove the event
                with self.__leaf_module_lock:
                    event = self.__ongoing_fetch_leaf_module_events.pop(current_submodule.ds_id, None)
                    if event is not None:
                        event.set()

    def _fetch_sub_module_impl(self, current_submodule: Module, forward: bool, is_leaf: bool) -> None:
        """Implementation of fetch_sub_module, separated for thread synchronization."""
        if logger.isEnabledFor(logging.DEBUG):
            debug_rank0(
                f"{self.__step_id}: M{current_submodule.ds_id}({type(current_submodule).__name__}) P{[p.ds_id for p in iter_params(current_submodule, recurse=is_leaf)]} "
                + str({
                    "avail": f"{self.__n_available_params:.1e}",
                    "queue_sz": f"{len(self.__param_queue or [])}",
                    "inflight": [p.ds_id for p in self.__inflight_param_registry],
                }))

        params_to_fetch = set(iter_params(current_submodule, recurse=is_leaf))
        fetch_numel = sum(
            [p.partition_numel() for p in params_to_fetch if p.ds_status == ZeroParamStatus.NOT_AVAILABLE])

        if fetch_numel > 0:
            event_name = __class__.FORWARD_FETCH_SUBMIT if forward else __class__.BACKWARD_FETCH_SUBMIT
            self._dump_param_ids(event_name, current_submodule.ds_id,
                                 [(p.ds_id, p.ds_shape)
                                  for p in params_to_fetch if p.ds_status == ZeroParamStatus.NOT_AVAILABLE])
            # self._dump_params(event_name, current_submodule, [p for p in params_to_fetch if p.ds_status == ZeroParamStatus.NOT_AVAILABLE])

            self.__profiler.start_event(event_name)
            # kick off all gather for params in the immediately required submodule
            #for param in params_to_fetch:
            if logger.isEnabledFor(logging.DEBUG):
                for param in params_to_fetch:
                    debug_rank0(f"-fetch: {param.ds_summary()}")
            self.__all_gather_params(params_to_fetch, forward)
            self.__profiler.stop_event(event_name, fetch_numel)

        wait_numel = 0
        wait_event_name = __class__.FORWARD_FETCH_WAIT if forward else __class__.BACKWARD_FETCH_WAIT
        self.__profiler.start_event(wait_event_name)
        fast_fetch = self.fast_sharding_for_leaf_module and is_leaf
        # wait for parameters in the immediately needed submodule to become available
        for param in params_to_fetch:
            param.ds_active_sub_modules.add(current_submodule.ds_id)
            if logger.isEnabledFor(logging.DEBUG):
                debug_rank0(f"-wait: {param.ds_summary()}")
            if param in self.__inflight_param_registry:
                wait_numel += param.partition_numel()
                with get_accelerator().stream(self.__allgather_stream):
                    while self.__ongoing_fetch_events and self.__ongoing_fetch_events[0].query():
                        self.__ongoing_fetch_events.popleft()
                    if len(self.__ongoing_fetch_events) > self.__max_ongoing_fetch_events:
                        self.__ongoing_fetch_events.popleft().synchronize()

                    self.__inflight_param_registry.pop(param).wait(handle_dependency=not fast_fetch)

                    if not get_accelerator().handles_memory_backpressure() and not fast_fetch:
                        event = get_accelerator().Event()
                        event.record()
                        self.__ongoing_fetch_events.append(event)

            assert param.ds_status == ZeroParamStatus.AVAILABLE, param.ds_summary()
        if not get_accelerator().resolves_data_dependency():
            get_accelerator().current_stream().wait_stream(self.__allgather_stream)
        if fast_fetch:
            AllGatherCoalescedHandle.free_buffer()
        self.__profiler.stop_event(wait_event_name, wait_numel)

        # kick off parameter prefetches for upcoming modules
        # don't prefetch if we dont have a completed model trace
        if self.is_complete_trace():
            # go through the parameters we need for the current module and pop them
            # off the fetch queue so that they aren't prefetched later.
            # if params have already been popped off the fetch queue by earlier
            # prefetches we won't look for them here
            discarded_from_prefetch_queue = set()
            params_not_already_fetched = set(
                filter(lambda p: self.__most_recent_step_id_param_fetched_for[p] < self.__step_id, params_to_fetch))
            while self.__param_queue and len(discarded_from_prefetch_queue) < len(params_not_already_fetched):
                param_in_trace = self.__param_queue.popleft()
                self.__most_recent_step_id_param_fetched_for[
                    param_in_trace.param] = param_in_trace.step_id_last_used_at
                discarded_from_prefetch_queue.add(param_in_trace.param)

            if discarded_from_prefetch_queue != params_not_already_fetched:
                raise RuntimeError(
                    f"tracing error at step {self.__step_id}: \n"
                    f"module id: {current_submodule.ds_id}, training: {current_submodule.training}\n"
                    f"expected the next {len(params_not_already_fetched)} parameters in the "
                    f"parameter fetch queue to be {tuple(p.ds_summary(use_debug_name=True) for p in params_not_already_fetched)} \n"
                    f"but got \n {tuple(p.ds_summary(use_debug_name=True) for p in discarded_from_prefetch_queue)}.")

            def _is_currently_on_nvme(param):
                if param.nvme_swapper is None:
                    return False

                return param.ds_tensor.final_location == OffloadDeviceEnum.nvme \
                    and param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE

            # kick off all gather for params in the next few submodules (prefetch)
            if self.__prefetch_bucket_sz > 0:
                max_params_to_prefetch = min(self.__max_n_available_params - self.__n_available_params,
                                             self.__prefetch_bucket_sz)
                params_to_prefetch = set()
                numel_prefetching = 0
                while self.__param_queue and numel_prefetching < max_params_to_prefetch:
                    param_in_trace: __class__.__ParamInTrace = self.__param_queue.popleft()

                    if _is_currently_on_nvme(param_in_trace.param):
                        # nvme prefetch is handled elsewhere. Need to break here to preserve fetch order
                        self.__param_queue.appendleft(param_in_trace)
                        break

                    do_prefetch = param_in_trace.param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                    if param_in_trace.param in params_to_prefetch:
                        # Avoid duplicates
                        do_prefetch = False

                    self.__most_recent_step_id_param_fetched_for[param_in_trace.param] = \
                        max(self.__most_recent_step_id_param_fetched_for[param_in_trace.param],
                            param_in_trace.step_id_last_used_at)

                    if do_prefetch:
                        params_to_prefetch.add(param_in_trace.param)
                        numel_prefetching += param_in_trace.param.ds_numel

                if numel_prefetching > 0:
                    event_name = __class__.FORWARD_PREFETCH_SUBMIT if forward else __class__.BACKWARD_PREFETCH_SUBMIT
                    self.__profiler.start_event(event_name)
                    if logger.isEnabledFor(logging.DEBUG):
                        for param in params_to_prefetch:
                            debug_rank0(f"-prefetch: {param.ds_summary()}")
                    self.__all_gather_params(params_to_prefetch, forward)
                    self.__profiler.stop_event(event_name, numel_prefetching)

                if self.__prefetch_nvme:
                    self.__prefetch_nvme_param_partitions()

        self.__step_id += 1

    @instrument_w_nvtx
    @torch.no_grad()
    def release_sub_module(self, submodule: Module, forward=False, deferred_release=None) -> None:
        """release the parameters of a sub module, assuming they meet conditions to
        be released."""
        #print_rank_0(f"release_sub_module {'fwd' if forward else 'bwd'}: {debug_module2name_id(submodule)}", force=False)
        if deferred_release is None:
            params_to_release = (self.__params_to_release(submodule, self.__step_id) if self.is_complete_trace() else
                                 set(p.ds_id for p in iter_params(submodule, recurse=z3_leaf_module(submodule))))

            free_data = not z3_leaf_module(submodule) or not self.fast_sharding_for_leaf_module
            if not free_data:
                # wait for the computation to finish and launch as early as possible.
                empty_buffer = torch.empty(1, device=torch.device(get_accelerator().current_device_name()))

            for param in iter_params(submodule, recurse=z3_leaf_module(submodule)):
                param.ds_active_sub_modules.discard(submodule.ds_id)
                if param.ds_id in params_to_release and not param.is_external_param:
                    self.__release_param(param, free_data)
                if not free_data:
                    if param.ds_id in params_to_release and not param.is_external_param:
                        # empty buffer ensures that all computations are complete
                        param.data = empty_buffer
            return

        params = tuple(iter_params(submodule, recurse=z3_leaf_module(submodule)))
        params_to_release = (self.__params_to_release(submodule, self.__step_id) if self.is_complete_trace() else set(
            p.ds_id for p in params))

        free_data = not z3_leaf_module(submodule) or not self.fast_sharding_for_leaf_module
        if not free_data:
            # wait for the computation to finish and launch as early as possible.
            empty_buffer = torch.empty(1, device=torch.device(get_accelerator().current_device_name()))

        # The direct multi-grad hook on this recompute invocation's inputs is its
        # local last-consumer boundary. Only the exact frozen invocation receives
        # a unique owner; empty-input and early-stop records remain for root fallback.
        deferred_params = {
            param
            for param in params if (deferred_release.protect_all_params or not param.requires_grad)
            and param.ds_id in params_to_release and not param.is_external_param
        }
        self.__attach_deferred_release(deferred_release, deferred_params, free_data, submodule.ds_id)

        for param in params:
            param.ds_active_sub_modules.discard(submodule.ds_id)

        for param in params:
            if param in deferred_params:
                continue
            if param.ds_id in params_to_release and not param.is_external_param:
                self.__release_param(param, free_data)
            if not free_data:
                if param.ds_id in params_to_release and not param.is_external_param and param not in deferred_params:
                    # empty buffer ensures that all computations are complete
                    param.data = empty_buffer

    @instrument_w_nvtx
    @torch.no_grad()
    def release_and_reset_all(self, module: Module) -> None:
        """release all module parameters"""
        if self.__outer_backward_graph_task_id is not None:
            self.release_outer_backward(self.__outer_backward_graph_task_id)

        for param in iter_params(module, recurse=True):
            if param in self.__inflight_param_registry:
                self.__inflight_param_registry.pop(param).wait()

            # TODO. make this throw if if there are still active submodules. currently
            # there's a hook execution issue
            param.ds_active_sub_modules.clear()
            self.__release_param(param)

        for param in iter_params(module, recurse=True):
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(f"{param.ds_summary()} expected to be released")

    def begin_outer_backward(self, graph_task_id: int) -> None:
        """Arm the outer GraphTask before execution descends into child modules."""
        if graph_task_id == -1:
            raise RuntimeError("ZeRO-3 cannot arm deferred release outside backward")

        # Same-engine concurrent backward is outside the existing ZeRO hook-state
        # contract. Root arming itself takes no frozen slow-path lock.
        active_graph_task_id = self.__outer_backward_graph_task_id
        if active_graph_task_id is None:
            self.__outer_backward_graph_task_id = graph_task_id
        elif active_graph_task_id != graph_task_id:
            raise RuntimeError("ZeRO-3 does not support overlapping backward calls on one engine")

    def has_active_outer_backward(self) -> bool:
        return self.__outer_backward_graph_task_id is not None

    def __get_deferred_release_lock(self):
        lock = self.__deferred_release_lock
        if lock is None:
            lock = threading.Lock()
            self.__deferred_release_lock = lock
        return lock

    def begin_deferred_release(self, protect_all_params: bool = False):
        """Create one frozen invocation record under the armed outer backward."""
        epoch_id = self.__outer_backward_graph_task_id
        if epoch_id is None:
            raise RuntimeError("ZeRO-3 checkpoint replay ran without a root backward epoch")

        deferred_release = __class__.__DeferredRelease(epoch_id, protect_all_params)
        with self.__get_deferred_release_lock():
            if self.__outer_backward_graph_task_id != epoch_id:
                raise RuntimeError("ZeRO-3 backward epoch changed during checkpoint replay")
            if self.__deferred_releases is None:
                self.__deferred_releases = set()
            self.__deferred_releases.add(deferred_release)
        return deferred_release

    def cancel_deferred_release(self, deferred_release) -> None:
        """Unpublish an invocation whose forward-boundary binding failed."""
        with self.__get_deferred_release_lock():
            if not deferred_release.active:
                return
            deferred_release.active = False
            if self.__deferred_releases is not None:
                self.__deferred_releases.discard(deferred_release)
                if not self.__deferred_releases:
                    self.__deferred_releases = None

    @instrument_w_nvtx
    @torch.no_grad()
    def defer_missing_post_backward(self, submodule: Module) -> None:
        """Transfer one executed no-grad invocation to the outer root fallback."""
        deferred_release = self.begin_deferred_release()
        params = tuple(iter_params(submodule, recurse=z3_leaf_module(submodule)))
        params_to_protect = {param for param in params if not param.ds_persist and not param.is_external_param}
        free_data = not z3_leaf_module(submodule) or not self.fast_sharding_for_leaf_module
        try:
            self.__attach_deferred_release(deferred_release, params_to_protect, free_data, submodule.ds_id)
        except Exception:
            self.cancel_deferred_release(deferred_release)
            raise

        # The unique record now replaces this invocation's module owner. This
        # mirrors the missing post-backward Function without changing fetch.
        for param in params:
            param.ds_active_sub_modules.discard(submodule.ds_id)

    def set_deferred_release_boundary_count(self, deferred_release, count: int) -> None:
        """Set the exact executable activation boundaries protecting one invocation."""
        if count < 1:
            raise RuntimeError("ZeRO-3 deferred release requires an activation boundary")
        with self.__get_deferred_release_lock():
            if not deferred_release.active or deferred_release.epoch_id != self.__outer_backward_graph_task_id:
                raise RuntimeError("ZeRO-3 deferred release does not belong to the active backward")
            if deferred_release.pending_boundaries is not None:
                raise RuntimeError("ZeRO-3 deferred release boundaries were set more than once")
            deferred_release.pending_boundaries = count

    def __attach_deferred_release(self, deferred_release, params: Set[Parameter], free_data: bool,
                                  submodule_id: int) -> None:
        """Promote only the otherwise-releasable frozen subset to unique ownership."""
        with self.__get_deferred_release_lock():
            if (not deferred_release.active or deferred_release.epoch_id != self.__outer_backward_graph_task_id):
                raise RuntimeError("ZeRO-3 deferred release does not belong to the active backward")
            if deferred_release.params is not None:
                raise RuntimeError("ZeRO-3 deferred release was attached more than once")

            if not params:
                deferred_release.active = False
                if self.__deferred_releases is not None:
                    self.__deferred_releases.discard(deferred_release)
                    if not self.__deferred_releases:
                        self.__deferred_releases = None
            else:
                deferred_release.params = set(params)
                deferred_release.free_data = free_data
                deferred_release.submodule_id = submodule_id
                for param in deferred_release.params:
                    param.ds_active_sub_modules.add(deferred_release)

    def finish_deferred_release(self, deferred_release) -> None:
        """Retire one invocation from its direct multi-grad input callback."""
        params_to_release = None
        free_data = True
        with self.__get_deferred_release_lock():
            if not deferred_release.active:
                return
            if deferred_release.pending_boundaries is None:
                raise RuntimeError("ZeRO-3 deferred release completed before boundary binding")
            deferred_release.pending_boundaries -= 1
            if deferred_release.pending_boundaries > 0:
                return
            if deferred_release.params is None:
                raise RuntimeError("ZeRO-3 deferred release completed before forward unwind")

            deferred_release.active = False
            params_to_release = deferred_release.params
            free_data = deferred_release.free_data
            if self.__deferred_releases is not None:
                self.__deferred_releases.discard(deferred_release)
                if not self.__deferred_releases:
                    self.__deferred_releases = None
            for param in params_to_release:
                param.ds_active_sub_modules.discard(deferred_release)

        for param in params_to_release:
            self.__release_param(param, free_data)

    @compiler.disable
    @torch.no_grad()
    def release_outer_backward(self, graph_task_id: int) -> None:
        """Release only invocation records missed by the local input boundary."""
        active_graph_task_id = self.__outer_backward_graph_task_id
        if active_graph_task_id is None:
            return
        if active_graph_task_id != graph_task_id:
            raise RuntimeError("ZeRO-3 outer backward callback does not match the armed epoch")

        # A frozen backward with no residual record reaches neither the lock nor
        # parameter/accounting work. Fully-trainable models never arm this state.
        if self.__deferred_releases is None:
            self.__outer_backward_graph_task_id = None
            return

        params_to_release = set()
        free_data_by_param = {}
        with self.__get_deferred_release_lock():
            active_graph_task_id = self.__outer_backward_graph_task_id
            if active_graph_task_id is None:
                return
            if active_graph_task_id != graph_task_id:
                raise RuntimeError("ZeRO-3 outer backward callback does not match the armed epoch")

            deferred_releases = self.__deferred_releases or set()
            self.__outer_backward_graph_task_id = None
            self.__deferred_releases = None
            for deferred_release in deferred_releases:
                if not deferred_release.active:
                    continue
                deferred_release.active = False
                if deferred_release.params is None:
                    continue
                for param in deferred_release.params:
                    param.ds_active_sub_modules.discard(deferred_release)
                    # A no-grad module input has no post-backward Function to
                    # retire the owner acquired by its backward fetch. At
                    # outer GraphTask completion every invocation in this
                    # backward is finished, so complete only this record's
                    # originating module release. Other module owners remain.
                    param.ds_active_sub_modules.discard(deferred_release.submodule_id)
                    params_to_release.add(param)
                    free_data_by_param[param] = free_data_by_param.get(param, True) and deferred_release.free_data

        # Partitioning, allocator work, and any offload side effects remain
        # outside the slow-path lock. Ordinary module owners are never removed.
        for param in params_to_release:
            self.__release_param(param, free_data_by_param[param])

    @instrument_w_nvtx
    def __all_gather_params(self, params: Set[Parameter], forward: bool) -> None:
        quantized_params = []
        nonquantized_params = []
        for param in params:
            if hasattr(param.ds_tensor, 'ds_quant_scale'):
                quantized_params.append(param)
            else:
                nonquantized_params.append(param)
        if quantized_params:
            self.__all_gather_params_(quantized_params, forward, quantize=True)
        if nonquantized_params:
            self.__all_gather_params_(nonquantized_params, forward, quantize=self.zero_quantized_weights)

    def __all_gather_params_(self, params: Set[Parameter], forward: bool, quantize: bool = False) -> None:
        """for each partitioned parameter, kick off an async allgather and store
        the work handle for the in flight parameters."""
        partitioned_params = []
        all_gather_numel = 0  # numel = num of elements
        for param in params:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                partitioned_params.append(param)
                all_gather_numel += param.ds_numel

        if partitioned_params:
            self.__n_available_params += all_gather_numel
            # here we need to handle a special case where some of the parameters have a valid hpz secondary tensor (e.g. they are not trainable so their secondary tensor never expire) but others do not.
            partitioned_params_with_secondary_tensors = [
                p for p in partitioned_params if p.ds_secondary_tensor is not None
            ]
            partitioned_params_without_secondary_tensors = [
                p for p in partitioned_params if p.ds_secondary_tensor is None
            ]
            for param_group in [
                    partitioned_params_with_secondary_tensors, partitioned_params_without_secondary_tensors
            ]:
                if not param_group:
                    continue
                with get_accelerator().stream(self.__allgather_stream):
                    event_name = __class__.FORWARD_ALL_GATHER if forward else __class__.BACKWARD_ALL_GATHER
                    self.__profiler.start_event(event_name)
                    handle = param_group[0].all_gather_coalesced(param_group, quantize=quantize)
                    self.__profiler.stop_event(event_name, all_gather_numel)
                for param in param_group:
                    assert param.ds_status == ZeroParamStatus.INFLIGHT, param.ds_summary()
                    self.__inflight_param_registry[param] = handle

            # Release swap buffers for persisted params on nvme since they will never be partitioned or evicted from GPU
            swap_persisted_params = [
                p for p in partitioned_params if p.ds_persist and p.ds_tensor.final_location == OffloadDeviceEnum.nvme
            ]
            if swap_persisted_params:
                swap_persisted_params[0].nvme_swapper.remove_partition_and_release_buffers(swap_persisted_params)

    @compiler.disable
    @instrument_w_nvtx
    def __release_param(self, param: Parameter, free_data: bool = True) -> None:
        if param.ds_status == ZeroParamStatus.AVAILABLE and not param.ds_active_sub_modules:
            if logger.isEnabledFor(logging.DEBUG):
                debug_rank0(f"-release: {param.ds_summary()}")
                print_rank_0(f"release: {debug_param2name_id_shape(param)}", force=False)
            param.partition(free_data=free_data)
            self.__n_available_params -= param.ds_numel

    @instrument_w_nvtx
    @functools.lru_cache(maxsize=None)
    def __params_to_release(self, submodule_to_release: Module, step_id: int) -> Set[int]:
        if not self.is_complete_trace():
            raise RuntimeError("expected trace to be complete")

        params_to_release = set(
            p.ds_id for p in iter_params(submodule_to_release, recurse=z3_leaf_module(submodule_to_release))
            if not p.ds_persist)

        # Problem: When prefetcher scans the param trace, it skips AVAILABLE params.
        # This creates issues if those params are released before the skipped uses:
        # 1) It hurts performance as the skipped uses are never prefetched.
        # 2) For nvme params, we run out of swap buffers because the prefetch order
        # diverges from the trace.
        # Solution: Don't release params whose reuse was skipped by prefetch. This is
        # possible because we detect such skips during prefetch and mark those params.
        for param in iter_params(submodule_to_release, recurse=z3_leaf_module(submodule_to_release)):
            if self.__most_recent_step_id_param_fetched_for[param] > step_id:
                params_to_release.discard(param.ds_id)

        # examine all modules within `max_reuse_dist_in_numel` of the current step,
        # if we see any of the candidate parameters to be released reoccur while
        # doing this, remove them from the set of parameters to release.
        params_traversed = 0
        for module in self.__submodule_order[step_id:]:
            if params_traversed >= self.__max_reuse_dist_in_numel:
                break
            for param in iter_params(module, recurse=z3_leaf_module(submodule_to_release)):
                params_to_release.discard(param.ds_id)
                params_traversed += param.ds_numel

        return params_to_release

    @instrument_w_nvtx
    def __prefetch_nvme_param_partitions(self) -> None:
        """swap in parameter partitions from nvme for those parameters that will be used
        after the ones that are already being prefetched into full parameters
        """
        if not self.is_complete_trace():
            return

        numel_in_flight = sum(param.ds_numel for param in self.__inflight_param_registry)

        numel_considered = 0
        swap_in_params = []
        for param_in_trace in self.__param_queue:
            param = param_in_trace.param
            if param.nvme_swapper is None:
                continue
            if (numel_considered > 2 * numel_in_flight
                    or len(swap_in_params) >= param.nvme_swapper.available_swap_in_buffers()):
                break
            if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
                swap_in_params.append(param)
            numel_considered += param.ds_numel

        if swap_in_params:
            swap_in_params[0].nvme_swapper.swap_in(swap_in_params, async_op=True)
