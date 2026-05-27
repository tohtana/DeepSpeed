# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
from types import SimpleNamespace

import pytest
import torch
from torch.fx import Graph

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.compile.config import CompileConfig
from deepspeed.compile.passes import prefetch as prefetch_pass
from deepspeed.compile.passes.zero3_compile import add_gather_and_release
from deepspeed.compile.util import get_deepcompile_handle, is_deepcompile_supported
from unit.common import DistributedTest

pytestmark = pytest.mark.skipif(not is_deepcompile_supported(),
                                reason="DeepCompile requires CUDA and supported PyTorch")


class TestDeepCompileZ3ReleaseStorage(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    def _device(self):
        return torch.device(get_accelerator().current_device_name())

    def _init_dc(self, symmetric_memory=False, bucket_bytes=1024):
        dc = get_deepcompile_handle()
        if symmetric_memory:
            group_name = getattr(dist.get_world_group(), "group_name", None)
            if group_name is None:
                pytest.skip("symmetric-memory process group name is unavailable")
            dist.enable_symm_mem_for_group(group_name)
        dc.init(dist.get_world_group(), CompileConfig(deepcompile=True, symmetric_memory=symmetric_memory),
                bucket_bytes)
        return dc

    def _cleanup_dc(self, dc, started_forward=False):
        try:
            if started_forward:
                dc.end_forward()
        finally:
            dc.cleanup()

    def _register_param(self, dc, graph_id, ds_id, shape, dtype=torch.float32, persistent=False):
        device = self._device()
        world_size = dist.get_world_size()
        true_numel = math.prod(shape)
        shard_numel = math.ceil(true_numel / world_size)
        rank = dist.get_rank()
        values = torch.arange(rank * shard_numel, (rank + 1) * shard_numel, device=device,
                              dtype=torch.float32).to(dtype)
        grad_buffer = torch.zeros_like(values)
        dc.register_z3_param(ds_id, list(shape), values, grad_buffer, persistent)
        dc.register_graph_z3(graph_id, [ds_id])
        return values

    def _gather_view_and_storage(self, shard, graph_id, ds_id, dtype=None):
        if dtype is None:
            gathered = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id)
        else:
            gathered = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id, dtype=dtype)
        gathered = torch.ops.dc.wait_allgather.default(gathered, graph_id, ds_id)
        flat = gathered.reshape(-1)
        view = flat.narrow(0, 0, max(flat.numel() - 1, 1))
        assert view.untyped_storage().data_ptr() == gathered.untyped_storage().data_ptr()
        storage = view.untyped_storage()
        assert storage.nbytes() >= gathered.numel() * gathered.element_size()
        return view, storage

    def _release(self, view, graph_id, ds_id, n_users, synchronize=True):
        torch.ops.dc.release_param.default(view, view, graph_id, ds_id, n_users)
        if synchronize:
            get_accelerator().synchronize()

    def _assert_alloc_returns_to_baseline(self, before_bytes, padded_bytes, tolerance_bytes=None):
        get_accelerator().empty_cache()
        current_bytes = get_accelerator().memory_allocated()
        if tolerance_bytes is None:
            tolerance_bytes = max(padded_bytes // 4, 1024)
        assert tolerance_bytes < padded_bytes
        assert current_bytes <= before_bytes + tolerance_bytes

    def _expected_view_sum(self, shape, dtype=torch.float32):
        world_size = dist.get_world_size()
        shard_numel = math.ceil(math.prod(shape) / world_size)
        values = torch.arange(0, world_size * shard_numel, dtype=torch.float32, device=self._device())
        values = values[:math.prod(shape)].to(dtype).reshape(-1)
        return values.narrow(0, 0, max(values.numel() - 1, 1)).sum()

    def test_storage_resized_to_zero_after_release_single_use(self):
        graph_id, ds_id = 9010, 9011
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4097])
            get_accelerator().empty_cache()
            before_bytes = get_accelerator().memory_allocated()
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            padded_bytes = storage.nbytes()
            self._release(view, graph_id, ds_id, 1)
            assert storage.nbytes() == 0
            del view
            self._assert_alloc_returns_to_baseline(before_bytes, padded_bytes)
        finally:
            self._cleanup_dc(dc)

    def test_storage_nonzero_until_final_release_when_multi_use(self):
        graph_id, ds_id = 9020, 9021
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [3])
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            before_release_nbytes = storage.nbytes()
            self._release(view, graph_id, ds_id, 2)
            assert storage.nbytes() == before_release_nbytes
            self._release(view, graph_id, ds_id, 2)
            assert storage.nbytes() == 0
        finally:
            self._cleanup_dc(dc)

    def test_persistent_param_storage_unchanged_across_release(self):
        graph_id, ds_id = 9030, 9031
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4], persistent=True)
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            before_ptr = storage.data_ptr()
            before_nbytes = storage.nbytes()
            self._release(view, graph_id, ds_id, 1)
            assert storage.data_ptr() == before_ptr
            assert storage.nbytes() == before_nbytes
        finally:
            self._cleanup_dc(dc)

    def test_dtype_mismatch_nonpersistent_releases_registered_storage(self):
        graph_id, ds_id = 9040, 9041
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [3], dtype=torch.bfloat16)
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id, dtype=torch.float32)
            assert view.dtype == torch.float32
            self._release(view, graph_id, ds_id, 1)
            assert storage.nbytes() == 0
        finally:
            self._cleanup_dc(dc)

    def test_prefetch_params_fused_resizes_each_buffer_independently(self):
        graph_id = 9050
        ds_id_a, ds_id_b = 9051, 9052
        dc = self._init_dc()
        try:
            shard_a = self._register_param(dc, graph_id, ds_id_a, [3], dtype=torch.bfloat16)
            shard_b = self._register_param(dc, graph_id, ds_id_b, [5], dtype=torch.bfloat16)
            dc.register_graph_z3(graph_id, [ds_id_a, ds_id_b])
            torch.ops.dc.prefetch_params_fused.default(graph_id, [shard_a, shard_b], [ds_id_a, ds_id_b],
                                                       [torch.float32, torch.float32])

            view_a, storage_a = self._gather_view_and_storage(shard_a, graph_id, ds_id_a, dtype=torch.float32)
            view_b, storage_b = self._gather_view_and_storage(shard_b, graph_id, ds_id_b, dtype=torch.float32)
            assert storage_a.nbytes() > 0
            assert storage_b.nbytes() > 0
            self._release(view_a, graph_id, ds_id_a, 1)
            assert storage_a.nbytes() == 0
            assert storage_b.nbytes() > 0
            self._release(view_b, graph_id, ds_id_b, 1)
            assert storage_b.nbytes() == 0
        finally:
            self._cleanup_dc(dc)

    def test_prefetch_params_fused_rejects_mixed_explicit_and_default_dtypes(self):
        get_deepcompile_handle()

        def build_ag_nodes(dtype_a=None, dtype_b=None):
            graph = Graph()
            param_a = graph.placeholder("param_a")
            param_b = graph.placeholder("param_b")
            kwargs_a = {} if dtype_a is None else {"dtype": dtype_a}
            kwargs_b = {} if dtype_b is None else {"dtype": dtype_b}
            ag_a = graph.call_function(torch.ops.dc.allgather_param.default,
                                       args=(param_a, 9060, 9061),
                                       kwargs=kwargs_a,
                                       name="ag_explicit")
            ag_b = graph.call_function(torch.ops.dc.allgather_param.default,
                                       args=(param_b, 9060, 9062),
                                       kwargs=kwargs_b,
                                       name="ag_default")
            new_graph = Graph()
            copy_a = new_graph.placeholder("copy_a")
            copy_b = new_graph.placeholder("copy_b")
            return new_graph, [copy_a, copy_b], [ag_a, ag_b]

        mixed_graph, mixed_params, mixed_ags = build_ag_nodes(torch.float32, None)
        with pytest.raises(AssertionError, match="ag_explicit.*ag_default"):
            prefetch_pass._insert_prefetch_params_fused(mixed_graph, 9060, mixed_params, mixed_ags, [9061, 9062])

        explicit_graph, explicit_params, explicit_ags = build_ag_nodes(torch.float32, torch.bfloat16)
        explicit_node = prefetch_pass._insert_prefetch_params_fused(explicit_graph, 9060, explicit_params,
                                                                    explicit_ags, [9061, 9062])
        assert explicit_node.target == torch.ops.dc.prefetch_params_fused.default
        assert len(explicit_node.args) == 4
        assert explicit_node.args[3] == [torch.float32, torch.bfloat16]

        default_graph, default_params, default_ags = build_ag_nodes(None, None)
        default_node = prefetch_pass._insert_prefetch_params_fused(default_graph, 9060, default_params, default_ags,
                                                                   [9061, 9062])
        assert default_node.target == torch.ops.dc.prefetch_params_fused.default
        assert len(default_node.args) == 3

    def test_release_node_keeps_gathered_param_live_as_input(self):
        graph_id, ds_id = 9065, 9066
        get_deepcompile_handle()

        graph = Graph()
        param = graph.placeholder("param")
        activation = graph.placeholder("activation")
        matmul = graph.call_function(torch.ops.aten.mm.default, args=(activation, param))
        graph.output(matmul)
        param.meta["val"] = torch.empty((4, 4), device=self._device(), dtype=torch.float32)

        param_manager = SimpleNamespace(ds_ids={"param": ds_id},
                                        params={"param": SimpleNamespace(dtype=torch.float32)})

        transformed = add_gather_and_release(graph_id, graph, param_manager, [param])

        release_nodes = [n for n in transformed.nodes if n.target == torch.ops.dc.release_param.default]
        assert len(release_nodes) == 1
        release_node = release_nodes[0]
        assert release_node.args[0].target == torch.ops.aten.mm.default
        assert release_node.args[1].target == torch.ops.dc.wait_allgather.default
        assert release_node.args[3] == ds_id

    def test_backward_side_gather_release_storage_resized(self):
        graph_id, ds_id = 9070, 9071
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [3])
            dc.start_backward(False)
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(view, graph_id, ds_id, 1)
            assert storage.nbytes() == 0
        finally:
            self._cleanup_dc(dc)

    def test_forward_release_then_backward_regather_allocates_fresh(self):
        fwd_graph_id, bwd_graph_id, ds_id = 9080, 9081, 9082
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, fwd_graph_id, ds_id, [3])
            dc.register_graph_z3(bwd_graph_id, [ds_id])
            fwd_view, fwd_storage = self._gather_view_and_storage(shard, fwd_graph_id, ds_id)
            self._release(fwd_view, fwd_graph_id, ds_id, 1)
            assert fwd_storage.nbytes() == 0

            dc.start_backward(False)
            bwd_view, bwd_storage = self._gather_view_and_storage(shard, bwd_graph_id, ds_id)
            assert bwd_storage.nbytes() > 0
            assert fwd_storage.nbytes() == 0
            self._release(bwd_view, bwd_graph_id, ds_id, 1)
            assert bwd_storage.nbytes() == 0
        finally:
            self._cleanup_dc(dc)

    def test_clear_all_gathered_params_then_regather_allocates_fresh(self):
        graph_id, ds_id = 9090, 9091
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [3])
            first_view, first_storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(first_view, graph_id, ds_id, 1)
            assert first_storage.nbytes() == 0
            dc.clear_all_gathered_params()

            second_view, second_storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            assert second_storage.nbytes() > 0
            assert first_storage.nbytes() == 0
            self._release(second_view, graph_id, ds_id, 1)
            assert second_storage.nbytes() == 0
        finally:
            self._cleanup_dc(dc)

    def test_long_running_consumer_kernel_does_not_use_after_free(self):
        graph_id, ds_id = 9100, 9101
        if not hasattr(torch.cuda, "_sleep"):  #ignore-cuda
            pytest.skip("CUDA sleep helper is unavailable")
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4097])
            get_accelerator().empty_cache()
            before_bytes = get_accelerator().memory_allocated()
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            padded_bytes = storage.nbytes()
            result = torch.empty((), device=self._device(), dtype=view.dtype)
            consumer_stream = get_accelerator().Stream()
            with get_accelerator().stream(consumer_stream):
                torch.cuda._sleep(int(1e8))  #ignore-cuda
                result.copy_(view.sum())
                self._release(view, graph_id, ds_id, 1, synchronize=False)

            scratch = torch.empty((padded_bytes // view.element_size()) + 1024,
                                  device=self._device(),
                                  dtype=view.dtype)
            scratch.fill_(17)
            get_accelerator().synchronize()
            assert torch.allclose(result, self._expected_view_sum([4097], dtype=view.dtype))
            assert storage.nbytes() == 0
            del scratch
            del view
            self._assert_alloc_returns_to_baseline(before_bytes, padded_bytes)
        finally:
            self._cleanup_dc(dc)

    def test_symm_mem_gather_release_workspace_unchanged(self):
        graph_id, ds_id, followup_ds_id = 9110, 9111, 9112
        dc = self._init_dc(symmetric_memory=True)
        started_forward = False
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4])
            followup_shard = self._register_param(dc, graph_id, followup_ds_id, [4])
            dc.register_graph_z3(graph_id, [ds_id, followup_ds_id])
            try:
                dc.start_forward()
                started_forward = True
            except RuntimeError as exc:
                pytest.skip(f"symmetric-memory setup is unavailable: {exc}")

            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(view, graph_id, ds_id, 1)
            assert storage.nbytes() == 0

            followup_view, followup_storage = self._gather_view_and_storage(followup_shard, graph_id, followup_ds_id)
            assert followup_storage.nbytes() > 0
            self._release(followup_view, graph_id, followup_ds_id, 1)
            assert followup_storage.nbytes() == 0
        finally:
            self._cleanup_dc(dc, started_forward=started_forward)
