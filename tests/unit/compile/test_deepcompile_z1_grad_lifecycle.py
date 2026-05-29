# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.compile import init_z1


class FakeHook:

    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class FakeAccelerator:

    def current_device_name(self):
        return "cpu"


class FakeDeepCompileHandle:

    def __init__(self):
        self.register_calls = []
        self.update_calls = []

    def init(self, *args, **kwargs):
        pass

    def register_param(self, param_id, shape, param, grad_buffer, offset):
        self.register_calls.append((param_id, grad_buffer, offset))

    def update_param_grad_buffer(self, param_id, grad_buffer, offset):
        self.update_calls.append((param_id, grad_buffer, offset))


class FakeOptimizer:

    def __init__(self):
        self.params = [
            torch.nn.Parameter(torch.ones(3)),
            torch.nn.Parameter(torch.ones(2)),
            torch.nn.Parameter(torch.ones(4)),
        ]
        self.bit16_groups = [self.params]
        self.params_in_partition = {0: self.params[:2]}
        self.first_offset = {0: 1}
        self.partition_size = {0: 4}
        self.gradient_accumulation_dtype = torch.float32
        self.all_grad_tensors = {}
        self.averaged_gradients = {}
        self.contiguous_gradients = True
        self.grad_hook = FakeHook()
        self._grad_acc_hooks = [self.grad_hook]
        self.param_id = {id(param): idx for idx, param in enumerate(self.params)}
        self.is_param_in_current_partition = {0: True, 1: True, 2: False}

    def get_param_id(self, param):
        return self.param_id[id(param)]

    def get_all_grad_tensors(self, tensor_list, dtype):
        return [torch.zeros_like(param, dtype=dtype) for param in tensor_list]

    def get_flat_partition(self,
                           tensor_list,
                           first_offset,
                           partition_size,
                           dtype,
                           device,
                           param_group_idx,
                           return_tensor_list=False):
        del dtype, device
        grad_tensors = self.all_grad_tensors[param_group_idx]
        result = []
        current_size = 0
        for tensor_idx, tensor in enumerate(grad_tensors):
            tensor_offset = first_offset if tensor_idx == 0 else 0
            num_elements = tensor.numel() - tensor_offset
            num_elements = min(num_elements, partition_size - current_size)
            result.append(tensor.view(-1).narrow(0, tensor_offset, num_elements))
            current_size += num_elements
            if current_size == partition_size:
                break
        if current_size < partition_size:
            result.append(torch.zeros(partition_size - current_size, dtype=grad_tensors[0].dtype))
        return result if return_tensor_list else torch.cat(result)


class FakeEngine:

    def __init__(self):
        self.optimizer = FakeOptimizer()
        self.data_parallel_group = object()
        self.launch_compile_passes = None

    def zero_reduce_bucket_size(self):
        return 1024


def test_deepcompile_z1_rebinds_step_local_grad_buffers(monkeypatch):
    fake_handle = FakeDeepCompileHandle()
    hooks = []

    monkeypatch.setattr(init_z1, "get_deepcompile_handle", lambda: fake_handle)
    monkeypatch.setattr(init_z1, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(init_z1, "add_pre_backward_hook", hooks.append)
    monkeypatch.setattr(init_z1, "make_backend", lambda *args, **kwargs: "backend")

    engine = FakeEngine()
    assert init_z1.init_z1(engine, "eager", object(), {}) == "backend"
    assert engine.optimizer.grad_hook.removed
    assert engine.optimizer._grad_acc_hooks == []
    assert all(call[1].numel() == 0 for call in fake_handle.register_calls)

    hooks[0](False)

    assert [call[0] for call in fake_handle.update_calls] == [0, 1]
    assert all(call[1].numel() == 0 for call in fake_handle.update_calls)
    assert engine.optimizer.averaged_gradients == {}
    fake_handle.update_calls.clear()

    hooks[0](True)

    current_buffers = engine.optimizer._deepcompile_z1_current_grad_buffers[0]
    assert engine.optimizer.averaged_gradients[0] is current_buffers
    assert [call[0] for call in fake_handle.update_calls[:2]] == [0, 1]
    assert fake_handle.update_calls[0][1] is current_buffers[0]
    assert fake_handle.update_calls[0][2] == 1
    assert fake_handle.update_calls[1][1] is current_buffers[1]
    assert fake_handle.update_calls[1][2] == 0

    engine.optimizer._deepcompile_z1_release_grad_buffers(0)

    assert engine.optimizer._deepcompile_z1_current_grad_buffers[0] is None
    assert [call[0] for call in fake_handle.update_calls[-2:]] == [0, 1]
    assert all(call[1].numel() == 0 for call in fake_handle.update_calls[-2:])
