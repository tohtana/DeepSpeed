# Copyright (c) 2025 Peng Du and Zhipeng Wang
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
try:
    from deepspeed.runtime.zero.muon.original_muon import MuonWithAuxAdam as BaseMuonWithAuxAdam
except ImportError:
    pass


class MuonWithAuxAdam(BaseMuonWithAuxAdam):

    def __init__(self, param_groups, adam_optimizer=None, adam_optimizer_kwargs=None):
        super().__init__(param_groups)
        self.aux_optimizer = None
        aux_param_groups = [group for group in self.param_groups if not group["use_muon"]]
        if aux_param_groups:
            assert adam_optimizer is not None, "An Adam optimizer is required for non-Muon parameter groups"
            self.aux_optimizer = adam_optimizer(aux_param_groups, **(adam_optimizer_kwargs or {}))
            for group, aux_group in zip(aux_param_groups, self.aux_optimizer.param_groups):
                for key, value in aux_group.items():
                    group.setdefault(key, value)
            self.aux_optimizer.param_groups = aux_param_groups
            self.aux_optimizer.state = self.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                # we move the muon update part to the deepspeed's optimizer since the parameter here is a flat version
                # thus not suitable for muon update
                for p in group["params"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(p.grad.reshape(p.shape), alpha=-group["lr"])

        if self.aux_optimizer is not None:
            aux_param_groups = [group for group in self.param_groups if not group["use_muon"]]
            for group in aux_param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
            self.aux_optimizer.param_groups = aux_param_groups
            self.aux_optimizer.state = self.state
            self.aux_optimizer.step()

        return loss
