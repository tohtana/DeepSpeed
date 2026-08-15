# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression test for a compiled ZeRO-3 parameter consumed by an eager resume."""

import argparse
import json

import torch
import torch.nn.functional as F

import deepspeed
from deepspeed import comm
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad

_LIVEOUT_SHAPES = []
_LIVEOUT_STORAGE_PTRS = []
_LIVEOUT_ALLOCATED_BYTES = []


class TrackingEmbedding(torch.nn.Embedding):

    def forward(self, input_ids):
        weight = self.weight
        torch._dynamo.graph_break()
        _LIVEOUT_SHAPES.append(list(weight.shape))
        _LIVEOUT_STORAGE_PTRS.append(weight.untyped_storage().data_ptr())
        _LIVEOUT_ALLOCATED_BYTES.append(get_accelerator().memory_allocated())
        return F.embedding(input_ids, weight)


class ParameterLiveOutModel(torch.nn.Module):

    def __init__(self, vocab_size=512, hidden_size=256):
        super().__init__()
        self.embedding = TrackingEmbedding(vocab_size, hidden_size)
        self.num_grid_per_side = 16

    def tensor_dependent_embedding(self, grid_thw):
        """Match the tensor-dependent Python loop that makes Dynamo skip this frame."""
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        indices = []
        for temporal, height, width in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, height).int()
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, width).int()
            image_indices = (h_idxs[:, None] * self.num_grid_per_side + w_idxs[None]).flatten()
            for _ in range(temporal):
                indices.extend(image_indices.tolist())

        index_tensor = torch.tensor(indices, dtype=torch.long, device=self.embedding.weight.device)
        return self.embedding(index_tensor)

    def forward(self, input_ids, grid_thw):
        outer_value = input_ids.float().sum()
        embedded = self.tensor_dependent_embedding(grid_thw)
        return embedded.float().sum() + outer_value * 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_config", type=str, default="ds_config_z3_deepcompile_no_persist.json")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--expect-parameter-offload", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1234)
    model = ParameterLiveOutModel()
    assert model.embedding.weight.numel() > 100000

    engine, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters())
    assert comm.get_world_size() == 2
    weight = engine.module.embedding.weight
    assert weight.ds_status == ZeroParamStatus.NOT_AVAILABLE
    assert weight.numel() == 0
    if args.expect_parameter_offload:
        assert weight.ds_tensor.device.type == "cpu"

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    engine.compile()

    device = get_accelerator().current_device_name()
    input_ids = torch.tensor([[0, 127, 256, 511], [511, 256, 127, 0]], device=device)
    grid_thw = torch.tensor([[1, 1, 4], [1, 2, 2]], device=device)
    before = safe_get_full_fp32_param(weight).detach().clone()
    losses = []
    gradient_l1 = []

    try:
        for _ in range(args.steps):
            loss = engine(input_ids, grid_thw)
            assert torch.isfinite(loss)
            engine.backward(loss)

            full_grad = safe_get_full_grad(weight)
            assert full_grad is not None
            assert list(full_grad.shape) == [model.embedding.num_embeddings, model.embedding.embedding_dim]
            assert torch.isfinite(full_grad).all()
            grad_l1 = full_grad.float().abs().sum().item()
            assert grad_l1 > 0
            gradient_l1.append(grad_l1)

            losses.append(loss.detach().float().item())
            engine.step()

        after = safe_get_full_fp32_param(weight).detach().clone()
        update_linf = (after - before).abs().max().item()
        assert update_linf > 0

        assert _LIVEOUT_SHAPES == [[model.embedding.num_embeddings, model.embedding.embedding_dim]] * args.steps
        assert len(set(_LIVEOUT_STORAGE_PTRS)) == 1

        steady_allocations = _LIVEOUT_ALLOCATED_BYTES[-3:]
        assert max(steady_allocations) - min(steady_allocations) <= 4 * 1024 * 1024
        assert sum(torch._dynamo.utils.counters["graph_break"].values()) > 0

        result = {
            "rank": comm.get_rank(),
            "world_size": comm.get_world_size(),
            "parameter_shard_device": weight.ds_tensor.device.type,
            "steps": args.steps,
            "consumer_ran_after_graph_break": True,
            "liveout_shape": _LIVEOUT_SHAPES[-1],
            "unique_liveout_storage_ptrs": len(set(_LIVEOUT_STORAGE_PTRS)),
            "steady_allocated_byte_spread": max(steady_allocations) - min(steady_allocations),
            "gradient_l1": gradient_l1,
            "update_linf": update_linf,
            "losses": losses,
        }
        print(f"PARAMETER_LIVEOUT_RESULT={json.dumps(result, sort_keys=True)}", flush=True)
        comm.barrier()
        if comm.get_rank() == 0:
            print("PARAMETER_LIVEOUT_PASS", flush=True)
    finally:
        engine.destroy()


if __name__ == "__main__":
    main()
