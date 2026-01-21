---
title: "AutoTP Training API"
tags: training tensor-parallelism
---

# AutoTP Training API

This tutorial covers the **AutoTP Training API** for combining tensor parallelism with ZeRO optimization during training. For inference-only tensor parallelism, see [Automatic Tensor Parallelism for HuggingFace Models](automatic-tensor-parallelism).

## Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Custom Layer Specifications](#custom-layer-specifications)
- [Limitations](#limitations)

## Introduction

The AutoTP Training API enables hybrid parallelism by combining:
- **Tensor Parallelism (TP)**: Split model weights across GPUs within a node
- **Data Parallelism (DP)**: Replicate model across GPU groups
- **ZeRO Optimization**: Memory-efficient optimizer states (Stage 0, 1, or 2)

Tensor parallelism (TP) splits the computations and parameters of large layers
across multiple GPUs so each rank holds only a shard of the weight matrix. This
is an efficient way to train large-scale transformer models by reducing per-GPU
memory pressure while keeping the layer math distributed across the TP group.


## Quick Start

### Basic Usage

AutoTP-specific steps are calling `set_autotp_mode(training=True)` before creating the model and wrapping the model with `deepspeed.tp_model_init(...)` to shard weights across TP ranks. Once initialized, the training loop itself does not change.

```python
import torch
import deepspeed
from deepspeed.module_inject.layers import set_autotp_mode

# 1. Enable training mode before model creation
set_autotp_mode(training=True)

# 2. Create your model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# 3. Create tensor parallel process groups
tp_size = 4
dp_size = world_size // tp_size
tp_group = create_tp_group(tp_size)

# 4. Apply tensor parallel sharding
model = deepspeed.tp_model_init(
    model,
    tp_size=tp_size,
    dtype=torch.bfloat16,
    tp_group=tp_group
)

# 5. Initialize DeepSpeed with ZeRO
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config,
    mpu=mpu  # Model parallel unit
)

# 6. Train as usual
for batch in dataloader:
    outputs = engine(input_ids=batch["input_ids"], labels=batch["labels"])
    engine.backward(outputs.loss)
    engine.step()
```

### Preset-based Sharding

If your model matches a built-in preset, set `tensor_parallel.preset_model` in the DeepSpeed config:

```json
{
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 1,
    "bf16": { "enabled": true },
    "zero_optimization": { "stage": 2 },
    "tensor_parallel": {
        "autotp_size": 4,
        "preset_model": "llama"
    }
}
```

For the list of available presets, see [supported models](../code-docs/training#autotp-supported-models).



## Custom Patterns

If you are training a custom model, define regex-based patterns and partition rules in `tensor_parallel.partition_config`:

```json
{
    "tensor_parallel": {
        "autotp_size": 4,
        "partition_config": {
            "use_default_specs": false,
            "layer_specs": [
                {
                    "patterns": [".*\\.o_proj\\.weight$", ".*\\.down_proj\\.weight$"],
                    "partition_type": "row"
                },
                {
                    "patterns": [".*\\.[qkv]_proj\\.weight$"],
                    "partition_type": "column"
                },
                {
                    "patterns": [".*\\.gate_up_proj\\.weight$"],
                    "partition_type": "column",
                    "shape": [2, -1],
                    "partition_dim": 0
                }
            ]
        }
    }
}
```

## Custom Layer Specifications

For models not covered by presets, define custom layer specs:

```json
{
    "tensor_parallel": {
        "autotp_size": 4,
        "partition_config": {
            "use_default_specs": false,
            "layer_specs": [
                {
                    "patterns": [".*\\.o_proj\\.weight$", ".*\\.down_proj\\.weight$"],
                    "partition_type": "row"
                },
                {
                    "patterns": [".*\\.[qkv]_proj\\.weight$"],
                    "partition_type": "column"
                },
                {
                    "patterns": [".*\\.gate_up_proj\\.weight$"],
                    "partition_type": "column",
                    "shape": [2, -1],
                    "partition_dim": 0
                }
            ]
        }
    }
}
```

### Fused Layers with Unequal Sub-parameters (GQA)

For Grouped Query Attention with different Q/K/V sizes:

```json
{
    "tensor_parallel": {
        "partition_config": {
            "layer_specs": [
                {
                    "patterns": [".*\\.qkv_proj\\.weight$"],
                    "partition_type": "column",
                    "shape": [[q_size, kv_size, kv_size], -1],
                    "partition_dim": 0
                }
            ]
        }
    }
}
```

## Limitations

1. **ZeRO Stage 3 not supported**: AutoTP currently only works with ZeRO stages 0, 1, and 2.

2. **TP size must divide model dimensions**: The tensor parallel size must evenly divide the attention head count and hidden dimensions.


## See Also

- [Automatic Tensor Parallelism for Inference](automatic-tensor-parallelism)
- [ZeRO Optimization](zero)
- [DeepSpeed Configuration](config-json)
