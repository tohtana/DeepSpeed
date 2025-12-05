# DeepSpeed Core API updates: PyTorch-style backward and low-precision master states

DeepSpeed is continuously evolving its core APIs to feel more natural to PyTorch users while giving them more control over performance and memory.

In this short update, we highlight two recent core improvements:

* **PyTorch-compatible backward API** – you can now use standard `tensor.backward(...)` patterns with DeepSpeed engines, including non-scalar outputs. ([PR](https://github.com/deepspeedai/DeepSpeed/pull/7665))
* **Low-precision master params / grads / optimizer states** – you can keep more state in bf16/fp16 to reduce memory usage and work better with `torch.autocast`. ([PR](https://github.com/deepspeedai/DeepSpeed/pull/7700))

These changes aim to make DeepSpeed feel closer to “vanilla PyTorch” while still providing ZeRO and mixed-precision benefits.


## 1. PyTorch-compatible backward API

Traditionally, DeepSpeed’s training loop relied on the engine’s backward API:

```python
loss = model_engine(batch)
model_engine.backward(loss)
model_engine.step()
```

This API had two notable constraints:

1. It only accepted a **scalar loss**.
2. You had to call **`model_engine.backward(loss)`**, rather than using the usual PyTorch `loss.backward()` style.

In plain PyTorch, many users rely on more flexible patterns like:

```python
output = model(batch)          # possibly non-scalar
output.backward(out_grad)      # custom gradient
```

This is useful when:

* You combine multiple models and losses.
* The loss is defined separately from the main model.
* You need to backprop through non-scalar tensors with custom gradients.

Previously, trying the same pattern with a DeepSpeed engine could skip internal preprocessing/postprocessing in DeepSpeed (e.g., loss scaling and ZeRO-related logic), potentially leading to incorrect behavior.

### What’s new

DeepSpeed now **intercepts the standard PyTorch `.backward()` call on tensors**, so you can use PyTorch-style backward while still getting the correct DeepSpeed behavior.

* You can call `.backward()` directly on tensors produced by a DeepSpeed engine.
* **Non-scalar outputs** are supported (with a matching gradient input), currently for **ZeROOptimizer-based** setups.
* When PyTorch’s internal hook APIs are not available, DeepSpeed will fall back to the traditional `model_engine.backward(loss)` path.

### Example: scalar loss with PyTorch-style backward

```python
import deepspeed
import torch

model = MyModel()
parameters = filter(lambda p: p.requires_grad, model.parameters())

engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=parameters,
    config="ds_config.json",
)

for batch in data_loader:
    inputs, labels = batch
    inputs = inputs.to(engine.device)
    labels = labels.to(engine.device)

    outputs = engine(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, labels)

    # New: use standard PyTorch-style backward
    loss.backward()

    # DeepSpeed step still drives the optimizer / ZeRO logic
    engine.step()
```

This looks and feels like standard PyTorch, but under the hood DeepSpeed still runs the same backward preprocessing and epilogue logic that you would get from `engine.backward(loss)`.

### Example: non-scalar output with custom gradient

DeepSpeed now supports non-scalar backward for ZeROOptimizer setups. For example:

```python
# outputs: [batch, hidden] – non-scalar
outputs = engine(inputs)

# Custom gradient of the same shape as outputs
grad_out = torch.ones_like(outputs) / outputs.numel()

# Backward from a non-scalar tensor
outputs.backward(grad_out)

engine.step()
```

This enables advanced scenarios like combining multiple models or passing custom gradients through intermediate activations, without having to manually call DeepSpeed’s internal prologue/epilogue APIs.


## 2. Low-precision master params, grads, and optimizer states

DeepSpeed optimizers historically maintained **FP32 master parameters, gradients, and optimizer states**, even when training in fp16 or bf16.

On large models, this can significantly increase memory usage, especially when combined with ZeRO.

### What’s new

We’ve introduced new options that allow you to keep more optimizer-related state in **lower precision**, especially for bf16:

* Existing (previously undocumented) option under `fp16`:

  * `fp16_master_weights_and_gradients` for ZeRO stage 1/2.
* New options under the `bf16` section:

  * `bf16_master_weights_and_grads`
  * `bf16_optimizer_states`

At a high level:

* `bf16_master_weights_and_grads = true`
  → keep master parameters and gradients in bf16 instead of fp32.
* `bf16_optimizer_states = true`
  → keep optimizer states (e.g., Adam moments) in bf16.

There is also a supported mixed configuration where `bf16_master_weights_and_grads == true` and `bf16_optimizer_states == false`, but **only when using CPU offload**.

Additionally:

* The same concept is extended beyond the original fp16+ZeRO1/2 support to **bf16** and **ZeRO3**.
* `torch.autocast` support (via the `torch_autocast` section) can now be combined with `bf16`/`fp16` in the same config, which was not supported before.

### Example: pure bf16 config with low-precision master state

Below is a simplified DeepSpeed config that keeps bf16 master weights, grads, and optimizer states, and uses `torch.autocast`:

```jsonc
{
  "train_batch_size": 1024,

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "bf16": {
    "enabled": true,
    "bf16_master_weights_and_grads": true,
    "bf16_optimizer_states": true
  },

  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "none"
    }
  },

  "torch_autocast": {
    "enabled": true,
    "dtype": "bfloat16"
  }
}
```

And a minimal training loop:

```python
import torch
import deepspeed

model = MyModel()
parameters = filter(lambda p: p.requires_grad, model.parameters())

engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=parameters,
    config="ds_config_bf16.json",
)

for batch in data_loader:
    inputs, labels = batch
    inputs = inputs.to(engine.device)
    labels = labels.to(engine.device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = engine(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

    loss.backward()
    engine.step()
```

This configuration:

* Runs the forward pass in **bf16** using `torch.autocast`.
* Keeps **master weights, grads, and optimizer states in bf16**, reducing memory footprint.
* Works with ZeRO stage 2 (and similarly with stage 3) for better scalability.

---

## Closing thoughts

These core API improvements are incremental but important steps toward making DeepSpeed:

* **More PyTorch-native** – training loops can increasingly look like standard PyTorch code.
* **More memory-efficient** – especially when combined with bf16/fp16 and ZeRO on large models.
* **Easier to compose** – enabling multi-model and custom-gradient workflows without custom DeepSpeed hooks.

We're excited to see how you use these APIs in your own training setups, and we welcome feedback and issues on GitHub as you try them out.

---

## Related Tests

For more usage examples, see the unit tests in the repository:

### PyTorch-compatible backward API

- [tests/unit/v1/zero/test_zero_user_backward.py](../../tests/unit/v1/zero/test_zero_user_backward.py)
  - `TestZeroUserBackwardBasic` – Basic `loss.backward()` with ZeRO stages 1, 2, and 3
  - `TestZeroUserBackwardNonScalar` – Non-scalar `tensor.backward(grad)` support
  - `TestZeroUserBackwardGradAccumulation` – Gradient accumulation with user backward
  - `TestZeroUserBackwardMultipleEngines` – Multiple engines with combined loss
  - `TestZeroUserBackwardSeparateLoss` – Separate loss function pattern
  - `TestZeroUserBackwardLeafModule` – Leaf module compatibility in ZeRO-3
  - `TestZeroUserBackwardWithScale` – `engine.scale()` method for fp16 loss scaling

### Low-precision master params/grads/optimizer states

- [tests/unit/v1/half_precision/test_bf16.py](../../tests/unit/v1/half_precision/test_bf16.py)
  - `TestBF16MasterWeightsGradients` – Tests `bf16_master_weights_and_grads` and `bf16_optimizer_states` options across ZeRO stages

- [tests/unit/v1/half_precision/test_with_autocast.py](../../tests/unit/v1/half_precision/test_with_autocast.py)
  - `TestTorchAutocastWithPrecisionModes` – Tests `torch_autocast` config combined with bf16/fp16 precision modes
