# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Llama kernel injection must find rope_theta wherever the installed transformers keeps it.

The value has moved twice. Older releases exposed ``config.rope_theta``; transformers 5.0
folded the rotary settings into ``config.rope_parameters`` and dropped the attribute, so an
injection policy that only knows the old spelling raises AttributeError against a stock
LlamaConfig. Older still, it lived on the attention module.
"""

from types import SimpleNamespace

import pytest

from deepspeed.module_inject.containers.llama import _get_rope_theta


def test_reads_the_legacy_config_attribute():
    self_attn = SimpleNamespace(config=SimpleNamespace(rope_theta=500000.0))

    assert _get_rope_theta(self_attn) == 500000.0


def test_reads_rope_parameters_when_the_attribute_is_gone():
    # transformers >= 5.0: the attribute is absent and the value sits in the dict.
    config = SimpleNamespace(rope_parameters={"rope_theta": 10000.0, "rope_type": "default"})
    self_attn = SimpleNamespace(config=config)

    assert _get_rope_theta(self_attn) == 10000.0


def test_prefers_rope_parameters_when_both_are_present():
    config = SimpleNamespace(rope_theta=500000.0, rope_parameters={"rope_theta": 10000.0})
    self_attn = SimpleNamespace(config=config)

    assert _get_rope_theta(self_attn) == 10000.0


def test_falls_back_to_the_module_attribute():
    # No config at all, the layout the policy handled before configs were attached.
    self_attn = SimpleNamespace(rope_theta=1000000.0)

    assert _get_rope_theta(self_attn) == 1000000.0


def test_falls_back_when_rope_parameters_carries_no_theta():
    config = SimpleNamespace(rope_parameters={"rope_type": "default"})
    self_attn = SimpleNamespace(config=config, rope_theta=250000.0)

    assert _get_rope_theta(self_attn) == 250000.0


def test_raises_when_nothing_carries_it():
    with pytest.raises(AttributeError):
        _get_rope_theta(SimpleNamespace(config=SimpleNamespace()))


def test_rejects_scaled_rope_parameters():
    config = SimpleNamespace(
        rope_parameters={
            "rope_theta": 500000.0,
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        })

    with pytest.raises(ValueError, match="only supports default RoPE, got rope_type='llama3'"):
        _get_rope_theta(SimpleNamespace(config=config))


def test_rejects_legacy_rope_scaling():
    config = SimpleNamespace(rope_theta=10000.0, rope_scaling={"type": "linear", "factor": 2.0})

    with pytest.raises(ValueError, match="only supports default RoPE, got rope_type='linear'"):
        _get_rope_theta(SimpleNamespace(config=config))
