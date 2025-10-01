# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import Optional

from pydantic import field_validator, field_serializer

from deepspeed.runtime.config_utils import DeepSpeedConfigModel


class UniversalOptimizerConfig(DeepSpeedConfigModel):
    """ Configure universal optimizer settings """
    enabled: bool = False

    force_model_dtype: Optional[torch.dtype] = None
    """ Desired model data type, will convert model to this type.
    """

    zero3_allgather_dtype: Optional[torch.dtype] = None
    """ Desired allgather data type, will convert communication to this type.
    """

    reduce_dtype: Optional[torch.dtype] = None
    """ Desired reduce_scatter data type, will convert communication to this type.
    """

    grad_accum_dtype: Optional[torch.dtype] = None
    """ Desired gradient accumulation data type, will convert gradient accumulation to this type.
    """

    optimizer_dtype: Optional[torch.dtype] = None
    """ Desired optimizer data type, will convert optimizer state to this type.
    """

    _dtype_fields = ("force_model_dtype", "zero3_allgather_dtype", "reduce_dtype", "grad_accum_dtype",
                     "optimizer_dtype")

    @classmethod
    def _coerce_dtype(cls, value):
        if value is None or isinstance(value, torch.dtype):
            return value

        from deepspeed.runtime.config import DtypeEnum

        if isinstance(value, DtypeEnum):
            return value.value

        if isinstance(value, str):
            candidate = value.strip()
            try:
                return DtypeEnum(candidate).value
            except ValueError:
                if candidate.startswith("torch."):
                    try:
                        return DtypeEnum(candidate.split(".", 1)[1]).value
                    except ValueError:
                        pass
                raise ValueError(f"Unsupported dtype specification '{value}'")

        raise TypeError(f"Unsupported dtype type '{type(value).__name__}'")

    @field_validator(*_dtype_fields, mode="before")
    @classmethod
    def validate_dtype(cls, value):
        return cls._coerce_dtype(value)

    @field_serializer(*_dtype_fields)
    def serialize_dtype(self, value):
        if value is None:
            return None
        return str(value)


def get_universal_optimizer_config(config_dict):
    if "universal_optimizer" in config_dict:
        return UniversalOptimizerConfig(**config_dict["universal_optimizer"])
    else:
        return UniversalOptimizerConfig()
