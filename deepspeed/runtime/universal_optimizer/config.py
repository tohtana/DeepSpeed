# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import Enum
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
import torch
from pydantic import Field
from typing import Optional


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


def get_universal_optimizer_config(config_dict):
    if "universal_optimizer" in config_dict:
        return UniversalOptimizerConfig(**config_dict["universal_optimizer"])
    else:
        return UniversalOptimizerConfig()
