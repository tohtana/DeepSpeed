# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Optional, Literal

from pydantic import Field, model_validator

from deepspeed.runtime.config_utils import DeepSpeedConfigModel

PassName = Literal["z1", "z3", "autosp"]
Zero3TuningStrategy = Literal["baseline", "agent"]


class CompileConfig(DeepSpeedConfigModel):
    """ Configure compile settings """

    deepcompile: bool = False
    """ Turn on/off the DeepCompile mode """

    free_activation: bool = False
    """ Turn on/off the free activation mode """

    free_activation_threshold: int = 10 * 1024 * 1024
    """ In free activation mode, activations no less than this threshold (in byte) are eagerly freed """

    offload_activation: bool = False
    """ Turn on/off the activation offloading """

    offload_opt_states: bool = False
    """ Turn on/off the optimizer states offloading """

    double_buffer: bool = True
    """ Turn on/off the double buffering """

    symmetric_memory: bool = False
    """ Turn on/off the symmetric memory """

    debug_log: bool = False
    """ Turn on/off the graph dumping """

    offload_parameters: bool = False
    """ Turn on/off the parameter offloading """

    sync_before_reduce: bool = False
    """ Turn on/off the sync before reduce """

    sync_after_reduce: bool = False
    """ Turn on/off the sync after reduce """

    sync_before_allgather: bool = False
    """ Turn on/off the sync before allgather """

    sync_after_allgather: bool = False
    """ Turn on/off the sync after allgather """

    keep_int_input_tensors: bool = True
    """ Keep real values for int tensors in InputStorage instead of using dummy values """

    keep_all_input_tensors: bool = False
    """ Keep real values for all input tensors in InputStorage instead of using dummy values """

    passes: Optional[List[PassName]] = None
    """ Composes different optimizations. """

    zero3_tuning_strategy: Zero3TuningStrategy = "baseline"
    """ Controls how ZeRO-3 warmup tuning decisions are made. """

    agent_command: Optional[List[str]] = None
    """ Command argv used to invoke an external agent when agent tuning is enabled. """

    agent_max_iterations: int = Field(3, gt=0)
    """ Maximum number of agent-directed tuning iterations per graph invocation. """

    agent_timeout_sec: int = Field(300, gt=0)
    """ Timeout in seconds for each external agent invocation. """

    @model_validator(mode="after")
    def validate_agent_mode(self):
        if self.zero3_tuning_strategy != "agent":
            return self

        if not self.agent_command or not isinstance(self.agent_command, list):
            raise ValueError("compile.agent_command must be a non-empty argv list when "
                             "compile.zero3_tuning_strategy='agent'")

        if any(not isinstance(arg, str) or arg == "" for arg in self.agent_command):
            raise ValueError("compile.agent_command must contain only non-empty strings when "
                             "compile.zero3_tuning_strategy='agent'")

        if self.offload_parameters:
            raise ValueError("compile.zero3_tuning_strategy='agent' conflicts with "
                             "compile.offload_parameters=True because the offload path has no warmup tuning step")

        if self.passes is not None:
            raise ValueError("compile.zero3_tuning_strategy='agent' conflicts with compile.passes because "
                             "custom pass composition overrides the default ZeRO-3 warmup schedule")

        return self
