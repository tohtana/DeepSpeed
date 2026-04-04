# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
from pathlib import Path

import pytest

from deepspeed.compile.agent_runner import AgentRunner, AgentRunnerConfig
from deepspeed.compile.config import CompileConfig


class TestCompileConfig:

    def test_valid_agent_config(self):
        config = CompileConfig(zero3_tuning_strategy="agent",
                               agent_command=["/bin/bash", "-lc", "cat"],
                               agent_max_iterations=2,
                               agent_timeout_sec=7)

        assert config.zero3_tuning_strategy == "agent"
        assert config.agent_command == ["/bin/bash", "-lc", "cat"]
        assert config.agent_max_iterations == 2
        assert config.agent_timeout_sec == 7

    def test_agent_config_requires_command(self):
        with pytest.raises(ValueError, match="compile.agent_command"):
            CompileConfig(zero3_tuning_strategy="agent")

    def test_agent_config_rejects_offload_conflict(self):
        with pytest.raises(ValueError, match="compile.offload_parameters"):
            CompileConfig(zero3_tuning_strategy="agent",
                          agent_command=["/bin/bash", "-lc", "cat"],
                          offload_parameters=True)

    def test_agent_config_rejects_passes_conflict(self):
        with pytest.raises(ValueError, match="compile.passes"):
            CompileConfig(zero3_tuning_strategy="agent", agent_command=["/bin/bash", "-lc", "cat"], passes=["autosp"])


class TestAgentRunner:

    def test_runner_invocation(self, tmp_path: Path):
        runner = AgentRunner(AgentRunnerConfig(command=["/bin/bash", "-lc", "cat"], timeout_sec=5, debug_log=False))

        result = runner.run("hello runner", tmp_path / "iter0")

        assert result.stdout == "hello runner"
        assert result.stderr == ""
        assert result.returncode == 0
        assert not result.timed_out

    def test_runner_timeout(self, tmp_path: Path):
        runner = AgentRunner(AgentRunnerConfig(command=["/bin/bash", "-lc", "sleep 5"], timeout_sec=1,
                                               debug_log=False))

        started = time.time()
        result = runner.run("ignored", tmp_path / "iter_timeout")
        elapsed = time.time() - started

        assert result.timed_out
        assert elapsed < 5

    def test_runner_cleanup_on_success(self, tmp_path: Path):
        runner = AgentRunner(AgentRunnerConfig(command=["/bin/bash", "-lc", "cat"], timeout_sec=5, debug_log=False))
        iteration_dir = tmp_path / "iter_cleanup"

        result = runner.run("cleanup", iteration_dir)
        runner.cleanup(iteration_dir, keep=result.returncode != 0 or result.timed_out)

        assert not iteration_dir.exists()

    def test_runner_retention_on_failure(self, tmp_path: Path):
        runner = AgentRunner(
            AgentRunnerConfig(command=["/bin/bash", "-lc", "echo boom 1>&2; exit 3"], timeout_sec=5, debug_log=False))
        iteration_dir = tmp_path / "iter_failure"

        result = runner.run("retain", iteration_dir)
        runner.cleanup(iteration_dir, keep=result.returncode != 0 or result.timed_out)

        assert result.returncode == 3
        assert iteration_dir.exists()

    def test_runner_retention_on_debug(self, tmp_path: Path):
        runner = AgentRunner(AgentRunnerConfig(command=["/bin/bash", "-lc", "cat"], timeout_sec=5, debug_log=True))
        iteration_dir = tmp_path / "iter_debug"

        result = runner.run("keep", iteration_dir)
        runner.cleanup(iteration_dir, keep=result.returncode != 0 or result.timed_out)

        assert iteration_dir.exists()
