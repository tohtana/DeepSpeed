# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import deepspeed.comm as dist


@dataclass
class AgentRunnerConfig:
    command: list[str]
    timeout_sec: int
    debug_log: bool
    terminate_grace_sec: int = 10


@dataclass
class AgentRunResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    prompt_path: str
    stdout_path: str
    stderr_path: str


class AgentRunner:

    def __init__(self, config: AgentRunnerConfig):
        self.config = config

    def run(self, prompt: str, iteration_dir: Path) -> AgentRunResult:
        iteration_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = iteration_dir / "prompt.txt"
        stdout_path = iteration_dir / "stdout.txt"
        stderr_path = iteration_dir / "stderr.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"DeepCompile agent command: {self.config.command}")

        timed_out = False
        proc = None
        with prompt_path.open("rb") as prompt_f, stdout_path.open("wb") as stdout_f, stderr_path.open(
                "wb") as stderr_f:
            proc = subprocess.Popen(self.config.command, stdin=prompt_f, stdout=stdout_f, stderr=stderr_f, shell=False)
            try:
                proc.wait(timeout=self.config.timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(timeout=self.config.terminate_grace_sec)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        return AgentRunResult(stdout=stdout,
                              stderr=stderr,
                              returncode=proc.returncode,
                              timed_out=timed_out,
                              prompt_path=str(prompt_path),
                              stdout_path=str(stdout_path),
                              stderr_path=str(stderr_path))

    def cleanup(self, iteration_dir: Path, keep: bool) -> None:
        if keep or self.config.debug_log or not iteration_dir.exists():
            return
        shutil.rmtree(iteration_dir, ignore_errors=True)
