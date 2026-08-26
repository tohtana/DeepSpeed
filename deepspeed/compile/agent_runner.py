# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import signal
import shutil
import subprocess
import time
from typing import List, Optional

import deepspeed.comm as dist


@dataclass
class AgentRunnerConfig:
    command: List[str]
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

    def run(self, prompt: str, iteration_dir: Path, role: Optional[str] = None,
            artifact_prefix: Optional[str] = None) -> AgentRunResult:
        iteration_dir.mkdir(parents=True, exist_ok=True)

        prefix = artifact_prefix if artifact_prefix is not None else role
        prefix = f"{prefix}_" if prefix else ""
        prompt_path = iteration_dir / f"{prefix}prompt.txt"
        stdout_path = iteration_dir / f"{prefix}stdout.txt"
        stderr_path = iteration_dir / f"{prefix}stderr.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"DeepCompile agent command: {self.config.command}")

        child_env = os.environ.copy()
        if role is not None:
            child_env["DEEPCOMPILE_AGENT_ROLE"] = role

        timed_out = False
        proc = None
        with prompt_path.open("rb") as prompt_f, stdout_path.open("wb") as stdout_f, stderr_path.open(
                "wb") as stderr_f:
            proc = subprocess.Popen(self.config.command,
                                    stdin=prompt_f,
                                    stdout=stdout_f,
                                    stderr=stderr_f,
                                    shell=False,
                                    env=child_env,
                                    start_new_session=True)
            try:
                proc.wait(timeout=self.config.timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                grace_deadline = time.monotonic() + self.config.terminate_grace_sec
                try:
                    proc.wait(timeout=self.config.terminate_grace_sec)
                except subprocess.TimeoutExpired:
                    pass
                remaining_grace = grace_deadline - time.monotonic()
                if remaining_grace > 0:
                    time.sleep(remaining_grace)
                # The wrapper may have exited while a Codex descendant ignored SIGTERM.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
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
