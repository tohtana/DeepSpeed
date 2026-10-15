# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import faulthandler
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback

import pytest

_DIAGNOSTICS_DIR_ENV = "DS_UNITTEST_DIAGNOSTICS_DIR"
_STALL_SECONDS_ENV = "DS_UNITTEST_DIAGNOSTICS_STALL_SECONDS"
_active_lock = threading.Lock()
_active = None
_fault_lock = threading.Lock()
_fault_handle = None
_fault_pid = None


def diagnostic_enabled():
    return bool(os.environ.get(_DIAGNOSTICS_DIR_ENV))


def _root():
    return Path(os.environ[_DIAGNOSTICS_DIR_ENV])


def _role():
    return os.environ.get("PYTEST_XDIST_WORKER", "controller")


def _identity():
    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "role": _role(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "rank": os.environ.get("RANK"),
        "world_size": os.environ.get("WORLD_SIZE"),
    }


def _ensure_faulthandler():
    global _fault_handle, _fault_pid
    pid = os.getpid()
    if _fault_pid == pid:
        return
    with _fault_lock:
        if _fault_pid == pid:
            return
        fault_path = _root() / "faulthandler" / f"{_role()}-{pid}.txt"
        fault_path.parent.mkdir(parents=True, exist_ok=True)
        _fault_handle = fault_path.open("a")
        _fault_pid = pid
        faulthandler.enable(file=_fault_handle, all_threads=True)
        try:
            faulthandler.register(signal.SIGUSR1, file=_fault_handle, all_threads=True, chain=False)
        except (AttributeError, OSError, RuntimeError):
            pass


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr) + "\n")
    os.replace(temporary, path)


def diagnostic_event(event, **payload):
    if not diagnostic_enabled():
        return
    try:
        _ensure_faulthandler()
        root = _root()
        path = root / "events" / f"{_role()}-{os.getpid()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event,
            "time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **_identity(),
            **payload,
        }
        line = json.dumps(record, sort_keys=True, default=repr) + "\n"
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, line.encode())
        finally:
            os.close(descriptor)
    except BaseException:
        pass


def diagnostic_pool_processes(pool):
    processes = []
    for process in getattr(pool, "_pool", ()):
        try:
            processes.append({
                "name": process.name,
                "pid": process.pid,
                "exitcode": process.exitcode,
                "alive": process.is_alive(),
            })
        except BaseException as error:
            processes.append({"error": repr(error)})
    return processes


def _set_active(nodeid, phase, item_started_ns=None):
    global _active
    now = time.monotonic_ns()
    state = {
        **_identity(),
        "nodeid": nodeid,
        "phase": phase,
        "item_started_monotonic_ns": item_started_ns or now,
        "phase_started_monotonic_ns": now,
        "updated_time_ns": time.time_ns(),
    }
    with _active_lock:
        _active = state
    if diagnostic_enabled():
        try:
            _write_json(_root() / "state" / f"{_role()}-{os.getpid()}.json", state)
        except BaseException:
            pass
    return state


def _clear_active(nodeid):
    global _active
    with _active_lock:
        if _active is None or _active.get("nodeid") == nodeid:
            _active = None
    if diagnostic_enabled():
        try:
            _write_json(_root() / "state" / f"{_role()}-{os.getpid()}.json", {
                **_identity(),
                "nodeid": None,
                "phase": "idle",
                "updated_time_ns": time.time_ns(),
            })
        except BaseException:
            pass


def _read_text(path, limit=256 * 1024):
    try:
        with path.open("r", errors="replace") as handle:
            return handle.read(limit)
    except BaseException as error:
        return f"<unavailable: {error!r}>\n"


def _capture_process_state(path):
    lines = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        path.write_text("/proc is unavailable on this platform\n")
        return
    for process_dir in sorted((entry for entry in proc_root.iterdir() if entry.name.isdigit()),
                              key=lambda entry: int(entry.name)):
        pid = process_dir.name
        status = _read_text(process_dir / "status", 64 * 1024)
        cmdline = _read_text(process_dir / "cmdline", 64 * 1024).replace("\0", " ")
        lines.extend((f"===== pid={pid} cmdline={cmdline} =====\n", status))
        for name in ("wchan", "syscall", "stack"):
            lines.append(f"--- {name} ---\n{_read_text(process_dir / name, 64 * 1024)}")
        task_root = process_dir / "task"
        try:
            task_dirs = sorted((entry for entry in task_root.iterdir() if entry.name.isdigit()),
                               key=lambda entry: int(entry.name))
        except BaseException:
            task_dirs = []
        for task_dir in task_dirs:
            lines.append(f"--- tid={task_dir.name} wchan ---\n{_read_text(task_dir / 'wchan', 16 * 1024)}")
            lines.append(f"--- tid={task_dir.name} syscall ---\n{_read_text(task_dir / 'syscall', 16 * 1024)}")
            lines.append(f"--- tid={task_dir.name} stack ---\n{_read_text(task_dir / 'stack', 64 * 1024)}")
    path.write_text("".join(lines))


def _capture_command(path, command):
    with path.open("w") as handle:
        try:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, timeout=20, check=False)
        except BaseException:
            traceback.print_exc(file=handle)


def _capture_active_map(path):
    states = []
    state_dir = _root() / "state"
    for state_path in sorted(state_dir.glob("*.json")):
        try:
            states.append(json.loads(state_path.read_text()))
        except BaseException as error:
            states.append({"path": str(state_path), "error": repr(error)})
    _write_json(path, states)


def diagnostic_snapshot(reason, **payload):
    if not diagnostic_enabled():
        return
    try:
        root = _root()
        timestamp = time.time_ns()
        label = f"{timestamp}-{_role()}-{os.getpid()}"
        snapshot_dir = root / "snapshots" / label
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        marker = {
            "reason": reason,
            "time_ns": timestamp,
            "monotonic_ns": time.monotonic_ns(),
            **_identity(),
            **payload,
        }
        _write_json(root / "markers" / f"{label}.json", marker)
        diagnostic_event("diagnostic_snapshot_start", reason=reason, snapshot=label, **payload)
        with (snapshot_dir / "python-stacks.txt").open("w") as handle:
            faulthandler.dump_traceback(file=handle, all_threads=True)
        _capture_active_map(snapshot_dir / "active-items.json")
        _capture_process_state(snapshot_dir / "process-state.txt")
        cgroup_payload = {}
        for name in ("memory.events", "memory.current", "memory.peak", "pids.current", "pids.events"):
            cgroup_payload[name] = _read_text(Path("/sys/fs/cgroup") / name, 64 * 1024)
        _write_json(snapshot_dir / "cgroup.json", cgroup_payload)
        _capture_command(
            snapshot_dir / "nvidia-smi-compute-apps.txt",
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader",
            ],
        )
        _capture_command(snapshot_dir / "nvidia-smi.txt", ["nvidia-smi", "-q", "-d", "MEMORY,UTILIZATION"])
        _capture_command(snapshot_dir / "kernel-log.txt", ["dmesg", "--ctime"])
        diagnostic_event("diagnostic_snapshot_end", reason=reason, snapshot=label)
    except BaseException as error:
        diagnostic_event("diagnostic_snapshot_error", reason=reason, error=repr(error))


class _DiagnosticsPlugin:

    def __init__(self):
        self._stop = threading.Event()
        self._item_started_ns = None
        self._triggered_state = None
        root = _root()
        root.mkdir(parents=True, exist_ok=True)
        _ensure_faulthandler()
        diagnostic_event("pytest_process_start", argv=list(os.sys.argv))
        self._watchdog = threading.Thread(target=self._watch_stalls, name="ds-unit-test-diagnostics", daemon=True)
        self._watchdog.start()

    def _watch_stalls(self):
        try:
            threshold = float(os.environ.get(_STALL_SECONDS_ENV, "180"))
        except ValueError:
            threshold = 180.0
        threshold = max(0.1, threshold)
        interval = min(1.0, max(0.05, threshold / 10.0))
        while not self._stop.wait(interval):
            with _active_lock:
                active = dict(_active) if _active is not None else None
            if active is None:
                continue
            state_key = (active["nodeid"], active["phase"], active["phase_started_monotonic_ns"])
            elapsed = (time.monotonic_ns() - active["phase_started_monotonic_ns"]) / 1_000_000_000
            if elapsed < threshold or state_key == self._triggered_state:
                continue
            self._triggered_state = state_key
            diagnostic_event("stalled_item", elapsed_seconds=elapsed, **active)
            diagnostic_snapshot("unchanged-item", elapsed_seconds=elapsed, **active)

    def _phase_start(self, item, phase):
        _set_active(item.nodeid, phase, self._item_started_ns)
        diagnostic_event("pytest_phase_start", nodeid=item.nodeid, phase=phase)

    def _phase_end(self, item, phase, outcome):
        excinfo = outcome.excinfo
        diagnostic_event(
            "pytest_phase_end",
            nodeid=item.nodeid,
            phase=phase,
            exception_type=getattr(excinfo[0], "__name__", None) if excinfo else None,
            exception=repr(excinfo[1]) if excinfo else None,
        )
        _set_active(item.nodeid, "between-phases", self._item_started_ns)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        self._item_started_ns = time.monotonic_ns()
        self._triggered_state = None
        _set_active(item.nodeid, "protocol", self._item_started_ns)
        diagnostic_event("pytest_item_start", nodeid=item.nodeid, nextitem=getattr(nextitem, "nodeid", None))
        outcome = yield
        excinfo = outcome.excinfo
        diagnostic_event(
            "pytest_item_end",
            nodeid=item.nodeid,
            exception_type=getattr(excinfo[0], "__name__", None) if excinfo else None,
            exception=repr(excinfo[1]) if excinfo else None,
        )
        _clear_active(item.nodeid)
        self._item_started_ns = None

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_setup(self, item):
        self._phase_start(item, "setup")
        outcome = yield
        self._phase_end(item, "setup", outcome)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_call(self, item):
        self._phase_start(item, "call")
        outcome = yield
        self._phase_end(item, "call", outcome)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_teardown(self, item, nextitem):
        self._phase_start(item, "teardown")
        outcome = yield
        self._phase_end(item, "teardown", outcome)

    def pytest_runtest_logreport(self, report):
        diagnostic_event(
            "pytest_phase_report",
            nodeid=report.nodeid,
            phase=report.when,
            outcome=report.outcome,
            duration=report.duration,
            worker_id=getattr(report, "worker_id", None),
        )

    def pytest_xdist_node_collection_finished(self, node, ids):
        diagnostic_event("pytest_worker_collection", worker=node.gateway.id, count=len(ids))

    def pytest_testnodedown(self, node, error):
        diagnostic_event(
            "pytest_worker_down",
            worker=node.gateway.id,
            error=repr(error),
            workeroutput=getattr(node, "workeroutput", None),
        )

    def pytest_handlecrashitem(self, crashitem, report, sched):
        diagnostic_event("pytest_crash_item", crashitem=crashitem, report=repr(report), scheduler=type(sched).__name__)

    def pytest_sessionfinish(self, session, exitstatus):
        diagnostic_event(
            "pytest_session_finish",
            exitstatus=int(exitstatus),
            shouldfail=repr(session.shouldfail),
            shouldstop=repr(session.shouldstop),
        )
        self._stop.set()

    def pytest_unconfigure(self, config):
        diagnostic_event("pytest_process_end")
        self._stop.set()


def register_pytest_diagnostics(config):
    if not diagnostic_enabled():
        return
    config.pluginmanager.register(_DiagnosticsPlugin(), "deepspeed-unit-test-diagnostics")
