# SPDX-FileCopyrightText: Copyright (c) 2026 MKA Project contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import PROJECT_ROOT, SHUTDOWN_GRACE_S

PID_FILE = PROJECT_ROOT / ".kimodo" / "run.pids.json"

_shutdown_event = threading.Event()
_shutdown_timer: threading.Timer | None = None


def shutdown_event() -> threading.Event:
    return _shutdown_event


def write_studio_pid(studio_pid: int | None = None) -> None:
    pid = studio_pid if studio_pid is not None else os.getpid()
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {}
    if PID_FILE.exists():
        try:
            payload = json.loads(PID_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload["studio_pid"] = pid
    payload["demo_pid"] = pid
    payload.setdefault("started_at", time.time())
    PID_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_pid_file() -> None:
    if PID_FILE.exists():
        PID_FILE.unlink()


def _kill_pid(pid: int, label: str) -> None:
    if pid <= 0 or pid == os.getpid():
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass


def stop_background_processes() -> None:
    from kimodo.demo.text_encoder_runtime import stop_text_encoder_subprocess

    stop_text_encoder_subprocess()

    if PID_FILE.exists():
        try:
            payload = json.loads(PID_FILE.read_text(encoding="utf-8"))
            for key in ("text_encoder_pid", "studio_pid", "demo_pid"):
                pid = int(payload.get(key) or 0)
                if pid and pid != os.getpid():
                    _kill_pid(pid, key)
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    clear_pid_file()


def cancel_shutdown_timer() -> None:
    global _shutdown_timer
    if _shutdown_timer is not None:
        _shutdown_timer.cancel()
        _shutdown_timer = None


def request_shutdown() -> None:
    if _shutdown_event.is_set():
        return
    _shutdown_event.set()
    stop_background_processes()


def schedule_shutdown(*, had_client: bool, client_count: int) -> None:
    from .config import is_pipeline_mode

    if is_pipeline_mode():
        return
    global _shutdown_timer

    if not had_client or client_count > 0:
        return

    def _fire() -> None:
        request_shutdown()

    cancel_shutdown_timer()
    _shutdown_timer = threading.Timer(SHUTDOWN_GRACE_S, _fire)
    _shutdown_timer.daemon = True
    _shutdown_timer.start()
