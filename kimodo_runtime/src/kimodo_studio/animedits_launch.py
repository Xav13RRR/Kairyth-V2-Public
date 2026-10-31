# SPDX-FileCopyrightText: Copyright (c) 2026 MKA Project contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .config import ANIMEDITS_PORT, PROJECT_ROOT, is_pipeline_mode

KIMODO_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
if not KIMODO_PYTHON.is_file():
    KIMODO_PYTHON = Path(sys.executable)

ANIMEDITS_URL = f"http://127.0.0.1:{ANIMEDITS_PORT}/"


class AnimEditsLaunchState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class AnimEditsLaunchJob:
    state: AnimEditsLaunchState = AnimEditsLaunchState.IDLE
    message: str = ""
    error: str | None = None
    url: str = ""
    bundle_path: str = ""
    pid: int = 0
    pipeline_mode: bool = False


_launch_job = AnimEditsLaunchJob()
_launch_lock = threading.Lock()


def _logger() -> logging.Logger:
    logger = logging.getLogger("kimodo_studio.animedits_launch")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def animedits_launch_status() -> dict:
    with _launch_lock:
        return {
            "state": _launch_job.state.value,
            "message": _launch_job.message,
            "error": _launch_job.error,
            "url": _launch_job.url,
            "bundle_path": _launch_job.bundle_path,
            "pid": _launch_job.pid,
            "pipeline_mode": _launch_job.pipeline_mode,
        }


def stop_animedits_processes() -> None:
    from animedits.lifecycle import stop_animedits_processes as _stop

    _stop()


def spawn_animedits(bundle_path: Path) -> subprocess.Popen:
    logger = _logger()
    if not KIMODO_PYTHON.is_file():
        raise FileNotFoundError(f"Python not found: {KIMODO_PYTHON}")

    stop_animedits_processes()

    env = os.environ.copy()
    env["ANIMEDITS_BUNDLE"] = str(bundle_path.resolve())
    env["ANIMEDITS_PORT"] = str(ANIMEDITS_PORT)
    cmd = [
        str(KIMODO_PYTHON),
        "-m",
        "animedits",
        "--bundle",
        str(bundle_path.resolve()),
        "--port",
        str(ANIMEDITS_PORT),
        "--no-browser",
    ]
    creationflags = 0
    if sys.platform == "win32" and not is_pipeline_mode():
        creationflags = subprocess.CREATE_NEW_CONSOLE
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        creationflags=creationflags,
    )
    logger.info("Spawned AnimEdits pid=%s bundle=%s url=%s", proc.pid, bundle_path, ANIMEDITS_URL)
    return proc


def start_animedits_launch(session) -> None:
    from .anim_export import export_animedits_bundle

    pipeline = is_pipeline_mode()
    with _launch_lock:
        if _launch_job.state == AnimEditsLaunchState.RUNNING:
            raise RuntimeError("AnimEdits 正在启动中")
        _launch_job.state = AnimEditsLaunchState.RUNNING
        _launch_job.message = "导出动画 bundle…"
        _launch_job.error = None
        _launch_job.url = ""
        _launch_job.bundle_path = ""
        _launch_job.pid = 0
        _launch_job.pipeline_mode = pipeline

    def _run() -> None:
        global _launch_job
        try:
            bundle_path = export_animedits_bundle(session)
            if pipeline:
                with _launch_lock:
                    _launch_job.state = AnimEditsLaunchState.DONE
                    _launch_job.message = "动画 bundle 已导出，等待流水线交接"
                    _launch_job.bundle_path = str(bundle_path)
                    _launch_job.url = "/edit/"
                return

            with _launch_lock:
                _launch_job.message = "启动 AnimEdits 服务…"
                _launch_job.bundle_path = str(bundle_path)
            proc = spawn_animedits(bundle_path)
            with _launch_lock:
                _launch_job.state = AnimEditsLaunchState.DONE
                _launch_job.message = "AnimEdits 已启动"
                _launch_job.url = ANIMEDITS_URL
                _launch_job.pid = int(proc.pid or 0)
        except Exception as exc:
            with _launch_lock:
                _launch_job.state = AnimEditsLaunchState.ERROR
                _launch_job.error = str(exc)
                _launch_job.message = str(exc)[:400]

    threading.Thread(target=_run, name="kimodo-studio-animedits-launch", daemon=True).start()
