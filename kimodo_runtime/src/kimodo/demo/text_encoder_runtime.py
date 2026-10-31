# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import socket
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TEXT_ENCODER_PORT = int(os.environ.get("TEXT_ENCODER_PORT", "9550"))
TEXT_ENCODER_WAIT_SECONDS = int(os.environ.get("KIMODO_TEXT_ENCODER_WAIT", "1800"))

_text_encoder_proc: subprocess.Popen | None = None
_last_start_error: str = ""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pid_file() -> Path:
    return _project_root() / ".kimodo" / "run.pids.json"


def _hf_cache_dir() -> Path:
    return _project_root() / ".mka_cache" / "huggingface"


def _read_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and token.strip():
        return token.strip()
    project_token = _project_root() / ".mka_cache" / "hf_token"
    if project_token.is_file():
        token = project_token.read_text(encoding="utf-8").strip()
        if token:
            return token
    user_token = Path.home() / ".cache" / "huggingface" / "token"
    if user_token.is_file():
        token = user_token.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def _encoder_subprocess_env() -> dict[str, str]:
    root = _project_root()
    env = os.environ.copy()
    hf_home = _hf_cache_dir()
    hub_cache = hf_home / "hub"
    env["HF_HOME"] = str(hf_home)
    env["HF_HUB_CACHE"] = str(hub_cache)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    env["TRANSFORMERS_CACHE"] = str(hub_cache)
    env.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TEXT_ENCODER_TMP_FOLDER", str(root / ".kimodo" / "text_encoder"))
    token = _read_hf_token()
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token
    return env


def _summarize_encoder_error(stderr: str) -> str:
    text = stderr or ""
    if "GatedRepoError" in text or "meta-llama/Meta-Llama-3-8B-Instruct" in text:
        return (
            "无法访问受限模型 meta-llama/Meta-Llama-3-8B-Instruct。"
            "请在 Hugging Face 接受 Llama-3 许可，并将 token 写入 .mka_cache/hf_token "
            "或运行 hf auth login 后重新 prefetch。"
        )
    if "Please log in" in text or "401 Client Error" in text:
        return (
            "Hugging Face 未认证。请将 token 写入 .mka_cache/hf_token "
            "或运行 hf auth login。"
        )
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        return lines[-1][:500]
    return ""


def get_text_encoder_start_error() -> str:
    return _last_start_error


def port_listening(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _update_pid_file_text_encoder(pid: int) -> None:
    pid_path = _pid_file()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {}
    if pid_path.exists():
        try:
            payload = json.loads(pid_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload["text_encoder_pid"] = pid
    payload.setdefault("demo_pid", os.getpid())
    payload.setdefault("started_at", time.time())
    pid_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def start_text_encoder_subprocess() -> subprocess.Popen | None:
    global _text_encoder_proc, _last_start_error
    _last_start_error = ""
    if port_listening(TEXT_ENCODER_PORT):
        return None

    root = _project_root()
    python_exe = root / "venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = Path(sys.executable)

    env = _encoder_subprocess_env()
    _hf_cache_dir().mkdir(parents=True, exist_ok=True)
    log_dir = root / ".kimodo"
    log_dir.mkdir(parents=True, exist_ok=True)
    stderr_log = log_dir / "text_encoder_stderr.log"

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    stderr_handle = open(stderr_log, "w", encoding="utf-8", errors="replace")
    _text_encoder_proc = subprocess.Popen(
        [str(python_exe), "-m", "kimodo.scripts.run_text_encoder_server"],
        cwd=str(root),
        env=env,
        stderr=stderr_handle,
        stdout=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    _update_pid_file_text_encoder(_text_encoder_proc.pid)
    return _text_encoder_proc


def wait_for_text_encoder(proc: subprocess.Popen | None = None, timeout: int | None = None) -> bool:
    global _last_start_error
    deadline = time.time() + (timeout if timeout is not None else TEXT_ENCODER_WAIT_SECONDS)
    url = f"http://127.0.0.1:{TEXT_ENCODER_PORT}/"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            stderr = ""
            log_path = _project_root() / ".kimodo" / "text_encoder_stderr.log"
            if log_path.is_file():
                stderr = log_path.read_text(encoding="utf-8", errors="replace")
            summary = _summarize_encoder_error(stderr)
            _last_start_error = summary or "text-encoder 子进程异常退出。"
            return False
        if port_listening(TEXT_ENCODER_PORT) and _http_ok(url):
            os.environ.setdefault("TEXT_ENCODER_URL", url)
            return True
        time.sleep(2)
    _last_start_error = f"text-encoder 在 {timeout or TEXT_ENCODER_WAIT_SECONDS}s 内未就绪。"
    return False


def ensure_text_encoder_running() -> bool:
    """Start and wait for the text-encoder service if it is not already up."""
    global _last_start_error
    if port_listening(TEXT_ENCODER_PORT):
        os.environ.setdefault("TEXT_ENCODER_URL", f"http://127.0.0.1:{TEXT_ENCODER_PORT}/")
        return True
    try:
        from mka.hf_env import llama_base_cache_error

        cache_err = llama_base_cache_error()
        if cache_err:
            _last_start_error = cache_err
            return False
    except ImportError:
        pass
    proc = start_text_encoder_subprocess()
    return wait_for_text_encoder(proc)


def stop_text_encoder_subprocess() -> None:
    """Stop the text-encoder subprocess started from this process or recorded in the pid file."""
    global _text_encoder_proc

    if _text_encoder_proc is not None and _text_encoder_proc.poll() is None:
        _text_encoder_proc.terminate()
        try:
            _text_encoder_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _text_encoder_proc.kill()
            _text_encoder_proc.wait(timeout=5)
    _text_encoder_proc = None

    pid_path = _pid_file()
    if not pid_path.exists():
        return
    try:
        payload = json.loads(pid_path.read_text(encoding="utf-8"))
        te_pid = int(payload.get("text_encoder_pid") or 0)
    except (OSError, ValueError, json.JSONDecodeError):
        return
    if te_pid <= 0 or te_pid == os.getpid():
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(te_pid), "/F", "/T"],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.kill(te_pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
