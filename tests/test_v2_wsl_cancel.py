"""No real WSL, model, or GPU: local Bash + mocked process-group signals."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from motionbuilder_app import stores

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "motionbuilder_app/wsl_job_runner.sh"
KILLER = ROOT / "motionbuilder_app/wsl_kill_job.sh"


def shell_path(path):
    path = Path(path).resolve()
    if os.name == "nt":
        return f"/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"
    return str(path)


@pytest.fixture
def local_bash():
    # Do not invoke Windows system bash.exe, which boots WSL and could touch a
    # user's old distro. Git Bash is sufficient for syntax and safe mock tests.
    if os.name == "nt":
        candidates = [Path("C:/Program Files/Git/bin/bash.exe"), Path("C:/Program Files/Git/usr/bin/bash.exe")]
        executable = next((path for path in candidates if path.is_file()), None)
    else:
        executable = shutil.which("bash")
    if not executable:
        pytest.skip("A non-WSL local Bash is not available")
    return str(executable)


def shell_run(bash, script, *arguments):
    return subprocess.run([bash, "--noprofile", "--norc", "-c", script, "synthetic-test", *arguments],
                          capture_output=True, text=True, timeout=10)


def test_shell_scripts_are_lf_and_parse(local_bash):
    for path in (RUNNER, KILLER):
        assert b"\r\n" not in path.read_bytes()
        result = subprocess.run([local_bash, "--noprofile", "--norc", "-n", shell_path(path)],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr


def test_late_runner_refuses_preexisting_cancel_before_child_launch(tmp_path, local_bash):
    pid_file, sentinel = tmp_path / "job/process.pid", tmp_path / "child-started"
    pid_file.parent.mkdir()
    marker = Path(str(pid_file) + ".cancel")
    marker.write_text("cancel")
    # setsid is deliberately unavailable/unneeded in this pre-launch path.
    result = shell_run(local_bash, 'source "$1" "$2" touch "$3"',
                       shell_path(RUNNER), shell_path(pid_file), shell_path(sentinel))
    assert result.returncode == 130, result.stderr
    assert marker.exists()
    assert not pid_file.exists()
    assert not sentinel.exists()


def test_killer_persists_intent_when_no_pid_was_published(tmp_path, local_bash):
    pid_file = tmp_path / "job/process.pid"
    result = shell_run(local_bash, 'source "$1" "$2"', shell_path(KILLER), shell_path(pid_file))
    assert result.returncode == 0, result.stderr
    assert Path(str(pid_file) + ".cancel").exists()
    assert not pid_file.exists()


def test_cancel_during_pid_publication_kills_group_and_marker_survives_next_stage(tmp_path, local_bash):
    pid_file, log = tmp_path / "job/process.pid", tmp_path / "signals.txt"
    marker = Path(str(pid_file) + ".cancel")
    # All signals are shell-function mocks; this never sends an actual signal to
    # any process. A printf hook makes cancellation deterministic at publication.
    driver = '''
TASK_TEST_MARKER="$3"
TASK_TEST_LOG="$4"
kill() { builtin printf '%s\n' "$*" >> "$TASK_TEST_LOG"; return 0; }
sleep() { :; }
setsid() { "$@"; }
printf() { builtin printf "$@"; : > "$TASK_TEST_MARKER"; }
source "$1" "$2" true
'''
    result = shell_run(local_bash, driver, shell_path(RUNNER), shell_path(pid_file), shell_path(marker), shell_path(log))
    assert result.returncode == 130, result.stderr
    signals = log.read_text()
    assert "-TERM -- -" in signals
    assert "-KILL -- -" in signals
    assert marker.exists()
    assert not pid_file.exists()
    next_stage = shell_run(local_bash, 'source "$1" "$2" false', shell_path(RUNNER), shell_path(pid_file))
    assert next_stage.returncode == 130
    assert marker.exists()
    assert log.read_text() == signals


def test_group_probe_targets_descendants_even_after_leader_exits(tmp_path, local_bash):
    pid_file, log = tmp_path / "process.pid", tmp_path / "signals.txt"
    pid_file.write_text("314159265")
    driver = '''
TASK_TEST_LOG="$3"
kill() { builtin printf '%s\n' "$*" >> "$TASK_TEST_LOG"; return 0; }
sleep() { :; }
source "$1" "$2"
'''
    result = shell_run(local_bash, driver, shell_path(KILLER), shell_path(pid_file), shell_path(log))
    assert result.returncode == 0, result.stderr
    signals = log.read_text().splitlines()
    assert "-TERM -- -314159265" in signals
    assert "-KILL -- -314159265" in signals
    assert "-0 -- -314159265" in signals
    assert "-0 314159265" not in signals  # leader-only probing misses grandchildren
    assert not pid_file.exists()
    assert Path(str(pid_file) + ".cancel").exists()


def test_new_job_does_not_inherit_cancel_marker_and_success_cleans_pid(tmp_path, local_bash):
    old_pid = tmp_path / "old-job/process.pid"
    old_pid.parent.mkdir()
    old_marker = Path(str(old_pid) + ".cancel")
    old_marker.write_text("cancel")
    new_pid = tmp_path / "new-job/process.pid"
    driver = '''
kill() { return 1; }
setsid() { "$@"; }
source "$1" "$2" true
'''
    result = shell_run(local_bash, driver, shell_path(RUNNER), shell_path(new_pid))
    assert result.returncode == 0, result.stderr
    assert not new_pid.exists()
    assert not Path(str(new_pid) + ".cancel").exists()
    assert old_marker.exists()


def test_windows_cancel_marks_before_wsl_kill_and_queued_cancel_starts_no_wsl(tmp_path, monkeypatch):
    monkeypatch.setattr(stores, "MOTIONS_ROOT", tmp_path / "motions")
    store = stores.MotionStore(None)
    for status in ("queued", "running"):
        root = stores.MOTIONS_ROOT / status
        root.mkdir()
        job = {"id": status, "status": status, "filename": "synthetic.mp4", "created_at": 1, "log": []}
        if status == "running":
            job["wsl_pid_file"] = str(root / "wsl-process.pid")
        store.jobs[status] = job
        marker = root / "wsl-process.pid.cancel"
        def check_marker(current):
            assert marker.is_file()
            assert current["status"] == "canceling"
        with mock.patch.object(store, "_kill_wsl_group", side_effect=check_marker) as kill:
            assert store.cancel(status)["status"] == "canceled"
        assert marker.exists()
        assert kill.call_count == (1 if status == "running" else 0)
