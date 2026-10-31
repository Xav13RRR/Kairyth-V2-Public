from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .kimodo_authoring import normalize_preview_scene
from .kimodo_runtime import runtime_status

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KIMODO_MOTIONS_ROOT = PROJECT_ROOT / "data" / "kimodo_motions"


def _clean_constraints(raw: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    cleaned = json.loads(json.dumps(raw, ensure_ascii=False))
    if not isinstance(cleaned, list):
        raise ValueError("Kimodo constraints must be a list")
    for index, item in enumerate(cleaned):
        if not isinstance(item, dict):
            raise ValueError(f"Kimodo constraint #{index + 1} must be an object")
        if not str(item.get("type", "")).strip():
            raise ValueError(f"Kimodo constraint #{index + 1} has no type")
    return cleaned


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class KimodoMotionStore:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.gpu_lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.threads: dict[str, threading.Thread] = {}
        KIMODO_MOTIONS_ROOT.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _restore(self) -> None:
        for manifest in KIMODO_MOTIONS_ROOT.glob("*/manifest.json"):
            try:
                row = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if row.get("status") in {"queued", "running", "canceling"}:
                row.update(status="interrupted", stage="上次服务退出时任务尚未完成", error="任务被中断")
                _write_json(manifest, row)
            row.setdefault("schema_version", 1)
            row.setdefault("preview_scene", {"scene_objects": []})
            self.jobs[row["id"]] = row

    def _save(self, job: dict[str, Any]) -> None:
        job["updated_at"] = time.time()
        _write_json(KIMODO_MOTIONS_ROOT / job["id"] / "manifest.json", job)

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in job.items() if key not in {"bundle"}}
        result["has_scene"] = bool(job.get("bundle"))
        return result

    def status(self) -> dict[str, Any]:
        return runtime_status()

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = (row for row in self.jobs.values() if not row.get("hidden_from_list") and row.get("status") != "canceled")
            return [self.public(row) for row in sorted(rows, key=lambda r: r.get("created_at", 0), reverse=True)]

    def get(self, motion_id: str) -> dict[str, Any]:
        with self.lock:
            if motion_id not in self.jobs:
                raise KeyError(motion_id)
            return self.jobs[motion_id]

    def rename(self, motion_id: str, name: str) -> dict[str, Any]:
        clean_name = name.strip()[:80]
        if not clean_name:
            raise ValueError("动作名称不能为空")
        with self.lock:
            if motion_id not in self.jobs:
                raise KeyError(motion_id)
            job = self.jobs[motion_id]
            job["name"] = clean_name
            self._save(job)
            return self.public(job)

    def bundle_path(self, job_or_id: dict[str, Any] | str) -> Path:
        job = self.get(job_or_id) if isinstance(job_or_id, str) else job_or_id
        name = job.get("bundle")
        if not name:
            raise FileNotFoundError("Kimodo motion bundle not ready")
        path = KIMODO_MOTIONS_ROOT / Path(job["id"]).name / Path(name).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def create(
        self,
        *,
        prompt: str = "",
        segments: list[dict[str, Any]] | None = None,
        name: str,
        duration_sec: float = 6.0,
        seed: int = 12345,
        diffusion_steps: int = 100,
        model_name: str = "Kimodo-SOMA-RP-v1.1",
        constraints: list[dict[str, Any]] | None = None,
        preview_scene: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if any(row.get("status") in {"queued", "running", "canceling"} for row in self.jobs.values()):
                raise ValueError("已有 Kimodo 动作正在生成，请等待完成或先终止任务")
        clean_segments: list[dict[str, Any]] = []
        if segments:
            for segment in segments:
                text = str(segment.get("prompt", "")).strip()
                if not text:
                    continue
                clean_segments.append({"prompt": text, "duration_sec": max(0.25, min(30.0, float(segment.get("duration_sec", 2.0))))})
        else:
            clean_prompt = (prompt or "").strip()
            if clean_prompt:
                clean_segments.append({"prompt": clean_prompt, "duration_sec": max(2.0, min(10.0, float(duration_sec)))})
        if not clean_segments:
            raise ValueError("Kimodo prompt cannot be empty")
        prompt_list = [item["prompt"] for item in clean_segments]
        duration_list = [float(item["duration_sec"]) for item in clean_segments]
        summary_prompt = " / ".join(prompt_list)
        total_duration = float(sum(duration_list))
        clean_constraints = _clean_constraints(constraints)
        clean_preview_scene = normalize_preview_scene(preview_scene)
        job_id = uuid.uuid4().hex[:12]
        root = KIMODO_MOTIONS_ROOT / job_id
        root.mkdir(parents=True, exist_ok=False)
        now = time.time()
        job = {
            "schema_version": 2,
            "id": job_id,
            "name": (name or summary_prompt).strip()[:80],
            "prompt": summary_prompt[:1200],
            "prompts": prompt_list,
            "durations_sec": duration_list,
            "segments": clean_segments,
            "duration_sec": total_duration,
            "seed": int(seed),
            "diffusion_steps": int(diffusion_steps),
            "model_name": (model_name or "Kimodo-SOMA-RP-v1.1").strip(),
            "constraints": clean_constraints,
            "constraints_count": len(clean_constraints),
            "preview_scene": clean_preview_scene,
            "status": "queued",
            "stage": "等待 Kimodo 生成",
            "progress": 1,
            "error": None,
            "bundle": None,
            "hidden_from_list": False,
            "created_at": now,
            "updated_at": now,
        }
        with self.lock:
            if any(row.get("status") in {"queued", "running", "canceling"} for row in self.jobs.values()):
                shutil.rmtree(root, ignore_errors=True)
                raise ValueError("已有 Kimodo 动作正在生成，请等待完成或先终止任务")
            self.jobs[job_id] = job
            self._save(job)
        worker_thread = threading.Thread(target=self._run, args=(job_id,), daemon=True, name=f"kimodo-{job_id}")
        with self.lock:
            self.threads[job_id] = worker_thread
        worker_thread.start()
        return self.public(job)

    def update_preview_scene(self, motion_id: str, preview_scene: dict[str, Any] | None) -> dict[str, Any]:
        cleaned = normalize_preview_scene(preview_scene)
        with self.lock:
            if motion_id not in self.jobs:
                raise KeyError(motion_id)
            job = self.jobs[motion_id]
            job["preview_scene"] = cleaned
            self._save(job)
            return self.public(job)

    @staticmethod
    def _creation_flags() -> int:
        if os.name != "nt":
            return 0
        return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))

    def _run_worker(self, job: dict[str, Any]) -> tuple[int, str]:
        command = [sys.executable, "-m", "motionbuilder_app.kimodo_worker", job["id"]]
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=self._creation_flags(),
            start_new_session=os.name != "nt",
        )
        with self.lock:
            self.processes[job["id"]] = process
        try:
            output, _ = process.communicate()
            return int(process.returncode or 0), output[-12000:]
        finally:
            with self.lock:
                self.processes.pop(job["id"], None)

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    timeout=15,
                    check=False,
                    capture_output=True,
                    creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
                )
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                process.terminate()
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _run(self, job_id: str) -> None:
        try:
            job = self.get(job_id)
        except KeyError:
            with self.lock:
                self.threads.pop(job_id, None)
            return
        try:
            with self.gpu_lock:
                with self.lock:
                    if job.get("status") in {"canceling", "canceled"}:
                        return
                    job.update(status="running", stage="Kimodo 生成中", progress=10, error=None)
                    self._save(job)
                code, output = self._run_worker(job)
                with self.lock:
                    if job.get("status") in {"canceling", "canceled"}:
                        return
                if code != 0:
                    raise RuntimeError(output.strip() or f"Kimodo worker exited with code {code}")
                result_path = KIMODO_MOTIONS_ROOT / job_id / "worker_result.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                bundle = KIMODO_MOTIONS_ROOT / job_id / Path(str(result["bundle"])).name
                if not bundle.is_file():
                    raise FileNotFoundError(bundle)
                with self.lock:
                    if job.get("status") in {"canceling", "canceled"}:
                        return
                    job.update(
                        status="completed",
                        stage="Kimodo 原生动作完成",
                        progress=100,
                        bundle=bundle.name,
                        frames=int(result["frames"]),
                        fps=float(result["fps"]),
                    )
                    self._save(job)
        except Exception as exc:
            with self.lock:
                if job.get("status") not in {"canceling", "canceled"}:
                    error_text = str(exc).strip() or "Kimodo worker failed"
                    job.update(status="failed", stage="Kimodo 生成失败", progress=100, error=error_text[-6000:])
                    self._save(job)
        finally:
            with self.lock:
                self.threads.pop(job_id, None)

    def cancel(self, motion_id: str) -> dict[str, Any]:
        job = self.get(motion_id)
        with self.lock:
            status = job.get("status")
            if status not in {"queued", "running", "canceling"}:
                return self.public(job)
            was_running = status in {"running", "canceling"}
            if status != "canceling":
                job.update(status="canceling", stage="正在终止任务", hidden_from_list=False)
            process = self.processes.get(motion_id)
            worker_thread = self.threads.get(motion_id)
            self._save(job)
        if process is not None:
            self._terminate_process_tree(process)
        if was_running and worker_thread is not None and worker_thread is not threading.current_thread():
            worker_thread.join(timeout=20)
            if worker_thread.is_alive():
                raise RuntimeError("Kimodo 任务未能在超时内停止")
        with self.lock:
            job.update(status="canceled", stage="任务已取消", hidden_from_list=True)
            self._save(job)
        return self.public(job)

    def shutdown(self) -> None:
        with self.lock:
            ids = [job_id for job_id, job in self.jobs.items() if job.get("status") in {"queued", "running", "canceling"}]
        for motion_id in ids:
            try:
                self.cancel(motion_id)
            except KeyError:
                pass

    def delete(self, motion_id: str) -> None:
        job = self.get(motion_id)
        if job.get("status") in {"queued", "running", "canceling"}:
            self.cancel(motion_id)
        root = (KIMODO_MOTIONS_ROOT / Path(motion_id).name).resolve()
        if root.parent != KIMODO_MOTIONS_ROOT.resolve() or not root.is_dir():
            raise KeyError(motion_id)
        shutil.rmtree(root)
        with self.lock:
            self.jobs.pop(motion_id, None)
