from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .edit_tracks import EditOperation, EditProject
from .meshpreps.binding_io import binding_preview_dict, load_binding
from .retarget import build_source_motion_bundle
from .backend_config import gvhmr_backend, require_backend
from .release_profile import require_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
CHARACTERS_ROOT = DATA_ROOT / "characters"
MOTIONS_ROOT = DATA_ROOT / "motions"
GVHMR_ROOT = PROJECT_ROOT / "GVHMR"
WSL_DISTRO = os.environ.get("MOTIONBUILDER_WSL_DISTRO", "Kairyth-V2-GVHMR")
WSL_PYTHON = os.environ.get("MOTIONBUILDER_WSL_PYTHON", "/home/gvhmr/miniforge3/envs/gvhmr/bin/python")
DETECTOR_PYTHON = os.environ.get("MOTIONBUILDER_DETECTOR_PYTHON", "/home/gvhmr/miniforge3/envs/kairyth-detector/bin/python")
DETECTOR_ROOT = PROJECT_ROOT / "detector_tool"


def windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class CharacterStore:
    def __init__(self) -> None:
        CHARACTERS_ROOT.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _with_ik_status(manifest: dict[str, Any]) -> dict[str, Any]:
        result = dict(manifest)
        if result.get("static_asset") or result.get("asset_kind") == "static_mesh":
            result["has_ik"] = False
            return result
        if "has_ik" in result:
            return result
        root = CHARACTERS_ROOT / Path(str(result.get("id", ""))).name
        binding = root / str(result.get("binding", ""))
        try:
            with np.load(binding, allow_pickle=False) as data:
                result["has_ik"] = "ik_profile_json" in data.files
        except (OSError, ValueError):
            result["has_ik"] = False
        return result

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in CHARACTERS_ROOT.glob("*/manifest.json"):
            try:
                rows.append(self._with_ik_status(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: row.get("updated_at", 0), reverse=True)

    def get(self, character_id: str) -> dict[str, Any]:
        path = CHARACTERS_ROOT / Path(character_id).name / "manifest.json"
        if not path.is_file():
            raise KeyError(character_id)
        return self._with_ik_status(json.loads(path.read_text(encoding="utf-8")))

    def binding_path(self, character_id: str) -> Path:
        manifest = self.get(character_id)
        if manifest.get("static_asset") or manifest.get("asset_kind") == "static_mesh":
            raise FileNotFoundError("static asset has no binding")
        path = CHARACTERS_ROOT / Path(character_id).name / manifest["binding"]
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def static_preview_path(self, character_id: str) -> Path:
        manifest = self.get(character_id)
        if not (manifest.get("static_asset") or manifest.get("asset_kind") == "static_mesh"):
            raise FileNotFoundError("not a static asset")
        path = CHARACTERS_ROOT / Path(character_id).name / manifest["preview"]
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def rename(self, character_id: str, name: str) -> dict[str, Any]:
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("角色名称不能为空")
        manifest = self.get(character_id)
        manifest["name"] = clean_name[:80]
        manifest["updated_at"] = time.time()
        root = CHARACTERS_ROOT / Path(character_id).name
        _json_write(root / "manifest.json", manifest)
        return manifest

    def export_path(self, character_id: str, suffix: str) -> Path:
        self.get(character_id)
        root = CHARACTERS_ROOT / Path(character_id).name / "exports"
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{Path(character_id).name}_binding.{suffix.lstrip('.')}"

    def create(self, name: str, source_binding: Path) -> dict[str, Any]:
        binding = load_binding(source_binding)
        if len(binding.joint_names) != 30:
            raise ValueError("仅支持 SOMA-30 角色资产")
        character_id = uuid.uuid4().hex[:12]
        root = CHARACTERS_ROOT / character_id
        version_path = root / "versions" / "v1" / "binding.npz"
        version_path.parent.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source_binding, version_path)
        now = time.time()
        has_ik = binding.ik_profile is not None
        manifest = {
            "id": character_id,
            "name": (name or "未命名角色").strip()[:80],
            "version": 1,
            "binding": "versions/v1/binding.npz",
            "vertices": int(len(binding.vertices)),
            "faces": int(len(binding.faces)),
            "joints": int(len(binding.joint_names)),
            "has_ik": has_ik,
            "ik_solver": binding.ik_profile.get("solver") if has_ik else None,
            "ik_profile_version": binding.ik_profile.get("version") if has_ik else None,
            "created_at": now,
            "updated_at": now,
        }
        _json_write(root / "manifest.json", manifest)
        return manifest

    def create_static(self, name: str, filename: str, data: bytes) -> dict[str, Any]:
        from .meshpreps.mesh_util import load_mesh, preview_mesh

        if not data:
            raise ValueError("model file is empty")
        suffix = Path(filename or "model.obj").suffix.lower()
        if suffix not in {".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl"}:
            raise ValueError("unsupported static model format")
        character_id = uuid.uuid4().hex[:12]
        root = CHARACTERS_ROOT / character_id
        static_dir = root / "static"
        static_dir.mkdir(parents=True, exist_ok=False)
        safe_name = Path(filename or f"static_asset{suffix}").name
        source_path = static_dir / safe_name
        try:
            source_path.write_bytes(data)
            mesh = load_mesh(source_path)
            preview_vertices, preview_faces = preview_mesh(mesh)
            preview_path = static_dir / "preview_mesh.npz"
            np.savez_compressed(
                preview_path,
                vertices=preview_vertices.astype(np.float32),
                faces=preview_faces.astype(np.int64),
            )
            now = time.time()
            clean_name = (name or Path(filename or "Static Asset").stem or "Static Asset").strip()[:80]
            manifest = {
                "id": character_id,
                "name": clean_name,
                "version": 1,
                "asset_kind": "static_mesh",
                "static_asset": True,
                "model": f"static/{safe_name}",
                "preview": "static/preview_mesh.npz",
                "vertices": int(mesh.vertices.shape[0]),
                "faces": int(mesh.faces.shape[0]),
                "joints": 0,
                "created_at": now,
                "updated_at": now,
            }
            _json_write(root / "manifest.json", manifest)
            return manifest
        except Exception as exc:
            shutil.rmtree(root, ignore_errors=True)
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"static model import failed: {exc}") from exc

    def preview(self, character_id: str) -> dict[str, Any]:
        manifest = self.get(character_id)
        if manifest.get("static_asset") or manifest.get("asset_kind") == "static_mesh":
            with np.load(self.static_preview_path(character_id), allow_pickle=False) as data:
                return {
                    "static_asset": True,
                    "kind": "static_mesh",
                    "vertices": data["vertices"].astype(np.float32).tolist(),
                    "faces": data["faces"].astype(np.int64).tolist(),
                    "n_verts": int(data["vertices"].shape[0]),
                    "n_faces": int(data["faces"].shape[0]),
                    "fps": 30,
                    "num_frames": 1,
                }
        return binding_preview_dict(load_binding(self.binding_path(character_id)))

    def auto_ik(self, character_id: str) -> dict[str, Any]:
        from .ik.profile import build_humanoid_ik_profile
        from .meshpreps.binding_io import save_binding

        manifest = self.get(character_id)
        if manifest.get("static_asset") or manifest.get("asset_kind") == "static_mesh":
            raise ValueError("static mesh assets do not support humanoid IK")
        binding_path = self.binding_path(character_id)
        binding = load_binding(binding_path)
        profile = build_humanoid_ik_profile(binding)
        binding.ik_profile = profile
        save_binding(binding_path, binding)
        manifest["has_ik"] = True
        manifest["ik_solver"] = profile["solver"]
        manifest["ik_profile_version"] = profile["version"]
        manifest["updated_at"] = time.time()
        root = CHARACTERS_ROOT / Path(character_id).name
        _json_write(root / "manifest.json", manifest)
        return manifest

    def delete(self, character_id: str) -> None:
        root = (CHARACTERS_ROOT / Path(character_id).name).resolve()
        if root.parent != CHARACTERS_ROOT.resolve() or not root.is_dir():
            raise KeyError(character_id)
        shutil.rmtree(root)


class MotionStore:
    def __init__(self, characters: CharacterStore) -> None:
        self.characters = characters
        self.lock = threading.RLock()
        self.gpu_lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.threads: dict[str, threading.Thread] = {}
        MOTIONS_ROOT.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _restore(self) -> None:
        for manifest_path in MOTIONS_ROOT.glob("*/manifest.json"):
            try:
                job = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            changed = False
            # Historical manifests may contain absolute paths from an older
            # deployment. A motion owns its files inside its own project folder.
            input_name = Path(str(job.get("input_path") or "")).name
            if input_name and input_name != ".":
                input_path = manifest_path.parent / input_name
                if job.get("input_path") != str(input_path):
                    job["input_path"] = str(input_path)
                    changed = True
            if job.get("pt_path"):
                result_name = Path(str(job.get("result_dir") or "")).name
                if not result_name or result_name == ".":
                    result_name = Path(input_name).stem
                pt_path = manifest_path.parent / "gvhmr" / result_name / "hmr4d_results.pt"
                if job["pt_path"] != str(pt_path):
                    job["pt_path"] = str(pt_path)
                    changed = True
            if job.get("result_dir"):
                result_name = Path(str(job["result_dir"])).name
                result_dir = manifest_path.parent / "gvhmr" / result_name
                if job["result_dir"] != str(result_dir):
                    job["result_dir"] = str(result_dir)
                    changed = True
            if job.get("wsl_pid_file"):
                pid_file = manifest_path.parent / "wsl-process.pid"
                if job["wsl_pid_file"] != str(pid_file):
                    job["wsl_pid_file"] = str(pid_file)
                    changed = True
            if job.get("status") in {"queued", "running", "canceling"}:
                job.update(status="interrupted", stage="上次运行被中断", error="服务退出时任务尚未完成")
                changed = True
            source_bundle = manifest_path.parent / "source_smplx22.npz"
            if source_bundle.is_file() and not job.get("source_bundle"):
                job["source_bundle"] = source_bundle.name
                changed = True
            if changed:
                _json_write(manifest_path, job)
            self.jobs[job["id"]] = job

    def _save(self, job: dict[str, Any]) -> None:
        job["updated_at"] = time.time()
        _json_write(MOTIONS_ROOT / job["id"] / "manifest.json", job)

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in job.items() if key not in {"input_path", "result_dir"}}
        result["video_url"] = f"/api/motions/{job['id']}/video"
        result["has_scene"] = bool(job.get("source_bundle") or job.get("current_bundle"))
        result["edit_count"] = len(self.edit_project(job).operations) if job.get("current_bundle") else 0
        return result

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

    def create(
        self,
        filename: str,
        stream: BinaryIO,
        *,
        root_mode: str,
        static_camera: bool,
        focal_length: int | None,
    ) -> dict[str, Any]:
        require_inference("gvhmr")
        if root_mode not in {"world", "locked"}:
            raise ValueError("根运动模式必须是 world 或 locked")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}:
            raise ValueError("不支持的视频格式")
        motion_id = uuid.uuid4().hex[:12]
        root = MOTIONS_ROOT / motion_id
        root.mkdir(parents=True, exist_ok=False)
        input_path = root / f"input{suffix}"
        with input_path.open("wb") as output:
            shutil.copyfileobj(stream, output, length=4 * 1024 * 1024)
        if input_path.stat().st_size > 8 * 1024 * 1024 * 1024:
            shutil.rmtree(root)
            raise ValueError("视频不能超过 8GB")
        now = time.time()
        job = {
            "id": motion_id,
            "filename": Path(filename).name,
            "status": "queued",
            "stage": "等待 GPU",
            "progress": 1,
            "error": None,
            "log": [],
            "static_camera": bool(static_camera),
            "root_mode": root_mode,
            "focal_length": focal_length,
            "input_path": str(input_path),
            "current_bundle": None,
            "source_bundle": None,
            "base_bundles": {},
            "revisions": [],
            "hidden_from_list": False,
            "created_at": now,
            "updated_at": now,
        }
        with self.lock:
            self.jobs[motion_id] = job
            self._save(job)
        worker_thread = threading.Thread(target=self._run, args=(motion_id,), daemon=True, name=f"motion-{motion_id}")
        with self.lock:
            self.threads[motion_id] = worker_thread
        worker_thread.start()
        return self.public(job)

    def _log(self, job: dict[str, Any], line: str) -> None:
        clean = line.strip()
        if not clean:
            return
        job["log"].append(clean[-1000:])
        job["log"] = job["log"][-120:]
        stages = [
            ("YoloV8 Tracking", "人物检测与跟踪", 12), ("ViTPose", "人体关键点估计", 30),
            ("HMR2 Feature", "人体特征提取", 48), ("[Preprocess] End", "预处理完成", 62),
            ("[HMR4D] Predicting", "生成世界动作", 75), ("Rendering Global", "生成 GVHMR 预览", 88),
            ("[Merge Videos]", "合成原始预览", 92),
        ]
        for marker, stage, progress in stages:
            if marker in clean and progress > job["progress"]:
                job.update(stage=stage, progress=progress)

    @staticmethod
    def _camera_solver_failed(job: dict[str, Any]) -> bool:
        tail = "\n".join(job.get("log", [])[-40:]).lower()
        return "solver_two_view.py" in tail and "nonetype" in tail and "matrix" in tail

    def _run_process(self, job: dict[str, Any], command: list[str], cwd: Path) -> None:
        # Cancellation and process creation/registration are one atomic transition.
        # Otherwise cancel() could miss a just-started detector or inference child.
        with self.lock:
            require_inference("gvhmr")
            if job.get("status") in {"canceling", "canceled"}:
                raise InterruptedError("任务已取消，未启动下一阶段")
            process = subprocess.Popen(
                command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self.processes[job["id"]] = process
        code = -1
        try:
            assert process.stdout is not None
            for line in process.stdout:
                with self.lock:
                    self._log(job, line)
            code = process.wait()
        finally:
            with self.lock:
                self.processes.pop(job["id"], None)
        if code != 0:
            raise RuntimeError(f"子进程退出代码 {code}")

    def _wsl_command(self, job: dict[str, Any], working_dir: Path, arguments: list[str]) -> list[str]:
        pid_file = MOTIONS_ROOT / job["id"] / "wsl-process.pid"
        job["wsl_pid_file"] = str(pid_file)
        return [
            "wsl.exe", "-d", job.get("backend_distro", WSL_DISTRO), "--cd", windows_to_wsl(working_dir), "--", "bash",
            windows_to_wsl(APP_ROOT / "wsl_job_runner.sh"), windows_to_wsl(pid_file), *arguments,
        ]

    def _kill_wsl_group(self, job: dict[str, Any]) -> None:
        pid_file = Path(job.get("wsl_pid_file", MOTIONS_ROOT / job["id"] / "wsl-process.pid"))
        deadline = time.time() + 1.5
        while not pid_file.is_file() and time.time() < deadline:
            time.sleep(0.05)
        try:
            subprocess.run(
                ["wsl.exe", "-d", job.get("backend_distro", WSL_DISTRO), "--", "bash", windows_to_wsl(APP_ROOT / "wsl_kill_job.sh"), windows_to_wsl(pid_file)],
                timeout=8, check=False, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _is_canceled(self, job: dict[str, Any]) -> bool:
        with self.lock:
            return job.get("status") in {"canceling", "canceled"}

    def _run(self, motion_id: str) -> None:
        try:
            job = self.get(motion_id)
        except KeyError:
            with self.lock:
                self.threads.pop(motion_id, None)
            return
        root = MOTIONS_ROOT / motion_id
        try:
            with self.gpu_lock:
                require_inference("gvhmr")
                with self.lock:
                    if job["status"] in {"canceling", "canceled"}:
                        return
                    job.update(status="running", stage="启动 GVHMR", progress=3)
                    self._save(job)
                input_path = Path(job["input_path"])
                output_root = root / "gvhmr"
                result_dir = output_root / input_path.stem
                backend = gvhmr_backend(default_root=GVHMR_ROOT, default_python=WSL_PYTHON,
                    default_distro=WSL_DISTRO, default_detector_python=DETECTOR_PYTHON, default_detector_root=DETECTOR_ROOT)
                require_backend(backend.status(cached_pose=(result_dir / "preprocess/vitpose.pt").is_file(),
                    cached_features=(result_dir / "preprocess/vit_features.pt").is_file()))
                job["backend_distro"] = backend.distro
                python_args = ["env", f"PYTHONPATH={windows_to_wsl(backend.root)}", "PYTHONDONTWRITEBYTECODE=1", backend.python, "-B"]
                tracking_file = result_dir / "preprocess" / "person_tracks.json"
                demo_args = [
                    *python_args,
                    windows_to_wsl(backend.root / "tools/demo/demo.py"), f"--video={windows_to_wsl(input_path)}",
                    f"--output_root={windows_to_wsl(output_root)}",
                ]
                # Prepare uses exactly GVHMR's frame-by-frame 30fps copy, not a
                # detector-side resampling of the uploaded source video.
                prepare = self._wsl_command(job, backend.root, [*demo_args, "--prepare_only"])
                self._run_process(job, prepare, PROJECT_ROOT)
                if self._is_canceled(job):
                    return
                with self.lock:
                    job.update(stage="独立人物检测与跟踪", progress=10)
                    self._save(job)
                detect = self._wsl_command(job, backend.detector_root, [
                    backend.detector_python, "-B", windows_to_wsl(backend.detector_root / "cli.py"),
                    f"--video={windows_to_wsl(result_dir / '0_input_video.mp4')}",
                    f"--weights={windows_to_wsl(backend.root / 'inputs/checkpoints/yolo/yolov8x.pt')}",
                    f"--output={windows_to_wsl(tracking_file)}",
                ])
                # Full process exit is required before GVHMR starts: no retained
                # YOLO object/CUDA context is shared with the inference runtime.
                self._run_process(job, detect, PROJECT_ROOT)
                if self._is_canceled(job):
                    return
                command = self._wsl_command(job, backend.root, [
                    *demo_args, f"--tracking_file={windows_to_wsl(tracking_file)}",
                ])
                if job["static_camera"]:
                    command.append("-s")
                if job["focal_length"]:
                    command.append(f"--f_mm={job['focal_length']}")
                try:
                    self._run_process(job, command, PROJECT_ROOT)
                except RuntimeError:
                    if self._is_canceled(job):
                        return
                    if job["static_camera"] or not self._camera_solver_failed(job):
                        raise
                    with self.lock:
                        job["camera_fallback"] = True
                        job.update(stage="动态相机求解失败，自动改用静态相机", progress=50)
                        self._log(job, "Camera solver failed; retrying this job with static-camera mode.")
                        self._save(job)
                    self._run_process(job, [*command, "-s"], PROJECT_ROOT)
                if self._is_canceled(job):
                    return
                pt = result_dir / "hmr4d_results.pt"
                if not pt.is_file():
                    raise FileNotFoundError(pt)

                with self.lock:
                    job.update(stage="提取 SMPL-X22 动作", progress=94, pt_path=str(pt), result_dir=str(result_dir))
                    self._save(job)
                extracted = root / "smplx22_motion.npz"
                extract_cmd = self._wsl_command(job, backend.root, [
                    *python_args,
                    windows_to_wsl(backend.extractor), f"--pt={windows_to_wsl(pt)}",
                    f"--video={windows_to_wsl(input_path)}", f"--output={windows_to_wsl(extracted)}",
                ])
                self._run_process(job, extract_cmd, PROJECT_ROOT)
                if self._is_canceled(job):
                    return
                source_bundle = root / "source_smplx22.npz"
                build_source_motion_bundle(extracted, source_bundle, root_mode=job["root_mode"])
                with self.lock:
                    if job.get("status") in {"canceling", "canceled"}:
                        return
                    job["source_bundle"] = source_bundle.name
                    job.update(status="completed", stage="GVHMR 原生动作已生成", progress=100)
                    self._save(job)
        except Exception as exc:
            with self.lock:
                if job["status"] not in {"canceling", "canceled"}:
                    job.update(status="error", stage="处理失败", error=str(exc))
                    self._save(job)
        finally:
            with self.lock:
                self.threads.pop(motion_id, None)

    def cancel(self, motion_id: str) -> dict[str, Any]:
        job = self.get(motion_id)
        with self.lock:
            status = job.get("status")
            if status not in {"queued", "running", "canceling"}:
                return self.public(job)
            # Persist intent *before* asking WSL to stop. A cold-started runner
            # may not have published its PID yet; it must refuse a late launch.
            pid_file = Path(job.get("wsl_pid_file", MOTIONS_ROOT / job["id"] / "wsl-process.pid"))
            cancel_file = Path(str(pid_file) + ".cancel")
            cancel_file.parent.mkdir(parents=True, exist_ok=True)
            cancel_file.write_text("cancel\n", encoding="ascii")
            was_running = status in {"running", "canceling"}
            if status != "canceling":
                job.update(status="canceling", stage="正在终止任务", hidden_from_list=False)
            process = self.processes.get(motion_id)
            worker_thread = self.threads.get(motion_id)
            self._save(job)
        if process is not None or job.get("wsl_pid_file"):
            self._kill_wsl_group(job)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if was_running and worker_thread is not None and worker_thread is not threading.current_thread():
            worker_thread.join(timeout=20)
            if worker_thread.is_alive():
                raise RuntimeError("视频动作任务未能在超时内停止")
        with self.lock:
            job.update(status="canceled", stage="任务已取消", hidden_from_list=True)
            self._save(job)
        return self.public(job)

    def shutdown(self) -> None:
        with self.lock:
            ids = [motion_id for motion_id, job in self.jobs.items() if job.get("status") in {"queued", "running", "canceling"}]
        for motion_id in ids:
            try:
                self.cancel(motion_id)
            except KeyError:
                pass

    def delete(self, motion_id: str) -> None:
        job = self.get(motion_id)
        if job.get("status") in {"queued", "running", "canceling"}:
            self.cancel(motion_id)
        root = (MOTIONS_ROOT / Path(motion_id).name).resolve()
        if root.parent != MOTIONS_ROOT.resolve() or not root.is_dir():
            raise KeyError(motion_id)
        shutil.rmtree(root)
        with self.lock:
            self.jobs.pop(motion_id, None)

    def bundle_path(self, job: dict[str, Any]) -> Path:
        name = job.get("current_bundle")
        if not name:
            raise FileNotFoundError("motion bundle not ready")
        path = MOTIONS_ROOT / job["id"] / Path(name).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def source_bundle_path(self, job: dict[str, Any]) -> Path:
        name = job.get("source_bundle")
        if not name:
            legacy = MOTIONS_ROOT / job["id"] / "smplx22_motion.npz"
            if legacy.is_file():
                generated = MOTIONS_ROOT / job["id"] / "source_smplx22.npz"
                legacy_mode = "locked" if job.get("root_mode") == "locked" else "world"
                build_source_motion_bundle(legacy, generated, root_mode=legacy_mode)
                job["source_bundle"] = generated.name
                self._save(job)
                name = generated.name
        if not name:
            raise FileNotFoundError("source motion not ready")
        path = MOTIONS_ROOT / Path(job["id"]).name / Path(name).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def edit_project(self, job: dict[str, Any]) -> EditProject:
        if not job.get("current_bundle"):
            return EditProject(MOTIONS_ROOT / job["id"] / "edits.json", 1, 30)
        with np.load(self.bundle_path(job), allow_pickle=False) as data:
            frames = int(data["world_motion"].shape[0])
            joints = int(data["world_motion"].shape[1])
        return EditProject(MOTIONS_ROOT / job["id"] / "edits.json", frames, joints)

    def effective_world(self, job: dict[str, Any]) -> tuple[np.lib.npyio.NpzFile, np.ndarray]:
        data = np.load(self.bundle_path(job), allow_pickle=False)
        project = self.edit_project(job)
        world = project.apply(data["world_motion"], data["parents"]) if project.operations else data["world_motion"]
        return data, world

    def bake(self, motion_id: str) -> dict[str, Any]:
        job = self.get(motion_id)
        project = self.edit_project(job)
        if not project.operations:
            return self.public(job)
        data, world = self.effective_world(job)
        revision = len(job["revisions"]) + 1
        name = f"revision_{revision}.npz"
        output = MOTIONS_ROOT / motion_id / name
        payload = {key: data[key] for key in data.files}
        data.close()
        payload["world_motion"] = world.astype(np.float32)
        np.savez_compressed(output, **payload)
        with self.lock:
            job["revisions"].append(job["current_bundle"])
            job["current_bundle"] = name
            self._save(job)
        project.clear()
        return self.public(job)

    def undo_bake(self, motion_id: str) -> dict[str, Any]:
        job = self.get(motion_id)
        with self.lock:
            if not job["revisions"]:
                raise ValueError("没有可撤销的烘焙版本")
            job["current_bundle"] = job["revisions"].pop()
            self._save(job)
        self.edit_project(job).clear()
        return self.public(job)

    def switch_root_mode(self, motion_id: str, root_mode: str) -> dict[str, Any]:
        job = self.get(motion_id)
        if root_mode not in job.get("base_bundles", {}):
            raise ValueError("root mode not available")
        if self.edit_project(job).operations:
            raise ValueError("切换根运动前请先烘焙或清除当前编辑")
        with self.lock:
            job["root_mode"] = root_mode
            job["current_bundle"] = job["base_bundles"][root_mode]
            job["revisions"] = []
            self._save(job)
        return self.public(job)
