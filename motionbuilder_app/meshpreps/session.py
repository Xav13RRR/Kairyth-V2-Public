from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np

from .bind_math import build_bind_matrices
from .bind_pipeline import run_skeleton, run_skin, run_skin_on_skeleton
from .binding_io import load_binding, save_binding
from .config import BINDING_METHOD, DEFAULT_BIND_TIMEOUT_S, DEFAULT_SEED, OUTPUT_DIR, WORK_DIR
from .mesh_align import apply_matrix4, resolve_align_matrix, suggest_align_transform
from .mesh_util import MeshData, load_mesh, preview_mesh
from .soma_binding import build_soma30_binding_from_skin, fit_soma_from_unirig_skel, soma_edit_preview_dict


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class JobPhase(str, Enum):
    PENDING = "pending"
    SKELETON = "skeleton"
    SOMA_EDIT = "soma_edit"
    SKIN = "skin"
    COMPLETE = "complete"


@dataclass
class MeshSession:
    mesh_id: str
    upload_path: Path
    base_vertices: np.ndarray
    base_faces: np.ndarray
    base_preview_vertices: np.ndarray
    base_preview_faces: np.ndarray
    preview_vertices: np.ndarray
    preview_faces: np.ndarray
    mesh: MeshData
    align_confirmed: bool = False
    mesh_transform: np.ndarray | None = None


@dataclass
class BindJob:
    job_id: str
    mesh_id: str
    state: JobState = JobState.PENDING
    phase: JobPhase = JobPhase.PENDING
    message: str = ""
    progress: int = 0
    automatic: bool = False
    work_dir: Path | None = None
    output_path: Path | None = None
    error: str | None = None
    skel_cache: dict | None = None
    soma_joints: np.ndarray | None = None
    soma_bind_matrices: np.ndarray | None = None
    soma_names: list[str] | None = None
    soma_parents: np.ndarray | None = None
    _thread: threading.Thread | None = field(default=None, repr=False)
    _automation_thread: threading.Thread | None = field(default=None, repr=False)


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._meshes: dict[str, MeshSession] = {}
        self._jobs: dict[str, BindJob] = {}
        self._active_procs: list = []
        self._shutdown_hooks: list[Callable[[], None]] = []

    def register_shutdown_hook(self, hook: Callable[[], None]) -> None:
        self._shutdown_hooks.append(hook)

    def kill_active_procs(self) -> None:
        import subprocess

        with self._lock:
            procs = list(self._active_procs)
        for proc in procs:
            try:
                if proc.poll() is None:
                    proc.kill()
            except (OSError, ProcessLookupError):
                pass
        with self._lock:
            self._active_procs.clear()

    def on_shutdown(self) -> None:
        self.kill_active_procs()
        for hook in self._shutdown_hooks:
            try:
                hook()
            except Exception:
                pass

    def add_mesh(self, upload_path: Path) -> MeshSession:
        mesh = load_mesh(upload_path)
        preview_vertices, preview_faces = preview_mesh(mesh)
        mesh_id = uuid.uuid4().hex[:12]
        session = MeshSession(
            mesh_id=mesh_id,
            upload_path=upload_path,
            base_vertices=mesh.vertices.copy(),
            base_faces=mesh.faces.copy(),
            base_preview_vertices=preview_vertices.copy(),
            base_preview_faces=preview_faces.copy(),
            preview_vertices=preview_vertices.copy(),
            preview_faces=preview_faces.copy(),
            mesh=MeshData(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy(),
                source_path=mesh.source_path,
            ),
        )
        with self._lock:
            self._meshes[mesh_id] = session
        return session

    def suggest_mesh_align(self, mesh_id: str) -> np.ndarray:
        session = self.get_mesh(mesh_id)
        if session is None:
            raise KeyError(mesh_id)
        return suggest_align_transform(session.base_vertices)

    def apply_mesh_align(
        self,
        mesh_id: str,
        *,
        matrix: np.ndarray | None = None,
        translation: np.ndarray | list[float] | None = None,
        quaternion: np.ndarray | list[float] | None = None,
        scale: float | None = None,
    ) -> MeshSession:
        session = self.get_mesh(mesh_id)
        if session is None:
            raise KeyError(mesh_id)
        from .mesh_align import validate_align_matrix

        resolved = resolve_align_matrix(
            matrix=matrix,
            translation=translation,
            quaternion=quaternion,
            scale=scale,
        )
        validate_align_matrix(resolved)
        aligned_vertices = apply_matrix4(session.base_vertices, resolved)
        session.mesh.vertices = aligned_vertices
        session.preview_vertices = apply_matrix4(session.base_preview_vertices, resolved)
        session.mesh_transform = resolved.copy()
        session.align_confirmed = True
        return session

    def reset_mesh_align(self, mesh_id: str) -> MeshSession:
        session = self.get_mesh(mesh_id)
        if session is None:
            raise KeyError(mesh_id)
        session.mesh.vertices = session.base_vertices.copy()
        session.preview_vertices = session.base_preview_vertices.copy()
        session.mesh_transform = None
        session.align_confirmed = False
        return session

    def get_mesh(self, mesh_id: str) -> MeshSession | None:
        with self._lock:
            return self._meshes.get(mesh_id)

    def mesh_align_status(self, mesh_id: str) -> dict | None:
        session = self.get_mesh(mesh_id)
        if session is None:
            return None
        return {
            "mesh_id": mesh_id,
            "align_confirmed": session.align_confirmed,
            "n_verts": int(session.mesh.vertices.shape[0]),
            "n_faces": int(session.mesh.faces.shape[0]),
            "source": session.upload_path.name,
        }

    def mesh_preview(self, mesh_id: str) -> dict | None:
        session = self.get_mesh(mesh_id)
        if session is None:
            return None
        return {
            "mesh_id": mesh_id,
            "vertices": session.preview_vertices.tolist(),
            "faces": session.preview_faces.tolist(),
            "n_verts": int(session.preview_vertices.shape[0]),
            "n_faces": int(session.preview_faces.shape[0]),
            "source": session.upload_path.name,
            "align_confirmed": session.align_confirmed,
        }

    def mesh_align_result(self, mesh_id: str) -> dict | None:
        status = self.mesh_align_status(mesh_id)
        if status is None:
            return None
        preview = self.mesh_preview(mesh_id)
        if preview is None:
            return None
        status["preview"] = {
            "vertices": preview["vertices"],
            "faces": preview["faces"],
        }
        return status

    def start_bind(self, mesh_id: str, *, seed: int = DEFAULT_SEED, timeout_s: float = DEFAULT_BIND_TIMEOUT_S) -> BindJob:
        session = self.get_mesh(mesh_id)
        if session is None:
            raise KeyError(mesh_id)
        if not session.align_confirmed:
            raise ValueError("mesh not aligned")
        job_id = uuid.uuid4().hex[:12]
        job = BindJob(job_id=job_id, mesh_id=mesh_id, state=JobState.PENDING, phase=JobPhase.PENDING)
        with self._lock:
            self._jobs[job_id] = job

        def _run() -> None:
            job.state = JobState.RUNNING
            job.phase = JobPhase.SKELETON
            job.progress = 8
            job.message = "UniRig 骨架绑定中…"
            work = WORK_DIR / job_id
            work.mkdir(parents=True, exist_ok=True)
            job.work_dir = work

            def on_log(msg: str) -> None:
                job.message = msg.splitlines()[-1][:200] if msg else job.message
                job.progress = min(45, max(12, job.progress + 1))

            try:
                with self._lock:
                    registry = self._active_procs
                skel = run_skeleton(
                    session.mesh,
                    work,
                    seed=seed,
                    timeout_s=timeout_s,
                    on_log=on_log,
                    proc_registry=registry,
                )
                job.skel_cache = skel
                job.state = JobState.DONE
                job.phase = JobPhase.SKELETON
                job.progress = 48
                job.message = "骨架绑定完成，请核对拟合并进入 SOMA 编辑"
            except Exception as exc:
                job.state = JobState.ERROR
                job.phase = JobPhase.PENDING
                job.error = str(exc)
                job.message = str(exc)[:400]

        thread = threading.Thread(target=_run, name=f"bind-{job_id}", daemon=True)
        job._thread = thread
        thread.start()
        return job

    def start_auto_bind(
        self,
        mesh_id: str,
        *,
        seed: int = DEFAULT_SEED,
        timeout_s: float = DEFAULT_BIND_TIMEOUT_S,
    ) -> BindJob:
        job = self.start_bind(mesh_id, seed=seed, timeout_s=timeout_s)
        job.automatic = True
        skeleton_thread = job._thread

        def _advance() -> None:
            try:
                if skeleton_thread is not None:
                    skeleton_thread.join()
                if job.state == JobState.ERROR:
                    return
                job.progress = 52
                job.message = "SOMA-30 自动映射与骨骼对齐中…"
                self.accept_skeleton(job.job_id)
                job.progress = 60
                job.message = "SOMA-30 对齐完成，正在启动 SkinTokens…"
                self.confirm_and_skin(job.job_id, timeout_s=timeout_s)
            except Exception as exc:
                job.state = JobState.ERROR
                job.error = str(exc)
                job.message = str(exc)[:400]

        automation = threading.Thread(target=_advance, name=f"auto-bind-{job.job_id}", daemon=True)
        job._automation_thread = automation
        automation.start()
        return job

    def _init_soma_from_skel(self, job: BindJob) -> None:
        if job.skel_cache is None:
            raise ValueError("skeleton data missing")
        joints, names, parents = fit_soma_from_unirig_skel(job.skel_cache)
        job.soma_joints = joints.copy()
        job.soma_names = names
        job.soma_parents = parents
        job.soma_bind_matrices = build_bind_matrices(joints, parents)

    def update_skeleton_joints(self, job_id: str, joints: np.ndarray) -> BindJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.phase != JobPhase.SKELETON or job.state != JobState.DONE:
            raise ValueError("skeleton not ready")
        if job.skel_cache is None:
            raise ValueError("skeleton data missing")
        arr = np.asarray(joints, dtype=np.float32)
        expected = np.asarray(job.skel_cache["joint_positions"], dtype=np.float32).shape
        if arr.shape != expected:
            raise ValueError(f"expected skeleton joints {expected}, got shape {arr.shape}")
        job.skel_cache = dict(job.skel_cache)
        job.skel_cache["joint_positions"] = arr.copy()
        job.message = "UniRig 骨架调整已保存，正在进入 SOMA 编辑"
        return job

    def accept_skeleton(self, job_id: str) -> BindJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.phase != JobPhase.SKELETON or job.state != JobState.DONE:
            raise ValueError("skeleton not ready")
        if job.skel_cache is None:
            raise ValueError("skeleton data missing")
        self._init_soma_from_skel(job)
        job.phase = JobPhase.SOMA_EDIT
        job.progress = max(job.progress, 55)
        job.message = "SOMA 骨架编辑：调整关节后确认蒙皮"
        return job

    def update_soma_joints(
        self,
        job_id: str,
        joints: np.ndarray,
        bind_matrices: np.ndarray | None = None,
    ) -> BindJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.phase != JobPhase.SOMA_EDIT:
            raise ValueError("not in soma edit phase")
        if job.soma_names is None:
            raise ValueError("soma skeleton missing")
        arr = np.asarray(joints, dtype=np.float32)
        if arr.shape != (len(job.soma_names), 3):
            raise ValueError(f"expected {len(job.soma_names)} joints, got shape {arr.shape}")
        job.soma_joints = arr.copy()
        if bind_matrices is not None:
            mats = np.asarray(bind_matrices, dtype=np.float32)
            if mats.shape != (len(job.soma_names), 4, 4):
                raise ValueError(
                    f"expected bind_matrices {(len(job.soma_names), 4, 4)}, got {mats.shape}"
                )
            job.soma_bind_matrices = mats.copy()
        elif job.soma_parents is not None:
            job.soma_bind_matrices = build_bind_matrices(arr, job.soma_parents)
        return job

    def confirm_and_skin(self, job_id: str, *, timeout_s: float = DEFAULT_BIND_TIMEOUT_S) -> BindJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.phase != JobPhase.SOMA_EDIT:
            raise ValueError("soma edit not ready")
        if job.work_dir is None or job.skel_cache is None:
            raise ValueError("skeleton data missing")
        if job.soma_joints is None or job.soma_names is None or job.soma_parents is None:
            raise ValueError("soma skeleton missing")
        if job.state == JobState.RUNNING:
            raise ValueError("job already running")

        job.state = JobState.RUNNING
        job.phase = JobPhase.SKIN
        job.progress = max(job.progress, 65)
        job.message = "SkinTokens 蒙皮中…"
        job.error = None

        soma_joints = job.soma_joints.copy()
        soma_names = list(job.soma_names)
        soma_parents = job.soma_parents.copy()
        soma_bind = job.soma_bind_matrices.copy() if job.soma_bind_matrices is not None else None

        def _run() -> None:
            out_dir = OUTPUT_DIR / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "binding.npz"

            def on_log(msg: str) -> None:
                job.message = msg.splitlines()[-1][:200] if msg else job.message
                job.progress = min(95, max(70, job.progress + 1))

            try:
                with self._lock:
                    registry = self._active_procs
                soma_skel = {
                    "vertices": job.skel_cache["vertices"],
                    "faces": job.skel_cache["faces"],
                    "joint_positions": soma_joints,
                    "joint_parents": soma_parents,
                    "joint_names": soma_names,
                }
                if soma_bind is not None:
                    soma_skel["bind_matrices"] = soma_bind
                skin = run_skin_on_skeleton(
                    job.work_dir,
                    soma_skel,
                    timeout_s=timeout_s,
                    on_log=on_log,
                    proc_registry=registry,
                )
                binding = build_soma30_binding_from_skin(
                    soma_skel,
                    skin,
                    method=BINDING_METHOD,
                    source_path=job.skel_cache.get("source_path"),
                )
                save_binding(out_path, binding)
                job.output_path = out_path
                job.state = JobState.DONE
                job.phase = JobPhase.COMPLETE
                job.progress = 100
                job.message = f"蒙皮完成：{out_path.name}"
            except Exception as exc:
                job.state = JobState.ERROR
                job.phase = JobPhase.SOMA_EDIT
                job.error = str(exc)
                job.message = str(exc)[:400]

        thread = threading.Thread(target=_run, name=f"skin-{job_id}", daemon=True)
        job._thread = thread
        thread.start()
        return job

    def get_job(self, job_id: str) -> BindJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def job_status(self, job_id: str) -> dict | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        return {
            "job_id": job.job_id,
            "mesh_id": job.mesh_id,
            "state": job.state.value,
            "phase": job.phase.value,
            "message": job.message,
            "error": job.error,
            "output": str(job.output_path) if job.output_path else None,
        }

    def skeleton_preview(self, job_id: str) -> dict | None:
        job = self.get_job(job_id)
        if job is None or job.skel_cache is None:
            return None
        from .skeleton_preview import skeleton_preview_dict

        preview = skeleton_preview_dict(job.skel_cache)
        preview["job_id"] = job_id
        return preview

    def soma_preview(self, job_id: str) -> dict | None:
        job = self.get_job(job_id)
        if job is None or job.skel_cache is None:
            return None
        if job.soma_joints is None or job.soma_names is None or job.soma_parents is None:
            return None
        preview = soma_edit_preview_dict(
            job.skel_cache,
            job.soma_joints,
            job.soma_names,
            job.soma_parents,
            bind_matrices=job.soma_bind_matrices,
            job_id=job_id,
        )
        return preview

    def binding_preview(self, job_id: str) -> dict | None:
        job = self.get_job(job_id)
        if job is None or job.phase != JobPhase.COMPLETE:
            return None
        if job.output_path is None or not job.output_path.is_file():
            return None
        from .binding_io import binding_preview_dict

        binding = load_binding(job.output_path)
        preview = binding_preview_dict(binding)
        preview["job_id"] = job_id
        preview["kind"] = "binding"
        preview["has_skin"] = True
        return preview

    def auto_ik(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.phase != JobPhase.COMPLETE or job.state != JobState.DONE:
            raise ValueError("binding not ready")
        if job.output_path is None or not job.output_path.is_file():
            raise ValueError("binding file missing")

        from ..ik.profile import build_humanoid_ik_profile

        binding = load_binding(job.output_path)
        profile = build_humanoid_ik_profile(binding)
        binding.ik_profile = profile
        save_binding(job.output_path, binding)
        job.message = "Two-Bone IK 绑定已完成"
        return {
            "ok": True,
            "has_ik": True,
            "profile": profile,
            "message": job.message,
        }

    def save_binding_weights(
        self,
        job_id: str,
        lbs_indices: np.ndarray,
        lbs_weights: np.ndarray,
    ) -> dict:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.phase != JobPhase.COMPLETE:
            raise ValueError("binding not ready")
        if job.output_path is None or not job.output_path.is_file():
            raise ValueError("binding file missing")

        from .binding_io import apply_lbs_to_binding, load_binding, save_binding, validate_lbs_arrays

        binding = load_binding(job.output_path)
        n_verts = int(binding.vertices.shape[0])
        n_joints = int(binding.joint_positions.shape[0])
        validate_lbs_arrays(lbs_indices, lbs_weights, n_verts=n_verts, n_joints=n_joints)
        updated = apply_lbs_to_binding(binding, lbs_indices, lbs_weights)
        save_binding(job.output_path, updated)
        job.message = "蒙皮权重已保存"
        return {"ok": True, "message": job.message}

    def complete_skin(
        self,
        job_id: str,
        lbs_indices: np.ndarray,
        lbs_weights: np.ndarray,
    ) -> dict:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.state != JobState.DONE:
            raise ValueError("job not finished")
        if job.phase != JobPhase.COMPLETE:
            raise ValueError("binding not ready")

        save_result = self.save_binding_weights(job_id, lbs_indices, lbs_weights)
        self.kill_active_procs()
        job.message = "蒙皮已保存，可加入角色资产库"
        return {
            **save_result,
            "binding": str(job.output_path),
            "message": job.message,
            "progress": job.progress,
            "automatic": job.automatic,
        }
