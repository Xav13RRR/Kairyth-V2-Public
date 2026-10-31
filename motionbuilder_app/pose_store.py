from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .backend_config import hybrik_backend, require_backend
from .pose_backend_client import write_request, start_worker, parse_event, validate_result, terminate_worker
from .skeleton.soma30 import SOMASkeleton30
from .release_profile import require_inference, unavailable_backend_status

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSES_ROOT = PROJECT_ROOT / "data" / "pose_assets"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
POSE_IMPORT_EXTENSIONS = {".npz", ".json"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _pose_constraints_from_world(world: np.ndarray, parents: np.ndarray) -> list[dict[str, Any]]:
    pose = np.asarray(world, dtype=np.float32)
    if pose.ndim == 3:
        pose = pose[None]
    if pose.ndim != 4 or pose.shape[0] != 1 or pose.shape[2:] != (4, 4):
        raise ValueError(f"Pose world_motion must be [30,4,4] or [1,30,4,4], got {pose.shape}")
    parents = np.asarray(parents, dtype=np.int64).reshape(-1)
    if len(parents) != pose.shape[1]:
        raise ValueError("Pose parent count does not match joint count")
    local_rots = []
    for index, parent in enumerate(parents):
        rot = pose[0, index, :3, :3]
        if parent >= 0:
            rot = pose[0, parent, :3, :3].T @ rot
        local_rots.append(rot)
    local_axis = Rotation.from_matrix(np.asarray(local_rots, dtype=np.float64)).as_rotvec().reshape(1, len(parents), 3).astype(np.float32)
    root_positions = pose[:, 0, :3, 3].astype(np.float32)
    return [
        {
            "type": "fullbody",
            "frame_indices": [0],
            "local_joints_rot": local_axis.tolist(),
            "root_positions": root_positions.tolist(),
            "smooth_root_2d": root_positions[:, [0, 2]].tolist(),
        }
    ]


def _soma_defaults() -> tuple[list[str], np.ndarray, np.ndarray]:
    skeleton = SOMASkeleton30(load=True)
    names = list(skeleton.bone_order_names)
    parents = np.asarray(skeleton.joint_parents, dtype=np.int64)
    positions = np.asarray(skeleton.neutral_joints, dtype=np.float32)
    bind = np.tile(np.eye(4, dtype=np.float32), (len(names), 1, 1))
    bind[:, :3, 3] = positions[: len(names)]
    return names, parents, bind


def _world_from_fullbody_constraint(payload: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    names, parents, bind = _soma_defaults()
    item = next((row for row in payload if isinstance(row, dict) and row.get("type") == "fullbody"), None)
    if not item:
        raise ValueError("JSON 姿势文件必须包含 type=fullbody 的 Kimodo constraint")
    local_axis = np.asarray(item.get("local_joints_rot"), dtype=np.float32)
    if local_axis.ndim == 3:
        local_axis = local_axis[0]
    if local_axis.shape != (len(names), 3):
        raise ValueError(f"fullbody local_joints_rot 需要是 [1,30,3] 或 [30,3]，当前为 {local_axis.shape}")
    roots = np.asarray(item.get("root_positions", [[0.0, 0.0, 0.0]]), dtype=np.float32)
    if roots.ndim == 1:
        roots = roots.reshape(1, 3)
    root = roots[0] if roots.size else np.zeros(3, dtype=np.float32)
    local_rots = Rotation.from_rotvec(local_axis.reshape(-1, 3)).as_matrix().astype(np.float32)
    neutral = bind[:, :3, 3].copy()
    local_offsets = neutral.copy()
    for index, parent in enumerate(parents):
        if parent >= 0:
            local_offsets[index] = neutral[index] - neutral[parent]
        else:
            local_offsets[index] = root
    world = np.tile(np.eye(4, dtype=np.float32), (1, len(names), 1, 1))
    for index, parent in enumerate(parents):
        local = np.eye(4, dtype=np.float32)
        local[:3, :3] = local_rots[index]
        local[:3, 3] = local_offsets[index]
        if parent >= 0:
            world[0, index] = world[0, parent] @ local
        else:
            world[0, index] = local
    return world.astype(np.float32), parents, bind.astype(np.float32), json.dumps(payload, ensure_ascii=False, indent=2)


class PoseStore:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.gpu_lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.processes = {}
        self.threads = {}
        self.closing = False
        POSES_ROOT.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _restore(self) -> None:
        for manifest in POSES_ROOT.glob("*/manifest.json"):
            try:
                row = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if row.get("status") in {"queued", "running"}:
                row.update(
                    status="interrupted",
                    stage="上次服务退出时图片 Pose 任务尚未完成",
                    error="任务被中断，请重新上传图片。",
                    progress=100,
                )
                _write_json(manifest, row)
            self.jobs[row["id"]] = row

    def _save(self, job: dict[str, Any]) -> None:
        job["updated_at"] = time.time()
        _write_json(POSES_ROOT / job["id"] / "manifest.json", job)

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in job.items() if key not in {"asset", "constraints", "raw", "image", "hidden_from_jobs"}}
        result["has_scene"] = bool(job.get("asset"))
        result["has_constraints"] = bool(job.get("constraints"))
        return result

    def status(self) -> dict[str, Any]:
        return unavailable_backend_status("hybrik") or hybrik_backend().status()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = [row for row in self.jobs.values() if not row.get("hidden_from_jobs")]
            return [self.public(row) for row in sorted(rows, key=lambda r: r.get("created_at", 0), reverse=True)]

    def list_library(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = [row for row in self.jobs.values() if row.get("confirmed") and row.get("status") == "completed"]
            return [self.public(row) for row in sorted(rows, key=lambda r: r.get("confirmed_at", r.get("created_at", 0)), reverse=True)]

    def get(self, pose_id: str) -> dict[str, Any]:
        with self.lock:
            if pose_id not in self.jobs:
                raise KeyError(pose_id)
            return self.jobs[pose_id]

    def root(self, pose_id: str) -> Path:
        root = (POSES_ROOT / Path(pose_id).name).resolve()
        if root.parent != POSES_ROOT.resolve() or not root.is_dir():
            raise KeyError(pose_id)
        return root

    def create(self, *, image_bytes: bytes, filename: str, name: str) -> dict[str, Any]:
        require_inference("hybrik")
        suffix = Path(filename or "image.png").suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            raise ValueError("仅支持 jpg/jpeg/png/webp/bmp 图片")
        pose_id = uuid.uuid4().hex[:12]
        root = POSES_ROOT / pose_id
        root.mkdir(parents=True, exist_ok=False)
        image_name = f"input{suffix}"
        image_path = root / image_name
        image_path.write_bytes(image_bytes)
        now = time.time()
        job = {
            "id": pose_id,
            "name": (name or Path(filename).stem or "HybrIK Pose").strip()[:80],
            "filename": filename or image_name,
            "image_name": image_name,
            "status": "queued",
            "stage": "等待 HybrIK 单图推理",
            "progress": 1,
            "error": None,
            "confirmed": False,
            "confirmed_at": None,
            "hidden_from_jobs": False,
            "asset": None,
            "constraints": None,
            "raw": None,
            "created_at": now,
            "updated_at": now,
        }
        with self.lock:
            if self.closing:
                raise ValueError("服务正在关闭")
            self.jobs[pose_id] = job
            self._save(job)
            thread = threading.Thread(target=self._run, args=(pose_id,), daemon=True, name=f"hybrik-pose-{pose_id}")
            self.threads[pose_id] = thread
            thread.start()
        return self.public(job)

    def import_file(self, *, file_bytes: bytes, filename: str, name: str) -> dict[str, Any]:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in POSE_IMPORT_EXTENSIONS:
            raise ValueError("仅支持导入 SOMA30 Pose .npz 或 Kimodo fullbody constraints .json")
        if not file_bytes:
            raise ValueError("姿势文件为空")
        pose_id = uuid.uuid4().hex[:12]
        root = POSES_ROOT / pose_id
        root.mkdir(parents=True, exist_ok=False)
        asset = root / "pose_soma30.npz"
        constraints = root / "constraints.json"
        source_name = filename or f"pose{suffix}"
        clean_name = (name or Path(source_name).stem or "Imported Pose").strip()[:80]
        constraints_text = ""
        if suffix == ".npz":
            with np.load(BytesIO(file_bytes), allow_pickle=False) as data:
                if "world_motion" not in data.files:
                    raise ValueError("NPZ 姿势文件缺少 world_motion")
                world = np.asarray(data["world_motion"], dtype=np.float32)
                if world.ndim == 3 and world.shape[-2:] == (4, 4):
                    world = world.reshape(1, world.shape[0], 4, 4)
                elif world.ndim == 4 and world.shape[-2:] == (4, 4):
                    world = world[:1]
                else:
                    raise ValueError(f"world_motion 需要是 [30,4,4] 或 [F,30,4,4]，当前为 {world.shape}")
                default_names, default_parents, default_bind = _soma_defaults()
                parents = np.asarray(data["parents"], dtype=np.int64).reshape(-1) if "parents" in data.files else default_parents
                names = np.asarray(data["names"], dtype="U64") if "names" in data.files else np.asarray(default_names, dtype="U64")
                target_bind = np.asarray(data["target_bind"], dtype=np.float32) if "target_bind" in data.files else default_bind
                if target_bind.ndim == 2 and target_bind.shape[1] == 16:
                    target_bind = target_bind.reshape(-1, 4, 4)
                if target_bind.ndim != 3 or target_bind.shape[-2:] != (4, 4):
                    raise ValueError(f"target_bind 需要是 [30,4,4] 或 [30,16]，当前为 {target_bind.shape}")
                if len(parents) != world.shape[1] or len(names) != world.shape[1] or target_bind.shape[0] != world.shape[1]:
                    raise ValueError("导入姿势的骨骼数量必须与 world_motion 一致")
                fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 30.0
                if "kimodo_constraints_json" in data.files:
                    constraints_text = str(np.asarray(data["kimodo_constraints_json"]).reshape(-1)[0])
                else:
                    constraints_text = json.dumps(_pose_constraints_from_world(world, parents), ensure_ascii=False, indent=2)
        else:
            try:
                payload = json.loads(file_bytes.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("JSON 姿势文件无法解析") from exc
            if not isinstance(payload, list):
                raise ValueError("JSON 姿势文件必须是 Kimodo constraints 数组")
            world, parents, target_bind, constraints_text = _world_from_fullbody_constraint(payload)
            names = np.asarray(_soma_defaults()[0], dtype="U64")
            fps = 30.0
        constraints.write_text(constraints_text, encoding="utf-8")
        np.savez_compressed(
            asset,
            world_motion=world.astype(np.float32),
            target_bind=target_bind.astype(np.float32),
            parents=np.asarray(parents, dtype=np.int64),
            names=np.asarray(names, dtype="U64"),
            fps=np.asarray([fps], dtype=np.float32),
            num_frames=np.asarray([1], dtype=np.int64),
            skeleton_type=np.asarray(["KIMODO-SOMA30-POSE"], dtype="U32"),
            source_kind=np.asarray(["imported_pose"], dtype="U32"),
            pose_name=np.asarray([clean_name], dtype=f"U{max(1, len(clean_name))}"),
            source_file_name=np.asarray([source_name], dtype=f"U{max(1, len(source_name))}"),
            kimodo_constraints_json=np.asarray([constraints_text], dtype=f"U{max(1, len(constraints_text))}"),
        )
        now = time.time()
        job = {
            "id": pose_id,
            "name": clean_name,
            "filename": source_name,
            "image_name": None,
            "status": "completed",
            "stage": "姿势文件已导入",
            "progress": 100,
            "error": None,
            "confirmed": True,
            "confirmed_at": now,
            "hidden_from_jobs": True,
            "asset": asset.name,
            "constraints": constraints.name,
            "raw": None,
            "frames": 1,
            "fps": fps,
            "joints": int(world.shape[1]),
            "created_at": now,
            "updated_at": now,
        }
        with self.lock:
            self.jobs[pose_id] = job
            self._save(job)
        return self.public(job)

    def _run(self, pose_id: str) -> None:
        job = self.get(pose_id)
        process = None
        try:
            with self.gpu_lock:
                require_inference("hybrik")
                backend = hybrik_backend()
                require_backend(backend.status())
                root = (POSES_ROOT / pose_id).resolve()
                with self.lock:
                    if self.closing or job["status"] == "cancelled": return
                    request = write_request(root, root / job["image_name"], backend,
                        name=job["name"], source_name=job.get("filename") or job["image_name"])
                    job.update(status="running", stage="HybrIK 独立研究进程推理中", progress=5, error=None)
                    self._save(job)
                    process = start_worker(root, backend, request)
                    self.processes[pose_id] = process
                assert process.stdout is not None
                error = None
                for line in process.stdout:
                    event = parse_event(line)
                    with self.lock:
                        if job["status"] == "cancelled": continue
                        if event and event["kind"] == "stage":
                            job.update(stage=event["message"], progress=min(99, event["progress"]))
                            self._save(job)
                        elif event and event["kind"] == "error": error = event["message"]
                        else:
                            job["log"] = [*job.get("log", []), line.strip()[:2000]][-60:]
                code = process.wait()
                if code != 0: raise RuntimeError(error or f"HybrIK worker exit {code}")
                result = validate_result(root)
                with self.lock:
                    if job["status"] == "cancelled" or self.closing: return
                    job.update(status="completed", stage="图片 Pose 已生成，等待确认导入 Pose 库", progress=100, **result)
                    self._save(job)
        except Exception as exc:
            with self.lock:
                if job["status"] != "cancelled":
                    job.update(status="failed", stage="图片 Pose 生成失败", progress=100, error=str(exc)[:1000])
                    self._save(job)
        finally:
            if process is not None and process.poll() is None: terminate_worker(process)
            with self.lock:
                self.processes.pop(pose_id, None)
                self.threads.pop(pose_id, None)

    def cancel(self, pose_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.get(pose_id)
            if job.get("status") not in {"queued", "running"}: return self.public(job)
            (POSES_ROOT / pose_id / "backend.cancel").write_text("cancel\n", encoding="ascii")
            job.update(status="cancelled", stage="图片 Pose 任务已取消", progress=100)
            self._save(job)
            process = self.processes.get(pose_id)
        if process is not None: terminate_worker(process)
        return self.public(job)

    def shutdown(self) -> None:
        with self.lock:
            self.closing = True
            active = [key for key, job in self.jobs.items() if job.get("status") in {"queued", "running"}]
        for key in active: self.cancel(key)
        for thread in list(self.threads.values()): thread.join(timeout=20)

    def confirm(self, pose_id: str, name: str | None = None) -> dict[str, Any]:
        job = self.get(pose_id)
        if job.get("status") != "completed" or not job.get("asset"):
            raise ValueError("Pose 尚未生成完成，不能导入 Pose 库")
        if name and name.strip():
            job["name"] = name.strip()[:80]
        job["confirmed"] = True
        job["confirmed_at"] = time.time()
        self._save(job)
        return self.public(job)

    def rename(self, pose_id: str, name: str) -> dict[str, Any]:
        job = self.get(pose_id)
        job["name"] = name.strip()[:80]
        self._save(job)
        return self.public(job)

    def create_edited(self, source_pose_id: str, *, name: str, world_motion: list[list[float]]) -> dict[str, Any]:
        source = self.get(source_pose_id)
        source_asset = self.asset_path(source)
        pose_id = uuid.uuid4().hex[:12]
        root = POSES_ROOT / pose_id
        root.mkdir(parents=True, exist_ok=False)
        asset = root / "pose_soma30.npz"
        constraints = root / "constraints.json"
        with np.load(source_asset, allow_pickle=False) as data:
            template = {key: data[key] for key in data.files if key not in {"world_motion", "kimodo_constraints_json", "pose_name", "source_kind", "num_frames", "parents", "names"}}
            parents = np.asarray(data["parents"], dtype=np.int64)
            names = np.asarray(data["names"])
            world = np.asarray(world_motion, dtype=np.float32)
            if world.ndim == 2 and world.shape[1] == 16:
                world = world.reshape(1, world.shape[0], 4, 4)
            elif world.ndim == 3 and world.shape[-2:] == (4, 4):
                world = world.reshape(1, world.shape[0], 4, 4)
            elif world.ndim == 4 and world.shape[0] == 1 and world.shape[-2:] == (4, 4):
                pass
            else:
                raise ValueError(f"Edited Pose world_motion has unsupported shape: {world.shape}")
            if world.shape[1] != len(parents):
                raise ValueError(f"Edited Pose joint count mismatch: expected {len(parents)}, got {world.shape[1]}")
            constraint = _pose_constraints_from_world(world, parents)
            constraints_text = json.dumps(constraint, ensure_ascii=False, indent=2)
            constraints.write_text(constraints_text, encoding="utf-8")
            clean_name = (name or f"{source.get('name', 'Pose')} Edited").strip()[:80]
            np.savez_compressed(
                asset,
                **template,
                world_motion=world.astype(np.float32),
                parents=parents,
                names=names,
                num_frames=np.asarray([1], dtype=np.int64),
                source_kind=np.asarray(["edited_pose"], dtype="U32"),
                pose_name=np.asarray([clean_name], dtype=f"U{max(1, len(clean_name))}"),
                kimodo_constraints_json=np.asarray([constraints_text], dtype=f"U{max(1, len(constraints_text))}"),
            )
        now = time.time()
        job = {
            "id": pose_id,
            "name": clean_name,
            "filename": f"Edited from {source.get('name') or source_pose_id}",
            "image_name": None,
            "status": "completed",
            "stage": "Pose 编辑已保存",
            "progress": 100,
            "error": None,
            "confirmed": True,
            "confirmed_at": now,
            "hidden_from_jobs": True,
            "asset": asset.name,
            "constraints": constraints.name,
            "raw": None,
            "frames": 1,
            "fps": float(template.get("fps", np.asarray([30.0], dtype=np.float32)).reshape(-1)[0]),
            "joints": int(world.shape[1]),
            "source_pose_id": source_pose_id,
            "created_at": now,
            "updated_at": now,
        }
        with self.lock:
            self.jobs[pose_id] = job
            self._save(job)
        return self.public(job)

    def create_from_motion_bundle(
        self,
        source_bundle: Path,
        *,
        source_kind: str,
        source_id: str,
        source_name: str,
        frame: int,
        name: str,
    ) -> dict[str, Any]:
        """Create an independent one-frame Pose asset from a completed motion bundle."""
        source_path = Path(source_bundle)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        pose_id = uuid.uuid4().hex[:12]
        root = POSES_ROOT / pose_id
        root.mkdir(parents=True, exist_ok=False)
        asset = root / "pose_soma30.npz"
        constraints = root / "constraints.json"
        try:
            with np.load(source_path, allow_pickle=False) as data:
                required = {"world_motion", "target_bind", "parents", "names"}
                missing = sorted(required.difference(data.files))
                if missing:
                    raise ValueError(f"Motion bundle is missing required fields: {', '.join(missing)}")
                source_world = np.asarray(data["world_motion"], dtype=np.float32)
                if source_world.ndim != 4 or source_world.shape[-2:] != (4, 4) or source_world.shape[0] < 1:
                    raise ValueError(f"Motion world_motion has unsupported shape: {source_world.shape}")
                frame_index = int(frame)
                if frame_index < 0 or frame_index >= source_world.shape[0]:
                    raise ValueError(f"Frame {frame_index} is outside motion range 0-{source_world.shape[0] - 1}")
                world = source_world[frame_index : frame_index + 1].copy()
                parents = np.asarray(data["parents"], dtype=np.int64).reshape(-1)
                names = np.asarray(data["names"], dtype="U64").reshape(-1)
                target_bind = np.asarray(data["target_bind"], dtype=np.float32)
                if target_bind.ndim == 2 and target_bind.shape[1] == 16:
                    target_bind = target_bind.reshape(-1, 4, 4)
                if target_bind.ndim != 3 or target_bind.shape[-2:] != (4, 4):
                    raise ValueError(f"Motion target_bind has unsupported shape: {target_bind.shape}")
                if len(parents) != world.shape[1] or len(names) != world.shape[1] or target_bind.shape[0] != world.shape[1]:
                    raise ValueError("Motion skeleton metadata does not match its world_motion joint count")
                fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 30.0
                default_skeleton = "KIMODO-SOMA30" if source_kind == "kimodo" else "SMPL-X22"
                skeleton_type = str(np.asarray(data["skeleton_type"]).reshape(-1)[0]) if "skeleton_type" in data.files else default_skeleton

            clean_name = (name or f"{source_name} Frame {frame_index + 1}").strip()[:80]
            if not clean_name:
                raise ValueError("Pose name cannot be empty")
            constraint = _pose_constraints_from_world(world, parents)
            constraints_text = json.dumps(constraint, ensure_ascii=False, indent=2)
            constraints.write_text(constraints_text, encoding="utf-8")
            np.savez_compressed(
                asset,
                world_motion=world.astype(np.float32),
                target_bind=target_bind.astype(np.float32),
                parents=parents,
                names=names,
                fps=np.asarray([fps], dtype=np.float32),
                num_frames=np.asarray([1], dtype=np.int64),
                skeleton_type=np.asarray([skeleton_type], dtype=f"U{max(1, len(skeleton_type))}"),
                source_kind=np.asarray([f"{source_kind}_motion_frame"], dtype="U32"),
                source_motion_id=np.asarray([source_id], dtype=f"U{max(1, len(source_id))}"),
                source_frame=np.asarray([frame_index], dtype=np.int64),
                pose_name=np.asarray([clean_name], dtype=f"U{max(1, len(clean_name))}"),
                kimodo_constraints_json=np.asarray([constraints_text], dtype=f"U{max(1, len(constraints_text))}"),
            )
            now = time.time()
            job = {
                "id": pose_id,
                "name": clean_name,
                "filename": f"Frame {frame_index + 1} from {source_name or source_id}",
                "image_name": None,
                "status": "completed",
                "stage": "Pose extracted from motion frame",
                "progress": 100,
                "error": None,
                "confirmed": True,
                "confirmed_at": now,
                "hidden_from_jobs": True,
                "asset": asset.name,
                "constraints": constraints.name,
                "raw": None,
                "frames": 1,
                "fps": fps,
                "joints": int(world.shape[1]),
                "source_kind": source_kind,
                "source_motion_id": source_id,
                "source_frame": frame_index,
                "created_at": now,
                "updated_at": now,
            }
            with self.lock:
                self.jobs[pose_id] = job
                self._save(job)
            return self.public(job)
        except Exception:
            with self.lock:
                self.jobs.pop(pose_id, None)
            shutil.rmtree(root, ignore_errors=True)
            raise

    def delete(self, pose_id: str) -> None:
        root = self.root(pose_id)
        shutil.rmtree(root)
        with self.lock:
            self.jobs.pop(pose_id, None)

    def delete_job(self, pose_id: str) -> None:
        job = self.get(pose_id)
        if job.get("confirmed"):
            job["hidden_from_jobs"] = True
            self._save(job)
            return
        if job.get("status") in {"queued", "running"}:
            self.cancel(pose_id)
            job.update(hidden_from_jobs=True)
            self._save(job)
            return
        self.delete(pose_id)

    def asset_path(self, job_or_id: dict[str, Any] | str) -> Path:
        job = self.get(job_or_id) if isinstance(job_or_id, str) else job_or_id
        name = job.get("asset")
        if not name:
            raise FileNotFoundError("Pose asset not ready")
        path = POSES_ROOT / Path(job["id"]).name / Path(name).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def constraints_path(self, job_or_id: dict[str, Any] | str) -> Path:
        job = self.get(job_or_id) if isinstance(job_or_id, str) else job_or_id
        name = job.get("constraints")
        if not name:
            raise FileNotFoundError("Pose constraints not ready")
        path = POSES_ROOT / Path(job["id"]).name / Path(name).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def image_path(self, job_or_id: dict[str, Any] | str) -> Path:
        job = self.get(job_or_id) if isinstance(job_or_id, str) else job_or_id
        path = POSES_ROOT / Path(job["id"]).name / Path(job.get("image_name", "")).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def constraints(self, pose_id: str) -> list[dict[str, Any]]:
        path = self.constraints_path(pose_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Pose constraints must be a list")
        return data
