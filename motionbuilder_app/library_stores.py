from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .edit_tracks import EditProject
from .ik_constraints import apply_ik_constraints
from .motion_math import local_from_world, world_from_local
from .retarget import retarget_kimodo_bundle_fixed, retarget_source_bundle_fixed
from .timeline import apply_ordered_cuts, blend_clip_boundary, resample_world, segment_world
from .virtual_root import build_virtual_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANIMATIONS_ROOT = PROJECT_ROOT / "data" / "animations"
COMPOSITIONS_ROOT = PROJECT_ROOT / "data" / "compositions"
RETARGET_PREVIEWS_ROOT = PROJECT_ROOT / "data" / "retarget_previews"
VIRTUAL_ROOT_PREVIEWS_ROOT = PROJECT_ROOT / "data" / "virtual_root_previews"
UPPER_LOWER_PREVIEWS_ROOT = PROJECT_ROOT / "data" / "upper_lower_previews"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _quat_to_matrix(rotation: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    q = np.asarray(rotation, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError("rotation must be [x, y, z, w]")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _apply_whole_animation_transforms(world: np.ndarray, parents: np.ndarray, adjustments: list[dict[str, Any]]) -> np.ndarray:
    if not adjustments:
        return np.asarray(world, dtype=np.float32)
    local = local_from_world(np.asarray(world, dtype=np.float64), np.asarray(parents, dtype=np.int64)).astype(np.float64)
    joints = local.shape[1]
    for item in adjustments:
        joint = int(item.get("joint", -1))
        if joint < 0 or joint >= joints:
            raise ValueError("joint out of range")
        translation = np.asarray(item.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
        if translation.size != 3 or not np.all(np.isfinite(translation)):
            raise ValueError("translation must be [x, y, z]")
        rot = _quat_to_matrix(item.get("rotation", [0, 0, 0, 1]))
        local[:, joint, :3, 3] += translation
        local[:, joint, :3, :3] = local[:, joint, :3, :3] @ rot
    return world_from_local(local, parents).astype(np.float32)


def _trim_animation_payload(payload: dict[str, np.ndarray], trim_start: int | None, trim_end: int | None) -> dict[str, np.ndarray]:
    frames = int(np.asarray(payload["world_motion"]).shape[0])
    start = 0 if trim_start is None else int(trim_start)
    end = frames if trim_end is None else int(trim_end)
    start = max(0, min(start, max(0, frames - 1)))
    end = max(start + 1, min(end, frames))
    if start == 0 and end == frames:
        return payload
    frame_keys = {
        "world_motion",
        "preview_world_motion",
        "virtual_root_motion",
        "root_motion",
        "hip_local_under_virtual_root",
        "virtual_root_ground_heights",
        "kimodo_smooth_root_motion",
    }
    for key in frame_keys:
        if key in payload:
            arr = np.asarray(payload[key])
            if arr.ndim >= 1 and arr.shape[0] == frames:
                payload[key] = arr[start:end]
    if "virtual_root_motion" in payload:
        root = np.asarray(payload["virtual_root_motion"], dtype=np.float32)
        payload["root_motion"] = (root[:, :3, 3] - root[[0], :3, 3]).astype(np.float32)
    elif "root_motion" in payload:
        root = np.asarray(payload["root_motion"], dtype=np.float32)
        if root.ndim == 2 and len(root):
            payload["root_motion"] = (root - root[[0]]).astype(np.float32)
    if "num_frames" in payload:
        payload["num_frames"] = np.asarray([end - start], dtype=np.int64)
    return payload


class AnimationStore:
    def __init__(self, characters, motions) -> None:
        self.characters = characters
        self.motions = motions
        ANIMATIONS_ROOT.mkdir(parents=True, exist_ok=True)

    def list(self, character_id: str | None = None, *, include_hidden: bool = False) -> list[dict[str, Any]]:
        rows = []
        for path in ANIMATIONS_ROOT.glob("*/manifest.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("hidden") and not include_hidden:
                    continue
                if character_id is None or row["character_id"] == character_id:
                    rows.append(self._with_root_motion_status(row, path.parent))
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: row.get("updated_at", 0), reverse=True)

    def get(self, animation_id: str) -> dict[str, Any]:
        path = ANIMATIONS_ROOT / Path(animation_id).name / "manifest.json"
        if not path.is_file():
            raise KeyError(animation_id)
        return self._with_root_motion_status(json.loads(path.read_text(encoding="utf-8")), path.parent)

    @staticmethod
    def _with_root_motion_status(row: dict[str, Any], root: Path) -> dict[str, Any]:
        result = dict(row)
        if result.get("has_root_motion") is True:
            return result
        bundle = root / Path(str(result.get("bundle", ""))).name
        result["has_root_motion"] = False
        try:
            with np.load(bundle, allow_pickle=False) as data:
                if "virtual_root_motion" in data.files or "root_motion" in data.files:
                    result["has_root_motion"] = True
                    return result
                if "world_motion" not in data.files or not len(data["world_motion"]):
                    return result
                world = np.asarray(data["world_motion"], dtype=np.float32)
                names = [str(item).lower() for item in data["names"].tolist()] if "names" in data.files else []
                root_index = names.index("hips") if "hips" in names else 0
                positions = world[:, root_index, :3, 3]
                if len(positions) > 1:
                    result["has_root_motion"] = bool(
                        np.max(np.linalg.norm(positions - positions[[0]], axis=1)) > 1e-4
                    )
        except (OSError, ValueError, KeyError, IndexError):
            pass
        return result

    def bundle_path(self, animation_id: str) -> Path:
        row = self.get(animation_id)
        path = ANIMATIONS_ROOT / Path(animation_id).name / Path(row["bundle"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def create(self, source_motion_id: str, character_id: str, name: str) -> dict[str, Any]:
        motion = self.motions.get(source_motion_id)
        source_path = self.motions.source_bundle_path(motion)
        self.characters.get(character_id)
        animation_id = uuid.uuid4().hex[:12]
        root = ANIMATIONS_ROOT / animation_id
        root.mkdir(parents=True, exist_ok=False)
        bundle = root / "retargeted.npz"
        retarget_source_bundle_fixed(source_path, self.characters.binding_path(character_id), bundle)
        with np.load(bundle, allow_pickle=False) as data:
            frames = int(data["world_motion"].shape[0])
            fps = float(data["fps"].reshape(-1)[0])
        now = time.time()
        row = {
            "id": animation_id,
            "name": (name or motion["filename"]).strip()[:80],
            "source_motion_id": source_motion_id,
            "character_id": character_id,
            "bundle": bundle.name,
            "frames": frames,
            "fps": fps,
            "created_at": now,
            "updated_at": now,
        }
        _write_json(root / "manifest.json", row)
        return row

    def create_from_preview(self, preview: dict[str, Any], preview_bundle: Path) -> dict[str, Any]:
        animation_id = uuid.uuid4().hex[:12]
        root = ANIMATIONS_ROOT / animation_id
        root.mkdir(parents=True, exist_ok=False)
        bundle = root / "retargeted.npz"
        shutil.copy2(preview_bundle, bundle)
        now = time.time()
        row = {
            "id": animation_id,
            "name": preview["name"],
            "source_motion_id": preview["source_motion_id"],
            "source_kind": preview.get("source_kind", "gvhmr"),
            "kimodo_motion_id": preview.get("kimodo_motion_id", ""),
            "character_id": preview["character_id"],
            "bundle": bundle.name,
            "frames": preview["frames"],
            "fps": preview["fps"],
            "created_at": now,
            "updated_at": now,
        }
        _write_json(root / "manifest.json", row)
        return row

    def create_from_virtual_root_preview(self, preview: dict[str, Any], preview_bundle: Path) -> dict[str, Any]:
        animation_id = uuid.uuid4().hex[:12]
        root = ANIMATIONS_ROOT / animation_id
        root.mkdir(parents=True, exist_ok=False)
        bundle = root / "virtual_root.npz"
        with np.load(preview_bundle, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files if key != "preview_world_motion"}
        np.savez_compressed(bundle, **payload)
        now = time.time()
        row = {
            "id": animation_id,
            "name": preview["name"],
            "source_motion_id": preview.get("source_motion_id", ""),
            "source_animation_id": preview["source_animation_id"],
            "character_id": preview["character_id"],
            "bundle": bundle.name,
            "frames": preview["frames"],
            "fps": preview["fps"],
            "virtual_root": True,
            "motion_kind": "virtual_root",
            "created_at": now,
            "updated_at": now,
        }
        _write_json(root / "manifest.json", row)
        return row

    def create_from_payload(
        self,
        character_id: str,
        name: str,
        payload: dict[str, np.ndarray],
        *,
        source_composition_id: str = "",
        motion_kind: str = "curve_baked",
        hidden: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.characters.get(character_id)
        animation_id = uuid.uuid4().hex[:12]
        root = ANIMATIONS_ROOT / animation_id
        root.mkdir(parents=True, exist_ok=False)
        bundle = root / "virtual_root.npz"
        np.savez_compressed(bundle, **payload)
        frames = int(np.asarray(payload["world_motion"]).shape[0])
        fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
        now = time.time()
        row = {
            "id": animation_id,
            "name": (name or "Curve Baked Motion").strip()[:80],
            "source_motion_id": "",
            "source_animation_id": "",
            "source_composition_id": source_composition_id,
            "character_id": character_id,
            "bundle": bundle.name,
            "frames": frames,
            "fps": fps,
            "virtual_root": "virtual_root_motion" in payload,
            "motion_kind": motion_kind,
            "hidden": hidden,
            "created_at": now,
            "updated_at": now,
        }
        if metadata:
            row.update(metadata)
        row["has_root_motion"] = bool(
            row.get("has_root_motion") or "virtual_root_motion" in payload or "root_motion" in payload
        )
        _write_json(root / "manifest.json", row)
        return row

    def rename(self, animation_id: str, name: str) -> dict[str, Any]:
        clean = (name or "").strip()
        if not clean:
            raise ValueError("动作名称不能为空")
        row = self.get(animation_id)
        row.update(name=clean[:80], updated_at=time.time())
        _write_json(ANIMATIONS_ROOT / Path(animation_id).name / "manifest.json", row)
        return row

    def adjust_whole_animation(
        self,
        animation_id: str,
        *,
        name: str,
        adjustments: list[dict[str, Any]],
        overwrite: bool = False,
        trim_start: int | None = None,
        trim_end: int | None = None,
    ) -> dict[str, Any]:
        row = self.get(animation_id)
        data, world = self.effective_world(animation_id)
        try:
            payload = {key: np.asarray(data[key]) for key in data.files}
        finally:
            data.close()
        payload["world_motion"] = _apply_whole_animation_transforms(world, payload["parents"], adjustments)
        payload = _trim_animation_payload(payload, trim_start, trim_end)
        clean_name = (name or row.get("name") or "Adjusted Motion").strip()[:80]
        if overwrite:
            bundle = self.bundle_path(animation_id)
            np.savez_compressed(bundle, **payload)
            row.update(
                name=clean_name,
                frames=int(payload["world_motion"].shape[0]),
                fps=float(np.asarray(payload["fps"]).reshape(-1)[0]),
                updated_at=time.time(),
            )
            self.edit_project(animation_id).clear()
            self.baked_edit_project(animation_id).clear()
            _write_json(ANIMATIONS_ROOT / Path(animation_id).name / "manifest.json", row)
            return row
        created = self.create_from_payload(
            row["character_id"],
            clean_name,
            payload,
            motion_kind=row.get("motion_kind", "adjusted_motion"),
            hidden=False,
        )
        created["source_animation_id"] = animation_id
        created["virtual_root"] = bool(row.get("virtual_root") or "virtual_root_motion" in payload)
        if created["virtual_root"]:
            created["motion_kind"] = row.get("motion_kind", "virtual_root")
        _write_json(ANIMATIONS_ROOT / created["id"] / "manifest.json", created)
        return created

    def delete(self, animation_id: str) -> None:
        root = (ANIMATIONS_ROOT / Path(animation_id).name).resolve()
        if root.parent != ANIMATIONS_ROOT.resolve() or not root.is_dir():
            raise KeyError(animation_id)
        shutil.rmtree(root)

    def edit_project(self, animation_id: str) -> EditProject:
        path = self.bundle_path(animation_id)
        with np.load(path, allow_pickle=False) as data:
            frames, joints = data["world_motion"].shape[:2]
        return EditProject(ANIMATIONS_ROOT / Path(animation_id).name / "edits.json", frames, joints)

    def baked_edit_project(self, animation_id: str) -> EditProject:
        path = self.bundle_path(animation_id)
        with np.load(path, allow_pickle=False) as data:
            frames, joints = data["world_motion"].shape[:2]
        return EditProject(ANIMATIONS_ROOT / Path(animation_id).name / "baked_edits.json", frames, joints)

    def effective_world(self, animation_id: str) -> tuple[np.lib.npyio.NpzFile, np.ndarray]:
        data = np.load(self.bundle_path(animation_id), allow_pickle=False)
        baked = self.baked_edit_project(animation_id)
        project = self.edit_project(animation_id)
        world = baked.apply(data["world_motion"], data["parents"]) if baked.operations else data["world_motion"]
        world = project.apply(world, data["parents"]) if project.operations else world
        return data, np.asarray(world, dtype=np.float32)

    def root_motion(self, animation_id: str, data: np.lib.npyio.NpzFile) -> np.ndarray | None:
        if "virtual_root_motion" in data.files:
            root = np.asarray(data["virtual_root_motion"], dtype=np.float32)
            return (root[:, :3, 3] - root[[0], :3, 3]).astype(np.float32)
        if "root_motion" in data.files:
            return np.asarray(data["root_motion"], dtype=np.float32)
        try:
            animation = self.get(animation_id)
            job = self.motions.get(animation["source_motion_id"])
            with np.load(self.motions.source_bundle_path(job), allow_pickle=False) as source:
                if "root_motion" in source.files:
                    return np.asarray(source["root_motion"], dtype=np.float32)
        except (KeyError, FileNotFoundError, OSError):
            pass
        # Legacy library entries may outlive their deleted source-motion job.
        # Recover only from a near-identical retargeted motion that still carries
        # a root track; the strict rotation threshold avoids borrowing an
        # unrelated trajectory merely because frame counts happen to match.
        try:
            current_world = np.asarray(data["world_motion"], dtype=np.float32)
            sample = np.linspace(0, len(current_world) - 1, min(32, len(current_world)), dtype=np.int64)
            best_score, best_root = float("inf"), None
            for manifest_path in ANIMATIONS_ROOT.glob("*/manifest.json"):
                if manifest_path.parent.name == Path(animation_id).name:
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                candidate_path = manifest_path.parent / Path(manifest["bundle"]).name
                with np.load(candidate_path, allow_pickle=False) as candidate:
                    if "root_motion" not in candidate.files or len(candidate["world_motion"]) != len(current_world):
                        continue
                    score = float(np.mean(np.abs(current_world[sample, :, :3, :3] - candidate["world_motion"][sample, :, :3, :3])))
                    if score < best_score:
                        best_score = score
                        best_root = np.asarray(candidate["root_motion"], dtype=np.float32)
            if best_root is not None and best_score < 0.01:
                return best_root
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return None
        return None

    def virtual_root_motion(self, data: np.lib.npyio.NpzFile) -> np.ndarray | None:
        if "virtual_root_motion" in data.files:
            return np.asarray(data["virtual_root_motion"], dtype=np.float32)
        return None


class CompositionStore:
    def __init__(self, animations: AnimationStore) -> None:
        self.animations = animations
        COMPOSITIONS_ROOT.mkdir(parents=True, exist_ok=True)

    def list(self, character_id: str | None = None) -> list[dict[str, Any]]:
        rows = []
        for path in COMPOSITIONS_ROOT.glob("*/manifest.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if character_id is None or row["character_id"] == character_id:
                    rows.append(row)
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: row.get("updated_at", 0), reverse=True)

    def get(self, composition_id: str) -> dict[str, Any]:
        path = COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json"
        if not path.is_file():
            raise KeyError(composition_id)
        row = json.loads(path.read_text(encoding="utf-8"))
        row.setdefault("schema_version", 2)
        row.setdefault("scene_objects", [])
        row.setdefault("ik_constraints", [])
        return row

    def create(self, character_id: str, name: str = "Untitled Composition") -> dict[str, Any]:
        self.animations.characters.get(character_id)
        composition_id = uuid.uuid4().hex[:12]
        now = time.time()
        row = {"id": composition_id, "name": name[:80], "character_id": character_id, "fps": 30.0, "segments": [], "transitions": [], "baked_transitions": [], "orientation_corrections": [], "cuts": [], "created_at": now, "updated_at": now}
        row["schema_version"] = 2
        row["scene_objects"] = []
        row["ik_constraints"] = []
        _write_json(COMPOSITIONS_ROOT / composition_id / "manifest.json", row)
        return row
    @staticmethod
    def _finite_vector(value: Any, length: int, default: list[float]) -> list[float]:
        array = np.asarray(default if value is None else value, dtype=np.float64).reshape(-1)
        if array.size != length or not np.isfinite(array).all():
            raise ValueError(f"expected {length} finite transform values")
        return array.astype(float).tolist()

    @classmethod
    def _normalize_scene_object(cls, item: dict[str, Any]) -> dict[str, Any]:
        asset_kind = str(item.get("asset_kind", "static_mesh")).lower()
        if asset_kind == "terrain_heightfield":
            asset_kind = "terrain"
        if asset_kind not in {"static_mesh", "terrain"}:
            raise ValueError("scene object kind must be static_mesh or terrain")
        source_asset_id = str(item.get("source_asset_id") or item.get("asset_id") or "").strip()
        if not source_asset_id:
            raise ValueError("scene object source asset is required")
        parent_type = str(item.get("parent_type", "world")).lower()
        if parent_type not in {"world", "bone", "ik_lock"}:
            raise ValueError("scene object parent must be world, bone, or ik_lock")
        transform = item.get("transform") or {}
        translation = cls._finite_vector(transform.get("translation", item.get("position")), 3, [0.0, 0.0, 0.0])
        quaternion = np.asarray(
            cls._finite_vector(transform.get("quaternion", item.get("quaternion")), 4, [0.0, 0.0, 0.0, 1.0]),
            dtype=np.float64,
        )
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-9)
        scale_value = transform.get("scale", item.get("scale", [1.0, 1.0, 1.0]))
        if np.isscalar(scale_value):
            scale_value = [float(scale_value)] * 3
        scale = np.clip(np.asarray(cls._finite_vector(scale_value, 3, [1.0, 1.0, 1.0])), 0.001, 1000.0).tolist()
        visibility = item.get("visibility") or {}
        ranges = []
        for value in visibility.get("ranges", item.get("visibility_ranges", [])):
            start, end = max(0, int(value.get("start_frame", 0))), max(0, int(value.get("end_frame", 0)))
            if end <= start:
                raise ValueError("scene visibility range end must be greater than start")
            ranges.append({"start_frame": start, "end_frame": end, "visible": bool(value.get("visible", False))})
        return {
            "id": str(item.get("id") or uuid.uuid4().hex[:12]),
            "name": str(item.get("name") or "Scene Object").strip()[:80],
            "asset_kind": asset_kind,
            "source_asset_id": source_asset_id,
            "asset_revision": str(item.get("asset_revision") or item.get("asset_hash") or ""),
            "parent_type": parent_type,
            "parent_id": str(item.get("parent_id") or ""),
            "parent_joint": int(item.get("parent_joint", item.get("joint", -1))),
            "transform": {"translation": translation, "quaternion": quaternion.astype(float).tolist(), "scale": scale},
            "visibility": {"default": bool(visibility.get("default", True)), "ranges": ranges},
            "editor_hidden": bool(item.get("editor_hidden", False)),
        }

    @classmethod
    def _normalize_ik_constraint(cls, item: dict[str, Any]) -> dict[str, Any]:
        solver = str(item.get("solver", "pbik")).lower()
        if solver not in {"two_bone", "pbik", "jacobian_dls", "fabrik"}:
            raise ValueError("IK constraint solver is invalid")
        start, end = max(0, int(item.get("start_frame", 0))), max(0, int(item.get("end_frame", 0)))
        if end < start:
            raise ValueError("IK constraint end frame must not precede start frame")
        position = cls._finite_vector(item.get("target_position"), 3, [0.0, 0.0, 0.0])
        raw_rotation = item.get("target_rotation")
        rotation = None
        if raw_rotation is not None:
            quaternion = np.asarray(cls._finite_vector(raw_rotation, 4, [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
            quaternion /= max(float(np.linalg.norm(quaternion)), 1e-9)
            rotation = quaternion.astype(float).tolist()
        keys = []
        for key in item.get("keys", []):
            key_rotation = key.get("rotation")
            if key_rotation is not None:
                quaternion = np.asarray(cls._finite_vector(key_rotation, 4, [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
                quaternion /= max(float(np.linalg.norm(quaternion)), 1e-9)
                key_rotation = quaternion.astype(float).tolist()
            keys.append({
                "frame": max(0, int(key.get("frame", 0))),
                "position": cls._finite_vector(key.get("position"), 3, position),
                "rotation": key_rotation,
            })
        keys.sort(key=lambda key: key["frame"])
        interpolation = str(item.get("interpolation", "cubic")).lower()
        if interpolation not in {"step", "linear", "cubic"}:
            raise ValueError("IK target interpolation must be step, linear, or cubic")
        target_space = str(item.get("target_space", "world")).lower()
        if target_space not in {"world", "object"}:
            raise ValueError("IK target space must be world or object")
        root_behavior = str(item.get("root_behavior", "pre_pull")).lower()
        if root_behavior not in {"pin", "pre_pull", "foot_priority", "free"}:
            raise ValueError("IK root behavior is invalid")
        effector_id = str(item.get("effector_id") or "").strip()
        joint_name = str(item.get("joint_name") or "").strip()
        if not effector_id and not joint_name:
            raise ValueError("IK constraint effector is required")
        return {
            "id": str(item.get("id") or uuid.uuid4().hex[:12]),
            "name": str(item.get("name") or joint_name or effector_id or "IK Lock").strip()[:80],
            "solver": solver,
            "effector_id": effector_id,
            "joint_name": joint_name,
            "start_frame": start,
            "end_frame": end,
            "blend_in_frames": int(np.clip(int(item.get("blend_in_frames", 0)), 0, 240)),
            "blend_out_frames": int(np.clip(int(item.get("blend_out_frames", 0)), 0, 240)),
            "target_space": target_space,
            "parent_object_id": str(item.get("parent_object_id") or ""),
            "target_position": position,
            "target_rotation": rotation,
            "keys": keys,
            "interpolation": interpolation,
            "root_behavior": root_behavior,
            "iterations": int(np.clip(int(item.get("iterations", 20)), 1, 100)),
            "tolerance": float(np.clip(float(item.get("tolerance", 0.002)), 0.00001, 0.05)),
            "enabled": bool(item.get("enabled", True)),
        }

    def update_scene_objects(self, composition_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        row = self.get(composition_id)
        normalized = [self._normalize_scene_object(item) for item in items]
        ids = [item["id"] for item in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError("scene object ids must be unique")
        row["scene_objects"] = normalized
        row["updated_at"] = time.time()
        _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
        return row

    def update_ik_constraints(self, composition_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        row = self.get(composition_id)
        normalized = [self._normalize_ik_constraint(item) for item in items]
        ids = [item["id"] for item in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError("IK constraint ids must be unique")
        row["ik_constraints"] = normalized
        row["updated_at"] = time.time()
        _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
        return row


    @staticmethod
    def _root_curve_bake_path(composition_id: str) -> Path:
        return COMPOSITIONS_ROOT / Path(composition_id).name / "root_curve_bake.npz"

    def _clear_root_curve_bake(self, composition_id: str, row: dict[str, Any] | None = None) -> dict[str, Any]:
        if row is None:
            row = self.get(composition_id)
        row.pop("root_curve_bake", None)
        path = self._root_curve_bake_path(composition_id)
        if path.is_file():
            path.unlink()
        return row

    def update(self, composition_id: str, *, name: str | None, segments: list[dict], cuts: list[dict] | None = None) -> dict[str, Any]:
        row = self.get(composition_id)
        self._clear_root_curve_bake(composition_id, row)
        validated = []
        for segment in segments:
            animation = self.animations.get(str(segment["animation_id"]))
            if animation["character_id"] != row["character_id"]:
                raise ValueError("timeline animation does not match character")
            speed = float(segment.get("speed", 1.0))
            if not 0.05 <= speed <= 8.0:
                raise ValueError("playback speed must be between 0.05 and 8.0")
            start_frame = max(0, min(int(segment.get("start_frame", 0)), max(0, animation["frames"] - 1)))
            end_frame = max(start_frame, min(int(segment.get("end_frame", animation["frames"] - 1)), max(0, animation["frames"] - 1)))
            source_frames = max(1, round((end_frame - start_frame + 1) * float(row["fps"]) / max(float(animation["fps"]), 1e-6) / speed))
            deleted_ranges = self._merge_ranges(segment.get("deleted_ranges", []), source_frames)
            output_frames = source_frames - sum(item["end_frame"] - item["start_frame"] for item in deleted_ranges)
            if output_frames < 1:
                raise ValueError("segment cannot delete all frames")
            validated.append({
                "id": str(segment.get("id") or uuid.uuid4().hex[:10]),
                "animation_id": animation["id"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "speed": speed,
                "reverse": bool(segment.get("reverse", False)),
                "root_motion": bool(segment.get("root_motion", False)),
                "mirror": bool(segment.get("mirror", False)),
                "group_id": str(segment.get("group_id") or ""),
                "deleted_ranges": deleted_ranges,
                "source_output_frames": source_frames,
                "output_frames": output_frames,
                "absolute_root": bool(segment.get("absolute_root", False)),
                "root_curve_batch": str(segment.get("root_curve_batch") or ""),
                "root_curve_index": int(segment.get("root_curve_index", -1)),
                "force_align_to_previous": bool(segment.get("force_align_to_previous", False)),
                "display_name": str(segment.get("display_name") or "").strip()[:80],
            })
        previous_transitions = list(row.get("transitions", []))
        row["segments"] = validated
        transitions = []
        indexes = {segment["id"]: index for index, segment in enumerate(validated)}
        signatures = {segment["id"]: self._segment_signature(segment) for segment in validated}
        for existing in previous_transitions:
            left_id, right_id = existing.get("left_segment_id"), existing.get("right_segment_id")
            valid = (
                left_id in indexes and right_id in indexes
                and indexes[right_id] == indexes[left_id] + 1
                and existing.get("left_index") == indexes[left_id]
                and existing.get("right_index") == indexes[right_id]
                and existing.get("left_signature") == signatures[left_id]
                and existing.get("right_signature") == signatures[right_id]
            )
            if valid:
                transitions.append(existing)
        row["transitions"] = transitions
        if cuts is not None:
            row["cuts"] = [{"start_frame": int(c["start_frame"]), "end_frame": int(c["end_frame"])} for c in cuts]
        if name is not None and name.strip():
            row["name"] = name.strip()[:80]
        row["updated_at"] = time.time()
        _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
        return {**row, "invalidated_transitions": len(previous_transitions) - len(transitions)}

    def rename_segments(self, composition_id: str, segment_ids: list[str], name: str) -> dict[str, Any]:
        row = self.get(composition_id)
        clean = str(name or "").strip()[:80]
        if not clean:
            raise ValueError("segment name cannot be empty")
        requested = {str(segment_id) for segment_id in segment_ids if str(segment_id)}
        existing = {str(segment["id"]) for segment in row.get("segments", [])}
        if not requested or not requested.issubset(existing):
            raise KeyError("segment not found")
        for segment in row["segments"]:
            if str(segment["id"]) in requested:
                segment["display_name"] = clean
        row["updated_at"] = time.time()
        _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
        return row

    def commit_to_animation(self, composition_id: str, name: str) -> dict[str, Any]:
        row = self.get(composition_id)
        clean = str(name or "").strip()[:80]
        if not clean:
            raise ValueError("animation name cannot be empty")
        path = self.compile(composition_id)
        with np.load(path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]).copy() for key in data.files}
        world = np.asarray(payload["world_motion"], dtype=np.float32)
        parents = np.asarray(payload["parents"], dtype=np.int64)
        frames, joints = world.shape[:2]
        baked = EditProject(COMPOSITIONS_ROOT / Path(composition_id).name / "baked_edits.json", frames, joints)
        pending = EditProject(COMPOSITIONS_ROOT / Path(composition_id).name / "edits.json", frames, joints)
        if baked.operations:
            world = np.asarray(baked.apply(world, parents), dtype=np.float32)
        if pending.operations:
            world = np.asarray(pending.apply(world, parents), dtype=np.float32)
        if row.get("ik_constraints"):
            names = self._decode_joint_names(payload["names"]) if "names" in payload else []
            world = apply_ik_constraints(world, parents, names, row["ik_constraints"])
        if "virtual_root_motion" not in payload:
            raise ValueError("composition has no virtual root motion")
        root = np.asarray(payload["virtual_root_motion"], dtype=np.float32)
        if len(root) != len(world):
            raise ValueError("virtual root frame count does not match composition")
        local_world = np.einsum(
            "fab,fjbc->fjac",
            np.linalg.inv(root.astype(np.float64)),
            world.astype(np.float64),
        ).astype(np.float32)
        payload["world_motion"] = local_world
        payload["virtual_root_motion"] = root
        payload["root_motion"] = (root[:, :3, 3] - root[[0], :3, 3]).astype(np.float32)
        payload["num_frames"] = np.asarray([len(world)], dtype=np.int64)
        metadata: dict[str, Any] = {}
        if row.get("scene_objects") or row.get("ik_constraints"):
            metadata["composition_scene"] = {
                "schema_version": int(row.get("schema_version", 2)),
                "scene_objects": row.get("scene_objects", []),
                "ik_constraints": row.get("ik_constraints", []),
            }
        if "terrain_heights" in payload and "terrain_size" in payload:
            metadata.update({
                "terrain_attached": True,
                "terrain_attachment": {
                    "source_terrain_id": str(payload.get("terrain_source_id", [""])[0]),
                    "source_terrain_name": str(payload.get("terrain_source_name", [""])[0]),
                    "source_revision": int(payload.get("terrain_source_revision", [0])[0]),
                    "snapshot_hash": str(payload.get("terrain_snapshot_hash", [""])[0]),
                    "display_by_default": True,
                },
            })
        return self.animations.create_from_payload(
            row["character_id"],
            clean,
            payload,
            source_composition_id=composition_id,
            motion_kind="composition_baked",
            hidden=False,
            metadata=metadata,
        )

    @staticmethod
    def _segment_signature(segment: dict[str, Any]) -> str:
        fields = {key: segment.get(key) for key in ("id", "animation_id", "start_frame", "end_frame", "speed", "reverse", "root_motion", "mirror", "group_id", "deleted_ranges", "output_frames", "absolute_root", "root_curve_batch", "root_curve_index", "force_align_to_previous")}
        return json.dumps(fields, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _transition_usage(row: dict[str, Any], segment_id: str, exclude_id: str | None = None) -> int:
        """Frames already reserved at either end of a segment by neighboring transitions."""
        used = 0
        for item in [*row.get("baked_transitions", []), *row.get("transitions", [])]:
            if item.get("id") == exclude_id or not item.get("enabled", True):
                continue
            if item.get("left_segment_id") == segment_id:
                used += max(0, int(item.get("left_frames", 0)))
            if item.get("right_segment_id") == segment_id:
                used += max(0, int(item.get("right_frames", 0)))
        return used

    def _transition_limit(
        self,
        row: dict[str, Any],
        segment: dict[str, Any],
        *,
        capacity: int | None = None,
        exclude_id: str | None = None,
    ) -> int:
        segment_frames = max(1, int(segment.get("output_frames", 1)))
        usable = min(segment_frames, int(capacity if capacity is not None else segment_frames))
        return max(0, usable - 1 - self._transition_usage(row, str(segment["id"]), exclude_id))

    def add_transition(self, composition_id: str, left_segment_id: str, right_segment_id: str) -> dict[str, Any]:
        row = self.get(composition_id)
        self._clear_root_curve_bake(composition_id, row)
        indexes = {segment["id"]: index for index, segment in enumerate(row["segments"])}
        if left_segment_id not in indexes or indexes.get(right_segment_id) != indexes[left_segment_id] + 1:
            raise ValueError("transition requires two adjacent segments")
        existing_transitions = [*row.get("transitions", []), *row.get("baked_transitions", [])]
        if any(item["left_segment_id"] == left_segment_id and item["right_segment_id"] == right_segment_id for item in existing_transitions):
            raise ValueError("transition already exists between these segments")
        left, right = row["segments"][indexes[left_segment_id]], row["segments"][indexes[right_segment_id]]
        default_left = max(1, int(np.floor(left["output_frames"] * 0.05 + 0.5))) if left["output_frames"] > 1 else 0
        default_right = max(1, int(np.floor(right["output_frames"] * 0.05 + 0.5))) if right["output_frames"] > 1 else 0
        transition = {
            "id": uuid.uuid4().hex[:12],
            "left_segment_id": left["id"], "right_segment_id": right["id"],
            "left_frames": min(default_left, self._transition_limit(row, left)),
            "right_frames": min(default_right, self._transition_limit(row, right)),
            "left_capacity": left["output_frames"], "right_capacity": right["output_frames"],
            "left_index": indexes[left_segment_id], "right_index": indexes[right_segment_id],
            "left_signature": self._segment_signature(left), "right_signature": self._segment_signature(right),
            "enabled": True,
        }
        row.setdefault("transitions", []).append(transition)
        row["updated_at"] = time.time()
        _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
        return row

    def update_transition(self, composition_id: str, transition_id: str, left_frames: int, right_frames: int) -> dict[str, Any]:
        row = self.get(composition_id)
        self._clear_root_curve_bake(composition_id, row)
        segments = {item["id"]: item for item in row["segments"]}
        for transition in row.get("transitions", []):
            if transition["id"] == transition_id:
                left = segments[transition["left_segment_id"]]
                right = segments[transition["right_segment_id"]]
                left_limit = min(left["output_frames"], int(transition.get("left_capacity", left["output_frames"])))
                right_limit = min(right["output_frames"], int(transition.get("right_capacity", right["output_frames"])))
                transition["left_frames"] = int(np.clip(left_frames, 0, self._transition_limit(row, left, capacity=left_limit, exclude_id=transition_id)))
                transition["right_frames"] = int(np.clip(right_frames, 0, self._transition_limit(row, right, capacity=right_limit, exclude_id=transition_id)))
                transition["enabled"] = True
                row["updated_at"] = time.time()
                _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
                return row
        raise KeyError(transition_id)

    def delete_transition(self, composition_id: str, transition_id: str) -> dict[str, Any]:
        row = self.get(composition_id)
        self._clear_root_curve_bake(composition_id, row)
        before = len(row.get("transitions", []))
        row["transitions"] = [item for item in row.get("transitions", []) if item["id"] != transition_id]
        if len(row["transitions"]) == before:
            raise KeyError(transition_id)
        row["updated_at"] = time.time()
        _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
        return row

    def apply_transition(self, composition_id: str, transition_id: str) -> dict[str, Any]:
        row = self.get(composition_id)
        self._clear_root_curve_bake(composition_id, row)
        selected = next((item for item in row.get("transitions", []) if item["id"] == transition_id), None)
        if selected is None:
            raise KeyError(transition_id)
        row["transitions"] = [item for item in row.get("transitions", []) if item["id"] != transition_id]
        row.setdefault("baked_transitions", []).append({**selected, "baked": True})
        row["updated_at"] = time.time()
        _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
        return row

    def add_orientation_correction(self, composition_id: str, start_frame: int, end_frame: int, mode: str = "range") -> dict[str, Any]:
        row = self.get(composition_id)
        self._clear_root_curve_bake(composition_id, row)
        total = max(1, sum(int(segment.get("output_frames", segment.get("source_output_frames", 1))) for segment in row["segments"]))
        start = int(np.clip(start_frame, 0, total - 1))
        end = start if mode == "frame" else int(np.clip(end_frame, start, total - 1))
        length = max(1, end - start + 1)
        correction = {
            "id": uuid.uuid4().hex[:12],
            "start_frame": start,
            "end_frame": end,
            "blend_in_frames": min(start, max(1, int(np.floor(length * 0.10 + 0.5)))),
            "blend_out_frames": min(max(0, total - 1 - end), max(1, int(np.floor(length * 0.10 + 0.5)))),
            "speed_threshold": 0.003,
            "enabled": True,
        }
        row.setdefault("orientation_corrections", []).append(correction)
        row["updated_at"] = time.time()
        _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
        return row

    def update_orientation_correction(self, composition_id: str, correction_id: str, blend_in_frames: int, blend_out_frames: int) -> dict[str, Any]:
        row = self.get(composition_id)
        self._clear_root_curve_bake(composition_id, row)
        total = max(1, sum(int(segment.get("output_frames", segment.get("source_output_frames", 1))) for segment in row["segments"]))
        for correction in row.get("orientation_corrections", []):
            if correction["id"] == correction_id:
                start, end = int(correction["start_frame"]), int(correction["end_frame"])
                correction["blend_in_frames"] = int(np.clip(blend_in_frames, 0, start))
                correction["blend_out_frames"] = int(np.clip(blend_out_frames, 0, max(0, total - 1 - end)))
                correction["enabled"] = True
                row["updated_at"] = time.time()
                _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
                return row
        raise KeyError(correction_id)

    def delete_orientation_correction(self, composition_id: str, correction_id: str) -> dict[str, Any]:
        row = self.get(composition_id)
        self._clear_root_curve_bake(composition_id, row)
        before = len(row.get("orientation_corrections", []))
        row["orientation_corrections"] = [item for item in row.get("orientation_corrections", []) if item["id"] != correction_id]
        if len(row["orientation_corrections"]) == before:
            raise KeyError(correction_id)
        row["updated_at"] = time.time()
        _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
        return row

    def apply_orientation_correction(self, composition_id: str, correction_id: str) -> dict[str, Any]:
        row = self.get(composition_id)
        self._clear_root_curve_bake(composition_id, row)
        selected = next((item for item in row.get("orientation_corrections", []) if item["id"] == correction_id), None)
        if selected is None:
            raise KeyError(correction_id)
        row["orientation_corrections"] = [item for item in row.get("orientation_corrections", []) if item["id"] != correction_id]
        row.setdefault("baked_orientation_corrections", []).append({**selected, "baked": True})
        row["updated_at"] = time.time()
        _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
        return row

    @staticmethod
    def _shortest_angle(delta: np.ndarray | float) -> np.ndarray | float:
        return (delta + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _yaw_from_roots(root: np.ndarray) -> np.ndarray:
        forward = root[:, :3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return np.asarray(np.arctan2(forward[:, 0], forward[:, 2]), dtype=np.float64)

    @staticmethod
    def _tangent_yaws(positions: np.ndarray, fallback: np.ndarray, threshold: float) -> np.ndarray:
        targets = np.zeros(len(positions), dtype=np.float64)
        last: float | None = None
        for i in range(len(positions)):
            prev_i, next_i = max(0, i - 1), min(len(positions) - 1, i + 1)
            delta = positions[next_i] - positions[prev_i]
            horizontal = np.asarray([delta[0], delta[2]], dtype=np.float64)
            speed = float(np.linalg.norm(horizontal) / max(1, next_i - prev_i))
            if speed >= threshold:
                yaw = float(np.arctan2(horizontal[0], horizontal[1]))
                if last is not None:
                    yaw = float(last + CompositionStore._shortest_angle(yaw - last))
                last = yaw
                targets[i] = yaw
            else:
                targets[i] = last if last is not None else float(fallback[i])
        return targets

    @staticmethod
    def _yaw_rotation(delta: float) -> np.ndarray:
        c, s = float(np.cos(delta)), float(np.sin(delta))
        return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)

    def _apply_orientation_corrections(self, world: np.ndarray, root: np.ndarray, corrections: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        if not corrections or len(root) == 0:
            return world, root
        n = len(root)
        for correction in corrections:
            if not correction.get("enabled", True):
                continue
            start = int(np.clip(correction.get("start_frame", 0), 0, n - 1))
            end = int(np.clip(correction.get("end_frame", start), start, n - 1))
            blend_in = int(np.clip(correction.get("blend_in_frames", 0), 0, start))
            blend_out = int(np.clip(correction.get("blend_out_frames", 0), 0, max(0, n - 1 - end)))
            first, last = start - blend_in, end + blend_out
            current_yaws = self._yaw_from_roots(root)
            targets = self._tangent_yaws(root[:, :3, 3], current_yaws, float(correction.get("speed_threshold", 0.003)))
            for frame in range(first, last + 1):
                if frame < start:
                    weight = (frame - first) / max(1, blend_in)
                elif frame > end:
                    weight = 1.0 - (frame - end) / max(1, blend_out)
                else:
                    weight = 1.0
                weight = float(np.clip(weight, 0.0, 1.0))
                delta_yaw = float(self._shortest_angle(targets[frame] - current_yaws[frame]) * weight)
                if abs(delta_yaw) < 1e-8:
                    continue
                rotation = self._yaw_rotation(delta_yaw)
                pivot = root[frame, :3, 3].copy()
                root[frame, :3, :3] = rotation @ root[frame, :3, :3]
                world[frame, :, :3, :3] = rotation @ world[frame, :, :3, :3]
                world[frame, :, :3, 3] = (world[frame, :, :3, 3] - pivot) @ rotation.T + pivot
        return world, root

    @staticmethod
    def _decode_joint_names(raw_names: Any) -> list[str]:
        names = np.asarray(raw_names).reshape(-1).tolist()
        decoded: list[str] = []
        for name in names:
            if isinstance(name, bytes):
                decoded.append(name.decode("utf-8", errors="ignore"))
            else:
                decoded.append(str(name))
        return decoded

    @staticmethod
    def _mirrored_name(name: str) -> str | None:
        for left, right in (("Left", "Right"), ("left", "right"), ("LEFT", "RIGHT")):
            if left in name:
                return name.replace(left, right, 1)
            if right in name:
                return name.replace(right, left, 1)
        return None

    @classmethod
    def mirror_world_clip_around_hip(cls, world: np.ndarray, names: list[str] | np.ndarray) -> np.ndarray:
        """Mirror a world-space clip around the per-frame Hip/Hips sagittal plane and swap left/right joints."""
        clip = np.asarray(world, dtype=np.float32)
        if clip.ndim != 4 or clip.shape[-2:] != (4, 4) or len(clip) == 0:
            return clip.copy()
        decoded = cls._decode_joint_names(names)
        if len(decoded) != clip.shape[1]:
            decoded = [str(index) for index in range(clip.shape[1])]
        lookup = {name: index for index, name in enumerate(decoded)}
        hip_index = next((lookup[name] for name in ("Hips", "Hip", "Pelvis", "pelvis", "hips") if name in lookup), 0)
        mirrored = clip.copy()
        reflect = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
        pivot = clip[:, hip_index, :3, 3].copy()
        mirrored[:, :, :3, 3] = pivot[:, None, :] + (clip[:, :, :3, 3] - pivot[:, None, :]) * reflect.diagonal()[None, None, :]
        mirrored[:, :, :3, :3] = np.einsum("ab,fjbc,cd->fjad", reflect, clip[:, :, :3, :3], reflect).astype(np.float32)
        visited: set[int] = set()
        for index, name in enumerate(decoded):
            if index in visited:
                continue
            other = cls._mirrored_name(name)
            other_index = lookup.get(other) if other else None
            if other_index is None or other_index == index or other_index in visited:
                continue
            mirrored[:, [index, other_index]] = mirrored[:, [other_index, index]]
            visited.update({index, other_index})
        return mirrored.astype(np.float32)

    @staticmethod
    def _merge_ranges(ranges: list[dict], frame_count: int) -> list[dict[str, int]]:
        normalized = []
        for item in ranges:
            start = int(np.clip(item.get("start_frame", 0), 0, frame_count))
            end = int(np.clip(item.get("end_frame", start), start, frame_count))
            if end > start:
                normalized.append((start, end))
        merged: list[list[int]] = []
        for start, end in sorted(normalized):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [{"start_frame": start, "end_frame": end} for start, end in merged]

    def delete_frames(self, composition_id: str, start_frame: int, end_frame: int) -> dict[str, Any]:
        """Delete a half-open visible range by splitting clips around removed frames."""
        row = self.get(composition_id)
        total = sum(int(segment.get("output_frames", segment.get("source_output_frames", 1))) for segment in row["segments"])
        start = int(np.clip(start_frame, 0, total))
        end = int(np.clip(end_frame, start, total))
        if end <= start:
            raise ValueError("delete end frame must be greater than start frame")
        if end - start >= total:
            raise ValueError("cannot delete every frame from a composition")

        cursor = 0
        segments = []
        for segment in row["segments"]:
            source_frames = int(segment.get("source_output_frames", segment.get("output_frames", 1)))
            old_ranges = self._merge_ranges(segment.get("deleted_ranges", []), source_frames)
            deleted = np.zeros(source_frames, dtype=bool)
            for item in old_ranges:
                deleted[item["start_frame"] : item["end_frame"]] = True
            surviving = np.flatnonzero(~deleted)
            visible_count = len(surviving)
            overlap_start = max(start, cursor)
            overlap_end = min(end, cursor + visible_count)
            keep_visible = np.ones(visible_count, dtype=bool)
            if overlap_end > overlap_start:
                keep_visible[overlap_start - cursor : overlap_end - cursor] = False
            kept = surviving[keep_visible]
            if len(kept):
                run_start = previous = int(kept[0])
                runs: list[tuple[int, int]] = []
                for value in kept[1:]:
                    value = int(value)
                    if value != previous + 1:
                        runs.append((run_start, previous + 1))
                        run_start = value
                    previous = value
                runs.append((run_start, previous + 1))
                for run_start, run_end in runs:
                    copy = self._split_segment_by_output_run(segment, run_start, run_end)
                    if copy is not None:
                        visible_index = int(np.searchsorted(surviving, run_start, side="left"))
                        if segments and cursor + visible_index >= end:
                            copy["force_align_to_previous"] = True
                        segments.append(copy)
            cursor += visible_count
        return self.update(composition_id, name=None, segments=segments, cuts=[])

    def split_at_frame(self, composition_id: str, frame: int) -> dict[str, Any]:
        row = self.get(composition_id)
        if not row["segments"]:
            raise ValueError("timeline is empty")
        total = sum(int(segment.get("output_frames", segment.get("source_output_frames", 1))) for segment in row["segments"])
        target = int(np.clip(frame, 0, max(0, total - 1)))
        cursor = 0
        segments: list[dict[str, Any]] = []
        split_done = False
        for segment in row["segments"]:
            count = int(segment.get("output_frames", segment.get("source_output_frames", 1)))
            if not split_done and cursor < target < cursor + count:
                source_frames = int(segment.get("source_output_frames", count))
                split = int(np.clip(round((target - cursor) * source_frames / max(1, count)), 1, max(1, source_frames - 1)))
                left = self._split_segment_by_output_run(segment, 0, split)
                right = self._split_segment_by_output_run(segment, split, source_frames)
                if left is not None:
                    segments.append(left)
                if right is not None:
                    right["force_align_to_previous"] = True
                    segments.append(right)
                split_done = True
            else:
                segments.append(dict(segment))
            cursor += count
        if not split_done:
            raise ValueError("瑜版挸澧犵敮褍婀悧鍥唽鏉堝湱鏅稉濠忕礉閺冪娀娓堕幋顏呮焽")
        return self.update(composition_id, name=None, segments=segments, cuts=[])

    @staticmethod
    def _split_segment_by_output_run(segment: dict[str, Any], run_start: int, run_end: int) -> dict[str, Any] | None:
        source_frames = max(1, int(segment.get("source_output_frames", segment.get("output_frames", 1))))
        run_start = int(np.clip(run_start, 0, source_frames))
        run_end = int(np.clip(run_end, run_start, source_frames))
        if run_end <= run_start:
            return None
        original_start = int(segment["start_frame"])
        original_end = int(segment["end_frame"])
        source_count = max(1, original_end - original_start + 1)
        start_offset = int(np.floor(run_start * source_count / source_frames))
        end_offset = int(np.ceil(run_end * source_count / source_frames)) - 1
        start_offset = int(np.clip(start_offset, 0, source_count - 1))
        end_offset = int(np.clip(end_offset, start_offset, source_count - 1))
        copy = dict(segment)
        copy["id"] = uuid.uuid4().hex[:10]
        copy["group_id"] = ""
        copy["deleted_ranges"] = []
        if copy.get("reverse", False):
            copy["start_frame"] = original_start + max(0, source_count - 1 - end_offset)
            copy["end_frame"] = original_start + max(0, source_count - 1 - start_offset)
        else:
            copy["start_frame"] = original_start + start_offset
            copy["end_frame"] = original_start + end_offset
        return copy

    @staticmethod
    def _sample_polyline_by_ratio(points: np.ndarray, ratios: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((len(ratios), 3), dtype=np.float32)
        if len(points) == 1:
            return np.repeat(points[:1], len(ratios), axis=0).astype(np.float32)
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(distances)])
        total = float(cumulative[-1])
        if total <= 1e-8:
            return np.repeat(points[:1], len(ratios), axis=0).astype(np.float32)
        targets = np.clip(ratios, 0.0, 1.0) * total
        sampled = np.stack([np.interp(targets, cumulative, points[:, axis]) for axis in range(3)], axis=1)
        return sampled.astype(np.float32)

    def apply_root_curve(self, composition_id: str, curve_points: list[list[float]], still_threshold: float = 0.002, control_points: list[list[float]] | None = None, exact_positions: bool = False, exact_transforms: list[list[float]] | None = None) -> dict[str, Any]:
        points = np.asarray(curve_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
            raise ValueError("root curve requires at least one xyz point")
        has_edit_control_points = control_points is not None
        edit_points = np.asarray(control_points if has_edit_control_points else curve_points, dtype=np.float32)
        if edit_points.ndim != 2 or edit_points.shape[1] != 3 or len(edit_points) < 1:
            raise ValueError("root curve control points must be xyz points")
        exact_root = None
        if exact_transforms is not None:
            exact_root = np.asarray(exact_transforms, dtype=np.float32)
            if exact_root.ndim != 2 or exact_root.shape[1] != 16:
                raise ValueError("exact root transforms must be flattened 4x4 matrices")
            exact_root = exact_root.reshape(-1, 4, 4)
        with np.load(self.compile(composition_id, include_transitions=False), allow_pickle=False) as data:
            if "virtual_root_motion" not in data.files:
                raise ValueError("composition has no virtual root motion")
            root = np.asarray(data["virtual_root_motion"], dtype=np.float32)
            old_positions = root[:, :3, 3].copy()
            world = np.asarray(data["world_motion"], dtype=np.float32)
            base_payload = {key: np.asarray(data[key]) for key in data.files if key not in {"world_motion", "virtual_root_motion", "root_motion"}}
        if len(old_positions) == 0:
            raise ValueError("timeline is empty")
        if exact_root is not None:
            if len(exact_root) != len(old_positions):
                raise ValueError("exact root transforms require one matrix per frame")
            old_root = root.astype(np.float64)
            root = exact_root.astype(np.float32)
            delta = np.einsum("fab,fbc->fac", root.astype(np.float64), np.linalg.inv(old_root))
            world = np.einsum("fab,fjbc->fjac", delta, world.astype(np.float64)).astype(np.float32)
            points = root[:, :3, 3].astype(np.float32)
        elif exact_positions:
            if len(points) != len(old_positions):
                raise ValueError("exact root curve requires one xyz point per frame")
            new_positions = points.astype(np.float32)
        else:
            deltas = np.linalg.norm(np.diff(old_positions, axis=0), axis=1)
            deltas[deltas < max(0.0, float(still_threshold))] = 0.0
            cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
            total = float(cumulative[-1])
            if total <= max(1e-8, float(still_threshold)):
                new_positions = np.repeat(points[:1], len(old_positions), axis=0)
            else:
                new_positions = self._sample_polyline_by_ratio(points, cumulative / total)
        row = self.get(composition_id)
        old_segments = list(row["segments"])
        old_index = {segment["id"]: index for index, segment in enumerate(old_segments)}
        old_transitions = [dict(item) for item in row.get("transitions", [])]
        old_baked_transitions = [dict(item) for item in row.get("baked_transitions", [])]
        if exact_root is None:
            delta = new_positions.astype(np.float32) - root[:, :3, 3]
            root[:, :3, 3] = new_positions.astype(np.float32)
            world[:, :, :3, 3] += delta[:, None, :]
        batch = uuid.uuid4().hex[:10]
        animations = []
        segments = []
        cursor = 0
        for index, original in enumerate(row["segments"]):
            count = int(original.get("output_frames", original.get("source_output_frames", 1)))
            if count <= 0:
                continue
            start, end = cursor, min(cursor + count, len(world))
            if end <= start:
                break
            local_world = np.einsum("fab,fjbc->fjac", np.linalg.inv(root[start:end].astype(np.float64)), world[start:end].astype(np.float64)).astype(np.float32)
            payload = {key: np.asarray(value) for key, value in base_payload.items()}
            payload["world_motion"] = local_world.astype(np.float32)
            payload["virtual_root_motion"] = root[start:end].astype(np.float32)
            payload["root_motion"] = (root[start:end, :3, 3] - root[start:start + 1, :3, 3]).astype(np.float32)
            payload["fps"] = np.asarray([row["fps"]], dtype=np.float32)
            payload["num_frames"] = np.asarray([end - start], dtype=np.int64)
            payload["root_curve_control_points"] = points.astype(np.float32)
            if has_edit_control_points:
                payload["root_curve_edit_control_points"] = edit_points.astype(np.float32)
            payload["root_curve_still_threshold"] = np.asarray([still_threshold], dtype=np.float32)
            animation = self.animations.create_from_payload(
                row["character_id"],
                f'{row.get("name", "Composition")} RootCurve {index + 1}',
                payload,
                source_composition_id=composition_id,
                motion_kind="curve_baked",
            )
            animations.append(animation)
            segments.append({
                "id": uuid.uuid4().hex[:10],
                "animation_id": animation["id"],
                "start_frame": 0,
                "end_frame": int(end - start - 1),
                "speed": 1.0,
                "reverse": False,
                "root_motion": True,
                "mirror": False,
                "group_id": str(original.get("group_id") or ""),
                "deleted_ranges": [],
                "source_output_frames": int(end - start),
                "output_frames": int(end - start),
                "absolute_root": True,
                "root_curve_batch": batch,
                "root_curve_index": index,
                "display_name": str(original.get("display_name") or "").strip()[:80],
            })
            cursor = end
        self._clear_root_curve_bake(composition_id, row)
        row["segments"] = segments
        def remap_transition(item: dict[str, Any]) -> dict[str, Any] | None:
            left_old, right_old = item.get("left_segment_id"), item.get("right_segment_id")
            left_index, right_index = old_index.get(left_old, -1), old_index.get(right_old, -1)
            if left_index < 0 or right_index != left_index + 1 or right_index >= len(segments):
                return None
            left, right = segments[left_index], segments[right_index]
            left_capacity, right_capacity = int(left["output_frames"]), int(right["output_frames"])
            updated = dict(item)
            updated["left_segment_id"] = left["id"]
            updated["right_segment_id"] = right["id"]
            updated["left_index"] = left_index
            updated["right_index"] = right_index
            updated["left_capacity"] = left_capacity
            updated["right_capacity"] = right_capacity
            updated["left_frames"] = int(np.clip(int(item.get("left_frames", 0)), 0, max(0, left_capacity - 1)))
            updated["right_frames"] = int(np.clip(int(item.get("right_frames", 0)), 0, max(0, right_capacity - 1)))
            updated["left_signature"] = self._segment_signature(left)
            updated["right_signature"] = self._segment_signature(right)
            updated["enabled"] = bool(item.get("enabled", True))
            return updated
        row["transitions"] = [mapped for item in old_transitions if (mapped := remap_transition(item)) is not None]
        row["baked_transitions"] = [mapped for item in old_baked_transitions if (mapped := remap_transition(item)) is not None]
        row["cuts"] = []
        row["root_curve_replaced_animation_ids"] = [animation["id"] for animation in animations]
        row["updated_at"] = time.time()
        _write_json(COMPOSITIONS_ROOT / Path(composition_id).name / "manifest.json", row)
        return {**row, "baked_animations": animations}

    def compile(self, composition_id: str, *, include_transitions: bool = True) -> Path:
        row = self.get(composition_id)
        if not row["segments"]:
            raise ValueError("timeline is empty")
        clips = []
        roots = []
        template = None
        previous_root = None
        previous_root_offset = np.zeros(3, dtype=np.float32)
        previous_segment = None
        for segment in row["segments"]:
            data, world = self.animations.effective_world(segment["animation_id"])
            if template is None:
                template = {key: data[key] for key in data.files if key not in {"world_motion", "virtual_root_motion", "root_motion"}}
            elif "terrain_heights" in data.files and "terrain_heights" not in template:
                for key in data.files:
                    if key.startswith("terrain_"):
                        template[key] = np.asarray(data[key]).copy()
            names = self._decode_joint_names(data["names"]) if "names" in data.files else []
            source_fps = float(data["fps"].reshape(-1)[0])
            clip = segment_world(world, start_frame=segment["start_frame"], end_frame=segment["end_frame"], speed=segment["speed"], reverse=segment["reverse"], source_fps=source_fps, output_fps=row["fps"])
            root_motion = self.animations.root_motion(segment["animation_id"], data) if segment.get("root_motion") else None
            source_virtual_root = self.animations.virtual_root_motion(data)
            virtual_root = source_virtual_root if segment.get("root_motion") else None
            if virtual_root is not None and len(virtual_root):
                start = int(np.clip(segment["start_frame"], 0, len(virtual_root) - 1))
                end = int(np.clip(segment["end_frame"], start, len(virtual_root) - 1))
                root = np.asarray(virtual_root[start : end + 1], dtype=np.float32)
                if segment["reverse"]:
                    root = root[::-1]
                root = resample_world(root[:, None], len(clip))[:, 0]
                clip = np.einsum("fab,fjbc->fjac", root, clip).astype(np.float32)
            elif root_motion is not None and len(root_motion):
                start = int(np.clip(segment["start_frame"], 0, len(root_motion) - 1))
                end = int(np.clip(segment["end_frame"], start, len(root_motion) - 1))
                root = np.asarray(root_motion[start : end + 1], dtype=np.float32)
                if segment["reverse"]:
                    root = root[::-1]
                source_t = np.arange(len(root), dtype=np.float64)
                target_t = np.linspace(0, max(0, len(root) - 1), len(clip))
                resampled_root = np.stack([np.interp(target_t, source_t, root[:, axis]) for axis in range(3)], axis=1)
                resampled_root -= resampled_root[[0]]
                clip[:, :, :3, 3] += resampled_root[:, None, :]
                root = np.repeat(np.eye(4, dtype=np.float32)[None], len(clip), axis=0)
                root[:, :3, 3] = resampled_root
            else:
                if source_virtual_root is not None and len(source_virtual_root):
                    start = int(np.clip(segment["start_frame"], 0, len(source_virtual_root) - 1))
                    end = int(np.clip(segment["end_frame"], start, len(source_virtual_root) - 1))
                    base = np.asarray(source_virtual_root[end if segment.get("reverse", False) else start], dtype=np.float32)
                    root = np.repeat(base[None], len(clip), axis=0)
                    clip = np.einsum("fab,fjbc->fjac", root, clip).astype(np.float32)
                else:
                    root = np.repeat(np.eye(4, dtype=np.float32)[None], len(clip), axis=0)
            if segment.get("mirror", False):
                clip = self.mirror_world_clip_around_hip(clip, names)
            clip = apply_ordered_cuts(clip, segment.get("deleted_ranges", []))
            root = apply_ordered_cuts(root, segment.get("deleted_ranges", []))
            is_contiguous_source = (
                previous_segment is not None
                and previous_segment.get("animation_id") == segment.get("animation_id")
                and bool(previous_segment.get("root_motion")) == bool(segment.get("root_motion"))
                and bool(previous_segment.get("reverse", False)) == bool(segment.get("reverse", False))
                and bool(previous_segment.get("mirror", False)) == bool(segment.get("mirror", False))
                and abs(float(previous_segment.get("speed", 1.0)) - float(segment.get("speed", 1.0))) < 1e-6
                and not previous_segment.get("deleted_ranges")
                and not segment.get("deleted_ranges")
                and (
                    (not segment.get("reverse", False) and int(previous_segment.get("end_frame", -1)) + 1 == int(segment.get("start_frame", -2)))
                    or (segment.get("reverse", False) and int(segment.get("end_frame", -1)) + 1 == int(previous_segment.get("start_frame", -2)))
                )
            )
            is_contiguous_curve_batch = (
                previous_segment is not None
                and previous_segment.get("absolute_root")
                and segment.get("absolute_root")
                and previous_segment.get("root_curve_batch")
                and previous_segment.get("root_curve_batch") == segment.get("root_curve_batch")
                and int(previous_segment.get("root_curve_index", -1)) + 1 == int(segment.get("root_curve_index", -2))
            )
            current_root_offset = np.zeros(3, dtype=np.float32)
            if previous_root is not None:
                force_align = bool(segment.get("force_align_to_previous", False))
                preserves_source_step = bool(segment.get("root_motion")) and (is_contiguous_source or is_contiguous_curve_batch)
                if force_align or (not segment.get("absolute_root") and not preserves_source_step):
                    current_root_offset = previous_root - root[0, :3, 3]
                elif preserves_source_step and (not segment.get("absolute_root") or np.any(np.abs(previous_root_offset) > 1e-8)):
                    current_root_offset = previous_root_offset.copy()
            if np.any(np.abs(current_root_offset) > 1e-8):
                offset = current_root_offset
                clip[:, :, :3, 3] += offset[None, None, :]
                root[:, :3, 3] += offset[None, :]
            previous_root = root[-1, :3, 3].copy()
            previous_root_offset = current_root_offset
            previous_segment = segment
            clips.append(clip)
            roots.append(root)
            data.close()
        parents = np.asarray(template["parents"], dtype=np.int64) if template is not None else np.array([], dtype=np.int64)
        indexes = {segment["id"]: index for index, segment in enumerate(row["segments"])}
        if include_transitions:
            occupied: dict[str, int] = {}
            for transition in [*row.get("baked_transitions", []), *row.get("transitions", [])]:
                index = indexes.get(transition.get("left_segment_id"), -1)
                if transition.get("enabled", True) and index >= 0 and indexes.get(transition.get("right_segment_id")) == index + 1:
                    left_id, right_id = str(transition["left_segment_id"]), str(transition["right_segment_id"])
                    left_frames = int(np.clip(transition.get("left_frames", 0), 0, max(0, len(clips[index]) - 1 - occupied.get(left_id, 0))))
                    right_frames = int(np.clip(transition.get("right_frames", 0), 0, max(0, len(clips[index + 1]) - 1 - occupied.get(right_id, 0))))
                    occupied[left_id] = occupied.get(left_id, 0) + left_frames
                    occupied[right_id] = occupied.get(right_id, 0) + right_frames
                    clips[index], clips[index + 1] = blend_clip_boundary(
                        clips[index], clips[index + 1], parents,
                        left_frames, right_frames,
                    )
                    root_parents = np.asarray([-1], dtype=np.int64)
                    left_root, right_root = blend_clip_boundary(
                        roots[index][:, None], roots[index + 1][:, None], root_parents,
                        left_frames, right_frames,
                    )
                    roots[index], roots[index + 1] = left_root[:, 0], right_root[:, 0]
        world = apply_ordered_cuts(np.concatenate(clips, axis=0), row.get("cuts", []))
        virtual_root_motion = apply_ordered_cuts(np.concatenate(roots, axis=0), row.get("cuts", []))
        bake_path = self._root_curve_bake_path(composition_id)
        if bake_path.is_file():
            with np.load(bake_path, allow_pickle=False) as bake:
                positions = np.asarray(bake["positions"], dtype=np.float32)
            if len(positions) == len(virtual_root_motion):
                delta = positions - virtual_root_motion[:, :3, 3]
                virtual_root_motion[:, :3, 3] = positions
                world[:, :, :3, 3] += delta[:, None, :]
        world, virtual_root_motion = self._apply_orientation_corrections(
            world,
            virtual_root_motion,
            [*row.get("baked_orientation_corrections", []), *row.get("orientation_corrections", [])],
        )
        assert template is not None
        template["world_motion"] = world.astype(np.float32)
        template["virtual_root_motion"] = virtual_root_motion.astype(np.float32)
        template["root_motion"] = (virtual_root_motion[:, :3, 3] - virtual_root_motion[[0], :3, 3]).astype(np.float32)
        template["fps"] = np.asarray([row["fps"]], dtype=np.float32)
        template["num_frames"] = np.asarray([len(world)], dtype=np.int64)
        output = COMPOSITIONS_ROOT / Path(composition_id).name / ("composed.npz" if include_transitions else "composed_no_transitions.npz")
        np.savez_compressed(output, **template)
        return output

    def delete(self, composition_id: str) -> None:
        root = (COMPOSITIONS_ROOT / Path(composition_id).name).resolve()
        if root.parent != COMPOSITIONS_ROOT.resolve() or not root.is_dir():
            raise KeyError(composition_id)
        shutil.rmtree(root)


class UpperLowerCompositePreviewStore:
    def __init__(self, animations: AnimationStore) -> None:
        self.animations = animations
        UPPER_LOWER_PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)

    def get(self, preview_id: str) -> dict[str, Any]:
        path = UPPER_LOWER_PREVIEWS_ROOT / Path(preview_id).name / "manifest.json"
        if not path.is_file():
            raise KeyError(preview_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def bundle_path(self, preview_id: str) -> Path:
        row = self.get(preview_id)
        path = UPPER_LOWER_PREVIEWS_ROOT / Path(preview_id).name / Path(row["bundle"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _upper_body_indices(names: list[str], parents: np.ndarray) -> list[int]:
        index = {name: i for i, name in enumerate(names)}
        upper_roots = [index[name] for name in ("Spine1", "Spine", "Spine2", "Chest") if name in index]
        if upper_roots:
            upper_set: set[int] = set()
            for joint in range(len(names)):
                current = joint
                while current >= 0:
                    if current in upper_roots:
                        upper_set.add(joint)
                        break
                    current = int(parents[current]) if current < len(parents) else -1
            return sorted(upper_set)
        lower_tokens = ("Leg", "Shin", "Foot", "Toe", "Hip", "Hips", "Pelvis")
        return [i for i, name in enumerate(names) if not any(token in name for token in lower_tokens)]

    def _build_payload(
        self,
        *,
        character_id: str,
        upper_animation_id: str,
        lower_animation_id: str,
        duration_sec: float,
    ) -> tuple[dict[str, np.ndarray], int, float]:
        upper_row = self.animations.get(upper_animation_id)
        lower_row = self.animations.get(lower_animation_id)
        if upper_row["character_id"] != character_id or lower_row["character_id"] != character_id:
            raise ValueError("upper and lower body animations must belong to the selected character")
        target_fps = 30.0
        frames = max(1, int(round(float(duration_sec) * target_fps)))
        upper_data, upper_world = self.animations.effective_world(upper_animation_id)
        lower_data, lower_world = self.animations.effective_world(lower_animation_id)
        try:
            upper_world = np.asarray(upper_world, dtype=np.float32)
            lower_world = np.asarray(lower_world, dtype=np.float32)
            upper_names = CompositionStore._decode_joint_names(upper_data["names"]) if "names" in upper_data.files else []
            lower_names = CompositionStore._decode_joint_names(lower_data["names"]) if "names" in lower_data.files else []
            if upper_names != lower_names:
                raise ValueError("upper and lower body animations must use the same skeleton")
            parents = np.asarray(lower_data["parents"], dtype=np.int64)
            if len(parents) != lower_world.shape[1] or upper_world.shape[1] != lower_world.shape[1]:
                raise ValueError("upper and lower body animations have incompatible joint counts")
            upper_clip = resample_world(upper_world, frames)
            lower_clip = resample_world(lower_world, frames)
            upper_local = local_from_world(upper_clip, parents)
            lower_local = local_from_world(lower_clip, parents)
            merged_local = lower_local.copy()
            upper_indices = self._upper_body_indices(lower_names, parents)
            if not upper_indices:
                raise ValueError("could not identify upper body joints from skeleton names")
            merged_local[:, upper_indices] = upper_local[:, upper_indices]
            merged_world = world_from_local(merged_local, parents).astype(np.float32)
            lower_virtual_root = self.animations.virtual_root_motion(lower_data)
            virtual_root = None
            if lower_virtual_root is not None and len(lower_virtual_root):
                virtual_root = resample_world(np.asarray(lower_virtual_root, dtype=np.float32)[:, None], frames)[:, 0]
            payload = {
                key: np.asarray(lower_data[key]).copy()
                for key in lower_data.files
                if key not in {
                    "world_motion",
                    "virtual_root_motion",
                    "root_motion",
                    "preview_world_motion",
                    "num_frames",
                    "fps",
                    "root_curve_control_points",
                    "root_curve_edit_control_points",
                }
            }
            payload["world_motion"] = merged_world.astype(np.float32)
            if virtual_root is not None:
                payload["virtual_root_motion"] = virtual_root.astype(np.float32)
                payload["root_motion"] = (virtual_root[:, :3, 3] - virtual_root[[0], :3, 3]).astype(np.float32)
            payload["fps"] = np.asarray([target_fps], dtype=np.float32)
            payload["num_frames"] = np.asarray([frames], dtype=np.int64)
            payload["motion_kind"] = np.asarray(["upper_lower_composite"], dtype="U64")
            payload["upper_body_source_animation_id"] = np.asarray([upper_animation_id], dtype="U64")
            payload["lower_body_source_animation_id"] = np.asarray([lower_animation_id], dtype="U64")
            payload["upper_lower_duration_sec"] = np.asarray([float(duration_sec)], dtype=np.float32)
            return payload, frames, target_fps
        finally:
            upper_data.close()
            lower_data.close()

    def create(self, character_id: str, upper_animation_id: str, lower_animation_id: str, duration_sec: float, name: str) -> dict[str, Any]:
        self.animations.characters.get(character_id)
        payload, frames, fps = self._build_payload(
            character_id=character_id,
            upper_animation_id=upper_animation_id,
            lower_animation_id=lower_animation_id,
            duration_sec=duration_sec,
        )
        preview_id = uuid.uuid4().hex[:12]
        root = UPPER_LOWER_PREVIEWS_ROOT / preview_id
        root.mkdir(parents=True, exist_ok=False)
        bundle = root / "upper_lower_preview.npz"
        np.savez_compressed(bundle, **payload)
        now = time.time()
        row = {
            "id": preview_id,
            "name": (name or "Upper Lower Composite").strip()[:80],
            "character_id": character_id,
            "upper_animation_id": upper_animation_id,
            "lower_animation_id": lower_animation_id,
            "duration_sec": float(duration_sec),
            "bundle": bundle.name,
            "frames": frames,
            "fps": fps,
            "created_at": now,
            "updated_at": now,
        }
        _write_json(root / "manifest.json", row)
        return row

    def commit(self, preview_id: str, name: str | None = None) -> dict[str, Any]:
        row = self.get(preview_id)
        with np.load(self.bundle_path(preview_id), allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]).copy() for key in data.files}
        animation = self.animations.create_from_payload(
            row["character_id"],
            (name or row["name"] or "Upper Lower Composite").strip()[:80],
            payload,
            motion_kind="upper_lower_composite",
            hidden=False,
        )
        self.delete(preview_id)
        return animation

    def delete(self, preview_id: str) -> None:
        root = (UPPER_LOWER_PREVIEWS_ROOT / Path(preview_id).name).resolve()
        if root.parent != UPPER_LOWER_PREVIEWS_ROOT.resolve() or not root.is_dir():
            raise KeyError(preview_id)
        shutil.rmtree(root)


class RetargetPreviewStore:
    def __init__(self, characters, motions, animations: AnimationStore, kimodo_motions=None) -> None:
        self.characters = characters
        self.motions = motions
        self.animations = animations
        self.kimodo_motions = kimodo_motions
        RETARGET_PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)

    def list(self, character_id: str | None = None) -> list[dict[str, Any]]:
        rows = []
        for path in RETARGET_PREVIEWS_ROOT.glob("*/manifest.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if character_id is None or row["character_id"] == character_id:
                    rows.append(row)
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        return rows

    def get(self, preview_id: str) -> dict[str, Any]:
        path = RETARGET_PREVIEWS_ROOT / Path(preview_id).name / "manifest.json"
        if not path.is_file():
            raise KeyError(preview_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def bundle_path(self, preview_id: str) -> Path:
        row = self.get(preview_id)
        path = RETARGET_PREVIEWS_ROOT / Path(preview_id).name / Path(row["bundle"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def create(
        self,
        source_motion_id: str,
        character_id: str,
        name: str,
        *,
        source_kind: str = "gvhmr",
        kimodo_motion_id: str = "",
    ) -> dict[str, Any]:
        source_kind = (source_kind or "gvhmr").strip().lower()
        if source_kind == "kimodo":
            if self.kimodo_motions is None:
                raise ValueError("Kimodo motion store is not available")
            motion = self.kimodo_motions.get(kimodo_motion_id or source_motion_id)
            source_path = self.kimodo_motions.bundle_path(motion)
            default_name = motion.get("name") or motion.get("prompt") or "Kimodo Motion"
            source_motion_id = ""
            kimodo_motion_id = motion["id"]
            retarget_fn = retarget_kimodo_bundle_fixed
        else:
            motion = self.motions.get(source_motion_id)
            source_path = self.motions.source_bundle_path(motion)
            default_name = motion.get("filename") or "GVHMR Motion"
            kimodo_motion_id = ""
            retarget_fn = retarget_source_bundle_fixed
        self.characters.get(character_id)
        preview_id = uuid.uuid4().hex[:12]
        root = RETARGET_PREVIEWS_ROOT / preview_id
        root.mkdir(parents=True, exist_ok=False)
        bundle = root / "preview.npz"
        retarget_fn(source_path, self.characters.binding_path(character_id), bundle)
        with np.load(bundle, allow_pickle=False) as data:
            frames = int(data["world_motion"].shape[0])
            fps = float(data["fps"].reshape(-1)[0])
        row = {
            "id": preview_id,
            "name": (name or default_name).strip()[:80],
            "source_motion_id": source_motion_id,
            "source_kind": source_kind,
            "kimodo_motion_id": kimodo_motion_id,
            "character_id": character_id,
            "bundle": bundle.name,
            "frames": frames,
            "fps": fps,
            "created_at": time.time(),
        }
        _write_json(root / "manifest.json", row)
        return row

    def commit(self, preview_id: str) -> dict[str, Any]:
        row = self.get(preview_id)
        animation = self.animations.create_from_preview(row, self.bundle_path(preview_id))
        self.delete(preview_id)
        return animation

    def delete(self, preview_id: str) -> None:
        root = (RETARGET_PREVIEWS_ROOT / Path(preview_id).name).resolve()
        if root.parent != RETARGET_PREVIEWS_ROOT.resolve() or not root.is_dir():
            raise KeyError(preview_id)
        shutil.rmtree(root)


class VirtualRootPreviewStore:
    def __init__(self, animations: AnimationStore) -> None:
        self.animations = animations
        VIRTUAL_ROOT_PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)

    def get(self, preview_id: str) -> dict[str, Any]:
        path = VIRTUAL_ROOT_PREVIEWS_ROOT / Path(preview_id).name / "manifest.json"
        if not path.is_file():
            raise KeyError(preview_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def bundle_path(self, preview_id: str) -> Path:
        row = self.get(preview_id)
        path = VIRTUAL_ROOT_PREVIEWS_ROOT / Path(preview_id).name / Path(row["bundle"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def create(
        self,
        source_animation_id: str,
        name: str,
        *,
        smoothing_frames: int = 9,
        position_tolerance: float = 0.005,
    ) -> dict[str, Any]:
        source = self.animations.get(source_animation_id)
        data, world = self.animations.effective_world(source_animation_id)
        try:
            if "virtual_root_motion" in data.files:
                original_world = np.einsum("fab,fjbc->fjac", data["virtual_root_motion"], world).astype(np.float32)
            else:
                original_world = np.asarray(world, dtype=np.float32)
                root = self.animations.root_motion(source_animation_id, data)
                if root is not None and len(root):
                    root_delta = np.asarray(root, dtype=np.float32)
                    if root_delta.shape[1] == 3:
                        original_world = original_world.copy()
                        original_world[:, :, :3, 3] += root_delta[:, None, :]
            names = [str(item) for item in data["names"].tolist()]
            target_bind = np.asarray(data["target_bind"], dtype=np.float32)
            reference_root_positions = None
            if "kimodo_smooth_root_motion" in data.files:
                lowered = [item.lower() for item in names]
                hip = lowered.index("hips") if "hips" in lowered else 0
                smooth_motion = np.asarray(data["kimodo_smooth_root_motion"], dtype=np.float32)
                if len(smooth_motion) == len(original_world) and smooth_motion.shape[1] == 3:
                    reference_root_positions = target_bind[hip, :3, 3][None] + smooth_motion
            result = build_virtual_root(
                original_world,
                np.asarray(data["parents"], dtype=np.int64),
                names,
                target_bind,
                rest_vertices=np.asarray(data["rest_vertices"], dtype=np.float32) if "rest_vertices" in data.files else None,
                lbs_indices=np.asarray(data["lbs_indices"], dtype=np.int64) if "lbs_indices" in data.files else None,
                lbs_weights=np.asarray(data["lbs_weights"], dtype=np.float32) if "lbs_weights" in data.files else None,
                reference_root_positions=reference_root_positions,
                smoothing_frames=smoothing_frames,
                position_tolerance=position_tolerance,
            )
            preview_id = uuid.uuid4().hex[:12]
            root_dir = VIRTUAL_ROOT_PREVIEWS_ROOT / preview_id
            root_dir.mkdir(parents=True, exist_ok=False)
            bundle = root_dir / "preview.npz"
            payload = {key: data[key] for key in data.files if key not in {"world_motion", "root_motion", "virtual_root_motion", "hip_local_under_virtual_root", "preview_world_motion"}}
            payload.update(
                world_motion=result.local_world_motion,
                preview_world_motion=result.preview_world_motion,
                virtual_root_motion=result.virtual_root_motion,
                hip_local_under_virtual_root=result.hip_local_under_virtual_root,
                root_motion=(result.virtual_root_motion[:, :3, 3] - result.virtual_root_motion[[0], :3, 3]).astype(np.float32),
                root_mode=np.asarray(["virtual_root"], dtype="U32"),
                virtual_root_version=np.asarray([result.version], dtype="U64"),
                virtual_root_ground_height=np.asarray([result.ground_height], dtype=np.float32),
                virtual_root_ground_heights=result.ground_heights.astype(np.float32),
                virtual_root_hip_index=np.asarray([result.hip_index], dtype=np.int64),
                num_frames=np.asarray([len(result.local_world_motion)], dtype=np.int64),
            )
            np.savez_compressed(bundle, **payload)
            now = time.time()
            row = {
                "id": preview_id,
                "name": (name or f"{source['name']} VirtualRoot").strip()[:80],
                "source_animation_id": source_animation_id,
                "source_motion_id": source.get("source_motion_id", ""),
                "character_id": source["character_id"],
                "bundle": bundle.name,
                "frames": int(result.local_world_motion.shape[0]),
                "fps": float(np.asarray(data["fps"]).reshape(-1)[0]),
                "virtual_root": True,
                "motion_kind": "virtual_root_preview",
                "smoothing_frames": int(smoothing_frames),
                "position_tolerance": float(position_tolerance),
                "created_at": now,
                "updated_at": now,
            }
            _write_json(root_dir / "manifest.json", row)
            return row
        finally:
            data.close()

    def commit(self, preview_id: str) -> dict[str, Any]:
        row = self.get(preview_id)
        animation = self.animations.create_from_virtual_root_preview(row, self.bundle_path(preview_id))
        self.delete(preview_id)
        return animation

    def replace_source(self, preview_id: str) -> dict[str, Any]:
        row = self.get(preview_id)
        source_id = row["source_animation_id"]
        source = self.animations.get(source_id)
        with np.load(self.bundle_path(preview_id), allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]).copy() for key in data.files if key != "preview_world_motion"}
        payload["motion_kind"] = np.asarray(["virtual_root"], dtype="U64")
        bundle = self.animations.bundle_path(source_id)
        np.savez_compressed(bundle, **payload)
        source.update(
            name=(row.get("name") or source.get("name") or "VirtualRoot 动作").strip()[:80],
            frames=int(np.asarray(payload["world_motion"]).shape[0]),
            fps=float(np.asarray(payload["fps"]).reshape(-1)[0]),
            virtual_root=True,
            motion_kind="virtual_root",
            updated_at=time.time(),
        )
        self.animations.edit_project(source_id).clear()
        self.animations.baked_edit_project(source_id).clear()
        _write_json(ANIMATIONS_ROOT / Path(source_id).name / "manifest.json", source)
        self.delete(preview_id)
        return source

    def delete(self, preview_id: str) -> None:
        root = (VIRTUAL_ROOT_PREVIEWS_ROOT / Path(preview_id).name).resolve()
        if root.parent != VIRTUAL_ROOT_PREVIEWS_ROOT.resolve() or not root.is_dir():
            raise KeyError(preview_id)
        shutil.rmtree(root)
