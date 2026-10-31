from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .gait_ik import GaitIKSettings, process_gait_ik
from .library_stores import AnimationStore, _write_json
from .terrain_store import TerrainStore
from .terrain_surface import PlaneSurface
from .timeline import resample_world


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAIT_IK_PREVIEWS_ROOT = PROJECT_ROOT / "data" / "gait_ik_previews"


def _retime_frame_array(values: np.ndarray, output_frames: int) -> np.ndarray:
    """Retimes auxiliary frame-domain payloads after a gait speed edit."""

    array = np.asarray(values)
    if len(array) == output_frames:
        return array.copy()
    if array.ndim >= 2 and array.shape[-2:] == (4, 4):
        if array.ndim == 3:
            return resample_world(array[:, None], output_frames)[:, 0].astype(array.dtype, copy=False)
        if array.ndim == 4:
            return resample_world(array, output_frames).astype(array.dtype, copy=False)
    target = np.linspace(0.0, max(0, len(array) - 1), output_frames)
    nearest = np.clip(np.rint(target).astype(np.int64), 0, max(0, len(array) - 1))
    if not np.issubdtype(array.dtype, np.number) or len(array) <= 1:
        return array[nearest].copy()
    source = np.arange(len(array), dtype=np.float64)
    flat = array.reshape(len(array), -1).astype(np.float64)
    sampled = np.stack(
        [np.interp(target, source, flat[:, column]) for column in range(flat.shape[1])], axis=1
    )
    if np.issubdtype(array.dtype, np.integer):
        sampled = np.rint(sampled)
    return sampled.reshape((output_frames, *array.shape[1:])).astype(array.dtype)


class GaitIKPreviewStore:
    def __init__(self, animations: AnimationStore, terrains: TerrainStore | None = None) -> None:
        self.animations = animations
        self.terrains = terrains
        GAIT_IK_PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)

    def get(self, preview_id: str) -> dict[str, Any]:
        path = GAIT_IK_PREVIEWS_ROOT / Path(preview_id).name / "manifest.json"
        if not path.is_file():
            raise KeyError(preview_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def bundle_path(self, preview_id: str) -> Path:
        row = self.get(preview_id)
        path = GAIT_IK_PREVIEWS_ROOT / Path(preview_id).name / Path(row["bundle"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def create(
        self,
        animation_id: str,
        name: str,
        settings: GaitIKSettings,
        *,
        terrain_id: str = "",
        initial_translation: list[float] | tuple[float, float, float] | np.ndarray | None = None,
        initial_yaw_degrees: float = 0.0,
    ) -> dict[str, Any]:
        source = self.animations.get(animation_id)
        if not source.get("has_root_motion"):
            raise ValueError("animation does not contain root motion")
        terrain: dict[str, Any] | None = None
        surface = PlaneSurface(0.0)
        if terrain_id:
            if self.terrains is None:
                raise ValueError("terrain storage is unavailable")
            terrain = self.terrains.get(terrain_id)
            surface = self.terrains.surface(terrain_id)

        data, effective_world = self.animations.effective_world(animation_id)
        try:
            world = np.asarray(effective_world, dtype=np.float32)
            reference_root_positions = None
            if "virtual_root_motion" in data.files:
                root = np.asarray(data["virtual_root_motion"], dtype=np.float32)
                world = np.einsum("fab,fjbc->fjac", root, world).astype(np.float32)
                reference_root_positions = root[:, :3, 3]
            else:
                root_motion = self.animations.root_motion(animation_id, data)
                if root_motion is not None and len(root_motion) == len(world):
                    root_delta = np.asarray(root_motion, dtype=np.float32)
                    if root_delta.ndim == 2 and root_delta.shape[1] == 3:
                        world = world.copy()
                        world[:, :, :3, 3] += root_delta[:, None, :]
                        names_lower = [str(item).lower() for item in data["names"].tolist()]
                        hip = names_lower.index("hips") if "hips" in names_lower else 0
                        reference_root_positions = data["target_bind"][hip, :3, 3][None] + root_delta
            names = [str(item) for item in data["names"].tolist()]
            result = process_gait_ik(
                world,
                np.asarray(data["parents"], dtype=np.int64),
                names,
                np.asarray(data["target_bind"], dtype=np.float32),
                float(np.asarray(data["fps"]).reshape(-1)[0]),
                settings=settings,
                rest_vertices=np.asarray(data["rest_vertices"], dtype=np.float32) if "rest_vertices" in data.files else None,
                lbs_indices=np.asarray(data["lbs_indices"], dtype=np.int64) if "lbs_indices" in data.files else None,
                lbs_weights=np.asarray(data["lbs_weights"], dtype=np.float32) if "lbs_weights" in data.files else None,
                reference_root_positions=reference_root_positions,
                surface=surface,
                initial_translation=initial_translation,
                initial_yaw_degrees=initial_yaw_degrees,
            )
            excluded = {
                "world_motion",
                "preview_world_motion",
                "root_motion",
                "virtual_root_motion",
                "hip_local_under_virtual_root",
                "virtual_root_ground_heights",
                "gait_contacts",
                "gait_contact_weights",
                "gait_pelvis_correction",
                "gait_root_ground_heights",
                "gait_foot_ground_heights",
                "gait_foot_ground_normals",
                "gait_terrain_hits",
                "gait_initial_transform_json",
                "gait_settings_json",
                "gait_metrics_json",
            }
            payload = {
                key: np.asarray(data[key])
                for key in data.files
                if key not in excluded and not key.startswith("terrain_")
            }
            source_frames = int(len(world))
            output_frames = int(len(result.local_world_motion))
            if output_frames != source_frames:
                for key, values in list(payload.items()):
                    if values.ndim and len(values) == source_frames and key not in {"names", "parents"}:
                        payload[key] = _retime_frame_array(values, output_frames)
            payload.update(
                world_motion=result.local_world_motion,
                preview_world_motion=result.preview_world_motion,
                virtual_root_motion=result.virtual_root_motion,
                root_motion=(
                    result.virtual_root_motion[:, :3, 3] - result.virtual_root_motion[[0], :3, 3]
                ).astype(np.float32),
                root_mode=np.asarray(["virtual_root"], dtype="U32"),
                virtual_root_version=np.asarray([result.version], dtype="U64"),
                virtual_root_ground_height=np.asarray([result.ground_height], dtype=np.float32),
                virtual_root_ground_heights=result.root_ground_heights.astype(np.float32),
                gait_contacts=result.contacts.astype(np.uint8),
                gait_contact_weights=result.contact_weights.astype(np.float32),
                gait_pelvis_correction=result.pelvis_correction.astype(np.float32),
                gait_root_ground_heights=result.root_ground_heights.astype(np.float32),
                gait_foot_ground_heights=result.foot_ground_heights.astype(np.float32),
                gait_foot_ground_normals=result.foot_ground_normals.astype(np.float32),
                gait_terrain_hits=result.terrain_hits.astype(np.uint8),
                gait_initial_transform_json=np.asarray(
                    [json.dumps(result.initial_transform, ensure_ascii=False, separators=(",", ":"))], dtype="U512"
                ),
                gait_settings_json=np.asarray(
                    [json.dumps(result.settings, ensure_ascii=False, separators=(",", ":"))], dtype="U2048"
                ),
                gait_metrics_json=np.asarray(
                    [json.dumps(result.metrics, ensure_ascii=False, separators=(",", ":"))], dtype="U2048"
                ),
                gait_ik_version=np.asarray([result.version], dtype="U64"),
                motion_kind=np.asarray(["gait_ik"], dtype="U32"),
                num_frames=np.asarray([len(result.local_world_motion)], dtype=np.int64),
            )
            if terrain is not None:
                terrain_heights = self.terrains.load_heights(terrain_id) if self.terrains is not None else surface.heights
                terrain_transform = terrain.get("transform") or {}
                payload.update(
                    terrain_heights=np.asarray(terrain_heights, dtype=np.float32),
                    terrain_size=np.asarray([terrain["size_x"], terrain["size_z"]], dtype=np.float32),
                    terrain_translation=np.asarray(terrain_transform.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32),
                    terrain_yaw_degrees=np.asarray([terrain_transform.get("yaw_degrees", 0.0)], dtype=np.float32),
                    terrain_uniform_scale=np.asarray([terrain_transform.get("uniform_scale", 1.0)], dtype=np.float32),
                    terrain_source_id=np.asarray([terrain["id"]], dtype="U64"),
                    terrain_source_name=np.asarray([terrain["name"]], dtype="U128"),
                    terrain_source_revision=np.asarray([int(terrain.get("revision", 1))], dtype=np.int64),
                    terrain_snapshot_hash=np.asarray([terrain.get("content_hash", "")], dtype="U128"),
                )
        finally:
            data.close()

        preview_id = uuid.uuid4().hex[:12]
        root_dir = GAIT_IK_PREVIEWS_ROOT / preview_id
        root_dir.mkdir(parents=True, exist_ok=False)
        bundle = root_dir / "preview.npz"
        np.savez_compressed(bundle, **payload)
        now = time.time()
        row = {
            "id": preview_id,
            "name": (name or f"{source['name']} Gait IK").strip()[:80],
            "source_animation_id": animation_id,
            "source_motion_id": source.get("source_motion_id", ""),
            "character_id": source["character_id"],
            "bundle": bundle.name,
            "frames": int(result.local_world_motion.shape[0]),
            "fps": float(np.asarray(payload["fps"]).reshape(-1)[0]),
            "virtual_root": True,
            "has_root_motion": True,
            "grounded_root": True,
            "motion_kind": "gait_ik_preview",
            "surface_kind": "heightfield" if terrain is not None else "plane",
            "terrain_id": terrain["id"] if terrain is not None else "",
            "terrain_name": terrain["name"] if terrain is not None else "",
            "terrain_revision": int(terrain.get("revision", 1)) if terrain is not None else 0,
            "terrain_snapshot_hash": terrain.get("content_hash", "") if terrain is not None else "",
            "terrain_transform": terrain.get("transform", {}) if terrain is not None else {},
            "initial_transform": result.initial_transform,
            "settings": result.settings,
            "playback_speed": float(result.settings.get("playback_speed", 1.0)),
            "metrics": result.metrics,
            "created_at": now,
            "updated_at": now,
        }
        _write_json(root_dir / "manifest.json", row)
        return row

    def commit(self, preview_id: str, *, include_terrain: bool = False) -> dict[str, Any]:
        row = self.get(preview_id)
        if include_terrain and not row.get("terrain_id"):
            raise ValueError("a terrain asset must be selected before saving with terrain")
        with np.load(self.bundle_path(preview_id), allow_pickle=False) as data:
            payload = {
                key: np.asarray(data[key]).copy()
                for key in data.files
                if key != "preview_world_motion" and (include_terrain or not key.startswith("terrain_"))
            }
        metadata = {
            "source_animation_id": row["source_animation_id"],
            "virtual_root": True,
            "has_root_motion": True,
            "grounded_root": True,
            "gait_settings": row.get("settings", {}),
            "gait_metrics": row.get("metrics", {}),
            "initial_transform": row.get("initial_transform", {}),
            "terrain_attached": bool(include_terrain),
        }
        if include_terrain:
            metadata["terrain_attachment"] = {
                "source_terrain_id": row.get("terrain_id", ""),
                "source_terrain_name": row.get("terrain_name", ""),
                "source_revision": int(row.get("terrain_revision", 0)),
                "snapshot_hash": row.get("terrain_snapshot_hash", ""),
                "transform": row.get("terrain_transform", {}),
                "initial_transform": row.get("initial_transform", {}),
                "display_by_default": True,
            }
        animation = self.animations.create_from_payload(
            row["character_id"],
            row["name"],
            payload,
            motion_kind="gait_ik",
            hidden=False,
            metadata=metadata,
        )
        self.delete(preview_id)
        return animation

    def delete(self, preview_id: str) -> None:
        root = (GAIT_IK_PREVIEWS_ROOT / Path(preview_id).name).resolve()
        if root.parent != GAIT_IK_PREVIEWS_ROOT.resolve() or not root.is_dir():
            raise KeyError(preview_id)
        shutil.rmtree(root)
