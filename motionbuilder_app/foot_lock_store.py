from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .foot_lock import FootLockSettings, process_foot_lock
from .library_stores import AnimationStore, _write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOOT_LOCK_PREVIEWS_ROOT = PROJECT_ROOT / "data" / "foot_lock_previews"


class FootLockPreviewStore:
    def __init__(self, animations: AnimationStore) -> None:
        self.animations = animations
        FOOT_LOCK_PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)

    def get(self, preview_id: str) -> dict[str, Any]:
        path = FOOT_LOCK_PREVIEWS_ROOT / Path(preview_id).name / "manifest.json"
        if not path.is_file():
            raise KeyError(preview_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def bundle_path(self, preview_id: str) -> Path:
        row = self.get(preview_id)
        path = FOOT_LOCK_PREVIEWS_ROOT / Path(preview_id).name / Path(row["bundle"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def create(
        self,
        animation_id: str,
        name: str,
        settings: FootLockSettings,
    ) -> dict[str, Any]:
        source = self.animations.get(animation_id)
        data, effective_world = self.animations.effective_world(animation_id)
        try:
            world = np.asarray(effective_world, dtype=np.float32)
            if "virtual_root_motion" in data.files:
                root = np.asarray(data["virtual_root_motion"], dtype=np.float32)
                if len(root) == len(world):
                    world = np.einsum("fab,fjbc->fjac", root, world).astype(np.float32)
            else:
                root_motion = self.animations.root_motion(animation_id, data)
                if root_motion is not None and len(root_motion) == len(world):
                    root_delta = np.asarray(root_motion, dtype=np.float32)
                    if root_delta.ndim == 2 and root_delta.shape[1] == 3:
                        world = world.copy()
                        world[:, :, :3, 3] += root_delta[:, None, :]
            names = [str(item) for item in data["names"].tolist()]
            result = process_foot_lock(
                world,
                np.asarray(data["parents"], dtype=np.int64),
                names,
                settings=settings,
            )
            excluded = {
                "world_motion",
                "preview_world_motion",
                "root_motion",
                "virtual_root_motion",
                "hip_local_under_virtual_root",
                "virtual_root_ground_heights",
                "foot_lock_residuals",
                "foot_lock_reached",
                "foot_lock_settings_json",
                "foot_lock_metrics_json",
                "foot_lock_targets_json",
            }
            payload = {key: np.asarray(data[key]).copy() for key in data.files if key not in excluded}
            payload.update(
                world_motion=result.local_world_motion,
                preview_world_motion=result.preview_world_motion,
                virtual_root_motion=result.virtual_root_motion,
                root_motion=np.zeros((len(result.virtual_root_motion), 3), dtype=np.float32),
                root_mode=np.asarray(["foot_lock_virtual_root"], dtype="U32"),
                virtual_root_version=np.asarray([result.version], dtype="U64"),
                foot_lock_residuals=result.residuals.astype(np.float32),
                foot_lock_reached=result.reached.astype(np.uint8),
                foot_lock_settings_json=np.asarray(
                    [json.dumps(result.settings, ensure_ascii=False, separators=(",", ":"))], dtype="U2048"
                ),
                foot_lock_metrics_json=np.asarray(
                    [json.dumps(result.metrics, ensure_ascii=False, separators=(",", ":"))], dtype="U2048"
                ),
                foot_lock_targets_json=np.asarray(
                    [json.dumps(result.reference_targets, ensure_ascii=False, separators=(",", ":"))], dtype="U2048"
                ),
                foot_lock_version=np.asarray([result.version], dtype="U64"),
                motion_kind=np.asarray(["foot_lock"], dtype="U32"),
                num_frames=np.asarray([len(result.local_world_motion)], dtype=np.int64),
            )
        finally:
            data.close()
        preview_id = uuid.uuid4().hex[:12]
        root_dir = FOOT_LOCK_PREVIEWS_ROOT / preview_id
        root_dir.mkdir(parents=True, exist_ok=False)
        bundle = root_dir / "preview.npz"
        np.savez_compressed(bundle, **payload)
        now = time.time()
        row = {
            "id": preview_id,
            "name": (name or f"{source['name']} Foot Lock").strip()[:80],
            "source_animation_id": animation_id,
            "source_motion_id": source.get("source_motion_id", ""),
            "character_id": source["character_id"],
            "bundle": bundle.name,
            "frames": int(len(result.local_world_motion)),
            "fps": float(np.asarray(payload["fps"]).reshape(-1)[0]),
            "virtual_root": True,
            "has_root_motion": True,
            "motion_kind": "foot_lock_preview",
            "settings": result.settings,
            "metrics": result.metrics,
            "reference_targets": result.reference_targets,
            "created_at": now,
            "updated_at": now,
        }
        _write_json(root_dir / "manifest.json", row)
        return row

    def commit(self, preview_id: str) -> dict[str, Any]:
        row = self.get(preview_id)
        with np.load(self.bundle_path(preview_id), allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]).copy() for key in data.files if key != "preview_world_motion"}
        animation = self.animations.create_from_payload(
            row["character_id"],
            row["name"],
            payload,
            motion_kind="foot_lock",
            hidden=False,
            metadata={
                "source_animation_id": row["source_animation_id"],
                "virtual_root": True,
                "has_root_motion": True,
                "foot_lock_settings": row.get("settings", {}),
                "foot_lock_metrics": row.get("metrics", {}),
                "foot_lock_reference_targets": row.get("reference_targets", {}),
                "virtual_root_source": "locked_feet_midpoint",
            },
        )
        self.delete(preview_id)
        return animation

    def delete(self, preview_id: str) -> None:
        root = (FOOT_LOCK_PREVIEWS_ROOT / Path(preview_id).name).resolve()
        if root.parent != FOOT_LOCK_PREVIEWS_ROOT.resolve() or not root.is_dir():
            raise KeyError(preview_id)
        shutil.rmtree(root)
