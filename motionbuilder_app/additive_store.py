from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .additive import (
    ADDITIVE_END_BEHAVIORS,
    ADDITIVE_REFERENCE_TYPES,
    build_additive_world,
    sample_base_transform_track,
)
from .library_stores import AnimationStore, CompositionStore, PROJECT_ROOT, _write_json


ADDITIVE_PREVIEWS_ROOT = PROJECT_ROOT / "data" / "additive_previews"


class AdditiveCompositePreviewStore:
    """Builds disposable full-body additive previews and commits them as animations."""

    def __init__(self, animations: AnimationStore) -> None:
        self.animations = animations
        ADDITIVE_PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)

    def get(self, preview_id: str) -> dict[str, Any]:
        path = ADDITIVE_PREVIEWS_ROOT / Path(preview_id).name / "manifest.json"
        if not path.is_file():
            raise KeyError(preview_id)
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    def bundle_path(self, preview_id: str) -> Path:
        row = self.get(preview_id)
        path = ADDITIVE_PREVIEWS_ROOT / Path(preview_id).name / Path(row["bundle"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _fps(data: np.lib.npyio.NpzFile) -> float:
        return float(np.asarray(data["fps"]).reshape(-1)[0])

    def _reference_pose(
        self,
        character_id: str,
        base_data: np.lib.npyio.NpzFile,
        joint_count: int,
    ) -> np.ndarray:
        if "bind_matrices" in base_data.files:
            matrices = np.asarray(base_data["bind_matrices"], dtype=np.float32)
            if matrices.shape == (joint_count, 4, 4):
                return matrices
        with np.load(self.animations.characters.binding_path(character_id), allow_pickle=False) as binding:
            matrices = np.asarray(binding["bind_matrices"], dtype=np.float32)
        if matrices.shape != (joint_count, 4, 4):
            raise ValueError("character reference pose is incompatible with the animation skeleton")
        return matrices

    def _build_payload(
        self,
        *,
        character_id: str,
        base_animation_id: str,
        additive_animation_id: str,
        base_speed: float,
        additive_speed: float,
        weight: float,
        reference_type: str,
        reference_frame: int,
        end_behavior: str,
    ) -> tuple[dict[str, np.ndarray], int, float]:
        base_row = self.animations.get(base_animation_id)
        additive_row = self.animations.get(additive_animation_id)
        if base_row["character_id"] != character_id or additive_row["character_id"] != character_id:
            raise ValueError("base and additive animations must belong to the selected character")
        if reference_type not in ADDITIVE_REFERENCE_TYPES:
            raise ValueError("unsupported additive reference type")
        if end_behavior not in ADDITIVE_END_BEHAVIORS:
            raise ValueError("unsupported additive end behavior")

        base_data, base_world = self.animations.effective_world(base_animation_id)
        additive_data, additive_world = self.animations.effective_world(additive_animation_id)
        try:
            base_names = CompositionStore._decode_joint_names(base_data["names"])
            additive_names = CompositionStore._decode_joint_names(additive_data["names"])
            if base_names != additive_names:
                raise ValueError("base and additive animations must use the same skeleton")
            parents = np.asarray(base_data["parents"], dtype=np.int64)
            additive_parents = np.asarray(additive_data["parents"], dtype=np.int64)
            if not np.array_equal(parents, additive_parents):
                raise ValueError("base and additive animations must use the same hierarchy")

            target_fps = self._fps(base_data)
            reference_world = None
            if reference_type == "ref_pose":
                reference_world = self._reference_pose(character_id, base_data, len(parents))
            output_world = build_additive_world(
                np.asarray(base_world, dtype=np.float32),
                np.asarray(additive_world, dtype=np.float32),
                parents,
                base_fps=target_fps,
                additive_fps=self._fps(additive_data),
                target_fps=target_fps,
                base_speed=float(base_speed),
                additive_speed=float(additive_speed),
                weight=float(weight),
                reference_type=reference_type,
                reference_frame=int(reference_frame),
                reference_world=reference_world,
                end_behavior=end_behavior,
            )
            frames = int(len(output_world))
            excluded = {
                "world_motion",
                "virtual_root_motion",
                "root_motion",
                "preview_world_motion",
                "num_frames",
                "fps",
                "root_curve_control_points",
                "root_curve_edit_control_points",
            }
            payload = {
                key: np.asarray(base_data[key]).copy()
                for key in base_data.files
                if key not in excluded
            }
            payload["world_motion"] = output_world
            if "virtual_root_motion" in base_data.files:
                virtual_root = sample_base_transform_track(base_data["virtual_root_motion"], frames)
                payload["virtual_root_motion"] = virtual_root.astype(np.float32)
                payload["root_motion"] = (
                    virtual_root[:, :3, 3] - virtual_root[[0], :3, 3]
                ).astype(np.float32)
            elif "root_motion" in base_data.files:
                root_motion = np.asarray(base_data["root_motion"], dtype=np.float32)
                if root_motion.ndim == 2 and root_motion.shape[1] == 3:
                    positions = np.linspace(0, len(root_motion) - 1, frames)
                    payload["root_motion"] = np.stack(
                        [np.interp(positions, np.arange(len(root_motion)), root_motion[:, axis]) for axis in range(3)],
                        axis=1,
                    ).astype(np.float32)
            payload["fps"] = np.asarray([target_fps], dtype=np.float32)
            payload["num_frames"] = np.asarray([frames], dtype=np.int64)
            payload["motion_kind"] = np.asarray(["additive_composite"], dtype="U64")
            payload["base_animation_id"] = np.asarray([base_animation_id], dtype="U64")
            payload["additive_animation_id"] = np.asarray([additive_animation_id], dtype="U64")
            payload["additive_base_speed"] = np.asarray([base_speed], dtype=np.float32)
            payload["additive_layer_speed"] = np.asarray([additive_speed], dtype=np.float32)
            payload["additive_weight"] = np.asarray([weight], dtype=np.float32)
            payload["additive_reference_type"] = np.asarray([reference_type], dtype="U32")
            payload["additive_reference_frame"] = np.asarray([reference_frame], dtype=np.int64)
            payload["additive_end_behavior"] = np.asarray([end_behavior], dtype="U16")
            return payload, frames, target_fps
        finally:
            base_data.close()
            additive_data.close()

    def create(
        self,
        character_id: str,
        base_animation_id: str,
        additive_animation_id: str,
        *,
        base_speed: float = 1.0,
        additive_speed: float = 1.0,
        weight: float = 1.0,
        reference_type: str = "first_frame",
        reference_frame: int = 0,
        end_behavior: str = "loop",
        name: str = "加法叠加动画",
    ) -> dict[str, Any]:
        self.animations.characters.get(character_id)
        payload, frames, fps = self._build_payload(
            character_id=character_id,
            base_animation_id=base_animation_id,
            additive_animation_id=additive_animation_id,
            base_speed=base_speed,
            additive_speed=additive_speed,
            weight=weight,
            reference_type=reference_type,
            reference_frame=reference_frame,
            end_behavior=end_behavior,
        )
        preview_id = uuid.uuid4().hex[:12]
        root = ADDITIVE_PREVIEWS_ROOT / preview_id
        root.mkdir(parents=True, exist_ok=False)
        bundle = root / "additive_preview.npz"
        np.savez_compressed(bundle, **payload)
        now = time.time()
        row = {
            "id": preview_id,
            "name": (name or "Additive Composite").strip()[:80],
            "character_id": character_id,
            "base_animation_id": base_animation_id,
            "additive_animation_id": additive_animation_id,
            "base_speed": float(base_speed),
            "additive_speed": float(additive_speed),
            "weight": float(weight),
            "reference_type": reference_type,
            "reference_frame": int(reference_frame),
            "end_behavior": end_behavior,
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
            (name or row["name"] or "Additive Composite").strip()[:80],
            payload,
            motion_kind="additive_composite",
            hidden=False,
        )
        self.delete(preview_id)
        return animation

    def delete(self, preview_id: str) -> None:
        root = (ADDITIVE_PREVIEWS_ROOT / Path(preview_id).name).resolve()
        if root.parent != ADDITIVE_PREVIEWS_ROOT.resolve() or not root.is_dir():
            raise KeyError(preview_id)
        shutil.rmtree(root)
