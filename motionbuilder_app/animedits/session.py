from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .edit import apply_joint_local_deltas, bake_overlays


@dataclass
class AnimEditsSession:
    bundle_path: Path
    rest_vertices: np.ndarray
    faces: np.ndarray
    lbs_indices: np.ndarray
    lbs_weights: np.ndarray
    target_bind: np.ndarray
    world_motion: np.ndarray
    parents: np.ndarray
    names: np.ndarray
    fps: float
    num_frames: int
    binding_path: str
    edit_overlays: dict[int, np.ndarray] = field(default_factory=dict)
    dirty: bool = False

    def meta_dict(self) -> dict:
        return {
            "num_frames": int(self.num_frames),
            "num_joints": int(self.world_motion.shape[1]),
            "fps": float(self.fps),
            "binding_path": self.binding_path,
            "bundle_path": str(self.bundle_path),
            "has_overlay": bool(self.edit_overlays),
            "dirty": bool(self.dirty),
        }

    def scene_dict(self) -> dict:
        bind_joints = self.target_bind[:, :3, 3].astype(np.float32)
        return {
            "vertices": self.rest_vertices.astype(np.float32).tolist(),
            "faces": self.faces.astype(np.int64).tolist(),
            "joints": bind_joints.tolist(),
            "parents": self.parents.astype(np.int64).tolist(),
            "names": [str(n) for n in self.names.tolist()],
            "bind_matrices": [m.reshape(-1).tolist() for m in self.target_bind.astype(np.float32)],
            "lbs_indices": self.lbs_indices.astype(np.int64).tolist(),
            "lbs_weights": self.lbs_weights.astype(np.float32).tolist(),
        }

    def edit_status_dict(self) -> dict:
        joints = [
            {"index": int(j), "name": str(self.names[j]) if j < len(self.names) else f"joint_{j}"}
            for j in sorted(self.edit_overlays.keys())
        ]
        return {
            "has_overlay": bool(self.edit_overlays),
            "dirty": bool(self.dirty),
            "joints": joints,
        }

    def set_overlay(self, joint_index: int, delta_local: np.ndarray) -> None:
        delta = np.asarray(delta_local, dtype=np.float32)
        if delta.shape == (16,):
            delta = delta.reshape(4, 4)
        if delta.shape != (4, 4):
            raise ValueError(f"delta_local must be [4,4] or 16 floats, got {delta.shape}")
        self.edit_overlays[int(joint_index)] = delta
        self.dirty = True

    def clear_overlays(self) -> None:
        self.edit_overlays.clear()
        self.dirty = False

    def effective_world(self, frame: int) -> np.ndarray:
        fi = max(0, min(int(frame), self.num_frames - 1))
        base = self.world_motion[fi]
        if not self.edit_overlays:
            return base
        return apply_joint_local_deltas(base, self.parents, self.edit_overlays)

    def bake_edits(self) -> None:
        if not self.edit_overlays:
            return
        self.world_motion = bake_overlays(self.world_motion, self.parents, self.edit_overlays)
        self.edit_overlays.clear()
        self.dirty = False
        self.save_bundle()

    def save_bundle(self) -> Path:
        out_path = Path(self.bundle_path).resolve()
        backup = out_path.with_name(f"{out_path.stem}_backup_{int(time.time() * 1000)}{out_path.suffix}")
        if out_path.is_file():
            backup.write_bytes(out_path.read_bytes())
        np.savez_compressed(
            out_path,
            rest_vertices=self.rest_vertices.astype(np.float32),
            faces=self.faces.astype(np.int64),
            lbs_indices=self.lbs_indices.astype(np.int64),
            lbs_weights=self.lbs_weights.astype(np.float32),
            target_bind=self.target_bind.astype(np.float32),
            world_motion=self.world_motion.astype(np.float32),
            parents=self.parents.astype(np.int64),
            names=np.array(self.names, dtype=object),
            fps=np.float32(self.fps),
            num_frames=np.int32(self.num_frames),
            binding_path=str(self.binding_path),
        )
        return out_path


def load_bundle(bundle_path: Path) -> AnimEditsSession:
    path = Path(bundle_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"bundle not found: {path}")
    data = np.load(path, allow_pickle=True)
    target_bind = np.asarray(data["target_bind"], dtype=np.float32)
    if target_bind.ndim == 2:
        target_bind = target_bind.reshape(-1, 4, 4)
    world = np.asarray(data["world_motion"], dtype=np.float32)
    if world.ndim == 2:
        world = world.reshape(-1, target_bind.shape[0], 4, 4)
    names = np.asarray(data["names"], dtype=object)
    return AnimEditsSession(
        bundle_path=path,
        rest_vertices=np.asarray(data["rest_vertices"], dtype=np.float32),
        faces=np.asarray(data["faces"], dtype=np.int64),
        lbs_indices=np.asarray(data["lbs_indices"], dtype=np.int64),
        lbs_weights=np.asarray(data["lbs_weights"], dtype=np.float32),
        target_bind=target_bind,
        world_motion=world,
        parents=np.asarray(data["parents"], dtype=np.int64),
        names=names,
        fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
        num_frames=int(np.asarray(data["num_frames"]).reshape(-1)[0]),
        binding_path=str(data.get("binding_path", "")),
    )
