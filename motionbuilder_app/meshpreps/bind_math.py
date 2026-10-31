from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BindingResult:
    vertices: np.ndarray
    faces: np.ndarray
    joint_positions: np.ndarray
    joint_parents: np.ndarray
    joint_names: list[str]
    lbs_indices: np.ndarray
    lbs_weights: np.ndarray
    bind_matrices: np.ndarray
    source_path: Path | None
    method: str
    align_offsets: np.ndarray | None = None
    base_target_bind: np.ndarray | None = None
    source_rest_bind: np.ndarray | None = None
    ik_profile: dict[str, Any] | None = None


def parents_to_array(parents_obj) -> np.ndarray:
    if parents_obj is None:
        return np.array([], dtype=np.int64)
    out: list[int] = []
    for p in parents_obj:
        if p is None:
            out.append(-1)
        else:
            out.append(int(p))
    return np.array(out, dtype=np.int64)


def normalize_parents(parents_obj) -> np.ndarray:
    return parents_to_array(parents_obj)


def top_k_skin(skin: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    n_verts, n_joints = skin.shape
    k = min(k, n_joints)
    top_idx = np.argsort(skin, axis=1)[:, -k:]
    top_w = np.take_along_axis(skin, top_idx, axis=1)
    top_w = top_w / np.maximum(top_w.sum(axis=1, keepdims=True), 1e-8)
    return top_idx.astype(np.int64), top_w.astype(np.float32)


def normalize_bind_matrices(bind_matrices) -> np.ndarray:
    mats = np.asarray(bind_matrices, dtype=np.float64)
    if mats.ndim == 2 and mats.shape == (4, 4):
        mats = mats.reshape(1, 4, 4)
    elif mats.ndim == 2 and mats.shape[1] == 16:
        mats = mats.reshape(-1, 4, 4)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"bind_matrices must be [J, 4, 4], got {mats.shape}")
    return mats.astype(np.float32)


def joint_positions_from_bind_matrices(bind_matrices: np.ndarray) -> np.ndarray:
    mats = normalize_bind_matrices(bind_matrices)
    return mats[:, :3, 3].astype(np.float32)


def bind_pose_motion(bind_matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-frame posed joints / global rotations matching meshpreps preview bind pose."""
    mats = normalize_bind_matrices(bind_matrices)
    n_joints = int(mats.shape[0])
    posed = mats[:, :3, 3].reshape(1, n_joints, 3).astype(np.float32)
    rots = mats[:, :3, :3].reshape(1, n_joints, 3, 3).astype(np.float32)
    return posed, rots


def build_bind_matrices(joint_positions: np.ndarray, joint_parents: np.ndarray) -> np.ndarray:
    """World-space bind pose 4x4 matrices from joint world positions and parent indices."""
    positions = np.asarray(joint_positions, dtype=np.float64)
    parents = np.asarray(joint_parents, dtype=np.int64)
    n = positions.shape[0]

    local_mats: list[np.ndarray] = []
    for j in range(n):
        local = np.eye(4, dtype=np.float64)
        p = int(parents[j])
        if p < 0:
            local[:3, 3] = positions[j]
        else:
            local[:3, 3] = positions[j] - positions[p]
        local_mats.append(local)

    world = np.zeros((n, 4, 4), dtype=np.float64)
    for j in range(n):
        p = int(parents[j])
        world[j] = local_mats[j] if p < 0 else world[p] @ local_mats[j]
    return world.astype(np.float32)


def apply_lbs_numpy(
    vertices: np.ndarray,
    lbs_indices: np.ndarray,
    lbs_weights: np.ndarray,
    bind_matrices: np.ndarray,
    joint_world_mats: np.ndarray,
) -> np.ndarray:
    """CPU LBS matching meshpreps rig_tools.js (_applyLbs)."""
    bind_inv = np.linalg.inv(bind_matrices)
    skin_mats = joint_world_mats @ bind_inv
    out = np.zeros_like(vertices, dtype=np.float64)
    n_verts = vertices.shape[0]
    for i in range(n_verts):
        v = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1.0], dtype=np.float64)
        acc = np.zeros(4, dtype=np.float64)
        for k in range(lbs_indices.shape[1]):
            j = int(lbs_indices[i, k])
            w = float(lbs_weights[i, k])
            if w < 1e-7:
                continue
            acc += w * (skin_mats[j] @ v)
        out[i] = acc[:3]
    return out.astype(np.float32)


def build_binding(skel: dict, skin: np.ndarray, *, method: str, source_path: Path | None) -> BindingResult:
    vertices = skel["vertices"]
    joints = skel["joint_positions"]
    parents = skel["joint_parents"]

    if skin.ndim != 2:
        raise ValueError(f"skin 维度异常：{skin.shape}")
    if skin.shape[0] == joints.shape[0] and skin.shape[1] != joints.shape[0]:
        skin = skin.T
    if skin.shape[0] != vertices.shape[0]:
        raise ValueError(f"skin 顶点数 {skin.shape[0]} 与 mesh {vertices.shape[0]} 不一致")
    if skin.shape[1] != joints.shape[0]:
        raise ValueError(f"skin 骨骼数 {skin.shape[1]} 与 skeleton {joints.shape[0]} 不一致")

    lbs_indices, lbs_weights = top_k_skin(skin, k=4)
    raw_bind = skel.get("bind_matrices")
    if raw_bind is not None:
        bind_matrices = normalize_bind_matrices(raw_bind)
        if bind_matrices.shape[0] != joints.shape[0]:
            raise ValueError(
                f"bind_matrices joints {bind_matrices.shape[0]} != skeleton {joints.shape[0]}"
            )
        joints = joint_positions_from_bind_matrices(bind_matrices)
    else:
        bind_matrices = build_bind_matrices(joints, parents)
    return BindingResult(
        vertices=vertices,
        faces=skel["faces"],
        joint_positions=joints,
        joint_parents=parents,
        joint_names=skel["joint_names"],
        lbs_indices=lbs_indices,
        lbs_weights=lbs_weights,
        bind_matrices=bind_matrices,
        source_path=source_path,
        method=method,
    )
