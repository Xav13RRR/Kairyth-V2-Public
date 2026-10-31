# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simplified humanoid skeleton (UniRig-style naming) fitted to an imported mesh."""

from __future__ import annotations

import numpy as np
import torch

from kimodo.skeleton.base import SkeletonBase

# Mixamo-like hierarchy used as UniRig stand-in until full UniRig integration.
HUMANOID_BONE_HIERARCHY: list[tuple[str, str | None]] = [
    ("Hips", None),
    ("Spine", "Hips"),
    ("Spine1", "Spine"),
    ("Spine2", "Spine1"),
    ("Neck", "Spine2"),
    ("Head", "Neck"),
    ("LeftShoulder", "Spine2"),
    ("LeftArm", "LeftShoulder"),
    ("LeftForeArm", "LeftArm"),
    ("LeftHand", "LeftForeArm"),
    ("RightShoulder", "Spine2"),
    ("RightArm", "RightShoulder"),
    ("RightForeArm", "RightArm"),
    ("RightHand", "RightForeArm"),
    ("LeftUpLeg", "Hips"),
    ("LeftLeg", "LeftUpLeg"),
    ("LeftFoot", "LeftLeg"),
    ("RightUpLeg", "Hips"),
    ("RightLeg", "RightUpLeg"),
    ("RightFoot", "RightLeg"),
]


def fit_humanoid_joints_to_mesh(vertices: np.ndarray) -> np.ndarray:
    """Place template joints using mesh AABB (Y-up, meters)."""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    cx = (vmin[0] + vmax[0]) * 0.5
    cz = (vmin[2] + vmax[2]) * 0.5
    h = max(vmax[1] - vmin[1], 1e-3)
    w = max(vmax[0] - vmin[0], 1e-3)
    d = max(vmax[2] - vmin[2], 1e-3)
    shoulder_w = w * 0.22
    hip_w = w * 0.14

    y = lambda t: vmin[1] + h * t  # noqa: E731

    positions = {
        "Hips": np.array([cx, y(0.52), cz], dtype=np.float32),
        "Spine": np.array([cx, y(0.58), cz], dtype=np.float32),
        "Spine1": np.array([cx, y(0.66), cz], dtype=np.float32),
        "Spine2": np.array([cx, y(0.74), cz], dtype=np.float32),
        "Neck": np.array([cx, y(0.82), cz], dtype=np.float32),
        "Head": np.array([cx, y(0.92), cz], dtype=np.float32),
        "LeftShoulder": np.array([cx - shoulder_w, y(0.76), cz], dtype=np.float32),
        "LeftArm": np.array([cx - shoulder_w * 1.4, y(0.68), cz + d * 0.05], dtype=np.float32),
        "LeftForeArm": np.array([cx - shoulder_w * 1.9, y(0.58), cz + d * 0.08], dtype=np.float32),
        "LeftHand": np.array([cx - shoulder_w * 2.3, y(0.50), cz + d * 0.1], dtype=np.float32),
        "RightShoulder": np.array([cx + shoulder_w, y(0.76), cz], dtype=np.float32),
        "RightArm": np.array([cx + shoulder_w * 1.4, y(0.68), cz + d * 0.05], dtype=np.float32),
        "RightForeArm": np.array([cx + shoulder_w * 1.9, y(0.58), cz + d * 0.08], dtype=np.float32),
        "RightHand": np.array([cx + shoulder_w * 2.3, y(0.50), cz + d * 0.1], dtype=np.float32),
        "LeftUpLeg": np.array([cx - hip_w, y(0.48), cz], dtype=np.float32),
        "LeftLeg": np.array([cx - hip_w, y(0.28), cz], dtype=np.float32),
        "LeftFoot": np.array([cx - hip_w, y(0.04), cz + d * 0.06], dtype=np.float32),
        "RightUpLeg": np.array([cx + hip_w, y(0.48), cz], dtype=np.float32),
        "RightLeg": np.array([cx + hip_w, y(0.28), cz], dtype=np.float32),
        "RightFoot": np.array([cx + hip_w, y(0.04), cz + d * 0.06], dtype=np.float32),
    }
    joints = np.stack([positions[name] for name, _ in HUMANOID_BONE_HIERARCHY], axis=0)
    # Root at origin for SkeletonBase convention
    root = joints[0].copy()
    joints -= root
    return joints


def compute_segment_weights(
    vertices: np.ndarray,
    joint_positions: np.ndarray,
    joint_parents: np.ndarray,
    num_influences: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-distance weights along bone segments."""
    n_verts = vertices.shape[0]
    n_joints = joint_positions.shape[0]
    dist = np.full((n_verts, n_joints), np.inf, dtype=np.float32)

    for j in range(n_joints):
        p = joint_positions[j]
        parent = int(joint_parents[j])
        if parent < 0:
            dist[:, j] = np.linalg.norm(vertices - p, axis=1)
        else:
            a = joint_positions[parent]
            b = p
            ab = b - a
            ab_len_sq = max(float(np.dot(ab, ab)), 1e-8)
            t = np.clip(np.dot(vertices - a, ab) / ab_len_sq, 0.0, 1.0)
            closest = a + t[:, None] * ab
            dist[:, j] = np.linalg.norm(vertices - closest, axis=1)

    inv = 1.0 / (dist + 1e-4)
    top_idx = np.argsort(inv, axis=1)[:, -num_influences:]
    top_w = np.take_along_axis(inv, top_idx, axis=1)
    top_w /= top_w.sum(axis=1, keepdims=True)
    return top_idx.astype(np.int64), top_w.astype(np.float32)


def build_bind_matrices(joint_positions: np.ndarray, joint_parents: np.ndarray) -> np.ndarray:
    n = joint_positions.shape[0]
    mats = np.tile(np.eye(4, dtype=np.float32)[None], (n, 1, 1))
    for j in range(n):
        mats[j, :3, 3] = joint_positions[j]
    return mats


class UniRigHumanoidSkeleton(SkeletonBase):
    """Minimal SkeletonBase for visualizing UniRig-style humanoid rigs."""

    name = "unirig_humanoid"
    bone_order_names_with_parents = HUMANOID_BONE_HIERARCHY
    left_foot_joint_names = ["LeftFoot"]
    right_foot_joint_names = ["RightFoot"]
    left_hand_joint_names = ["LeftHand"]
    right_hand_joint_names = ["RightHand"]
    hip_joint_names = ["RightUpLeg", "LeftUpLeg"]

    def __init__(self, joint_positions: np.ndarray):
        super().__init__(folder=None, load=False)
        self.register_buffer(
            "neutral_joints",
            torch.tensor(joint_positions, dtype=torch.float32),
            persistent=False,
        )


def _pick_joint_names(candidates: list[str], joint_names: list[str]) -> list[str]:
    lower = {n.lower(): n for n in joint_names}
    out = []
    for c in candidates:
        if c in joint_names:
            out.append(c)
        elif c.lower() in lower:
            out.append(lower[c.lower()])
    return out


class GenericBindingSkeleton(SkeletonBase):
    """SkeletonBase built dynamically from a UniRigBinding joint hierarchy."""

    name = "generic_binding"

    def __init__(
        self,
        joint_names: list[str],
        joint_parents: np.ndarray,
        joint_positions: np.ndarray,
    ):
        hierarchy: list[tuple[str, str | None]] = []
        for i, name in enumerate(joint_names):
            p = int(joint_parents[i])
            parent_name = None if p < 0 else joint_names[p]
            hierarchy.append((name, parent_name))

        type(self).bone_order_names_with_parents = hierarchy
        self.left_foot_joint_names = _pick_joint_names(
            ["LeftFoot", "lfoot"],
            joint_names,
        ) or [joint_names[-1]]
        self.right_foot_joint_names = _pick_joint_names(
            ["RightFoot", "rfoot"],
            joint_names,
        ) or [joint_names[-1]]
        self.left_hand_joint_names = _pick_joint_names(
            ["LeftHand", "lhand"],
            joint_names,
        ) or [joint_names[0]]
        self.right_hand_joint_names = _pick_joint_names(
            ["RightHand", "rhand"],
            joint_names,
        ) or [joint_names[0]]
        right_hip = _pick_joint_names(["RightUpLeg", "rthigh"], joint_names)
        left_hip = _pick_joint_names(["LeftUpLeg", "lthigh"], joint_names)
        self.hip_joint_names = (right_hip[:1] + left_hip[:1]) or joint_names[:2]

        root_idx = int(np.where(joint_parents < 0)[0][0])
        rel = joint_positions.astype(np.float32) - joint_positions[root_idx].astype(np.float32)

        super().__init__(folder=None, load=False)
        self.register_buffer(
            "neutral_joints",
            torch.tensor(rel, dtype=torch.float32),
            persistent=False,
        )
