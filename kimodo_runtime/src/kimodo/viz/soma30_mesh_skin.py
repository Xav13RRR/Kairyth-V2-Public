# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from kimodo.rigging.soma30_retarget import (
    retarget_motion_to_target_bind,
    retargeted_joint_world_positions,
)
from kimodo.rigging.types import RemappedSOMABinding


def apply_lbs_numpy(
    vertices: np.ndarray,
    lbs_indices: np.ndarray,
    lbs_weights: np.ndarray,
    skin_matrices: np.ndarray,
) -> np.ndarray:
    """LBS with precomputed skin matrices (joint_world @ bind_inv baked into skin_matrices)."""
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
            acc += w * (skin_matrices[j] @ v)
        out[i] = acc[:3]
    return out.astype(np.float32)


class Soma30MeshSkin:
    """LBS for meshpreps SOMA-30 binding with retargeted animation."""

    def __init__(self, binding: RemappedSOMABinding) -> None:
        if not binding.has_soma30_rig():
            raise ValueError("RemappedSOMABinding missing SOMA-30 rig fields")
        self.binding = binding
        verts = np.asarray(binding.soma30_rest_vertices, dtype=np.float32).copy()
        scale = float(binding.soma_scale)
        verts *= scale
        if binding.soma_root_offset is not None:
            verts = verts + np.asarray(binding.soma_root_offset, dtype=np.float32)
        self.rest_vertices_mesh = verts
        self.faces = np.asarray(binding.faces, dtype=np.int64)
        self.lbs_indices = np.asarray(binding.soma30_lbs_indices, dtype=np.int64)
        self.lbs_weights = np.asarray(binding.soma30_lbs_weights, dtype=np.float32)
        self.target_bind = np.asarray(binding.soma30_bind_matrices, dtype=np.float32)
        if self.target_bind.ndim == 2:
            self.target_bind = self.target_bind.reshape(-1, 4, 4)
        self.bind_inv = np.linalg.inv(self.target_bind)

    def rest_vertices(self) -> np.ndarray:
        return self.rest_vertices_mesh.copy()

    def skin_frames(
        self,
        posed_joints: np.ndarray,
        global_rot_mats: np.ndarray,
        start: int,
        end: int,
    ) -> list[np.ndarray]:
        all_skin = retarget_motion_to_target_bind(
            self.binding,
            posed_joints,
            global_rot_mats,
        )
        n_frames = all_skin.shape[0]
        start = max(0, start)
        end = min(n_frames, end)
        out: list[np.ndarray] = []
        for fi in range(start, end):
            verts = apply_lbs_numpy(
                self.rest_vertices_mesh,
                self.lbs_indices,
                self.lbs_weights,
                all_skin[fi],
            )
            out.append(verts)
        return out

    def retargeted_joint_positions(
        self,
        posed_joints: np.ndarray,
        global_rot_mats: np.ndarray,
    ) -> np.ndarray:
        """World joint positions [F, J, 3] on target bind hierarchy after retarget."""
        return retargeted_joint_world_positions(self.binding, posed_joints, global_rot_mats)
