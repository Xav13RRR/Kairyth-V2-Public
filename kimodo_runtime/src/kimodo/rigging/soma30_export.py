# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol

import numpy as np

from .soma_remap import (
    MAX_INFLUENCES,
    _build_rest_transform,
    _load_soma_skin_meta,
    _nearest_skin_ancestor,
    _solve_bind_vertices,
)
from .soma30_retarget import compute_align_offsets, source_bind_matrices, source_rest_preview_dict
from .soma_retarget_profile import SOMA_WEIGHT_EXCLUDE
from .types import RemappedSOMABinding


class Soma30BindingLike(Protocol):
    vertices: np.ndarray
    faces: np.ndarray
    joint_names: list[str]
    lbs_indices: np.ndarray
    lbs_weights: np.ndarray


def _build_soma30_to_soma77_lut(joint_names: list[str], soma_names: list[str]) -> list[int]:
    soma_name_to_idx = {n: i for i, n in enumerate(soma_names)}
    lut: list[int] = []
    for name in joint_names:
        if name not in soma_name_to_idx:
            raise ValueError(f"SOMA-30 joint {name!r} not found in SOMA-77 skeleton")
        lut.append(soma_name_to_idx[name])
    return lut


def _remap_soma30_weights_sanitized(
    lbs_indices: np.ndarray,
    lbs_weights: np.ndarray,
    soma30_to_soma77: list[int],
    soma_names: list[str],
    soma_parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_verts = lbs_indices.shape[0]
    n_soma = len(soma_names)
    soma_w = np.zeros((n_verts, n_soma), dtype=np.float64)
    lut = np.array(soma30_to_soma77, dtype=np.int64)
    n_influences = lbs_indices.shape[1]
    for k in range(n_influences):
        bone_idx = lbs_indices[:, k].astype(np.int64)
        soma_j = lut[bone_idx]
        np.add.at(soma_w, (np.arange(n_verts), soma_j), lbs_weights[:, k].astype(np.float64))

    exclude_idx = [i for i, name in enumerate(soma_names) if name in SOMA_WEIGHT_EXCLUDE]
    for j in exclude_idx:
        col = soma_w[:, j]
        if not np.any(col > 1e-8):
            continue
        target = _nearest_skin_ancestor(j, soma_names, soma_parents)
        if target != j:
            soma_w[:, target] += col
        soma_w[:, j] = 0.0

    top_idx = np.argsort(soma_w, axis=1)[:, -MAX_INFLUENCES:]
    top_w = np.take_along_axis(soma_w, top_idx, axis=1).astype(np.float32)
    row_sum = top_w.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-8] = 1.0
    top_w = top_w / row_sum
    return top_idx.astype(np.int64), top_w


def meshpreps_binding_to_remapped(binding: Soma30BindingLike) -> RemappedSOMABinding:
    """Convert meshpreps SOMA-30 binding to Kimodo RemappedSOMABinding (SOMA-77)."""
    soma_bind, soma_names, soma_parents = _load_soma_skin_meta()
    rest_transform = _build_rest_transform(soma_names)
    soma30_to_soma77 = _build_soma30_to_soma77_lut(binding.joint_names, soma_names)

    top_idx, top_w = _remap_soma30_weights_sanitized(
        np.asarray(binding.lbs_indices, dtype=np.int64),
        np.asarray(binding.lbs_weights, dtype=np.float32),
        soma30_to_soma77,
        soma_names,
        soma_parents,
    )

    rest_target = np.asarray(binding.vertices, dtype=np.float32)
    bind_vertices = _solve_bind_vertices(rest_target, top_idx, top_w, soma_bind, rest_transform)

    bone_map = {name: name for name in binding.joint_names if name in soma_names}
    bone_map_modes = {name: "name" for name in bone_map}

    joints = np.asarray(binding.joint_positions, dtype=np.float32)
    parents = np.asarray(binding.joint_parents, dtype=np.int64)
    bind_m = np.asarray(binding.bind_matrices, dtype=np.float32)
    if bind_m.ndim == 2:
        bind_m = bind_m.reshape(-1, 4, 4)
    source_rest = source_bind_matrices()
    align_offsets = None
    base_target = bind_m.copy()
    if getattr(binding, "align_offsets", None) is not None:
        align_offsets = np.asarray(binding.align_offsets, dtype=np.float32).copy()
    if getattr(binding, "base_target_bind", None) is not None:
        base_target = np.asarray(binding.base_target_bind, dtype=np.float32).copy()
    if getattr(binding, "source_rest_bind", None) is not None:
        source_rest = np.asarray(binding.source_rest_bind, dtype=np.float32).copy()

    return RemappedSOMABinding(
        vertices=bind_vertices,
        faces=np.asarray(binding.faces, dtype=np.int64).copy(),
        lbs_indices=top_idx,
        lbs_weights=top_w,
        bind_rig_transform=soma_bind.astype(np.float32),
        bone_map=bone_map,
        bone_map_modes=bone_map_modes,
        soma_scale=1.0,
        soma_root_offset=np.zeros(3, dtype=np.float32),
        soma30_joint_positions=joints.copy(),
        soma30_joint_parents=parents.copy(),
        soma30_joint_names=list(binding.joint_names),
        soma30_bind_matrices=bind_m.copy(),
        soma30_lbs_indices=np.asarray(binding.lbs_indices, dtype=np.int64).copy(),
        soma30_lbs_weights=np.asarray(binding.lbs_weights, dtype=np.float32).copy(),
        soma30_rest_vertices=np.asarray(binding.vertices, dtype=np.float32).copy(),
        soma30_source_rest_bind=source_rest.copy(),
        soma30_base_target_bind=base_target.copy(),
        soma30_align_offsets=align_offsets,
    )
