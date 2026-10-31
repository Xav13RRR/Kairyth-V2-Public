# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .types import ImportedMesh


def _normalize_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts = vertices.astype(np.float32)
    faces = faces.astype(np.int64)
    # Center XZ, ground Y
    center = verts.mean(axis=0)
    verts[:, 0] -= center[0]
    verts[:, 2] -= center[2]
    min_y = verts[:, 1].min()
    verts[:, 1] -= min_y
    # Scale to ~1.7m height
    height = max(verts[:, 1].max() - verts[:, 1].min(), 1e-6)
    scale = 1.7 / height
    verts *= scale
    return verts, faces


def load_mesh_file(path: str | Path) -> ImportedMesh:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load triangle mesh from {path}")

    vertices, faces = _normalize_mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    return ImportedMesh(vertices=vertices, faces=faces, source_path=path)
