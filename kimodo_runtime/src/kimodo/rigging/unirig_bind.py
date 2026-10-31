# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from .mesh_import import ImportedMesh, load_mesh_file
from .types import UniRigBinding
from .unirig_runner import DEFAULT_SEED, DEFAULT_TIMEOUT_S, run_unirig_bind


def bind_mesh_unirig(
    mesh_path: str | Path | ImportedMesh,
    *,
    seed: int = DEFAULT_SEED,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> UniRigBinding:
    if isinstance(mesh_path, ImportedMesh):
        mesh = mesh_path
    else:
        mesh = load_mesh_file(mesh_path)
    return run_unirig_bind(mesh, seed=seed, timeout_s=timeout_s)
