# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ImportedMesh:
    vertices: "np.ndarray"  # [N, 3]
    faces: "np.ndarray"  # [F, 3]
    source_path: Path


@dataclass
class UniRigBinding:
    """UniRig/MIA-style rig on the imported mesh (stage 2 output)."""

    vertices: "np.ndarray"
    faces: "np.ndarray"
    joint_positions: "np.ndarray"  # [J, 3] rest pose
    joint_parents: "np.ndarray"  # [J] int, -1 root
    joint_names: list[str]
    lbs_indices: "np.ndarray"  # [N, W]
    lbs_weights: "np.ndarray"  # [N, W]
    bind_matrices: "np.ndarray"  # [J, 4, 4]
    source_path: Optional[Path] = None
    method: str = "unirig"


@dataclass
class RemappedSOMABinding:
    """Mesh skinned to SOMA-77 for Kimodo generation (stage 3 output)."""

    vertices: "np.ndarray"
    faces: "np.ndarray"
    lbs_indices: "np.ndarray"  # [N, W] indices into SOMA-77
    lbs_weights: "np.ndarray"
    bind_rig_transform: "np.ndarray"  # [77, 4, 4]
    bone_map: dict[str, str]
    bone_map_modes: dict[str, str] | None = None
    soma_scale: float = 1.0
    soma_root_offset: "np.ndarray" = None  # type: ignore
    # meshpreps SOMA-30 target rig (absolute bind pose for retarget + LBS)
    soma30_joint_positions: "np.ndarray | None" = None
    soma30_joint_parents: "np.ndarray | None" = None
    soma30_joint_names: list[str] | None = None
    soma30_bind_matrices: "np.ndarray | None" = None
    soma30_lbs_indices: "np.ndarray | None" = None
    soma30_lbs_weights: "np.ndarray | None" = None
    soma30_rest_vertices: "np.ndarray | None" = None
    soma30_align_offsets: "np.ndarray | None" = None
    soma30_source_rest_bind: "np.ndarray | None" = None
    soma30_base_target_bind: "np.ndarray | None" = None

    def __post_init__(self) -> None:
        import numpy as np

        if self.soma_root_offset is None:
            self.soma_root_offset = np.zeros(3, dtype=np.float32)
        if self.bone_map_modes is None:
            self.bone_map_modes = {}

    def has_soma30_rig(self) -> bool:
        return (
            self.soma30_bind_matrices is not None
            and self.soma30_lbs_indices is not None
            and self.soma30_lbs_weights is not None
            and self.soma30_joint_parents is not None
            and self.soma30_rest_vertices is not None
        )
