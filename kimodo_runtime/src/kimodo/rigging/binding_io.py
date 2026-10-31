# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import RemappedSOMABinding, UniRigBinding


def save_unirig_binding(path: str | Path, binding: UniRigBinding) -> Path:
    """Save UniRig+SkinTokens binding for standalone storage / later Kimodo load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parents = binding.joint_parents.astype(np.int64)
    names = np.array(binding.joint_names, dtype=object)
    source = str(binding.source_path) if binding.source_path else ""
    np.savez_compressed(
        path,
        vertices=binding.vertices.astype(np.float32),
        faces=binding.faces.astype(np.int64),
        joints=binding.joint_positions.astype(np.float32),
        parents=parents,
        names=names,
        lbs_indices=binding.lbs_indices.astype(np.int64),
        lbs_weights=binding.lbs_weights.astype(np.float32),
        bind_matrices=binding.bind_matrices.astype(np.float32),
        method=np.array([binding.method], dtype=object),
        source_mesh=np.array([source], dtype=object),
    )
    return path


def load_unirig_binding(path: str | Path) -> UniRigBinding:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    method = str(data["method"][0]) if "method" in data.files else "unirig + skintokens"
    source_raw = data["source_mesh"][0] if "source_mesh" in data.files else ""
    source_path = Path(str(source_raw)) if source_raw else None
    names_raw = data["names"]
    joint_names = [str(n) for n in list(names_raw)]
    return UniRigBinding(
        vertices=np.asarray(data["vertices"], dtype=np.float32),
        faces=np.asarray(data["faces"], dtype=np.int64),
        joint_positions=np.asarray(data["joints"], dtype=np.float32),
        joint_parents=np.asarray(data["parents"], dtype=np.int64),
        joint_names=joint_names,
        lbs_indices=np.asarray(data["lbs_indices"], dtype=np.int64),
        lbs_weights=np.asarray(data["lbs_weights"], dtype=np.float32),
        bind_matrices=np.asarray(data["bind_matrices"], dtype=np.float32),
        source_path=source_path,
        method=method,
    )


def save_remapped_binding(path: str | Path, binding: RemappedSOMABinding) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "vertices": binding.vertices,
        "faces": binding.faces,
        "lbs_indices": binding.lbs_indices,
        "lbs_weights": binding.lbs_weights,
        "bind_rig_transform": binding.bind_rig_transform,
        "bone_map_json": json.dumps(binding.bone_map, ensure_ascii=False),
        "bone_map_modes_json": json.dumps(binding.bone_map_modes or {}, ensure_ascii=False),
        "soma_scale": np.array([binding.soma_scale], dtype=np.float32),
        "soma_root_offset": binding.soma_root_offset,
    }
    if binding.has_soma30_rig():
        payload["soma30_joint_positions"] = binding.soma30_joint_positions
        payload["soma30_joint_parents"] = binding.soma30_joint_parents
        payload["soma30_joint_names_json"] = json.dumps(
            binding.soma30_joint_names or [],
            ensure_ascii=False,
        )
        payload["soma30_bind_matrices"] = binding.soma30_bind_matrices
        payload["soma30_lbs_indices"] = binding.soma30_lbs_indices
        payload["soma30_lbs_weights"] = binding.soma30_lbs_weights
        payload["soma30_rest_vertices"] = binding.soma30_rest_vertices
        if binding.soma30_align_offsets is not None:
            payload["soma30_align_offsets"] = binding.soma30_align_offsets
        if binding.soma30_source_rest_bind is not None:
            payload["soma30_source_rest_bind"] = binding.soma30_source_rest_bind
        if binding.soma30_base_target_bind is not None:
            payload["soma30_base_target_bind"] = binding.soma30_base_target_bind
    np.savez_compressed(path, **payload)
    return path


def load_remapped_binding(path: str | Path) -> RemappedSOMABinding:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    bone_map = json.loads(str(data["bone_map_json"]))
    modes: dict[str, str] = {}
    if "bone_map_modes_json" in data.files:
        modes = json.loads(str(data["bone_map_modes_json"]))

    soma30_kwargs: dict = {}
    if "soma30_bind_matrices" in data.files:
        if "soma30_joint_names_json" in data.files:
            soma30_names = json.loads(str(data["soma30_joint_names_json"]))
        else:
            soma30_names = [str(n) for n in list(data["soma30_joint_names"])]
        soma30_kwargs = {
            "soma30_joint_positions": np.array(data["soma30_joint_positions"]),
            "soma30_joint_parents": np.array(data["soma30_joint_parents"]),
            "soma30_joint_names": soma30_names,
            "soma30_bind_matrices": np.array(data["soma30_bind_matrices"]),
            "soma30_lbs_indices": np.array(data["soma30_lbs_indices"]),
            "soma30_lbs_weights": np.array(data["soma30_lbs_weights"]),
            "soma30_rest_vertices": np.array(data["soma30_rest_vertices"]),
        }
        if "soma30_align_offsets" in data.files:
            soma30_kwargs["soma30_align_offsets"] = np.array(data["soma30_align_offsets"])
        if "soma30_source_rest_bind" in data.files:
            soma30_kwargs["soma30_source_rest_bind"] = np.array(data["soma30_source_rest_bind"])
        if "soma30_base_target_bind" in data.files:
            soma30_kwargs["soma30_base_target_bind"] = np.array(data["soma30_base_target_bind"])

    return RemappedSOMABinding(
        vertices=np.array(data["vertices"]),
        faces=np.array(data["faces"]),
        lbs_indices=np.array(data["lbs_indices"]),
        lbs_weights=np.array(data["lbs_weights"]),
        bind_rig_transform=np.array(data["bind_rig_transform"]),
        bone_map=bone_map,
        bone_map_modes=modes,
        soma_scale=float(np.array(data["soma_scale"]).reshape(-1)[0]),
        soma_root_offset=np.array(data["soma_root_offset"]),
        **soma30_kwargs,
    )
