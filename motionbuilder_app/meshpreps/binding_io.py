from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bind_math import BindingResult, joint_positions_from_bind_matrices


def validate_lbs_arrays(
    lbs_indices: np.ndarray,
    lbs_weights: np.ndarray,
    *,
    n_verts: int,
    n_joints: int,
    tol: float = 1e-5,
) -> None:
    idx = np.asarray(lbs_indices, dtype=np.int64)
    w = np.asarray(lbs_weights, dtype=np.float32)
    if idx.ndim != 2 or idx.shape[1] != 4:
        raise ValueError(f"lbs_indices must be [V, 4], got {idx.shape}")
    if w.shape != idx.shape:
        raise ValueError(f"lbs_weights shape {w.shape} != lbs_indices shape {idx.shape}")
    if idx.shape[0] != n_verts:
        raise ValueError(f"expected {n_verts} vertices, got {idx.shape[0]}")
    if (idx < 0).any() or (idx >= n_joints).any():
        raise ValueError("joint index out of range")
    if (w < 0).any():
        raise ValueError("negative weights not allowed")
    row_sums = w.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol, rtol=0):
        bad = int(np.argmax(np.abs(row_sums - 1.0)))
        raise ValueError(f"weights must sum to 1 per vertex (max error at row {bad})")


def apply_lbs_to_binding(
    binding: BindingResult,
    lbs_indices: np.ndarray,
    lbs_weights: np.ndarray,
) -> BindingResult:
    idx = np.asarray(lbs_indices, dtype=np.int64)
    w = np.asarray(lbs_weights, dtype=np.float32)
    validate_lbs_arrays(
        idx,
        w,
        n_verts=int(binding.vertices.shape[0]),
        n_joints=int(binding.joint_positions.shape[0]),
    )
    return BindingResult(
        vertices=binding.vertices,
        faces=binding.faces,
        joint_positions=binding.joint_positions,
        joint_parents=binding.joint_parents,
        joint_names=binding.joint_names,
        lbs_indices=idx.copy(),
        lbs_weights=w.copy(),
        bind_matrices=binding.bind_matrices,
        source_path=binding.source_path,
        method=binding.method,
        align_offsets=binding.align_offsets,
        base_target_bind=binding.base_target_bind,
        source_rest_bind=binding.source_rest_bind,
        ik_profile=binding.ik_profile,
    )


def save_binding(path: str | Path, binding: BindingResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = str(binding.source_path) if binding.source_path else ""
    payload = {
        "vertices": binding.vertices.astype(np.float32),
        "faces": binding.faces.astype(np.int64),
        "joints": binding.joint_positions.astype(np.float32),
        "parents": binding.joint_parents.astype(np.int64),
        "names": np.array(binding.joint_names, dtype=object),
        "lbs_indices": binding.lbs_indices.astype(np.int64),
        "lbs_weights": binding.lbs_weights.astype(np.float32),
        "bind_matrices": binding.bind_matrices.astype(np.float32),
        "method": np.array([binding.method], dtype=object),
        "source_mesh": np.array([source], dtype=object),
    }
    if binding.align_offsets is not None:
        payload["align_offsets"] = binding.align_offsets.astype(np.float32)
    if binding.base_target_bind is not None:
        payload["base_target_bind"] = binding.base_target_bind.astype(np.float32)
    if binding.source_rest_bind is not None:
        payload["source_rest_bind"] = binding.source_rest_bind.astype(np.float32)
    if binding.ik_profile is not None:
        payload["ik_profile_json"] = np.asarray(
            [json.dumps(binding.ik_profile, ensure_ascii=False, separators=(",", ":"))],
            dtype=np.str_,
        )
    np.savez_compressed(path, **payload)
    return path


def load_binding(path: str | Path) -> BindingResult:
    data = np.load(Path(path), allow_pickle=True)
    method = str(data["method"][0]) if "method" in data.files else "unirig + skintokens"
    source_raw = data["source_mesh"][0] if "source_mesh" in data.files else ""
    source_path = Path(str(source_raw)) if source_raw else None
    bind_matrices = np.asarray(data["bind_matrices"], dtype=np.float32)
    joint_positions = joint_positions_from_bind_matrices(bind_matrices)
    extra: dict = {}
    if "align_offsets" in data.files:
        extra["align_offsets"] = np.asarray(data["align_offsets"], dtype=np.float32)
    if "base_target_bind" in data.files:
        extra["base_target_bind"] = np.asarray(data["base_target_bind"], dtype=np.float32)
    if "source_rest_bind" in data.files:
        extra["source_rest_bind"] = np.asarray(data["source_rest_bind"], dtype=np.float32)
    if "ik_profile_json" in data.files:
        try:
            extra["ik_profile"] = json.loads(str(np.asarray(data["ik_profile_json"]).reshape(-1)[0]))
        except (IndexError, TypeError, ValueError, json.JSONDecodeError):
            extra["ik_profile"] = None
    return BindingResult(
        vertices=np.asarray(data["vertices"], dtype=np.float32),
        faces=np.asarray(data["faces"], dtype=np.int64),
        joint_positions=joint_positions,
        joint_parents=np.asarray(data["parents"], dtype=np.int64),
        joint_names=[str(n) for n in list(data["names"])],
        lbs_indices=np.asarray(data["lbs_indices"], dtype=np.int64),
        lbs_weights=np.asarray(data["lbs_weights"], dtype=np.float32),
        bind_matrices=bind_matrices,
        source_path=source_path,
        method=method,
        **extra,
    )


def binding_preview_dict(binding: BindingResult) -> dict:
    joints = joint_positions_from_bind_matrices(binding.bind_matrices)
    return {
        "vertices": binding.vertices.tolist(),
        "faces": binding.faces.tolist(),
        "joints": joints.tolist(),
        "parents": binding.joint_parents.tolist(),
        "names": binding.joint_names,
        "lbs_indices": binding.lbs_indices.tolist(),
        "lbs_weights": binding.lbs_weights.tolist(),
        "bind_matrices": [m.reshape(-1).tolist() for m in binding.bind_matrices],
        "method": binding.method,
        "n_verts": int(binding.vertices.shape[0]),
        "n_joints": int(binding.joint_positions.shape[0]),
        "has_ik": binding.ik_profile is not None,
        "ik_profile": binding.ik_profile,
    }
