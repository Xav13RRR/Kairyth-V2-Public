from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .bind_math import BindingResult, build_binding


def fit_soma_from_unirig_skel(skel: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    from ..rigging.soma_fit import fit_soma30_positions

    fit = fit_soma30_positions(
        skel["joint_names"],
        skel["joint_positions"],
        skel.get("joint_parents"),
    )
    return (
        np.asarray(fit.joints, dtype=np.float32),
        list(fit.names),
        np.asarray(fit.parents, dtype=np.int64),
    )


def _nearest_skin_ancestor(
    joint_idx: int,
    soma_names: list[str],
    soma_parents: np.ndarray,
    skin_names: frozenset[str],
) -> int:
    j = int(joint_idx)
    seen: set[int] = set()
    while j >= 0 and j not in seen:
        seen.add(j)
        if soma_names[j] in skin_names:
            return j
        j = int(soma_parents[j])
    return 0


def _sanitize_soma_skin(
    skin: np.ndarray,
    soma_names: list[str],
    soma_parents: np.ndarray,
) -> np.ndarray:
    """Roll non-skin SOMA joint weights up to the nearest skin ancestor."""
    from ..rigging.soma_retarget_profile import SOMA_NON_SKIN_JOINTS, SOMA_SKIN_JOINTS

    if skin.ndim != 2:
        raise ValueError(f"skin 维度异常：{skin.shape}")
    n_verts, n_joints = skin.shape
    if n_joints != len(soma_names):
        raise ValueError(f"skin 骨骼数 {n_joints} 与 SOMA {len(soma_names)} 不一致")

    out = skin.astype(np.float64, copy=True)
    exclude_idx = [i for i, name in enumerate(soma_names) if name in SOMA_NON_SKIN_JOINTS]
    for j in exclude_idx:
        col = out[:, j]
        if not np.any(col > 1e-8):
            continue
        target = _nearest_skin_ancestor(j, soma_names, soma_parents, SOMA_SKIN_JOINTS)
        if target != j:
            out[:, target] += col
        out[:, j] = 0.0

    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-8] = 1.0
    return (out / row_sum).astype(np.float32)


def build_soma30_binding_from_skin(
    soma_skel: dict,
    skin: np.ndarray,
    *,
    method: str,
    source_path: Path | None,
) -> BindingResult:
    """Build SOMA-30 binding from SkinTokens weights inferred on the same skeleton."""
    joints = np.asarray(soma_skel["joint_positions"], dtype=np.float32)
    parents = np.asarray(soma_skel["joint_parents"], dtype=np.int64)
    names = list(soma_skel["joint_names"])
    vertices = np.asarray(soma_skel["vertices"], dtype=np.float32)

    if skin.shape[0] == joints.shape[0] and skin.shape[1] != joints.shape[0]:
        skin = skin.T
    if skin.shape[0] != vertices.shape[0]:
        raise ValueError(f"skin 顶点数 {skin.shape[0]} 与 mesh {vertices.shape[0]} 不一致")
    if skin.shape[1] != joints.shape[0]:
        raise ValueError(f"skin 骨骼数 {skin.shape[1]} 与 SOMA {joints.shape[0]} 不一致")

    skin = _sanitize_soma_skin(skin, names, parents)
    return build_binding(
        soma_skel,
        skin,
        method=method,
        source_path=source_path,
    )


def soma_edit_preview_dict(
    skel: dict,
    soma_joints: np.ndarray,
    soma_names: list[str],
    soma_parents: np.ndarray,
    *,
    bind_matrices: np.ndarray | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    soma: dict[str, Any] = {
        "joints": soma_joints.tolist(),
        "parents": soma_parents.tolist(),
        "names": soma_names,
        "n_joints": len(soma_names),
    }
    if bind_matrices is not None:
        soma["bind_matrices"] = [m.reshape(-1).tolist() for m in bind_matrices]
    return {
        "kind": "soma_edit",
        "has_skin": False,
        "vertices": skel["vertices"].tolist(),
        "faces": skel["faces"].tolist(),
        "n_verts": int(skel["vertices"].shape[0]),
        "n_faces": int(skel["faces"].shape[0]),
        "soma": soma,
        "job_id": job_id,
    }
