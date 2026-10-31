from __future__ import annotations

import numpy as np


def _load_soma_rest_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    from ..rigging.soma_fit import _load_soma_rest

    names, rest, parents = _load_soma_rest()
    return names, np.asarray(rest, dtype=np.float64), np.asarray(parents, dtype=np.int64)


def soma30_reference_dict() -> dict:
    names, rest, parents = _load_soma_rest_data()
    return {
        "joints": rest.astype(np.float32).tolist(),
        "parents": parents.tolist(),
        "names": names,
        "n_joints": len(names),
    }


def matrix_from_trs(
    translation: np.ndarray | list[float],
    quaternion: np.ndarray | list[float],
    scale: float,
) -> np.ndarray:
    """Build 4x4 matching Three.js Matrix4.compose(position, quaternion, scale)."""
    tx, ty, tz = (float(v) for v in translation)
    x, y, z, w = (float(v) for v in quaternion)
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    s = max(float(scale), 1e-6)

    m = np.eye(4, dtype=np.float64)
    m[0, 0] = (1.0 - (yy + zz)) * s
    m[1, 0] = (xy + wz) * s
    m[2, 0] = (xz - wy) * s
    m[0, 1] = (xy - wz) * s
    m[1, 1] = (1.0 - (xx + zz)) * s
    m[2, 1] = (yz + wx) * s
    m[0, 2] = (xz + wy) * s
    m[1, 2] = (yz - wx) * s
    m[2, 2] = (1.0 - (xx + yy)) * s
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def resolve_align_matrix(
    *,
    matrix: np.ndarray | None = None,
    translation: np.ndarray | list[float] | None = None,
    quaternion: np.ndarray | list[float] | None = None,
    scale: float | None = None,
) -> np.ndarray:
    if translation is not None and quaternion is not None and scale is not None:
        return matrix_from_trs(translation, quaternion, scale)
    if matrix is not None:
        return np.asarray(matrix, dtype=np.float64)
    raise ValueError("align requires transform or matrix")


def apply_matrix4(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform matching Three.js Matrix4 (column vectors: p' = M @ p)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix, got {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError("matrix contains non-finite values")
    ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
    hom = np.hstack([vertices.astype(np.float64), ones])
    out = (m @ hom.T).T
    return out[:, :3].astype(np.float32)


def validate_align_matrix(matrix: np.ndarray) -> None:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError("matrix must be 4x4")
    if not np.isfinite(m).all():
        raise ValueError("matrix contains non-finite values")
    col_norms = np.linalg.norm(m[:3, :3], axis=0)
    if float(col_norms.max()) > 1000.0 or float(col_norms.min()) < 1e-6:
        raise ValueError("matrix scale out of safe range")


def _yaw_matrix(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _xz_pca_angle(vertices: np.ndarray, reference_axis: np.ndarray | None = None) -> float:
    xz = vertices[:, [0, 2]].astype(np.float64)
    xz = xz - xz.mean(axis=0, keepdims=True)
    if xz.shape[0] < 3:
        return 0.0
    cov = xz.T @ xz / max(xz.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    if reference_axis is not None:
        ref = np.asarray(reference_axis, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(ref))
        if norm > 1e-8 and float(np.dot(axis, ref / norm)) < 0.0:
            axis = -axis
    return float(np.arctan2(axis[1], axis[0]))


def _soma_yaw_reference(soma: np.ndarray, name_to_idx: dict[str, int]) -> float:
    for left_key, right_key in (("LeftLeg", "RightLeg"), ("LeftShoulder", "RightShoulder")):
        if left_key in name_to_idx and right_key in name_to_idx:
            lv = soma[name_to_idx[left_key]]
            rv = soma[name_to_idx[right_key]]
            vec = rv - lv
            if float(np.linalg.norm(vec[[0, 2]])) > 1e-6:
                return float(np.arctan2(vec[2], vec[0]))
    return 0.0


def suggest_align_transform(vertices: np.ndarray) -> np.ndarray:
    """Rigid+uniform scale mapping normalized mesh into SOMA30 rest bounds."""
    verts = np.asarray(vertices, dtype=np.float64)
    names, soma = _load_soma_rest_data()[:2]
    name_to_idx = {n: i for i, n in enumerate(names)}

    mesh_min = verts.min(axis=0)
    mesh_max = verts.max(axis=0)
    soma_min = soma.min(axis=0)
    soma_max = soma.max(axis=0)

    mesh_h = max(float(mesh_max[1] - mesh_min[1]), 1e-6)
    soma_h = max(float(soma_max[1] - soma_min[1]), 1e-6)
    scale = soma_h / mesh_h

    scaled = verts * scale
    s_min = scaled.min(axis=0)
    s_max = scaled.max(axis=0)
    mesh_cx = 0.5 * (s_min[0] + s_max[0])
    mesh_cz = 0.5 * (s_min[2] + s_max[2])
    soma_cx = 0.5 * (soma_min[0] + soma_max[0])
    soma_cz = 0.5 * (soma_min[2] + soma_max[2])
    tx = soma_cx - mesh_cx
    ty = float(soma_min[1] - s_min[1])
    tz = soma_cz - mesh_cz

    soma_yaw = _soma_yaw_reference(soma, name_to_idx)
    soma_axis = np.array([np.cos(soma_yaw), np.sin(soma_yaw)], dtype=np.float64)
    mesh_yaw = _xz_pca_angle(scaled, soma_axis)
    yaw = soma_yaw - mesh_yaw
    r_yaw = _yaw_matrix(yaw)

    s_mat = np.eye(4, dtype=np.float64)
    s_mat[0, 0] = s_mat[1, 1] = s_mat[2, 2] = scale
    t_mat = np.eye(4, dtype=np.float64)
    t_mat[0, 3] = tx
    t_mat[1, 3] = ty
    t_mat[2, 3] = tz
    return t_mat @ r_yaw @ s_mat
