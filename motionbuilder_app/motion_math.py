from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def topo_order(parents: np.ndarray) -> list[int]:
    parents = np.asarray(parents, dtype=np.int64)
    depth = np.zeros(len(parents), dtype=np.int64)
    for j in range(len(parents)):
        p, seen = int(parents[j]), set()
        while p >= 0 and p not in seen:
            seen.add(p)
            depth[j] += 1
            p = int(parents[p])
    return list(np.argsort(depth, kind="stable"))


def world_from_local(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local = np.asarray(local, dtype=np.float64)
    single = local.ndim == 3
    if single:
        local = local[None]
    out = np.zeros_like(local)
    for j in topo_order(parents):
        p = int(parents[j])
        out[:, j] = local[:, j] if p < 0 else out[:, p] @ local[:, j]
    return out[0].astype(np.float32) if single else out.astype(np.float32)


def local_from_world(world: np.ndarray, parents: np.ndarray) -> np.ndarray:
    world = np.asarray(world, dtype=np.float64)
    single = world.ndim == 3
    if single:
        world = world[None]
    out = np.zeros_like(world)
    for j in topo_order(parents):
        p = int(parents[j])
        out[:, j] = world[:, j] if p < 0 else np.linalg.inv(world[:, p]) @ world[:, j]
    return out[0].astype(np.float32) if single else out.astype(np.float32)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    arr = np.asarray(axis_angle, dtype=np.float64)
    shape = arr.shape[:-1]
    return Rotation.from_rotvec(arr.reshape(-1, 3)).as_matrix().reshape(*shape, 3, 3).astype(np.float32)


def quaternion_weighted(quaternion_xyzw: np.ndarray, weight: float) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    q /= norm
    rotations = Rotation.from_quat(np.stack([[0.0, 0.0, 0.0, 1.0], q]))
    return Slerp([0.0, 1.0], rotations)([float(np.clip(weight, 0.0, 1.0))]).as_matrix()[0].astype(np.float32)


def interpolate_quaternions(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    pair = np.stack([np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)])
    pair /= np.maximum(np.linalg.norm(pair, axis=1, keepdims=True), 1e-8)
    return Slerp([0.0, 1.0], Rotation.from_quat(pair))([float(amount)]).as_quat()[0].astype(np.float32)


def world_to_skin_matrices(world: np.ndarray, target_bind: np.ndarray) -> np.ndarray:
    return (np.asarray(world, dtype=np.float64) @ np.linalg.inv(np.asarray(target_bind, dtype=np.float64))).astype(np.float32)


def apply_lbs_numpy(
    vertices: np.ndarray,
    lbs_indices: np.ndarray,
    lbs_weights: np.ndarray,
    skin_matrices: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    idx = np.asarray(lbs_indices, dtype=np.int64)
    weights = np.asarray(lbs_weights, dtype=np.float32)
    hom = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)
    gathered = np.asarray(skin_matrices, dtype=np.float32)[idx]
    transformed = np.einsum("vkij,vj->vki", gathered, hom)
    return np.einsum("vk,vki->vi", weights, transformed)[..., :3].astype(np.float32)
