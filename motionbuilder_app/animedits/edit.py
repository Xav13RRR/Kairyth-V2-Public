from __future__ import annotations

import numpy as np


def _topo_order(parents: np.ndarray) -> list[int]:
    n = len(parents)
    depth = np.zeros(n, dtype=np.int64)
    for j in range(n):
        d = 0
        p = int(parents[j])
        while p >= 0:
            d += 1
            p = int(parents[p])
        depth[j] = d
    return list(np.argsort(depth))


def local_from_world(world: np.ndarray, parents: np.ndarray) -> np.ndarray:
    n = world.shape[0]
    local = np.zeros_like(world)
    for j in range(n):
        p = int(parents[j])
        if p < 0:
            local[j] = world[j]
        else:
            local[j] = np.linalg.inv(world[p]) @ world[j]
    return local


def world_from_local(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    n = local.shape[0]
    world = np.zeros_like(local)
    for j in _topo_order(parents):
        p = int(parents[j])
        world[j] = local[j] if p < 0 else world[p] @ local[j]
    return world


def apply_joint_local_deltas(
    world: np.ndarray,
    parents: np.ndarray,
    deltas: dict[int, np.ndarray],
) -> np.ndarray:
    if not deltas:
        return world
    local = local_from_world(world, parents)
    for j in sorted(deltas.keys()):
        delta = np.asarray(deltas[j], dtype=np.float64)
        if delta.shape == (3, 3):
            m = np.eye(4, dtype=np.float64)
            m[:3, :3] = delta
            delta = m
        local[j] = local[j] @ delta
    return world_from_local(local, parents).astype(np.float32)


def bake_overlays(
    base_world: np.ndarray,
    parents: np.ndarray,
    overlays: dict[int, np.ndarray],
) -> np.ndarray:
    if not overlays:
        return base_world.copy()
    out = np.zeros_like(base_world)
    for fi in range(base_world.shape[0]):
        out[fi] = apply_joint_local_deltas(base_world[fi], parents, overlays)
    return out
