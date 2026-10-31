from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from scipy import sparse
from scipy.sparse.linalg import splu

from .motion_math import apply_lbs_numpy, world_to_skin_matrices


VIRTUAL_ROOT_VERSION = "virtual-root-y-up-kimodo-smooth-v3"


@dataclass(frozen=True)
class VirtualRootResult:
    preview_world_motion: np.ndarray
    local_world_motion: np.ndarray
    virtual_root_motion: np.ndarray
    hip_local_under_virtual_root: np.ndarray
    ground_height: float
    ground_heights: np.ndarray
    hip_index: int
    version: str = VIRTUAL_ROOT_VERSION


def _hip_index(names: list[str]) -> int:
    lowered = [name.lower() for name in names]
    for candidate in ("hips", "pelvis", "hip"):
        if candidate in lowered:
            return lowered.index(candidate)
    return 0


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window = max(1, int(window))
    if window <= 1 or len(values) <= 2:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, [(radius, radius), (0, 0)], mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(values.shape[1])], axis=1)


class _KimodoTrajectorySmoother:
    """Kimodo-style acceleration-minimizing smoother with margin constraints."""

    def __init__(self, margins: np.ndarray, pos_weight: float = 0.0, admm_iters: int = 500, alpha_overrelax: float = 1.8) -> None:
        self.margin_vals = np.asarray(margins, dtype=np.float64)
        self.pos_weight = float(pos_weight)
        self.admm_iters = int(admm_iters)
        self.alpha_overrelax = float(alpha_overrelax)
        n = len(self.margin_vals)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for i in range(1, n - 1):
            data.extend([-1.0, 2.0, -1.0])
            rows.extend([i, i, i])
            cols.extend([i - 1, i, i + 1])
        a = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        identity = sparse.eye(n)
        system = self.pos_weight * identity + a.T @ a
        diag_max = max(float(np.max(np.abs(system.diagonal()))), 1e-8)
        self.admm_stepsize = 0.25 * np.sqrt(diag_max)
        self.system_lu = splu((system + self.admm_stepsize * identity).tocsc())

    def smooth(self, targets: np.ndarray, x0: np.ndarray) -> np.ndarray:
        x_target = np.asarray(targets, dtype=np.float64).copy()
        x = np.asarray(x0, dtype=np.float64).copy()
        z = np.zeros_like(x)
        u = np.zeros_like(x)
        for _ in range(self.admm_iters):
            z[:] = x + u - x_target
            norms = np.linalg.norm(z, axis=1)
            mask = norms > self.margin_vals
            if np.any(mask):
                z[mask] *= (self.margin_vals[mask] / np.maximum(norms[mask], 1e-8))[:, None]
            z[:] += x_target
            u[:] += self.alpha_overrelax * (x - z)
            rhs = self.pos_weight * x_target + self.admm_stepsize * (z - u)
            for axis in range(x.shape[1]):
                x[:, axis] = self.system_lu.solve(rhs[:, axis])
        return x


def _kimodo_smooth_signal(points: np.ndarray, margin: float = 0.06) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 2:
        return points.copy()
    smoothed = np.repeat(points.mean(axis=0, keepdims=True), len(points), axis=0)
    levels = max(int(np.floor(np.log2(len(points)))) - 4, 1)
    step = 2**levels
    margins = np.full(len(points), float(margin), dtype=np.float64)
    while True:
        coarse_count = len(smoothed[::step])
        smoother = _KimodoTrajectorySmoother(margins[::step])
        smoothed[::step] = smoother.smooth(points[::step], smoothed[::step])
        if step == 1:
            break
        next_step = step // 2
        interleaved = len(smoothed[next_step::step])
        if interleaved == coarse_count and coarse_count >= 2:
            smoothed[next_step::step][-1] = smoothed[::step][-1] + (smoothed[::step][-1] - smoothed[::step][-2]) / 2
            interleaved -= 1
        if interleaved > 0:
            smoothed[next_step::step][:interleaved] = (smoothed[::step][:-1] + smoothed[::step][1:]) / 2
        step = next_step
    return smoothed


def _threshold_position_jitter(points: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Hold samples inside the dead zone and identify positions that must stay fixed."""
    points = np.asarray(points, dtype=np.float64)
    stable = points.copy()
    anchored = np.zeros(len(points), dtype=bool)
    if not len(points):
        return stable, anchored
    anchored[0] = True
    tolerance = max(0.0, float(tolerance))
    if tolerance <= 0.0:
        return stable, anchored
    for frame in range(1, len(stable)):
        if float(np.linalg.norm(points[frame] - stable[frame - 1])) < tolerance:
            stable[frame] = stable[frame - 1]
            anchored[frame] = True
    return stable, anchored


def _continuous_yaw(yaw: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(yaw, dtype=np.float64))


def _yaw_from_matrix_y_up(rotations: np.ndarray) -> np.ndarray:
    forward = np.asarray(rotations[:, :3, 2], dtype=np.float64)
    forward[:, 1] = 0.0
    norms = np.linalg.norm(forward, axis=1)
    yaw = np.zeros(len(forward), dtype=np.float64)
    valid = norms > 1e-7
    yaw[valid] = np.arctan2(forward[valid, 0], forward[valid, 2])
    for i in range(1, len(yaw)):
        if not valid[i]:
            yaw[i] = yaw[i - 1]
    return _continuous_yaw(yaw)


def _ground_heights(
    world: np.ndarray,
    target_bind: np.ndarray,
    *,
    rest_vertices: np.ndarray | None,
    lbs_indices: np.ndarray | None,
    lbs_weights: np.ndarray | None,
    smoothing_frames: int,
) -> np.ndarray:
    if (
        rest_vertices is not None
        and lbs_indices is not None
        and lbs_weights is not None
        and len(rest_vertices)
    ):
        vertices = np.asarray(rest_vertices, dtype=np.float32)
        idx = np.asarray(lbs_indices, dtype=np.int64)
        weights = np.asarray(lbs_weights, dtype=np.float32)
        heights = np.empty(len(world), dtype=np.float64)
        for frame in range(len(world)):
            skin = world_to_skin_matrices(world[frame], target_bind)
            heights[frame] = float(np.min(apply_lbs_numpy(vertices, idx, weights, skin)[:, 1]))
    else:
        heights = np.min(np.asarray(world, dtype=np.float64)[:, :, 1, 3], axis=1)
    return _moving_average(heights[:, None], max(1, min(int(smoothing_frames), 9)))[:, 0]


def build_virtual_root(
    world_motion: np.ndarray,
    parents: np.ndarray,
    names: list[str],
    target_bind: np.ndarray,
    *,
    rest_vertices: np.ndarray | None = None,
    lbs_indices: np.ndarray | None = None,
    lbs_weights: np.ndarray | None = None,
    reference_root_positions: np.ndarray | None = None,
    smoothing_frames: int = 9,
    position_tolerance: float = 0.005,
) -> VirtualRootResult:
    """Create a stable Y-up locomotion root while preserving body motion below it.

    The output keeps the existing skeleton untouched. ``local_world_motion`` is the
    original animation expressed under the generated VirtualRoot, while
    ``virtual_root_motion`` stores the World -> VirtualRoot transform.
    """
    world = np.asarray(world_motion, dtype=np.float32)
    if world.ndim != 4 or world.shape[-2:] != (4, 4) or len(world) < 1:
        raise ValueError("world_motion must have shape [frames, joints, 4, 4]")
    parents = np.asarray(parents, dtype=np.int64)
    target_bind = np.asarray(target_bind, dtype=np.float32)
    hip = _hip_index(names)
    ground_y = _ground_heights(
        world,
        target_bind,
        rest_vertices=rest_vertices,
        lbs_indices=lbs_indices,
        lbs_weights=lbs_weights,
        smoothing_frames=smoothing_frames,
    )

    hip_global = world[:, hip]
    hip_position = hip_global[:, :3, 3].astype(np.float64)
    reference = None
    if reference_root_positions is not None:
        reference = np.asarray(reference_root_positions, dtype=np.float64)
        if reference.shape[:1] != hip_position.shape[:1] or reference.shape[-1] != 3:
            reference = None
    raw_xz = (reference[:, [0, 2]] if reference is not None else hip_position[:, [0, 2]])
    thresholded_xz, anchored_xz = _threshold_position_jitter(raw_xz, position_tolerance)
    if reference is not None:
        stable_xz = thresholded_xz
    else:
        stable_xz = _kimodo_smooth_signal(thresholded_xz)
        # The acceleration smoother may move a stationary sample by its margin.
        # Reapply dead-zone anchors so smoothing cannot turn jitter into motion.
        stable_xz[anchored_xz] = thresholded_xz[anchored_xz]

    yaw = _yaw_from_matrix_y_up(hip_global[:, :3, :3])
    yaw = _moving_average(yaw[:, None], smoothing_frames)[:, 0]
    yaw = _continuous_yaw(yaw)

    virtual_root = np.repeat(np.eye(4, dtype=np.float32)[None], len(world), axis=0)
    virtual_root[:, 0, 3] = stable_xz[:, 0].astype(np.float32)
    virtual_root[:, 1, 3] = ground_y.astype(np.float32)
    virtual_root[:, 2, 3] = stable_xz[:, 1].astype(np.float32)
    virtual_root[:, :3, :3] = Rotation.from_rotvec(np.column_stack([np.zeros_like(yaw), yaw, np.zeros_like(yaw)])).as_matrix().astype(np.float32)

    inv_root = np.linalg.inv(virtual_root.astype(np.float64)).astype(np.float32)
    local_world = np.einsum("fab,fjbc->fjac", inv_root, world).astype(np.float32)
    hip_local = np.einsum("fab,fbc->fac", inv_root, hip_global).astype(np.float32)

    # Preview is reconstructed from the generated hierarchy and should match the
    # original motion, aside from smoothing/threshold decisions in the root.
    preview_world = np.einsum("fab,fjbc->fjac", virtual_root, local_world).astype(np.float32)

    return VirtualRootResult(
        preview_world_motion=preview_world,
        local_world_motion=local_world,
        virtual_root_motion=virtual_root.astype(np.float32),
        hip_local_under_virtual_root=hip_local,
        ground_height=float(ground_y[0]),
        ground_heights=ground_y.astype(np.float32),
        hip_index=hip,
    )
