from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .ik.full_body import IKTarget, solve_full_body_frame


FOOT_LOCK_VERSION = "foot-lock-full-body-v1"


@dataclass(frozen=True)
class FootLockSettings:
    reference_frame: int = 0
    foot_mode: str = "both"
    lock_rotation: bool = False
    solver: str = "pbik"
    root_behavior: str = "foot_priority"
    iterations: int = 24
    tolerance: float = 0.002

    def normalized(self, frame_count: int) -> "FootLockSettings":
        foot_mode = str(self.foot_mode).lower()
        if foot_mode not in {"left", "right", "both"}:
            raise ValueError("foot mode must be left, right, or both")
        solver = str(self.solver).lower()
        solver = {"full_body": "pbik", "dls": "jacobian_dls", "hierarchical_fabrik": "fabrik"}.get(solver, solver)
        if solver not in {"pbik", "jacobian_dls", "fabrik"}:
            raise ValueError("foot lock solver must be pbik, jacobian_dls, or fabrik")
        root_behavior = str(self.root_behavior).lower()
        if root_behavior not in {"pin", "pre_pull", "foot_priority", "free"}:
            raise ValueError("invalid foot lock root behavior")
        return FootLockSettings(
            reference_frame=int(np.clip(self.reference_frame, 0, max(0, frame_count - 1))),
            foot_mode=foot_mode,
            lock_rotation=bool(self.lock_rotation),
            solver=solver,
            root_behavior=root_behavior,
            iterations=int(np.clip(self.iterations, 1, 100)),
            tolerance=float(np.clip(self.tolerance, 1e-5, 0.05)),
        )


@dataclass(frozen=True)
class FootLockResult:
    local_world_motion: np.ndarray
    preview_world_motion: np.ndarray
    virtual_root_motion: np.ndarray
    residuals: np.ndarray
    reached: np.ndarray
    settings: dict[str, object]
    metrics: dict[str, float]
    reference_targets: dict[str, list[float]]
    version: str = FOOT_LOCK_VERSION


def _default_stiffness(names: list[str]) -> np.ndarray:
    values = np.zeros(len(names), dtype=np.float64)
    configured = {
        "Hips": 0.45,
        "Spine1": 0.75,
        "Spine2": 0.82,
        "Chest": 0.9,
        "LeftLeg": 0.08,
        "RightLeg": 0.08,
        "LeftShin": 0.02,
        "RightShin": 0.02,
    }
    for index, name in enumerate(names):
        values[index] = configured.get(str(name), 0.15)
    return values


def _virtual_root_from_locked_feet(
    solved_world: np.ndarray,
    hip_index: int,
    midpoint: np.ndarray,
) -> np.ndarray:
    frame_count = len(solved_world)
    root = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    root[:, :3, 3] = np.asarray(midpoint, dtype=np.float64)[None]
    forward = np.einsum(
        "fij,j->fi", solved_world[:, hip_index, :3, :3].astype(np.float64), np.asarray([0.0, 0.0, 1.0])
    )
    forward[:, 1] = 0.0
    length = np.linalg.norm(forward, axis=1)
    valid = length > 1e-8
    forward[valid] /= length[valid, None]
    forward[~valid] = np.asarray([0.0, 0.0, 1.0])
    yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    yaw_vectors = np.stack([np.zeros_like(yaw), yaw, np.zeros_like(yaw)], axis=1)
    root[:, :3, :3] = Rotation.from_rotvec(yaw_vectors).as_matrix()
    return root.astype(np.float32)


def process_foot_lock(
    world_motion: np.ndarray,
    parents: np.ndarray,
    names: list[str],
    *,
    settings: FootLockSettings | None = None,
) -> FootLockResult:
    world_motion = np.asarray(world_motion, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    if world_motion.ndim != 4 or world_motion.shape[2:] != (4, 4):
        raise ValueError("foot lock expects [frames,joints,4,4] world motion")
    if len(world_motion) < 1 or len(parents) != world_motion.shape[1] or len(names) != len(parents):
        raise ValueError("foot lock skeleton arrays are inconsistent")
    index = {str(name): position for position, name in enumerate(names)}
    required = {"Hips", "LeftFoot", "RightFoot"}
    missing = sorted(required.difference(index))
    if missing:
        raise ValueError(f"humanoid foot lock joints missing: {', '.join(missing)}")
    settings = (settings or FootLockSettings()).normalized(len(world_motion))
    reference = settings.reference_frame
    foot_indexes = {"left": index["LeftFoot"], "right": index["RightFoot"]}
    reference_positions = {
        side: world_motion[reference, joint, :3, 3].astype(np.float64).copy()
        for side, joint in foot_indexes.items()
    }
    reference_rotations = {
        side: world_motion[reference, joint, :3, :3].astype(np.float64).copy()
        for side, joint in foot_indexes.items()
    }
    active_sides = ["left", "right"] if settings.foot_mode == "both" else [settings.foot_mode]
    stiffness = _default_stiffness(names)
    solved = np.empty_like(world_motion)
    residuals = np.zeros((len(world_motion), len(active_sides)), dtype=np.float32)
    reached = np.zeros((len(world_motion), len(active_sides)), dtype=np.uint8)
    iterations = np.zeros(len(world_motion), dtype=np.int64)
    for frame in range(len(world_motion)):
        targets = [
            IKTarget(
                joint=foot_indexes[side],
                position=reference_positions[side],
                rotation=reference_rotations[side] if settings.lock_rotation else None,
                position_weight=1.0,
                rotation_weight=1.0 if settings.lock_rotation else 0.0,
            )
            for side in active_sides
        ]
        result = solve_full_body_frame(
            world_motion[frame],
            parents,
            index["Hips"],
            targets,
            solver=settings.solver,
            iterations=settings.iterations,
            tolerance=settings.tolerance,
            root_behavior=settings.root_behavior,
            stiffness=stiffness,
        )
        solved[frame] = result.world
        residuals[frame] = result.residuals
        reached[frame] = result.reached.astype(np.uint8)
        iterations[frame] = result.iterations
    midpoint = 0.5 * (reference_positions["left"] + reference_positions["right"])
    virtual_root = _virtual_root_from_locked_feet(solved, index["Hips"], midpoint)
    inverse_root = np.linalg.inv(virtual_root.astype(np.float64)).astype(np.float32)
    local_world = np.einsum("fab,fjbc->fjac", inverse_root, solved).astype(np.float32)
    reconstructed = np.einsum("fab,fjbc->fjac", virtual_root, local_world).astype(np.float32)
    if not np.isfinite(reconstructed).all():
        raise ValueError("foot lock produced non-finite transforms")
    metrics = {
        "max_residual": float(np.max(residuals)) if residuals.size else 0.0,
        "mean_residual": float(np.mean(residuals)) if residuals.size else 0.0,
        "unreachable_ratio": float(1.0 - np.mean(reached)) if reached.size else 0.0,
        "mean_iterations": float(np.mean(iterations)) if len(iterations) else 0.0,
        "virtual_root_position_drift": float(np.max(np.ptp(virtual_root[:, :3, 3], axis=0))),
    }
    return FootLockResult(
        local_world_motion=local_world,
        preview_world_motion=reconstructed,
        virtual_root_motion=virtual_root,
        residuals=residuals,
        reached=reached,
        settings=asdict(settings),
        metrics=metrics,
        reference_targets={side: reference_positions[side].astype(float).tolist() for side in ("left", "right")},
    )
