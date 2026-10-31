from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..motion_math import local_from_world, world_from_local


@dataclass(frozen=True)
class IKTarget:
    joint: int
    position: np.ndarray
    rotation: np.ndarray | None = None
    position_weight: float = 1.0
    rotation_weight: float = 0.0


@dataclass(frozen=True)
class FullBodyIKSolution:
    world: np.ndarray
    residuals: np.ndarray
    reached: np.ndarray
    iterations: int
    solver: str


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length > 1e-10:
        return vector / length
    fallback = np.asarray(fallback, dtype=np.float64)
    return fallback / max(float(np.linalg.norm(fallback)), 1e-10)


def _rotation_between(left: np.ndarray, right: np.ndarray, amount: float = 1.0) -> np.ndarray:
    left = _unit(left, np.asarray([0.0, 0.0, 1.0]))
    right = _unit(right, left)
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    axis = np.cross(left, right)
    if float(np.linalg.norm(axis)) < 1e-9:
        axis = np.cross(left, np.asarray([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-9:
            axis = np.cross(left, np.asarray([0.0, 1.0, 0.0]))
        angle = np.pi
    else:
        axis = _unit(axis, np.asarray([0.0, 1.0, 0.0]))
        angle = float(np.arccos(dot))
    return Rotation.from_rotvec(axis * angle * float(np.clip(amount, 0.0, 1.0))).as_matrix()


def _path_to_root(parents: np.ndarray, joint: int, root: int) -> list[int]:
    path: list[int] = []
    current = int(joint)
    seen: set[int] = set()
    while current >= 0 and current not in seen:
        seen.add(current)
        path.append(current)
        if current == root:
            return path
        current = int(parents[current])
    return []


def _is_ancestor(parents: np.ndarray, ancestor: int, joint: int) -> bool:
    current = int(joint)
    seen: set[int] = set()
    while current >= 0 and current not in seen:
        if current == ancestor:
            return True
        seen.add(current)
        current = int(parents[current])
    return False


def _set_world_rotation(
    local: np.ndarray,
    world: np.ndarray,
    parents: np.ndarray,
    joint: int,
    desired_world_rotation: np.ndarray,
) -> None:
    parent = int(parents[joint])
    if parent < 0:
        local[joint, :3, :3] = desired_world_rotation
        return
    # Transpose is only an inverse for a perfectly orthonormal matrix. Imported motion can carry tiny affine drift.
    parent_linear = world[parent, :3, :3]
    try:
        local[joint, :3, :3] = np.linalg.solve(parent_linear, desired_world_rotation)
    except np.linalg.LinAlgError:
        local[joint, :3, :3] = np.linalg.pinv(parent_linear) @ desired_world_rotation


def _apply_root_pull(
    local: np.ndarray,
    world: np.ndarray,
    parents: np.ndarray,
    root: int,
    targets: list[IKTarget],
    behavior: str,
) -> np.ndarray:
    amount = {"pin": 0.0, "pre_pull": 0.55, "foot_priority": 1.0, "free": 1.0}.get(behavior, 0.55)
    if amount <= 0.0 or not targets:
        return world
    weights = np.asarray([max(0.0, target.position_weight) for target in targets], dtype=np.float64)
    if float(weights.sum()) <= 1e-9:
        return world
    deltas = np.stack(
        [np.asarray(target.position, dtype=np.float64) - world[target.joint, :3, 3] for target in targets]
    )
    delta = np.average(deltas, axis=0, weights=weights) * amount
    parent = int(parents[root])
    if parent < 0:
        local_delta = delta
    else:
        # Transpose is only an inverse for a perfectly orthonormal matrix. Imported motion can carry tiny affine drift.
        parent_linear = world[parent, :3, :3]
        try:
            local_delta = np.linalg.solve(parent_linear, delta)
        except np.linalg.LinAlgError:
            local_delta = np.linalg.pinv(parent_linear) @ delta
    local[root, :3, 3] += local_delta
    return world_from_local(local, parents).astype(np.float64)


def _apply_target_rotations(
    local: np.ndarray,
    world: np.ndarray,
    parents: np.ndarray,
    targets: list[IKTarget],
) -> np.ndarray:
    for target in targets:
        if target.rotation is None or target.rotation_weight <= 1e-8:
            continue
        desired = np.asarray(target.rotation, dtype=np.float64)
        rotations = Rotation.from_matrix(np.stack([world[target.joint, :3, :3], desired]))
        blended = Slerp([0.0, 1.0], rotations)(
            [float(np.clip(target.rotation_weight, 0.0, 1.0))]
        ).as_matrix()[0]
        _set_world_rotation(local, world, parents, target.joint, blended)
        world = world_from_local(local, parents).astype(np.float64)
    return world


def _solve_ccd(
    input_world: np.ndarray,
    parents: np.ndarray,
    root: int,
    targets: list[IKTarget],
    iterations: int,
    tolerance: float,
    root_behavior: str,
    stiffness: np.ndarray,
) -> tuple[np.ndarray, int]:
    local = local_from_world(input_world, parents).astype(np.float64)
    world = world_from_local(local, parents).astype(np.float64)
    world = _apply_root_pull(local, world, parents, root, targets, root_behavior)
    completed = 0
    for iteration in range(iterations):
        completed = iteration + 1
        for target in targets:
            path = _path_to_root(parents, int(parents[target.joint]), root)
            for joint in path:
                pivot = world[joint, :3, 3]
                current = world[target.joint, :3, 3] - pivot
                desired = np.asarray(target.position, dtype=np.float64) - pivot
                if min(float(np.linalg.norm(current)), float(np.linalg.norm(desired))) <= 1e-9:
                    continue
                freedom = 1.0 - float(np.clip(stiffness[joint], 0.0, 0.98))
                delta = _rotation_between(current, desired, min(0.72, target.position_weight * freedom))
                _set_world_rotation(local, world, parents, joint, delta @ world[joint, :3, :3])
                world = world_from_local(local, parents).astype(np.float64)
        residuals = np.asarray(
            [np.linalg.norm(world[target.joint, :3, 3] - target.position) for target in targets]
        )
        if len(residuals) == 0 or float(np.max(residuals)) <= tolerance:
            break
    world = _apply_target_rotations(local, world, parents, targets)
    return world, completed


def _solve_jacobian_dls(
    input_world: np.ndarray,
    parents: np.ndarray,
    root: int,
    targets: list[IKTarget],
    iterations: int,
    tolerance: float,
    root_behavior: str,
    stiffness: np.ndarray,
) -> tuple[np.ndarray, int]:
    local = local_from_world(input_world, parents).astype(np.float64)
    world = world_from_local(local, parents).astype(np.float64)
    world = _apply_root_pull(local, world, parents, root, targets, root_behavior)
    active = sorted(
        {joint for target in targets for joint in _path_to_root(parents, int(parents[target.joint]), root)},
        key=lambda joint: len(_path_to_root(parents, joint, root)),
        reverse=True,
    )
    completed = 0
    for iteration in range(iterations):
        completed = iteration + 1
        errors = []
        jacobian = np.zeros((len(targets) * 3, len(active) * 3), dtype=np.float64)
        for target_index, target in enumerate(targets):
            effector = world[target.joint, :3, 3]
            error = (np.asarray(target.position, dtype=np.float64) - effector) * target.position_weight
            errors.extend(error.tolist())
            for joint_index, joint in enumerate(active):
                if not _is_ancestor(parents, joint, target.joint):
                    continue
                offset = effector - world[joint, :3, 3]
                for axis in range(3):
                    world_axis = world[joint, :3, axis]
                    jacobian[target_index * 3 : target_index * 3 + 3, joint_index * 3 + axis] = (
                        np.cross(world_axis, offset) * target.position_weight
                    )
        error_vector = np.asarray(errors, dtype=np.float64)
        if not len(error_vector) or float(np.max(np.linalg.norm(error_vector.reshape(-1, 3), axis=1))) <= tolerance:
            break
        damping = 0.08
        lhs = jacobian.T @ jacobian + np.eye(jacobian.shape[1]) * damping * damping
        try:
            delta = np.linalg.solve(lhs, jacobian.T @ error_vector)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(lhs, jacobian.T @ error_vector, rcond=None)[0]
        for joint_index, joint in enumerate(active):
            vector = delta[joint_index * 3 : joint_index * 3 + 3]
            vector *= 1.0 - float(np.clip(stiffness[joint], 0.0, 0.98))
            length = float(np.linalg.norm(vector))
            if length > 0.18:
                vector *= 0.18 / length
            if float(np.linalg.norm(vector)) <= 1e-10:
                continue
            world_delta = world[joint, :3, :3] @ Rotation.from_rotvec(vector).as_matrix() @ world[joint, :3, :3].T
            _set_world_rotation(local, world, parents, joint, world_delta @ world[joint, :3, :3])
            world = world_from_local(local, parents).astype(np.float64)
    world = _apply_target_rotations(local, world, parents, targets)
    return world, completed


def _fabrik_positions(points: np.ndarray, target: np.ndarray, iterations: int, tolerance: float) -> np.ndarray:
    solved = np.asarray(points, dtype=np.float64).copy()
    lengths = np.linalg.norm(np.diff(solved, axis=0), axis=1)
    origin = solved[0].copy()
    target = np.asarray(target, dtype=np.float64)
    if float(np.linalg.norm(target - origin)) >= float(lengths.sum()):
        direction = _unit(target - origin, solved[-1] - origin)
        for index, length in enumerate(lengths):
            solved[index + 1] = solved[index] + direction * length
        return solved
    for _ in range(iterations):
        solved[-1] = target
        for index in range(len(solved) - 2, -1, -1):
            direction = _unit(solved[index] - solved[index + 1], points[index] - points[index + 1])
            solved[index] = solved[index + 1] + direction * lengths[index]
        solved[0] = origin
        for index in range(len(solved) - 1):
            direction = _unit(solved[index + 1] - solved[index], points[index + 1] - points[index])
            solved[index + 1] = solved[index] + direction * lengths[index]
        if float(np.linalg.norm(solved[-1] - target)) <= tolerance:
            break
    return solved


def _solve_fabrik(
    input_world: np.ndarray,
    parents: np.ndarray,
    root: int,
    targets: list[IKTarget],
    iterations: int,
    tolerance: float,
    root_behavior: str,
    stiffness: np.ndarray,
) -> tuple[np.ndarray, int]:
    local = local_from_world(input_world, parents).astype(np.float64)
    world = world_from_local(local, parents).astype(np.float64)
    world = _apply_root_pull(local, world, parents, root, targets, root_behavior)
    completed = 0
    for iteration in range(iterations):
        completed = iteration + 1
        for target in targets:
            reverse_path = _path_to_root(parents, target.joint, root)
            if not reverse_path:
                continue
            chain = list(reversed(reverse_path))
            points = world[chain, :3, 3]
            solved = _fabrik_positions(points, target.position, max(2, iterations // 2), tolerance)
            for segment in range(len(chain) - 1):
                joint = chain[segment]
                freedom = 1.0 - float(np.clip(stiffness[joint], 0.0, 0.98))
                delta = _rotation_between(
                    world[chain[segment + 1], :3, 3] - world[joint, :3, 3],
                    solved[segment + 1] - solved[segment],
                    min(1.0, target.position_weight * freedom),
                )
                _set_world_rotation(local, world, parents, joint, delta @ world[joint, :3, :3])
                world = world_from_local(local, parents).astype(np.float64)
        residuals = np.asarray(
            [np.linalg.norm(world[target.joint, :3, 3] - target.position) for target in targets]
        )
        if len(residuals) == 0 or float(np.max(residuals)) <= tolerance:
            break
    world = _apply_target_rotations(local, world, parents, targets)
    return world, completed


def solve_full_body_frame(
    world: np.ndarray,
    parents: np.ndarray,
    root: int,
    targets: Iterable[IKTarget],
    *,
    solver: str = "pbik",
    iterations: int = 20,
    tolerance: float = 1e-3,
    root_behavior: str = "pre_pull",
    stiffness: np.ndarray | None = None,
) -> FullBodyIKSolution:
    world = np.asarray(world, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    resolved_targets = [
        IKTarget(
            joint=int(target.joint),
            position=np.asarray(target.position, dtype=np.float64).reshape(3),
            rotation=None if target.rotation is None else np.asarray(target.rotation, dtype=np.float64).reshape(3, 3),
            position_weight=float(np.clip(target.position_weight, 0.0, 1.0)),
            rotation_weight=float(np.clip(target.rotation_weight, 0.0, 1.0)),
        )
        for target in targets
    ]
    if world.ndim != 3 or world.shape[1:] != (4, 4) or len(world) != len(parents):
        raise ValueError("full-body IK expects one [joints,4,4] world pose")
    if not 0 <= int(root) < len(parents):
        raise ValueError("full-body IK root is invalid")
    for target in resolved_targets:
        if not 0 <= target.joint < len(parents):
            raise ValueError("full-body IK target joint is invalid")
        if not _path_to_root(parents, target.joint, int(root)):
            raise ValueError("full-body IK target is not a descendant of the root")
        if not np.isfinite(target.position).all():
            raise ValueError("full-body IK target contains non-finite values")
    stiffness_values = np.zeros(len(parents), dtype=np.float64)
    if stiffness is not None:
        stiffness_values[:] = np.clip(np.asarray(stiffness, dtype=np.float64).reshape(len(parents)), 0.0, 0.98)
    iterations = int(np.clip(iterations, 1, 100))
    tolerance = float(np.clip(tolerance, 1e-6, 0.25))
    canonical = {
        "full_body": "pbik",
        "position_based": "pbik",
        "jacobian": "jacobian_dls",
        "dls": "jacobian_dls",
        "hierarchical_fabrik": "fabrik",
    }.get(str(solver).lower(), str(solver).lower())
    implementations = {
        "pbik": _solve_ccd,
        "jacobian_dls": _solve_jacobian_dls,
        "fabrik": _solve_fabrik,
    }
    if canonical not in implementations:
        raise ValueError("full-body IK solver must be pbik, jacobian_dls, or fabrik")
    solved, completed = implementations[canonical](
        world, parents, int(root), resolved_targets, iterations, tolerance, root_behavior, stiffness_values
    )
    residuals = np.asarray(
        [np.linalg.norm(solved[target.joint, :3, 3] - target.position) for target in resolved_targets],
        dtype=np.float32,
    )
    return FullBodyIKSolution(
        world=solved.astype(np.float32),
        residuals=residuals,
        reached=(residuals <= tolerance),
        iterations=completed,
        solver=canonical,
    )
