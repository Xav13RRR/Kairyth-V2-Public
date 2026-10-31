from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .ik.full_body import IKTarget, solve_full_body_frame
from .ik.two_bone import solve_two_bone_positions
from .motion_math import local_from_world, world_from_local


_EFFECTOR_NAMES = {
    "left_foot": "LeftFoot",
    "right_foot": "RightFoot",
    "left_hand": "LeftHand",
    "right_hand": "RightHand",
    "head": "Head",
}
_TWO_BONE_CHAINS = {
    "LeftFoot": ("LeftLeg", "LeftShin", "LeftFoot"),
    "RightFoot": ("RightLeg", "RightShin", "RightFoot"),
    "LeftHand": ("LeftArm", "LeftForeArm", "LeftHand"),
    "RightHand": ("RightArm", "RightForeArm", "RightHand"),
}


def _smooth_weight(frame: int, start: int, end: int, blend_in: int, blend_out: int) -> float:
    if start <= frame <= end:
        return 1.0
    if blend_in > 0 and start - blend_in <= frame < start:
        t = (frame - (start - blend_in)) / blend_in
        return float(0.5 - 0.5 * np.cos(np.pi * t))
    if blend_out > 0 and end < frame <= end + blend_out:
        t = (frame - end) / blend_out
        return float(0.5 + 0.5 * np.cos(np.pi * t))
    return 0.0


def _catmull_rom(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, t: float) -> np.ndarray:
    return 0.5 * (
        2.0 * b
        + (-a + c) * t
        + (2.0 * a - 5.0 * b + 4.0 * c - d) * t * t
        + (-a + 3.0 * b - 3.0 * c + d) * t * t * t
    )


def _sample_constraint(constraint: dict[str, Any], frame: int) -> tuple[np.ndarray, np.ndarray | None]:
    keys = sorted(constraint.get("keys") or [], key=lambda item: int(item.get("frame", 0)))
    fallback_position = np.asarray(constraint.get("target_position", [0.0, 0.0, 0.0]), dtype=np.float64)
    fallback_rotation = constraint.get("target_rotation")
    if not keys:
        rotation = None if fallback_rotation is None else Rotation.from_quat(fallback_rotation).as_matrix()
        return fallback_position, rotation
    if frame <= int(keys[0].get("frame", 0)):
        chosen = keys[0]
        rotation = chosen.get("rotation", fallback_rotation)
        return np.asarray(chosen.get("position", fallback_position), dtype=np.float64), None if rotation is None else Rotation.from_quat(rotation).as_matrix()
    if frame >= int(keys[-1].get("frame", 0)):
        chosen = keys[-1]
        rotation = chosen.get("rotation", fallback_rotation)
        return np.asarray(chosen.get("position", fallback_position), dtype=np.float64), None if rotation is None else Rotation.from_quat(rotation).as_matrix()
    right_index = next(index for index, key in enumerate(keys) if int(key.get("frame", 0)) >= frame)
    left_index = right_index - 1
    left, right = keys[left_index], keys[right_index]
    left_frame, right_frame = int(left.get("frame", 0)), int(right.get("frame", 0))
    amount = (frame - left_frame) / max(1, right_frame - left_frame)
    left_position = np.asarray(left.get("position", fallback_position), dtype=np.float64)
    right_position = np.asarray(right.get("position", fallback_position), dtype=np.float64)
    interpolation = str(constraint.get("interpolation", "cubic")).lower()
    if interpolation == "step":
        position = left_position
    elif interpolation == "linear":
        position = (1.0 - amount) * left_position + amount * right_position
    else:
        before = np.asarray(keys[max(0, left_index - 1)].get("position", left_position), dtype=np.float64)
        after = np.asarray(keys[min(len(keys) - 1, right_index + 1)].get("position", right_position), dtype=np.float64)
        position = _catmull_rom(before, left_position, right_position, after, float(amount))
    left_rotation = left.get("rotation", fallback_rotation)
    right_rotation = right.get("rotation", fallback_rotation)
    rotation = None
    if left_rotation is not None and right_rotation is not None:
        pair = Rotation.from_quat(np.stack([left_rotation, right_rotation]))
        rotation = Slerp([0.0, 1.0], pair)([float(amount)]).as_matrix()[0]
    return position, rotation


def _rotation_between(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    current /= max(float(np.linalg.norm(current)), 1e-9)
    desired /= max(float(np.linalg.norm(desired)), 1e-9)
    dot = float(np.clip(np.dot(current, desired), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    axis = np.cross(current, desired)
    if float(np.linalg.norm(axis)) < 1e-8:
        axis = np.cross(current, np.asarray([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-8:
            axis = np.cross(current, np.asarray([0.0, 1.0, 0.0]))
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    return Rotation.from_rotvec(axis * np.arccos(dot)).as_matrix()


def _set_world_rotation(local: np.ndarray, world: np.ndarray, parents: np.ndarray, joint: int, rotation: np.ndarray) -> None:
    parent = int(parents[joint])
    if parent < 0:
        local[joint, :3, :3] = rotation
        return
    # Transpose is only an inverse for a perfectly orthonormal matrix. Imported motion can carry tiny affine drift.
    parent_linear = world[parent, :3, :3]
    try:
        local[joint, :3, :3] = np.linalg.solve(parent_linear, rotation)
    except np.linalg.LinAlgError:
        local[joint, :3, :3] = np.linalg.pinv(parent_linear) @ rotation


def _solve_two_bone(
    world: np.ndarray,
    parents: np.ndarray,
    chain: tuple[int, int, int],
    target: np.ndarray,
) -> np.ndarray:
    start_index, middle_index, end_index = chain
    local = local_from_world(world, parents).astype(np.float64)
    solved_world = np.asarray(world, dtype=np.float64).copy()
    start, middle, end = (solved_world[index, :3, 3] for index in chain)
    direction = end - start
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    projected = start + direction * float(np.dot(middle - start, direction))
    bend = middle - projected
    if float(np.linalg.norm(bend)) < 1e-7:
        bend = np.asarray([0.0, 0.0, 1.0])
    pole = middle + bend / max(float(np.linalg.norm(bend)), 1e-9) * max(float(np.linalg.norm(end - start)) * 0.6, 0.05)
    positions = solve_two_bone_positions(start, middle, end, target, pole)
    delta = _rotation_between(middle - start, positions.middle.astype(np.float64) - start)
    _set_world_rotation(local, solved_world, parents, start_index, delta @ solved_world[start_index, :3, :3])
    solved_world = world_from_local(local, parents).astype(np.float64)
    middle, end = solved_world[middle_index, :3, 3], solved_world[end_index, :3, 3]
    delta = _rotation_between(end - middle, positions.end.astype(np.float64) - middle)
    _set_world_rotation(local, solved_world, parents, middle_index, delta @ solved_world[middle_index, :3, :3])
    return world_from_local(local, parents).astype(np.float32)


def apply_ik_constraints(
    world_motion: np.ndarray,
    parents: np.ndarray,
    names: list[str],
    constraints: list[dict[str, Any]],
) -> np.ndarray:
    world_motion = np.asarray(world_motion, dtype=np.float32)
    if not np.isfinite(world_motion).all():
        raise ValueError("IK input contains non-finite transforms; animation was not updated")
    if not constraints or not len(world_motion):
        return world_motion.copy()
    parents = np.asarray(parents, dtype=np.int64)
    index = {str(name): position for position, name in enumerate(names)}
    hips = index.get("Hips", 0)
    result = world_motion.copy()
    for frame in range(len(result)):
        active: list[tuple[dict[str, Any], int, np.ndarray, np.ndarray | None, float]] = []
        for constraint in constraints:
            start = int(constraint.get("start_frame", 0))
            end = int(constraint.get("end_frame", len(result) - 1))
            weight = _smooth_weight(frame, start, end, int(constraint.get("blend_in_frames", 0)), int(constraint.get("blend_out_frames", 0)))
            if weight <= 1e-8 or constraint.get("enabled", True) is False:
                continue
            joint_name = str(constraint.get("joint_name") or _EFFECTOR_NAMES.get(str(constraint.get("effector_id", "")).lower(), ""))
            if joint_name not in index:
                continue
            position, rotation = _sample_constraint(constraint, frame)
            active.append((constraint, index[joint_name], position, rotation, weight))
        if not active:
            continue
        frame_world = result[frame]
        two_bone = [entry for entry in active if str(entry[0].get("solver", "pbik")).lower() == "two_bone"]
        full_body = [entry for entry in active if str(entry[0].get("solver", "pbik")).lower() != "two_bone"]
        for constraint, joint, target, _, weight in two_bone:
            joint_name = names[joint]
            chain_names = _TWO_BONE_CHAINS.get(joint_name)
            if not chain_names or any(name not in index for name in chain_names):
                continue
            blended = frame_world[joint, :3, 3] * (1.0 - weight) + target * weight
            frame_world = _solve_two_bone(frame_world, parents, tuple(index[name] for name in chain_names), blended)
            if not np.isfinite(frame_world).all():
                raise ValueError(
                    f"IK solver two_bone produced non-finite transforms at frame {frame} "
                    f"for target {joint_name}; animation was not updated"
                )
        if full_body:
            groups: dict[tuple[str, str], list[tuple[dict[str, Any], int, np.ndarray, np.ndarray | None, float]]] = {}
            for entry in full_body:
                constraint = entry[0]
                key = (
                    str(constraint.get("solver", "pbik")).lower(),
                    str(constraint.get("root_behavior", "pre_pull")).lower(),
                )
                groups.setdefault(key, []).append(entry)
            for (solver, root_behavior), entries in groups.items():
                targets = []
                for constraint, joint, target, rotation, weight in entries:
                    blended = frame_world[joint, :3, 3] * (1.0 - weight) + target * weight
                    targets.append(IKTarget(joint, blended, rotation, weight, weight if rotation is not None else 0.0))
                solved = solve_full_body_frame(
                    frame_world,
                    parents,
                    hips,
                    targets,
                    solver=solver,
                    iterations=max(int(entry[0].get("iterations", 20)) for entry in entries),
                    tolerance=min(float(entry[0].get("tolerance", 0.002)) for entry in entries),
                    root_behavior=root_behavior,
                )
                if not np.isfinite(solved.world).all() or not np.isfinite(solved.residuals).all():
                    target_names = ", ".join(names[entry[1]] for entry in entries)
                    raise ValueError(
                        f"IK solver {solver} produced non-finite output at frame {frame} "
                        f"for targets {target_names}; animation was not updated"
                    )
                frame_world = solved.world
        result[frame] = frame_world
    return result.astype(np.float32)
