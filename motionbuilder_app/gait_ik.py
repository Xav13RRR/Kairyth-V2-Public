from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .ik.two_bone import solve_two_bone_positions
from .motion_math import local_from_world, world_from_local
from .virtual_root import build_virtual_root


GAIT_IK_VERSION = "gait-ik-flat-ground-v1"


@dataclass(frozen=True)
class GaitIKSettings:
    stride_scale: float = 1.0
    root_distance_scale: float = 1.0
    foot_lock_strength: float = 1.0
    contact_height: float = 0.06
    contact_velocity: float = 0.25
    plant_blend_frames: int = 4
    sole_offset: float = 0.0
    pelvis_max_correction: float = 0.12
    pelvis_smoothing_frames: int = 7
    ground_rotation_strength: float = 1.0
    root_smoothing_frames: int = 9
    root_position_tolerance: float = 0.005

    def normalized(self) -> "GaitIKSettings":
        return GaitIKSettings(
            stride_scale=float(np.clip(self.stride_scale, 0.25, 2.5)),
            root_distance_scale=float(np.clip(self.root_distance_scale, 0.0, 3.0)),
            foot_lock_strength=float(np.clip(self.foot_lock_strength, 0.0, 1.0)),
            contact_height=float(np.clip(self.contact_height, 0.001, 0.25)),
            contact_velocity=float(np.clip(self.contact_velocity, 0.01, 3.0)),
            plant_blend_frames=int(np.clip(self.plant_blend_frames, 0, 30)),
            sole_offset=float(np.clip(self.sole_offset, -0.1, 0.25)),
            pelvis_max_correction=float(np.clip(self.pelvis_max_correction, 0.0, 0.5)),
            pelvis_smoothing_frames=int(np.clip(self.pelvis_smoothing_frames, 1, 61)),
            ground_rotation_strength=float(np.clip(self.ground_rotation_strength, 0.0, 1.0)),
            root_smoothing_frames=int(np.clip(self.root_smoothing_frames, 1, 61)),
            root_position_tolerance=float(np.clip(self.root_position_tolerance, 0.0, 0.2)),
        )


@dataclass(frozen=True)
class GaitIKResult:
    local_world_motion: np.ndarray
    preview_world_motion: np.ndarray
    virtual_root_motion: np.ndarray
    contacts: np.ndarray
    contact_weights: np.ndarray
    pelvis_correction: np.ndarray
    metrics: dict[str, float]
    settings: dict[str, float | int]
    ground_height: float
    version: str = GAIT_IK_VERSION


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window = max(1, int(window))
    if window <= 1 or len(values) < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window, dtype=np.float64) / window, mode="valid")


def _cleanup_contacts(mask: np.ndarray, min_length: int = 2, max_gap: int = 2) -> np.ndarray:
    clean = np.asarray(mask, dtype=bool).copy()
    if not len(clean):
        return clean
    start = 0
    while start < len(clean):
        value = clean[start]
        end = start + 1
        while end < len(clean) and clean[end] == value:
            end += 1
        length = end - start
        if value and length < min_length:
            clean[start:end] = False
        elif not value and length <= max_gap and start > 0 and end < len(clean):
            clean[start:end] = True
        start = end
    return clean


def _contact_weights(mask: np.ndarray, blend_frames: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    weights = mask.astype(np.float64)
    blend_frames = max(0, int(blend_frames))
    if blend_frames <= 0:
        return weights.astype(np.float32)
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        end = index + 1
        while end < len(mask) and mask[end]:
            end += 1
        length = end - index
        ramp = min(blend_frames, max(0, length // 2))
        for offset in range(ramp):
            amount = 0.5 - 0.5 * np.cos(np.pi * (offset + 1) / (ramp + 1))
            weights[index + offset] = min(weights[index + offset], amount)
            weights[end - 1 - offset] = min(weights[end - 1 - offset], amount)
        index = end
    return weights.astype(np.float32)


def detect_foot_contacts(
    world: np.ndarray,
    foot_index: int,
    toe_index: int | None,
    fps: float,
    *,
    height_threshold: float,
    velocity_threshold: float,
) -> np.ndarray:
    foot = np.asarray(world[:, foot_index, :3, 3], dtype=np.float64)
    sample = foot if toe_index is None else 0.5 * (foot + np.asarray(world[:, toe_index, :3, 3], dtype=np.float64))
    heights = foot[:, 1] if toe_index is None else np.minimum(foot[:, 1], world[:, toe_index, 1, 3])
    baseline = float(np.percentile(heights, 8.0))
    velocity = np.zeros(len(sample), dtype=np.float64)
    if len(sample) > 1:
        step = np.linalg.norm(np.diff(sample, axis=0), axis=1) * max(float(fps), 1e-6)
        velocity[1:] = step
        velocity[0] = step[0]
    raw = (heights <= baseline + float(height_threshold)) & (velocity <= float(velocity_threshold))
    clean = _cleanup_contacts(raw)
    if not np.any(clean) and len(clean):
        fallback = heights <= baseline + max(float(height_threshold), 0.02)
        clean = _cleanup_contacts(fallback)
    return clean


def _contact_targets(positions: np.ndarray, contacts: np.ndarray, sole_offset: float) -> np.ndarray:
    targets = np.asarray(positions, dtype=np.float64).copy()
    index = 0
    while index < len(contacts):
        if not contacts[index]:
            index += 1
            continue
        end = index + 1
        while end < len(contacts) and contacts[end]:
            end += 1
        anchor = np.median(targets[index:end], axis=0)
        anchor[1] = max(float(anchor[1]), float(sole_offset))
        targets[index:end] = anchor
        index = end
    return targets


def _rotation_between(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    left = np.asarray(current, dtype=np.float64)
    right = np.asarray(desired, dtype=np.float64)
    left /= max(float(np.linalg.norm(left)), 1e-9)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-9:
        axis = np.cross(left, np.asarray([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(left, np.asarray([0.0, 1.0, 0.0]))
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    axis = np.cross(left, right)
    sine = float(np.linalg.norm(axis))
    axis /= max(sine, 1e-9)
    return Rotation.from_rotvec(axis * np.arctan2(sine, dot)).as_matrix()


def _set_world_rotation(local: np.ndarray, world: np.ndarray, parents: np.ndarray, joint: int, rotation: np.ndarray) -> None:
    parent = int(parents[joint])
    local[joint, :3, :3] = rotation if parent < 0 else world[parent, :3, :3].T @ rotation


def _solve_leg(
    frame_world: np.ndarray,
    parents: np.ndarray,
    chain: tuple[int, int, int],
    target: np.ndarray,
    preferred_bend: np.ndarray,
    target_bind: np.ndarray,
    rotation_weight: float,
) -> np.ndarray:
    start_index, middle_index, end_index = chain
    local = local_from_world(frame_world, parents).astype(np.float64)
    world = np.asarray(frame_world, dtype=np.float64).copy()
    start = world[start_index, :3, 3]
    middle = world[middle_index, :3, 3]
    end = world[end_index, :3, 3]
    direction = end - start
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    projected = start + direction * float(np.dot(middle - start, direction))
    bend = middle - projected
    if float(np.linalg.norm(bend)) < 1e-6:
        bend = np.asarray(preferred_bend, dtype=np.float64)
    bend /= max(float(np.linalg.norm(bend)), 1e-9)
    pole = middle + bend * max(float(np.linalg.norm(middle - start) + np.linalg.norm(end - middle)) * 0.6, 0.05)
    solved = solve_two_bone_positions(start, middle, end, target, pole)

    delta = _rotation_between(middle - start, solved.middle.astype(np.float64) - start)
    _set_world_rotation(local, world, parents, start_index, delta @ world[start_index, :3, :3])
    world = world_from_local(local, parents).astype(np.float64)

    middle = world[middle_index, :3, 3]
    end = world[end_index, :3, 3]
    delta = _rotation_between(end - middle, solved.end.astype(np.float64) - middle)
    _set_world_rotation(local, world, parents, middle_index, delta @ world[middle_index, :3, :3])
    world = world_from_local(local, parents).astype(np.float64)

    if rotation_weight > 1e-6:
        current_rotation = world[end_index, :3, :3]
        bind_rotation = np.asarray(target_bind[end_index, :3, :3], dtype=np.float64)
        semantic_local = bind_rotation.T
        forward = current_rotation @ semantic_local[:, 2]
        forward[1] = 0.0
        if float(np.linalg.norm(forward)) < 1e-7:
            forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        forward /= float(np.linalg.norm(forward))
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(up, forward)
        right /= max(float(np.linalg.norm(right)), 1e-9)
        desired_rotation = np.column_stack([right, up, forward]) @ bind_rotation
        rotations = Rotation.from_matrix(np.stack([current_rotation, desired_rotation]))
        desired = Slerp([0.0, 1.0], rotations)([float(np.clip(rotation_weight, 0.0, 1.0))]).as_matrix()[0]
        _set_world_rotation(local, world, parents, end_index, desired)
        world = world_from_local(local, parents).astype(np.float64)
    return world.astype(np.float32)


def _slide_distance(world: np.ndarray, foot_index: int, contacts: np.ndarray) -> float:
    positions = np.asarray(world[:, foot_index, :3, 3], dtype=np.float64)[:, [0, 2]]
    if len(positions) <= 1:
        return 0.0
    active = np.asarray(contacts[:-1], dtype=bool) & np.asarray(contacts[1:], dtype=bool)
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0)[active], axis=1))) if np.any(active) else 0.0


def process_gait_ik(
    world_motion: np.ndarray,
    parents: np.ndarray,
    names: list[str],
    target_bind: np.ndarray,
    fps: float,
    *,
    settings: GaitIKSettings | None = None,
    rest_vertices: np.ndarray | None = None,
    lbs_indices: np.ndarray | None = None,
    lbs_weights: np.ndarray | None = None,
    reference_root_positions: np.ndarray | None = None,
) -> GaitIKResult:
    settings = (settings or GaitIKSettings()).normalized()
    world_motion = np.asarray(world_motion, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    target_bind = np.asarray(target_bind, dtype=np.float32)
    index = {str(name): position for position, name in enumerate(names)}
    required = {"Hips", "LeftLeg", "LeftShin", "LeftFoot", "RightLeg", "RightShin", "RightFoot"}
    missing = sorted(required.difference(index))
    if missing:
        raise ValueError(f"humanoid gait IK joints missing: {', '.join(missing)}")

    root_result = build_virtual_root(
        world_motion,
        parents,
        names,
        target_bind,
        rest_vertices=rest_vertices,
        lbs_indices=lbs_indices,
        lbs_weights=lbs_weights,
        reference_root_positions=reference_root_positions,
        smoothing_frames=settings.root_smoothing_frames,
        position_tolerance=settings.root_position_tolerance,
    )
    virtual_root = np.asarray(root_result.virtual_root_motion, dtype=np.float32).copy()
    origin_xz = virtual_root[0, [0, 2], 3].copy()
    virtual_root[:, [0, 2], 3] = origin_xz + (
        virtual_root[:, [0, 2], 3] - origin_xz
    ) * settings.root_distance_scale
    virtual_root[:, 1, 3] = 0.0
    grounded_world = np.einsum("fab,fjbc->fjac", virtual_root, root_result.local_world_motion).astype(np.float32)

    sides = (
        ("left", (index["LeftLeg"], index["LeftShin"], index["LeftFoot"]), index.get("LeftToeBase"), np.asarray([0.0, 0.0, 1.0])),
        ("right", (index["RightLeg"], index["RightShin"], index["RightFoot"]), index.get("RightToeBase"), np.asarray([0.0, 0.0, 1.0])),
    )
    contacts = np.zeros((len(grounded_world), 2), dtype=bool)
    weights = np.zeros((len(grounded_world), 2), dtype=np.float32)
    targets: list[np.ndarray] = []
    hips_positions = grounded_world[:, index["Hips"], :3, 3].astype(np.float64)
    for side_index, (_, chain, toe_index, _) in enumerate(sides):
        foot_index = chain[2]
        contacts[:, side_index] = detect_foot_contacts(
            grounded_world,
            foot_index,
            toe_index,
            fps,
            height_threshold=settings.contact_height,
            velocity_threshold=settings.contact_velocity,
        )
        weights[:, side_index] = _contact_weights(contacts[:, side_index], settings.plant_blend_frames)
        foot_positions = grounded_world[:, foot_index, :3, 3].astype(np.float64)
        swing = foot_positions.copy()
        swing[:, [0, 2]] = hips_positions[:, [0, 2]] + (
            foot_positions[:, [0, 2]] - hips_positions[:, [0, 2]]
        ) * settings.stride_scale
        locked = _contact_targets(foot_positions, contacts[:, side_index], settings.sole_offset)
        amount = (weights[:, side_index] * settings.foot_lock_strength)[:, None]
        targets.append((1.0 - amount) * swing + amount * locked)

    pelvis = np.zeros(len(grounded_world), dtype=np.float64)
    for frame in range(len(grounded_world)):
        correction = 0.0
        for side_index, (_, chain, _, _) in enumerate(sides):
            start = grounded_world[frame, chain[0], :3, 3].astype(np.float64)
            target = targets[side_index][frame]
            upper = float(np.linalg.norm(target_bind[chain[1], :3, 3] - target_bind[chain[0], :3, 3]))
            lower = float(np.linalg.norm(target_bind[chain[2], :3, 3] - target_bind[chain[1], :3, 3]))
            reach = max(upper + lower - 1e-4, 1e-4)
            horizontal = float(np.linalg.norm((target - start)[[0, 2]]))
            if horizontal >= reach:
                continue
            highest_reachable = float(target[1] + np.sqrt(max(reach * reach - horizontal * horizontal, 0.0)))
            correction = min(correction, highest_reachable - float(start[1]))
        pelvis[frame] = np.clip(correction, -settings.pelvis_max_correction, settings.pelvis_max_correction)
    pelvis = _moving_average(pelvis, settings.pelvis_smoothing_frames).astype(np.float32)

    corrected = grounded_world.copy()
    for frame in range(len(corrected)):
        local = local_from_world(corrected[frame], parents).astype(np.float64)
        local[index["Hips"], 1, 3] += float(pelvis[frame])
        corrected[frame] = world_from_local(local, parents)
        for side_index, (_, chain, _, preferred) in enumerate(sides):
            corrected[frame] = _solve_leg(
                corrected[frame],
                parents,
                chain,
                targets[side_index][frame],
                preferred,
                target_bind,
                settings.ground_rotation_strength * float(weights[frame, side_index]),
            )

    inv_root = np.linalg.inv(virtual_root.astype(np.float64)).astype(np.float32)
    local_world = np.einsum("fab,fjbc->fjac", inv_root, corrected).astype(np.float32)
    before_left = _slide_distance(grounded_world, sides[0][1][2], contacts[:, 0])
    before_right = _slide_distance(grounded_world, sides[1][1][2], contacts[:, 1])
    after_left = _slide_distance(corrected, sides[0][1][2], contacts[:, 0])
    after_right = _slide_distance(corrected, sides[1][1][2], contacts[:, 1])
    metrics = {
        "left_slide_before": before_left,
        "right_slide_before": before_right,
        "left_slide_after": after_left,
        "right_slide_after": after_right,
        "slide_before": before_left + before_right,
        "slide_after": after_left + after_right,
        "left_contact_ratio": float(np.mean(contacts[:, 0])) if len(contacts) else 0.0,
        "right_contact_ratio": float(np.mean(contacts[:, 1])) if len(contacts) else 0.0,
    }
    return GaitIKResult(
        local_world_motion=local_world,
        preview_world_motion=corrected.astype(np.float32),
        virtual_root_motion=virtual_root.astype(np.float32),
        contacts=contacts.astype(np.uint8),
        contact_weights=weights.astype(np.float32),
        pelvis_correction=pelvis,
        metrics=metrics,
        settings=asdict(settings),
        ground_height=float(root_result.ground_height),
    )


# Keep the original import surface stable while routing callers through the
# terrain-aware solver. The v1 helpers above remain useful for compatibility.
from .gait_ik_v2 import (  # noqa: E402,F401
    GAIT_IK_VERSION as GAIT_IK_VERSION,
    GaitIKResult as GaitIKResult,
    GaitIKSettings as GaitIKSettings,
    detect_foot_contacts as detect_foot_contacts,
    process_gait_ik as process_gait_ik,
)
