from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .ik.two_bone import solve_two_bone_positions
from .motion_math import local_from_world, world_from_local
from .terrain_surface import PlaneSurface
from .timeline import resample_world
from .virtual_root import build_virtual_root


GAIT_IK_VERSION = "gait-ik-terrain-v2"


@dataclass(frozen=True)
class GaitIKSettings:
    playback_speed: float = 1.0
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
    terrain_smoothing_frames: int = 5
    max_slope_degrees: float = 55.0

    def normalized(self) -> "GaitIKSettings":
        return GaitIKSettings(
            playback_speed=float(np.clip(self.playback_speed, 0.05, 4.0)),
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
            terrain_smoothing_frames=int(np.clip(self.terrain_smoothing_frames, 1, 61)),
            max_slope_degrees=float(np.clip(self.max_slope_degrees, 1.0, 89.0)),
        )


@dataclass(frozen=True)
class GaitIKResult:
    local_world_motion: np.ndarray
    preview_world_motion: np.ndarray
    virtual_root_motion: np.ndarray
    contacts: np.ndarray
    contact_weights: np.ndarray
    pelvis_correction: np.ndarray
    root_ground_heights: np.ndarray
    foot_ground_heights: np.ndarray
    foot_ground_normals: np.ndarray
    terrain_hits: np.ndarray
    metrics: dict[str, float]
    settings: dict[str, float | int]
    initial_transform: dict[str, float | list[float]]
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
    if values.ndim == 1:
        padded = np.pad(values, (radius, radius), mode="edge")
        return np.convolve(padded, np.ones(window, dtype=np.float64) / window, mode="valid")
    return np.stack([_moving_average(values[:, axis], window) for axis in range(values.shape[1])], axis=1)


def _fill_missing(values: np.ndarray, hits: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    valid = np.asarray(hits, dtype=bool) & np.isfinite(result)
    if not np.any(valid):
        result.fill(float(fallback))
        return result
    indexes = np.arange(len(result))
    result[~valid] = np.interp(indexes[~valid], indexes[valid], result[valid])
    return result


def _cleanup_contacts(mask: np.ndarray, min_length: int = 2, max_gap: int = 2) -> np.ndarray:
    clean = np.asarray(mask, dtype=bool).copy()
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
        ramp = min(blend_frames, max(0, (end - index) // 2))
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
    ground_heights: np.ndarray | None = None,
    terrain_hits: np.ndarray | None = None,
) -> np.ndarray:
    foot = np.asarray(world[:, foot_index, :3, 3], dtype=np.float64)
    sample = foot if toe_index is None else 0.5 * (foot + np.asarray(world[:, toe_index, :3, 3], dtype=np.float64))
    heights = foot[:, 1] if toe_index is None else np.minimum(foot[:, 1], world[:, toe_index, 1, 3])
    velocity = np.zeros(len(sample), dtype=np.float64)
    if len(sample) > 1:
        step = np.linalg.norm(np.diff(sample, axis=0), axis=1) * max(float(fps), 1e-6)
        velocity[1:] = step
        velocity[0] = step[0]
    if ground_heights is None:
        baseline = float(np.percentile(heights, 8.0))
        relative = heights - baseline
        hits = np.ones(len(heights), dtype=bool)
    else:
        ground = np.asarray(ground_heights, dtype=np.float64)
        relative = heights - ground
        hits = np.asarray(terrain_hits, dtype=bool) if terrain_hits is not None else np.isfinite(ground)
    raw = hits & (relative <= float(height_threshold)) & (relative >= -max(float(height_threshold), 0.03)) & (velocity <= float(velocity_threshold))
    clean = _cleanup_contacts(raw)
    if ground_heights is None and not np.any(clean) and len(clean):
        baseline = float(np.percentile(heights, 8.0))
        clean = _cleanup_contacts(heights <= baseline + max(float(height_threshold), 0.02))
    return clean


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
    target_normal: np.ndarray,
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
        normal = np.asarray(target_normal, dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        current_rotation = world[end_index, :3, :3]
        bind_rotation = np.asarray(target_bind[end_index, :3, :3], dtype=np.float64)
        semantic_local = bind_rotation.T
        forward = current_rotation @ semantic_local[:, 2]
        forward -= normal * float(np.dot(forward, normal))
        if float(np.linalg.norm(forward)) < 1e-7:
            forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            forward -= normal * float(np.dot(forward, normal))
        forward /= max(float(np.linalg.norm(forward)), 1e-9)
        right = np.cross(normal, forward)
        right /= max(float(np.linalg.norm(right)), 1e-9)
        forward = np.cross(right, normal)
        desired_rotation = np.column_stack([right, normal, forward]) @ bind_rotation
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


def _locked_surface_targets(
    positions: np.ndarray,
    contacts: np.ndarray,
    surface,
    sole_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float64)
    samples = surface.sample_many(positions[:, [0, 2]])
    targets = positions.copy()
    targets[:, 1] = np.where(samples.hits, np.maximum(targets[:, 1], samples.heights + sole_offset), targets[:, 1])
    normals = samples.normals.copy()
    index = 0
    while index < len(contacts):
        if not contacts[index]:
            index += 1
            continue
        end = index + 1
        while end < len(contacts) and contacts[end]:
            end += 1
        anchor_xz = np.median(positions[index:end, [0, 2]], axis=0)
        height, normal, hit = surface.sample(float(anchor_xz[0]), float(anchor_xz[1]))
        if hit:
            anchor = np.asarray([anchor_xz[0], height + sole_offset, anchor_xz[1]], dtype=np.float64)
            targets[index:end] = anchor
            normals[index:end] = normal
        index = end
    return targets, normals, samples.heights, samples.hits


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
    surface=None,
    initial_translation: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    initial_yaw_degrees: float = 0.0,
) -> GaitIKResult:
    settings = (settings or GaitIKSettings()).normalized()
    surface = surface or PlaneSurface(0.0)
    initial_translation = np.asarray(initial_translation if initial_translation is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    if initial_translation.shape != (3,) or not np.isfinite(initial_translation).all():
        raise ValueError("initial translation must contain three finite values")
    initial_yaw_degrees = float(initial_yaw_degrees)
    world_motion = np.asarray(world_motion, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    target_bind = np.asarray(target_bind, dtype=np.float32)
    index = {str(name): position for position, name in enumerate(names)}
    required = {"Hips", "LeftLeg", "LeftShin", "LeftFoot", "RightLeg", "RightShin", "RightFoot"}
    missing = sorted(required.difference(index))
    if missing:
        raise ValueError(f"humanoid gait IK joints missing: {', '.join(missing)}")

    source_frame_count = int(len(world_motion))
    output_frame_count = max(1, int(round(source_frame_count / settings.playback_speed)))
    if output_frame_count != source_frame_count:
        world_motion = resample_world(world_motion, output_frame_count).astype(np.float32)
        if reference_root_positions is not None:
            reference = np.asarray(reference_root_positions, dtype=np.float64)
            if len(reference) == source_frame_count:
                source_t = np.arange(source_frame_count, dtype=np.float64)
                target_t = np.linspace(0.0, max(0, source_frame_count - 1), output_frame_count)
                reference_root_positions = np.stack(
                    [np.interp(target_t, source_t, reference[:, axis]) for axis in range(3)], axis=1
                ).astype(np.float32)

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
    virtual_root = np.asarray(root_result.virtual_root_motion, dtype=np.float64).copy()
    yaw = np.deg2rad(initial_yaw_degrees)
    yaw_rotation = Rotation.from_rotvec([0.0, yaw, 0.0]).as_matrix()
    origin_xz = virtual_root[0, [0, 2], 3].copy()
    relative = virtual_root[:, [0, 2], 3] - origin_xz
    relative *= settings.root_distance_scale
    rotated = relative @ yaw_rotation[np.ix_([0, 2], [0, 2])].T
    virtual_root[:, 0, 3] = initial_translation[0] + rotated[:, 0]
    virtual_root[:, 2, 3] = initial_translation[2] + rotated[:, 1]
    virtual_root[:, :3, :3] = np.einsum("ab,fbc->fac", yaw_rotation, virtual_root[:, :3, :3])
    root_samples = surface.sample_many(virtual_root[:, [0, 2], 3])
    root_ground = _fill_missing(root_samples.heights, root_samples.hits, fallback=float(initial_translation[1]))
    root_ground = _moving_average(root_ground, settings.terrain_smoothing_frames)
    virtual_root[:, 1, 3] = root_ground
    virtual_root = virtual_root.astype(np.float32)
    grounded_world = np.einsum("fab,fjbc->fjac", virtual_root, root_result.local_world_motion).astype(np.float32)

    sides = (
        ((index["LeftLeg"], index["LeftShin"], index["LeftFoot"]), index.get("LeftToeBase"), np.asarray([0.0, 0.0, 1.0])),
        ((index["RightLeg"], index["RightShin"], index["RightFoot"]), index.get("RightToeBase"), np.asarray([0.0, 0.0, 1.0])),
    )
    frame_count = len(grounded_world)
    contacts = np.zeros((frame_count, 2), dtype=bool)
    weights = np.zeros((frame_count, 2), dtype=np.float32)
    terrain_hits = np.zeros((frame_count, 2), dtype=bool)
    foot_ground_heights = np.full((frame_count, 2), np.nan, dtype=np.float32)
    foot_ground_normals = np.repeat(np.asarray([[[0.0, 1.0, 0.0]]], dtype=np.float32), frame_count * 2, axis=0).reshape(frame_count, 2, 3)
    targets: list[np.ndarray] = []
    target_normals: list[np.ndarray] = []
    hips_positions = grounded_world[:, index["Hips"], :3, 3].astype(np.float64)
    slope_cos = float(np.cos(np.deg2rad(settings.max_slope_degrees)))
    for side_index, (chain, toe_index, _) in enumerate(sides):
        foot_index = chain[2]
        foot_positions = grounded_world[:, foot_index, :3, 3].astype(np.float64)
        raw_samples = surface.sample_many(foot_positions[:, [0, 2]])
        valid_hits = raw_samples.hits & (raw_samples.normals[:, 1] >= slope_cos)
        foot_ground_heights[:, side_index] = raw_samples.heights.astype(np.float32)
        foot_ground_normals[:, side_index] = raw_samples.normals.astype(np.float32)
        terrain_hits[:, side_index] = valid_hits
        contacts[:, side_index] = detect_foot_contacts(
            grounded_world,
            foot_index,
            toe_index,
            fps,
            height_threshold=settings.contact_height,
            velocity_threshold=settings.contact_velocity,
            ground_heights=raw_samples.heights,
            terrain_hits=valid_hits,
        )
        weights[:, side_index] = _contact_weights(contacts[:, side_index], settings.plant_blend_frames)
        swing = foot_positions.copy()
        swing[:, [0, 2]] = hips_positions[:, [0, 2]] + (foot_positions[:, [0, 2]] - hips_positions[:, [0, 2]]) * settings.stride_scale
        swing_samples = surface.sample_many(swing[:, [0, 2]])
        swing[:, 1] = np.where(swing_samples.hits, np.maximum(swing[:, 1], swing_samples.heights + settings.sole_offset), swing[:, 1])
        locked, normals, _, _ = _locked_surface_targets(foot_positions, contacts[:, side_index], surface, settings.sole_offset)
        amount = (weights[:, side_index] * settings.foot_lock_strength)[:, None]
        targets.append((1.0 - amount) * swing + amount * locked)
        target_normals.append(normals)

    pelvis = np.zeros(frame_count, dtype=np.float64)
    overreach = np.zeros((frame_count, 2), dtype=bool)
    for frame in range(frame_count):
        lower_bound = -np.inf
        upper_bound = np.inf
        for side_index, (chain, _, _) in enumerate(sides):
            start = grounded_world[frame, chain[0], :3, 3].astype(np.float64)
            target = targets[side_index][frame]
            upper = float(np.linalg.norm(target_bind[chain[1], :3, 3] - target_bind[chain[0], :3, 3]))
            lower = float(np.linalg.norm(target_bind[chain[2], :3, 3] - target_bind[chain[1], :3, 3]))
            reach = max((upper + lower) * 0.995, 1e-4)
            horizontal = float(np.linalg.norm((target - start)[[0, 2]]))
            if horizontal >= reach:
                overreach[frame, side_index] = True
                continue
            vertical = float(np.sqrt(max(reach * reach - horizontal * horizontal, 0.0)))
            lower_bound = max(lower_bound, float(target[1] - vertical))
            upper_bound = min(upper_bound, float(target[1] + vertical))
        current = float(grounded_world[frame, index["Hips"], 1, 3])
        desired = current
        if lower_bound <= upper_bound:
            desired = float(np.clip(current, lower_bound, upper_bound))
        elif np.isfinite(lower_bound) and np.isfinite(upper_bound):
            desired = 0.5 * (lower_bound + upper_bound)
        pelvis[frame] = np.clip(desired - current, -settings.pelvis_max_correction, settings.pelvis_max_correction)
    pelvis = _moving_average(pelvis, settings.pelvis_smoothing_frames).astype(np.float32)

    corrected = grounded_world.copy()
    for frame in range(frame_count):
        local = local_from_world(corrected[frame], parents).astype(np.float64)
        local[index["Hips"], 1, 3] += float(pelvis[frame])
        corrected[frame] = world_from_local(local, parents)
        for side_index, (chain, _, preferred) in enumerate(sides):
            corrected[frame] = _solve_leg(
                corrected[frame],
                parents,
                chain,
                targets[side_index][frame],
                preferred,
                target_bind,
                target_normals[side_index][frame],
                settings.ground_rotation_strength * float(weights[frame, side_index]),
            )

    inv_root = np.linalg.inv(virtual_root.astype(np.float64)).astype(np.float32)
    local_world = np.einsum("fab,fjbc->fjac", inv_root, corrected).astype(np.float32)
    before = [_slide_distance(grounded_world, sides[i][0][2], contacts[:, i]) for i in range(2)]
    after = [_slide_distance(corrected, sides[i][0][2], contacts[:, i]) for i in range(2)]
    penetrations: list[np.ndarray] = []
    slopes: list[np.ndarray] = []
    for side_index, (chain, _, _) in enumerate(sides):
        foot = corrected[:, chain[2], :3, 3]
        samples = surface.sample_many(foot[:, [0, 2]])
        gap = foot[:, 1] - samples.heights - settings.sole_offset
        penetrations.append(np.where(samples.hits, gap, np.nan))
        slopes.append(np.rad2deg(np.arccos(np.clip(samples.normals[:, 1], -1.0, 1.0))))
    finite_penetration = np.concatenate([p[np.isfinite(p)] for p in penetrations]) if any(np.isfinite(p).any() for p in penetrations) else np.asarray([0.0])
    metrics = {
        "playback_speed": float(settings.playback_speed),
        "source_frames": float(source_frame_count),
        "output_frames": float(frame_count),
        "left_slide_before": before[0],
        "right_slide_before": before[1],
        "left_slide_after": after[0],
        "right_slide_after": after[1],
        "slide_before": before[0] + before[1],
        "slide_after": after[0] + after[1],
        "left_contact_ratio": float(np.mean(contacts[:, 0])) if frame_count else 0.0,
        "right_contact_ratio": float(np.mean(contacts[:, 1])) if frame_count else 0.0,
        "terrain_no_hit_ratio": float(1.0 - np.mean(terrain_hits)) if frame_count else 0.0,
        "overreach_ratio": float(np.mean(overreach)) if frame_count else 0.0,
        "max_penetration": float(max(0.0, -float(np.nanmin(finite_penetration)))),
        "max_slope_degrees": float(max(np.nanmax(slopes[0]), np.nanmax(slopes[1]))) if frame_count else 0.0,
        "max_pelvis_correction": float(np.max(np.abs(pelvis))) if frame_count else 0.0,
    }
    initial = {
        "translation": [float(initial_translation[0]), float(root_ground[0]), float(initial_translation[2])],
        "yaw_degrees": initial_yaw_degrees,
    }
    return GaitIKResult(
        local_world_motion=local_world,
        preview_world_motion=corrected.astype(np.float32),
        virtual_root_motion=virtual_root,
        contacts=contacts.astype(np.uint8),
        contact_weights=weights.astype(np.float32),
        pelvis_correction=pelvis,
        root_ground_heights=root_ground.astype(np.float32),
        foot_ground_heights=foot_ground_heights,
        foot_ground_normals=foot_ground_normals,
        terrain_hits=terrain_hits.astype(np.uint8),
        metrics=metrics,
        settings=asdict(settings),
        initial_transform=initial,
        ground_height=float(root_ground[0]),
    )
