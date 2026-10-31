from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .motion_math import local_from_world, world_from_local
from .timeline import resample_world


ADDITIVE_REFERENCE_TYPES = {"first_frame", "animation_frame", "ref_pose"}
ADDITIVE_END_BEHAVIORS = {"loop", "hold"}


def _sample_world_positions(world: np.ndarray, positions: np.ndarray, *, loop: bool) -> np.ndarray:
    source = np.asarray(world, dtype=np.float32)
    if source.ndim != 4 or source.shape[2:] != (4, 4) or not len(source):
        raise ValueError("world motion must be [F, J, 4, 4]")
    sample_positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    if len(source) == 1:
        return np.repeat(source, len(sample_positions), axis=0)
    if loop:
        sample_positions = np.mod(sample_positions, len(source))
        lower = np.floor(sample_positions).astype(np.int64)
        upper = (lower + 1) % len(source)
    else:
        sample_positions = np.clip(sample_positions, 0.0, len(source) - 1.0)
        lower = np.floor(sample_positions).astype(np.int64)
        upper = np.minimum(lower + 1, len(source) - 1)
    alpha = sample_positions - lower
    out = np.repeat(np.eye(4, dtype=np.float32)[None, None], len(sample_positions), axis=0)
    out = np.repeat(out, source.shape[1], axis=1)
    for joint in range(source.shape[1]):
        for frame, (left, right, weight) in enumerate(zip(lower, upper, alpha, strict=True)):
            left_matrix = source[left, joint]
            right_matrix = source[right, joint]
            out[frame, joint, :3, 3] = (
                (1.0 - weight) * left_matrix[:3, 3] + weight * right_matrix[:3, 3]
            )
            if left == right or weight <= 1e-12:
                out[frame, joint, :3, :3] = left_matrix[:3, :3]
            else:
                rotations = Rotation.from_matrix(
                    np.stack([left_matrix[:3, :3], right_matrix[:3, :3]])
                )
                out[frame, joint, :3, :3] = Slerp([0.0, 1.0], rotations)([weight]).as_matrix()[0]
    return out


def additive_output_frames(
    base_frames: int,
    *,
    base_fps: float,
    target_fps: float,
    base_speed: float,
) -> int:
    if base_frames < 1:
        raise ValueError("base animation is empty")
    if min(base_fps, target_fps, base_speed) <= 0:
        raise ValueError("animation rates must be greater than zero")
    return max(1, round(base_frames * target_fps / base_fps / base_speed))


def build_additive_world(
    base_world: np.ndarray,
    additive_world: np.ndarray,
    parents: np.ndarray,
    *,
    base_fps: float,
    additive_fps: float,
    target_fps: float = 30.0,
    base_speed: float = 1.0,
    additive_speed: float = 1.0,
    weight: float = 1.0,
    reference_type: str = "first_frame",
    reference_frame: int = 0,
    reference_world: np.ndarray | None = None,
    end_behavior: str = "loop",
) -> np.ndarray:
    """Apply a full-skeleton local-space additive layer to a base animation.

    Output duration follows the speed-adjusted base animation. The additive
    animation is independently sampled and either loops or holds its last pose.
    Horizontal translation of the skeleton root comes only from the base layer.
    """

    base = np.asarray(base_world, dtype=np.float32)
    layer = np.asarray(additive_world, dtype=np.float32)
    hierarchy = np.asarray(parents, dtype=np.int64)
    if base.ndim != 4 or layer.ndim != 4 or base.shape[2:] != (4, 4) or layer.shape[2:] != (4, 4):
        raise ValueError("animation motion must be [F, J, 4, 4]")
    if not len(base) or not len(layer):
        raise ValueError("animations must contain at least one frame")
    if base.shape[1] != layer.shape[1] or len(hierarchy) != base.shape[1]:
        raise ValueError("additive animations require compatible skeletons")
    if reference_type not in ADDITIVE_REFERENCE_TYPES:
        raise ValueError("unsupported additive reference type")
    if end_behavior not in ADDITIVE_END_BEHAVIORS:
        raise ValueError("unsupported additive end behavior")
    if not 0.0 <= float(weight) <= 2.0:
        raise ValueError("additive weight must be between 0 and 2")
    if min(base_fps, additive_fps, target_fps, base_speed, additive_speed) <= 0:
        raise ValueError("animation rates must be greater than zero")

    frames = additive_output_frames(
        len(base), base_fps=base_fps, target_fps=target_fps, base_speed=base_speed
    )
    sampled_base = resample_world(base, frames)
    layer_positions = np.arange(frames, dtype=np.float64) * additive_speed * additive_fps / target_fps
    sampled_layer = _sample_world_positions(layer, layer_positions, loop=end_behavior == "loop")
    base_local = local_from_world(sampled_base, hierarchy).astype(np.float64)
    layer_local = local_from_world(sampled_layer, hierarchy).astype(np.float64)

    if reference_type == "ref_pose":
        if reference_world is None:
            raise ValueError("reference pose matrices are required")
        ref = np.asarray(reference_world, dtype=np.float32)
        if ref.ndim == 3:
            ref = ref[None]
        if ref.shape != (1, base.shape[1], 4, 4):
            raise ValueError("reference pose must be [J, 4, 4]")
        reference_local = local_from_world(ref, hierarchy)[0].astype(np.float64)
    else:
        frame = 0 if reference_type == "first_frame" else int(reference_frame)
        frame = int(np.clip(frame, 0, len(layer) - 1))
        reference_local = local_from_world(layer[frame], hierarchy).astype(np.float64)

    output_local = base_local.copy()
    blend_weight = float(weight)
    for joint in range(base.shape[1]):
        reference_rotation = reference_local[joint, :3, :3]
        layer_rotation = layer_local[:, joint, :3, :3]
        delta_rotation = np.einsum("ab,fbc->fac", reference_rotation.T, layer_rotation)
        if blend_weight <= 1e-12:
            weighted_delta = np.repeat(np.eye(3, dtype=np.float64)[None], frames, axis=0)
        else:
            delta = Rotation.from_matrix(delta_rotation)
            if np.isclose(blend_weight, 1.0):
                weighted_delta = delta.as_matrix()
            else:
                weighted_delta = Rotation.from_rotvec(delta.as_rotvec() * blend_weight).as_matrix()
        output_local[:, joint, :3, :3] = np.einsum(
            "fab,fbc->fac", base_local[:, joint, :3, :3], weighted_delta
        )
        translation_delta = layer_local[:, joint, :3, 3] - reference_local[joint, :3, 3]
        if int(hierarchy[joint]) < 0:
            translation_delta[:, 0] = 0.0
            translation_delta[:, 2] = 0.0
        output_local[:, joint, :3, 3] = (
            base_local[:, joint, :3, 3] + translation_delta * blend_weight
        )
    return world_from_local(output_local, hierarchy).astype(np.float32)


def sample_base_transform_track(track: np.ndarray, output_frames: int) -> np.ndarray:
    values = np.asarray(track, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError("transform track must be [F, 4, 4]")
    return resample_world(values[:, None], output_frames)[:, 0]
