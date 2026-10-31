from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .motion_math import local_from_world, world_from_local


def resample_world(world: np.ndarray, output_frames: int) -> np.ndarray:
    world = np.asarray(world, dtype=np.float32)
    output_frames = max(1, int(output_frames))
    if len(world) == output_frames:
        return world.copy()
    if len(world) == 1:
        return np.repeat(world, output_frames, axis=0)
    source_t = np.arange(len(world), dtype=np.float64)
    target_t = np.linspace(0, len(world) - 1, output_frames)
    out = np.repeat(np.eye(4, dtype=np.float32)[None, None], output_frames, axis=0)
    out = np.repeat(out, world.shape[1], axis=1)
    for joint in range(world.shape[1]):
        out[:, joint, :3, :3] = Slerp(source_t, Rotation.from_matrix(world[:, joint, :3, :3]))(target_t).as_matrix()
        for axis in range(3):
            out[:, joint, axis, 3] = np.interp(target_t, source_t, world[:, joint, axis, 3])
    return out


def segment_world(
    world: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    speed: float,
    reverse: bool,
    source_fps: float,
    output_fps: float,
) -> np.ndarray:
    if speed <= 0:
        raise ValueError("speed must be greater than zero")
    start = int(np.clip(start_frame, 0, len(world) - 1))
    end = int(np.clip(end_frame, start, len(world) - 1))
    clip = np.asarray(world[start : end + 1], dtype=np.float32)
    if reverse:
        clip = clip[::-1]
    output_frames = max(1, round(len(clip) * float(output_fps) / max(float(source_fps), 1e-6) / speed))
    return resample_world(clip, output_frames)


def blend_clip_boundary(
    left: np.ndarray,
    right: np.ndarray,
    parents: np.ndarray,
    left_frames: int,
    right_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fade a clip boundary without changing the total frame count."""
    left = np.asarray(left, dtype=np.float32).copy()
    right = np.asarray(right, dtype=np.float32).copy()
    left_frames = int(np.clip(left_frames, 0, max(0, len(left) - 1)))
    right_frames = int(np.clip(right_frames, 0, max(0, len(right) - 1)))
    count = left_frames + right_frames
    if count < 2:
        return left, right

    start_world = left[-left_frames] if left_frames else left[-1]
    end_world = right[right_frames - 1] if right_frames else right[0]
    start_local = local_from_world(start_world, parents).astype(np.float64)
    end_local = local_from_world(end_world, parents).astype(np.float64)
    blended = np.repeat(start_local[None], count, axis=0)
    weights = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, count))
    for frame, weight in enumerate(weights):
        blended[frame, :, :3, 3] = (1.0 - weight) * start_local[:, :3, 3] + weight * end_local[:, :3, 3]
        for joint in range(blended.shape[1]):
            rotations = Rotation.from_matrix(np.stack([start_local[joint, :3, :3], end_local[joint, :3, :3]]))
            blended[frame, joint, :3, :3] = Slerp([0.0, 1.0], rotations)([float(weight)]).as_matrix()[0]
    transition = world_from_local(blended, parents)
    if left_frames:
        left[-left_frames:] = transition[:left_frames]
    if right_frames:
        right[:right_frames] = transition[left_frames:]
    return left, right


def apply_ordered_cuts(world: np.ndarray, cuts: list[dict]) -> np.ndarray:
    """Delete half-open frame ranges (start included, end excluded)."""
    result = np.asarray(world, dtype=np.float32)
    for cut in cuts:
        if len(result) <= 1:
            break
        start = int(np.clip(cut.get("start_frame", 0), 0, len(result) - 1))
        end = int(np.clip(cut.get("end_frame", start), start, len(result)))
        if end <= start:
            continue
        keep = np.ones(len(result), dtype=bool)
        keep[start:end] = False
        result = result[keep]
        if not len(result):
            raise ValueError("cannot delete every frame from a composition")
    return result
