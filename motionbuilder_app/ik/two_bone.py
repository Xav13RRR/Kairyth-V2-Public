from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TwoBoneSolution:
    start: np.ndarray
    middle: np.ndarray
    end: np.ndarray
    target: np.ndarray
    clamped_distance: float
    reached: bool


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length > 1e-10:
        return vector / length
    fallback_length = float(np.linalg.norm(fallback))
    if fallback_length <= 1e-10:
        raise ValueError("zero-length direction")
    return fallback / fallback_length


def solve_two_bone_positions(
    start: np.ndarray,
    middle: np.ndarray,
    end: np.ndarray,
    target: np.ndarray,
    pole: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> TwoBoneSolution:
    """Solve the geometric three-point form of a Two-Bone IK chain.

    Bone lengths are preserved. Unreachable targets are clamped just inside the
    valid reach interval so callers can safely derive joint rotations.
    """

    a = np.asarray(start, dtype=np.float64).reshape(3)
    b = np.asarray(middle, dtype=np.float64).reshape(3)
    c = np.asarray(end, dtype=np.float64).reshape(3)
    requested = np.asarray(target, dtype=np.float64).reshape(3)
    pole_point = np.asarray(pole, dtype=np.float64).reshape(3)

    upper = float(np.linalg.norm(b - a))
    lower = float(np.linalg.norm(c - b))
    if min(upper, lower) <= epsilon:
        raise ValueError("Two-Bone IK requires non-zero bone lengths")

    requested_delta = requested - a
    requested_distance = float(np.linalg.norm(requested_delta))
    fallback_direction = _unit(c - a, np.asarray([0.0, -1.0, 0.0], dtype=np.float64))
    direction = _unit(requested_delta, fallback_direction)
    min_reach = abs(upper - lower) + epsilon
    max_reach = max(upper + lower - epsilon, min_reach)
    distance = float(np.clip(requested_distance, min_reach, max_reach))
    solved_end = a + direction * distance

    pole_delta = pole_point - a
    bend = pole_delta - direction * float(np.dot(pole_delta, direction))
    if float(np.linalg.norm(bend)) <= epsilon:
        current_bend = b - a
        bend = current_bend - direction * float(np.dot(current_bend, direction))
    if float(np.linalg.norm(bend)) <= epsilon:
        fallback_pole = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        bend = fallback_pole - direction * float(np.dot(fallback_pole, direction))
    bend_direction = _unit(bend, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))

    along = (upper * upper - lower * lower + distance * distance) / (2.0 * distance)
    height_sq = max(upper * upper - along * along, 0.0)
    solved_middle = a + direction * along + bend_direction * np.sqrt(height_sq)
    reached = min_reach <= requested_distance <= max_reach
    return TwoBoneSolution(
        start=a.astype(np.float32),
        middle=solved_middle.astype(np.float32),
        end=solved_end.astype(np.float32),
        target=requested.astype(np.float32),
        clamped_distance=distance,
        reached=reached,
    )
