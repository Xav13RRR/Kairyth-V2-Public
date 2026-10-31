from __future__ import annotations

import math

import pytest

from motionbuilder_app.kimodo_authoring import normalize_constraints, normalize_preview_scene


def _rotations(frame_count: int, joint_count: int = 30) -> list[list[list[float]]]:
    return [[[0.0, 0.0, 0.0] for _ in range(joint_count)] for _ in range(frame_count)]


def test_original_root_and_fullbody_constraints_are_supported() -> None:
    constraints = normalize_constraints(
        [
            {
                "type": "root2d",
                "frame_indices": [0, 1, 2],
                "smooth_root_2d": [[0, 0], [0.1, 0], [0.2, 0.1]],
                "global_root_heading": [[1, 0], [1, 0], [0, 1]],
            },
            {
                "type": "fullbody",
                "frame_indices": [2],
                "root_positions": [[0.2, 1.0, 0.1]],
                "local_joints_rot": _rotations(1),
                "smooth_root_2d": [[0.2, 0.1]],
            },
        ],
        total_frames=4,
    )
    assert constraints[0]["type"] == "root2d"
    assert constraints[1]["type"] == "fullbody"
    assert len(constraints[1]["local_joints_rot"][0]) == 30


def test_new_end_effector_constraint_types_are_not_accepted() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_constraints(
            [
                {
                    "type": "end-effector",
                    "frame_indices": [0],
                    "root_positions": [[0, 1, 0]],
                    "local_joints_rot": _rotations(1),
                    "joint_names": ["LeftHand"],
                }
            ]
        )


def test_constraint_validation_rejects_out_of_range_frames() -> None:
    with pytest.raises(ValueError, match="outside"):
        normalize_constraints(
            [{"type": "root2d", "frame_indices": [4], "smooth_root_2d": [[0, 0]]}],
            total_frames=4,
        )


def test_preview_scene_uses_uniform_scale_and_normalized_quaternion() -> None:
    scene = normalize_preview_scene(
        {
            "scene_objects": [
                {
                    "id": "terrain-a",
                    "name": "Slope",
                    "asset_kind": "terrain",
                    "source_asset_id": "terrain-source",
                    "transform": {
                        "translation": [1, 2, 3],
                        "quaternion": [0, 2, 0, 2],
                        "scale": [2, 9, 4],
                    },
                }
            ]
        }
    )
    transform = scene["scene_objects"][0]["transform"]
    assert transform["uniform_scale"] == 2
    assert math.isclose(sum(value * value for value in transform["quaternion"]), 1.0)
