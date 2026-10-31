from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import motionbuilder_app.ik_constraints as ik_constraints
import motionbuilder_app.library_stores as library_stores
from motionbuilder_app.ik_constraints import apply_ik_constraints
from motionbuilder_app.library_stores import CompositionStore
from motionbuilder_app.server import app


class _Characters:
    @staticmethod
    def get(character_id: str) -> dict:
        return {"id": character_id}


def _world_frame() -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = ["Hips", "LeftLeg", "LeftShin", "LeftFoot", "RightLeg", "RightShin", "RightFoot"]
    parents = np.asarray([-1, 0, 1, 2, 0, 4, 5], dtype=np.int64)
    positions = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [-0.2, 0.9, 0.0],
            [-0.2, 0.5, 0.05],
            [-0.2, 0.1, 0.1],
            [0.2, 0.9, 0.0],
            [0.2, 0.5, 0.05],
            [0.2, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    frame = np.repeat(np.eye(4, dtype=np.float32)[None], len(names), axis=0)
    frame[:, :3, 3] = positions
    return frame, parents, names


def test_scene_objects_and_ik_constraints_persist(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(library_stores, "COMPOSITIONS_ROOT", tmp_path / "compositions")
    store = CompositionStore(SimpleNamespace(characters=_Characters()))
    row = store.create("character-a", "Scene")
    row = store.update_scene_objects(
        row["id"],
        [
            {
                "id": "crate",
                "name": "Crate",
                "asset_kind": "static_mesh",
                "source_asset_id": "asset-a",
                "parent_type": "ik_lock",
                "parent_id": "left-lock",
                "transform": {"translation": [1, 2, 3], "quaternion": [0, 0, 0, 1], "scale": 2},
                "visibility": {
                    "default": True,
                    "ranges": [{"start_frame": 8, "end_frame": 16, "visible": False}],
                },
            }
        ],
    )
    assert row["scene_objects"][0]["transform"]["scale"] == [2.0, 2.0, 2.0]
    assert row["scene_objects"][0]["visibility"]["ranges"][0] == {
        "start_frame": 8,
        "end_frame": 16,
        "visible": False,
    }
    row = store.update_ik_constraints(
        row["id"],
        [
            {
                "id": "left-lock",
                "name": "Left Foot Lock",
                "solver": "jacobian_dls",
                "joint_name": "LeftFoot",
                "start_frame": 5,
                "end_frame": 20,
                "blend_in_frames": 4,
                "blend_out_frames": 7,
                "target_position": [-0.2, 0.1, 0.4],
                "keys": [
                    {"frame": 5, "position": [-0.2, 0.1, 0.2]},
                    {"frame": 20, "position": [-0.2, 0.1, 0.4]},
                ],
            }
        ],
    )
    saved = store.get(row["id"])
    assert saved["schema_version"] == 2
    assert saved["ik_constraints"][0]["solver"] == "jacobian_dls"
    assert [key["frame"] for key in saved["ik_constraints"][0]["keys"]] == [5, 20]


def test_two_bone_constraint_moves_effector_and_blends() -> None:
    frame, parents, names = _world_frame()
    motion = np.repeat(frame[None], 7, axis=0)
    target = np.asarray([-0.2, 0.12, 0.35], dtype=np.float32)
    result = apply_ik_constraints(
        motion,
        parents,
        names,
        [
            {
                "solver": "two_bone",
                "joint_name": "LeftFoot",
                "start_frame": 3,
                "end_frame": 3,
                "blend_in_frames": 2,
                "blend_out_frames": 2,
                "target_position": target.tolist(),
            }
        ],
    )
    original_error = np.linalg.norm(motion[3, 3, :3, 3] - target)
    solved_error = np.linalg.norm(result[3, 3, :3, 3] - target)
    assert solved_error < original_error
    assert np.allclose(result[0], motion[0])
    assert not np.allclose(result[2], motion[2])


def test_full_body_blend_in_preserves_rigid_bone_transforms() -> None:
    joint_count = 8
    names = ["Hips", *[f"Joint{index}" for index in range(1, joint_count - 1)], "RightHand"]
    parents = np.arange(joint_count, dtype=np.int64) - 1
    parents[0] = -1
    frame = np.repeat(np.eye(4, dtype=np.float64)[None], joint_count, axis=0)
    for joint in range(joint_count):
        frame[joint, :3, 3] = [0.03 * joint, 1.0 + 0.1 * joint, 0.02 * np.sin(joint)]
        angle = 0.01 * joint
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        tiny_input_drift = np.diag(
            [1.0 + 2e-6 * (joint + 1), 1.0 - 1e-6 * (joint + 1), 1.0 + 1.5e-6 * (joint + 1)]
        )
        frame[joint, :3, :3] = rotation @ tiny_input_drift
    motion = np.repeat(frame[None], 8, axis=0).astype(np.float32)

    result = apply_ik_constraints(
        motion,
        parents,
        names,
        [
            {
                "solver": "pbik",
                "joint_name": "RightHand",
                "start_frame": 5,
                "end_frame": 5,
                "blend_in_frames": 5,
                "target_position": [2.0, 2.5, 1.0],
                "iterations": 24,
                "root_behavior": "pre_pull",
            }
        ],
    )

    first_active_frame = 1
    assert np.isfinite(result).all()
    for joint, parent in enumerate(parents):
        if parent < 0:
            continue
        original_length = np.linalg.norm(
            motion[first_active_frame, joint, :3, 3] - motion[first_active_frame, parent, :3, 3]
        )
        solved_length = np.linalg.norm(
            result[first_active_frame, joint, :3, 3] - result[first_active_frame, parent, :3, 3]
        )
        assert np.isclose(solved_length, original_length, rtol=2e-4, atol=1e-6)
    rotations = result[first_active_frame, :, :3, :3]
    gram = rotations.transpose(0, 2, 1) @ rotations
    assert np.max(np.linalg.norm(gram - np.eye(3), axis=(1, 2))) < 1e-3

def test_final_scene_routes_are_registered() -> None:
    paths = app.openapi()["paths"]
    assert "put" in paths["/api/compositions/{composition_id}/scene-objects"]
    assert "put" in paths["/api/compositions/{composition_id}/ik-constraints"]


def test_mixed_full_body_solvers_are_executed_independently(monkeypatch) -> None:
    frame, parents, names = _world_frame()
    calls: list[str] = []

    def fake_solver(world, _parents, _root, _targets, **kwargs):
        calls.append(kwargs["solver"])
        return SimpleNamespace(
            world=np.asarray(world, dtype=np.float32).copy(),
            residuals=np.zeros(len(_targets), dtype=np.float32),
        )

    monkeypatch.setattr(ik_constraints, "solve_full_body_frame", fake_solver)
    ik_constraints.apply_ik_constraints(
        frame[None],
        parents,
        names,
        [
            {
                "solver": "pbik",
                "joint_name": "LeftFoot",
                "start_frame": 0,
                "end_frame": 0,
                "target_position": [-0.2, 0.1, 0.2],
            },
            {
                "solver": "fabrik",
                "joint_name": "RightFoot",
                "start_frame": 0,
                "end_frame": 0,
                "target_position": [0.2, 0.1, 0.2],
            },
        ],
    )

    assert calls == ["pbik", "fabrik"]


def test_final_scene_frontend_is_cache_busted_and_failure_isolated() -> None:
    root = Path(__file__).resolve().parents[1]
    studio = (root / "motionbuilder_app" / "static" / "studio.js").read_text(encoding="utf-8")
    index = (root / "motionbuilder_app" / "static" / "index.html").read_text(encoding="utf-8")
    final_scene = (root / "motionbuilder_app" / "static" / "final_scene_tools.js").read_text(encoding="utf-8")
    assert 'import { createFinalSceneTools }' not in studio
    assert 'await import("/app-static/final_scene_tools.js?v=6")' in studio
    assert '"/static/ik_controls.js?v=ik-selection-2"' in studio
    assert '"/static/full_body_ik.js?v=ik-selection-2"' in studio
    assert "Final scene tools failed to initialize" in studio
    assert "studio.js?v=110-release-capabilities" in index
    assert "export function uniformScaleMultiplier" in final_scene
    assert '"Uniform Scale"' in final_scene
    assert "applyUniformScaleDrag();" in final_scene
    assert "ikLockCaptureBtn" not in final_scene
    assert "readonly disabled" in final_scene
    assert "selectIKControl" in final_scene
    assert "selectedIKControlTransform" in final_scene
    assert "alignConstraintToSelectedControl" in final_scene
    assert "updateIKActionAvailability" in final_scene
    assert "finalIKKeyFrame" in final_scene
    assert "final-ik-blend-in" in final_scene
    assert "dragIKBlend" in final_scene


def test_studio_module_has_no_duplicate_top_level_functions() -> None:
    root = Path(__file__).resolve().parents[1]
    studio = (root / "motionbuilder_app" / "static" / "studio.js").read_text(encoding="utf-8")
    function_names = re.findall(
        r"^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)",
        studio,
        flags=re.MULTILINE,
    )
    duplicates = sorted(name for name, count in Counter(function_names).items() if count > 1)
    assert duplicates == []
