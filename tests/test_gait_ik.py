from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from motionbuilder_app import gait_store
from motionbuilder_app.gait_ik import GaitIKSettings, detect_foot_contacts, process_gait_ik
from motionbuilder_app.gait_store import GaitIKPreviewStore
from motionbuilder_app.library_stores import _apply_whole_animation_transforms
from motionbuilder_app.meshpreps.bind_math import build_bind_matrices
from motionbuilder_app.motion_math import local_from_world
from motionbuilder_app.server import app
from motionbuilder_app.skeleton import SOMASkeleton30


def _moving_soma(frames: int = 24) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    skeleton = SOMASkeleton30(load=True)
    names = list(skeleton.bone_order_names)
    parents = np.asarray(skeleton.joint_parents, dtype=np.int64)
    bind = build_bind_matrices(np.asarray(skeleton.neutral_joints, dtype=np.float32), parents).astype(np.float32)
    world = np.repeat(bind[None], frames, axis=0)
    world[:, :, 0, 3] += np.linspace(0.0, 0.3, frames, dtype=np.float32)[:, None]
    world[:, :, 1, 3] += (0.03 * np.sin(np.linspace(0.0, 2.0 * np.pi, frames)))[:, None]
    return world, parents, names, bind


def test_gait_ik_grounds_root_reduces_slide_and_preserves_humanoid_bones() -> None:
    world, parents, names, bind = _moving_soma()
    result = process_gait_ik(
        world,
        parents,
        names,
        bind,
        30.0,
        settings=GaitIKSettings(root_smoothing_frames=1, contact_height=0.08, contact_velocity=0.4),
    )

    np.testing.assert_allclose(result.virtual_root_motion[:, 1, 3], 0.0, atol=1e-7)
    reconstructed = np.einsum("fab,fjbc->fjac", result.virtual_root_motion, result.local_world_motion)
    np.testing.assert_allclose(reconstructed, result.preview_world_motion, atol=2e-5)
    assert result.contacts.shape == (len(world), 2)
    assert result.contact_weights.shape == (len(world), 2)
    assert np.isfinite(result.preview_world_motion).all()
    assert result.metrics["slide_after"] <= result.metrics["slide_before"] + 1e-6

    for chain in (("LeftLeg", "LeftShin", "LeftFoot"), ("RightLeg", "RightShin", "RightFoot")):
        indices = [names.index(name) for name in chain]
        for start, end in zip(indices, indices[1:]):
            expected = np.linalg.norm(bind[end, :3, 3] - bind[start, :3, 3])
            actual = np.linalg.norm(
                result.preview_world_motion[:, end, :3, 3] - result.preview_world_motion[:, start, :3, 3],
                axis=1,
            )
            np.testing.assert_allclose(actual, expected, atol=2e-4)

    source_local = local_from_world(world, parents)
    result_local = local_from_world(result.local_world_motion, parents)
    for name in ("Jaw", "LeftEye", "RightEye"):
        index = names.index(name)
        np.testing.assert_allclose(result_local[:, index], source_local[:, index], atol=2e-5)


def test_contact_detection_uses_height_and_speed_then_cleans_short_spikes() -> None:
    world = np.repeat(np.eye(4, dtype=np.float32)[None, None], 8, axis=0)
    world[:, 0, 1, 3] = [0.0, 0.0, 0.0, 0.4, 0.4, 0.0, 0.0, 0.0]
    world[:, 0, 0, 3] = [0.0, 0.0, 0.0, 0.2, 0.4, 0.4, 0.4, 0.4]
    contacts = detect_foot_contacts(
        world,
        0,
        None,
        30.0,
        height_threshold=0.05,
        velocity_threshold=0.2,
    )
    assert contacts.tolist() == [True, True, True, False, False, False, True, True]


def test_whole_animation_adjustment_applies_translation_and_rotation_in_local_space() -> None:
    parents = np.asarray([-1, 0], dtype=np.int64)
    local = np.repeat(np.eye(4, dtype=np.float32)[None, None], 2, axis=1)
    local[0, 1, 0, 3] = 1.0
    world = np.repeat(local, 3, axis=0)
    half = np.sqrt(0.5)
    adjusted = _apply_whole_animation_transforms(
        world,
        parents,
        [{"joint": 1, "translation": [0.0, 0.25, 0.0], "rotation": [0.0, 0.0, half, half]}],
    )
    adjusted_local = local_from_world(adjusted, parents)
    expected_position = np.repeat(np.asarray([[1.0, 0.25, 0.0]]), 3, axis=0)
    expected_rotation = np.repeat(np.asarray([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]]), 3, axis=0)
    np.testing.assert_allclose(adjusted_local[:, 1, :3, 3], expected_position, atol=1e-6)
    np.testing.assert_allclose(adjusted_local[:, 1, :3, :3], expected_rotation, atol=1e-5)


def test_gait_preview_commit_is_save_new_and_removes_only_preview(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gait_store, "GAIT_IK_PREVIEWS_ROOT", tmp_path / "gait_previews")
    preview_id = "preview123"
    root = gait_store.GAIT_IK_PREVIEWS_ROOT / preview_id
    root.mkdir(parents=True)
    np.savez_compressed(
        root / "preview.npz",
        world_motion=np.repeat(np.eye(4, dtype=np.float32)[None, None], 2, axis=0),
        preview_world_motion=np.repeat(np.eye(4, dtype=np.float32)[None, None], 2, axis=0),
        fps=np.asarray([30.0], dtype=np.float32),
    )
    manifest = {
        "id": preview_id,
        "name": "Walk Grounded",
        "source_animation_id": "original-animation",
        "character_id": "character",
        "bundle": "preview.npz",
        "settings": {"foot_lock_strength": 1.0},
        "metrics": {"slide_after": 0.01},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    class Animations:
        def __init__(self) -> None:
            self.calls = []

        def create_from_payload(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"id": "new-animation", "name": args[1]}

    animations = Animations()
    saved = GaitIKPreviewStore(animations).commit(preview_id)

    assert saved["id"] == "new-animation"
    assert len(animations.calls) == 1
    args, kwargs = animations.calls[0]
    assert args[:2] == ("character", "Walk Grounded")
    assert kwargs["motion_kind"] == "gait_ik"
    assert kwargs["hidden"] is False
    assert kwargs["metadata"]["source_animation_id"] == "original-animation"
    assert "preview_world_motion" not in args[2]
    assert not root.exists()


def test_gait_routes_and_ik_editor_contracts_are_exposed() -> None:
    paths = app.openapi()["paths"]
    assert "/api/gait-ik-previews" in paths
    assert "/api/gait-ik-previews/{preview_id}/scene" in paths
    assert "/api/gait-ik-previews/{preview_id}/frames" in paths
    assert "/api/gait-ik-previews/{preview_id}/commit" in paths
    assert not any("replace" in path for path in paths if "gait-ik-previews" in path)

    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    html = (static_root / "index.html").read_text(encoding="utf-8")
    studio = (static_root / "studio.js").read_text(encoding="utf-8")
    for element_id in (
        "gaitIKBtn",
        "gaitIKModal",
        "commitGaitIKBtn",
        "animationAdjustPoseIK",
        "animationAdjustRotate",
        "finalPoseIK",
        "finalIKSolver",
    ):
        assert f'id="{element_id}"' in html
    assert 'ground_rotation_strength:$("gaitIKGroundRotation").checked?1:0' in studio
    assert 'setFinalPoseMode("ik")' in studio
    assert 'setAnimationAdjustPoseMode("ik")' in studio
