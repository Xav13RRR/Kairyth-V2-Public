from pathlib import Path

import numpy as np

from motionbuilder_app import additive_store
from motionbuilder_app.additive import build_additive_world
from motionbuilder_app.additive_store import AdditiveCompositePreviewStore
from motionbuilder_app.motion_math import local_from_world, world_from_local


def _two_joint_motion(frames: int) -> tuple[np.ndarray, np.ndarray]:
    parents = np.asarray([-1, 0], dtype=np.int64)
    local = np.repeat(np.eye(4, dtype=np.float32)[None, None], frames, axis=0)
    local = np.repeat(local, 2, axis=1)
    local[:, 1, 1, 3] = 1.0
    return world_from_local(local, parents), parents


def test_additive_duration_follows_base_speed_and_weight_zero_is_base() -> None:
    base, parents = _two_joint_motion(4)
    base[:, 0, 0, 3] = np.arange(4, dtype=np.float32)
    layer, _ = _two_joint_motion(2)
    result = build_additive_world(
        base,
        layer,
        parents,
        base_fps=30,
        additive_fps=30,
        target_fps=30,
        base_speed=0.5,
        additive_speed=1.5,
        weight=0.0,
    )
    assert result.shape == (8, 2, 4, 4)
    np.testing.assert_allclose(result[0], base[0], atol=1e-6)
    np.testing.assert_allclose(result[-1], base[-1], atol=1e-6)


def test_additive_layer_loops_and_preserves_base_horizontal_root_motion() -> None:
    base, parents = _two_joint_motion(4)
    base[:, 0, 0, 3] = np.arange(4, dtype=np.float32)
    layer_local = local_from_world(_two_joint_motion(2)[0], parents)
    angle = np.pi / 2
    layer_local[1, 1, :3, :3] = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    layer = world_from_local(layer_local, parents)
    result = build_additive_world(
        base,
        layer,
        parents,
        base_fps=30,
        additive_fps=30,
        target_fps=30,
        base_speed=1.0,
        additive_speed=1.0,
        weight=1.0,
        reference_type="first_frame",
        end_behavior="loop",
    )
    output_local = local_from_world(result, parents)
    np.testing.assert_allclose(result[:, 0, 0, 3], base[:, 0, 0, 3], atol=1e-6)
    np.testing.assert_allclose(output_local[0, 1, :3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(output_local[1, 1, :3, :3], layer_local[1, 1, :3, :3], atol=1e-5)
    np.testing.assert_allclose(output_local[2, 1, :3, :3], np.eye(3), atol=1e-5)


def test_additive_preview_store_commits_as_visible_new_animation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(additive_store, "ADDITIVE_PREVIEWS_ROOT", tmp_path / "additive_previews")
    base, parents = _two_joint_motion(4)
    layer, _ = _two_joint_motion(2)
    names = np.asarray(["Hips", "Spine1"], dtype="U32")
    paths = {"base": tmp_path / "base.npz", "layer": tmp_path / "layer.npz"}
    for key, world in (("base", base), ("layer", layer)):
        np.savez_compressed(
            paths[key],
            world_motion=world,
            bind_matrices=base[0],
            parents=parents,
            names=names,
            fps=np.asarray([30], dtype=np.float32),
            num_frames=np.asarray([len(world)], dtype=np.int64),
        )

    committed: dict = {}

    class Characters:
        def get(self, character_id):
            return {"id": character_id}

    class Animations:
        characters = Characters()

        def get(self, animation_id):
            return {"id": animation_id, "character_id": "character"}

        def effective_world(self, animation_id):
            data = np.load(paths[animation_id], allow_pickle=False)
            return data, np.asarray(data["world_motion"], dtype=np.float32)

        def create_from_payload(self, character_id, name, payload, **kwargs):
            committed.update(character_id=character_id, name=name, payload=payload, kwargs=kwargs)
            return {"id": "new-animation", "character_id": character_id, "name": name}

    store = AdditiveCompositePreviewStore(Animations())
    preview = store.create(
        "character",
        "base",
        "layer",
        base_speed=2.0,
        additive_speed=0.5,
        reference_type="ref_pose",
        name="Walk + Breath",
    )
    assert preview["frames"] == 2
    assert store.bundle_path(preview["id"]).is_file()
    animation = store.commit(preview["id"], "Walk + Breath Saved")
    assert animation["id"] == "new-animation"
    assert committed["kwargs"] == {"motion_kind": "additive_composite", "hidden": False}
    assert committed["payload"]["world_motion"].shape[0] == 2
    assert not (additive_store.ADDITIVE_PREVIEWS_ROOT / preview["id"]).exists()
