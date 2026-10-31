from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from motionbuilder_app import gait_store, terrain_store
from motionbuilder_app.gait_ik import GaitIKSettings, process_gait_ik
from motionbuilder_app.gait_store import GaitIKPreviewStore
from motionbuilder_app.meshpreps.bind_math import build_bind_matrices
from motionbuilder_app.server import app
from motionbuilder_app.skeleton import SOMASkeleton30
from motionbuilder_app.terrain_store import TerrainStore
from motionbuilder_app.terrain_surface import HeightFieldSurface


def _moving_soma(frames: int = 24) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    skeleton = SOMASkeleton30(load=True)
    names = list(skeleton.bone_order_names)
    parents = np.asarray(skeleton.joint_parents, dtype=np.int64)
    bind = build_bind_matrices(np.asarray(skeleton.neutral_joints, dtype=np.float32), parents).astype(np.float32)
    world = np.repeat(bind[None], frames, axis=0)
    world[:, :, 0, 3] += np.linspace(0.0, 0.35, frames, dtype=np.float32)[:, None]
    return world, parents, names, bind


def test_heightfield_surface_bilinear_sampling_normals_and_bounds() -> None:
    heights = np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    surface = HeightFieldSurface(heights, 2.0, 2.0)
    height, normal, hit = surface.sample(0.0, 0.0)
    assert hit is True
    assert height == 1.0
    assert normal[1] > 0.5
    outside = surface.sample_many(np.asarray([[2.0, 0.0], [0.0, -2.0]], dtype=np.float32))
    assert outside.hits.tolist() == [False, False]
    assert np.isnan(outside.heights).all()


def test_heightfield_surface_applies_translation_yaw_and_uniform_scale() -> None:
    heights = np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    surface = HeightFieldSurface(
        heights,
        2.0,
        2.0,
        translation=[2.0, 3.0, -1.0],
        yaw_degrees=90.0,
        uniform_scale=2.0,
    )
    height, normal, hit = surface.sample(2.0, -1.0)
    assert hit is True
    assert height == 5.0
    assert normal[1] > 0.5
    assert surface.sample(6.1, -1.0)[2] is False


def test_terrain_store_crud_revision_and_scene(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(terrain_store, "TERRAINS_ROOT", tmp_path / "terrains")
    store = TerrainStore()
    row = store.create("Test Ground", size_x=6.0, size_z=4.0, resolution=33)
    assert row["revision"] == 1
    assert store.list()[0]["id"] == row["id"]

    heights = store.load_heights(row["id"])
    heights[:, :] = np.linspace(0.0, 0.4, 33, dtype=np.float32)[None, :]
    updated = store.update_heights(
        row["id"],
        heights,
        expected_revision=1,
        transform={"translation": [1.0, 0.5, -2.0], "yaw_degrees": 35.0, "uniform_scale": 1.5},
    )
    assert updated["revision"] == 2
    assert updated["transform"]["translation"] == [1.0, 0.5, -2.0]
    assert store.surface(row["id"]).uniform_scale == 1.5
    assert updated["max_height"] == np.float32(0.4)
    scene = store.scene(row["id"])
    assert len(scene["vertices"]) == 33 * 33
    assert len(scene["faces"]) == 32 * 32 * 2
    assert scene["heights"][0][-1] == np.float32(0.4)

    store.delete(row["id"])
    assert store.list() == []


def test_gait_solver_tracks_heightfield_and_initial_yaw() -> None:
    world, parents, names, bind = _moving_soma()
    heights = np.repeat(np.linspace(0.0, 0.8, 65, dtype=np.float32)[None, :], 65, axis=0)
    surface = HeightFieldSurface(heights, 4.0, 4.0)
    result = process_gait_ik(
        world,
        parents,
        names,
        bind,
        30.0,
        settings=GaitIKSettings(root_smoothing_frames=1, terrain_smoothing_frames=1),
        surface=surface,
        initial_translation=[0.0, 0.0, 0.0],
        initial_yaw_degrees=0.0,
    )
    assert result.version == "gait-ik-terrain-v2"
    assert np.ptp(result.root_ground_heights) > 0.02
    np.testing.assert_allclose(result.virtual_root_motion[:, 1, 3], result.root_ground_heights, atol=1e-6)
    assert result.terrain_hits.shape == (len(world), 2)
    assert result.foot_ground_normals.shape == (len(world), 2, 3)
    assert result.initial_transform["yaw_degrees"] == 0.0


def test_gait_commit_can_embed_terrain_snapshot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gait_store, "GAIT_IK_PREVIEWS_ROOT", tmp_path / "previews")
    root = gait_store.GAIT_IK_PREVIEWS_ROOT / "preview-terrain"
    root.mkdir(parents=True)
    np.savez_compressed(
        root / "preview.npz",
        world_motion=np.repeat(np.eye(4, dtype=np.float32)[None, None], 2, axis=0),
        preview_world_motion=np.repeat(np.eye(4, dtype=np.float32)[None, None], 2, axis=0),
        fps=np.asarray([30.0], dtype=np.float32),
        terrain_heights=np.zeros((3, 3), dtype=np.float32),
        terrain_size=np.asarray([2.0, 2.0], dtype=np.float32),
        terrain_source_id=np.asarray(["terrain-1"]),
    )
    manifest = {
        "id": "preview-terrain",
        "name": "Walk on Ground",
        "source_animation_id": "walk",
        "character_id": "soma",
        "bundle": "preview.npz",
        "terrain_id": "terrain-1",
        "terrain_name": "Ground",
        "terrain_revision": 3,
        "terrain_snapshot_hash": "abc",
        "initial_transform": {"translation": [0.0, 0.0, 0.0], "yaw_degrees": 0.0},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    class Animations:
        def __init__(self) -> None:
            self.call = None

        def create_from_payload(self, *args, **kwargs):
            self.call = (args, kwargs)
            return {"id": "saved", "name": args[1]}

    animations = Animations()
    saved = GaitIKPreviewStore(animations).commit("preview-terrain", include_terrain=True)
    assert saved["id"] == "saved"
    args, kwargs = animations.call
    assert "terrain_heights" in args[2]
    assert kwargs["metadata"]["terrain_attached"] is True
    assert kwargs["metadata"]["terrain_attachment"]["source_terrain_id"] == "terrain-1"


def test_terrain_api_and_frontend_contracts_are_exposed() -> None:
    paths = app.openapi()["paths"]
    assert "/api/terrains" in paths
    assert "/api/terrains/{terrain_id}/scene" in paths
    assert "/api/terrains/{terrain_id}/heightfield" in paths

    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    html = (static_root / "index.html").read_text(encoding="utf-8")
    studio = (static_root / "studio.js").read_text(encoding="utf-8")
    terrain_ui = (static_root / "terrain_workspace.js").read_text(encoding="utf-8")
    assert "terrain_workspace.js?v=1" in html
    for element_id in (
        "terrainLibraryTab",
        "terrainEditorModal",
        "terrainOverallPanel",
        "terrainOverallScaleValue",
        "gaitIKTerrainPanel",
        "gaitIKStartTranslate",
        "gaitIKStartRotate",
        "gaitIKIncludeTerrain",
        "finalTerrainToggle",
    ):
        assert element_id in terrain_ui
    assert 'data-terrain-tool="sculpt_invert"' in terrain_ui
    assert 'data-terrain-tool="overall"' in terrain_ui
    assert "tabs.insertBefore(button, rigTab || languageSwitch)" in terrain_ui
    assert "terrain-focus-active" in terrain_ui
    assert 'document.documentElement.classList.toggle("terrain-focus-active", active)' in terrain_ui
    assert "menu.getBoundingClientRect().width" in terrain_ui
    assert "renderSelectedTerrainPreview" in terrain_ui
    assert "new THREE.PerspectiveCamera(38" in terrain_ui
    assert "new THREE.Vector3(1, Math.SQRT2, 1)" in terrain_ui
    assert "geometry.computeBoundingSphere()" in terrain_ui
    assert "data-terrain-thumbnail" not in terrain_ui
    assert "REV ${terrain.revision}" not in terrain_ui
    terrain_css = (static_root / "terrain_workspace.css").read_text(encoding="utf-8")
    assert "#tab-terrain .page-title.asset-title" in terrain_css
    assert "border: 0 !important" in terrain_css
    assert "html.terrain-focus-active::after" in terrain_css
    assert "overflow: hidden !important" in terrain_css
    assert "justify-content: flex-start" in terrain_css
    assert "width: max-content" in terrain_css
    assert "TerrainWorkspace?.gaitPayload" in studio
    assert "include_terrain" in studio
    assert "syncPlayerTerrain" in studio
