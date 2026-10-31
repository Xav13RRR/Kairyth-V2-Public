from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from motionbuilder_app.director_store import DirectorProjectStore
from motionbuilder_app.server import app


def _camera(camera_id: str = "camera-main") -> dict:
    return {
        "id": camera_id,
        "number": "CAM 01",
        "name": "Main Camera",
        "transform": {
            "translation": [2.0, 1.5, 3.0],
            "quaternion": [0.0, 0.0, 0.0, 1.0],
            "uniform_scale": 1.0,
        },
        "fov": 45,
    }


def _project_payload() -> dict:
    return {
        "duration_frames": 12,
        "fps": 24,
        "aspect_preset": "16:9",
        "resolution_preset": "720p",
        "cameras": [_camera()],
        "camera_paths": [
            {
                "id": "path-main",
                "name": "Main Rail",
                "kind": "catmull_rom",
                "points": [
                    {"id": "point-a", "position": [0, 1, 3]},
                    {"id": "point-b", "position": [2, 1, 3]},
                ],
            }
        ],
        "camera_cuts": [
            {
                "id": "shot-001",
                "number": "S001",
                "name": "Opening",
                "camera_id": "camera-main",
                "start_frame": 0,
                "end_frame": 12,
            }
        ],
        "markers": [{"id": "marker-a", "frame": 4, "name": "Beat", "color": "#e4ff4e"}],
        "time_warps": [{"id": "warp-a", "name": "Slow", "start_frame": 5, "end_frame": 9, "rate": 0.5}],
    }


def test_director_project_crud_revision_and_snapshots(tmp_path: Path) -> None:
    store = DirectorProjectStore(tmp_path / "projects", tmp_path / "exports")
    project = store.create("Sequence A", _project_payload())

    assert project["revision"] == 1
    assert project["camera_cuts"][0]["camera_id"] == "camera-main"
    assert store.list()[0]["shot_count"] == 1

    snapshot = store.create_snapshot(project["id"], "Approved", "manual")
    changed = {**project, "name": "Sequence B"}
    updated = store.update(project["id"], changed, expected_revision=1)
    assert updated["revision"] == 2
    assert updated["name"] == "Sequence B"

    with pytest.raises(RuntimeError, match="revision conflict"):
        store.update(project["id"], changed, expected_revision=1)

    restored = store.restore_snapshot(project["id"], snapshot["id"])
    assert restored["name"] == "Sequence A"
    assert restored["revision"] == 3
    assert any(row["kind"] == "pre_restore" for row in store.list_snapshots(project["id"]))


def test_director_project_rejects_missing_camera_cut_reference(tmp_path: Path) -> None:
    store = DirectorProjectStore(tmp_path / "projects", tmp_path / "exports")
    payload = _project_payload()
    payload["camera_cuts"][0]["camera_id"] = "missing"
    with pytest.raises(ValueError, match="missing camera"):
        store.create("Broken", payload)


def test_director_segments_support_blackout_and_reject_overlap(tmp_path: Path) -> None:
    store = DirectorProjectStore(tmp_path / "projects", tmp_path / "exports")
    payload = _project_payload()
    payload["camera_cuts"] = [
        {
            "id": "shot-001",
            "number": "S001",
            "name": "Opening",
            "camera_id": "camera-main",
            "start_frame": 0,
            "end_frame": 6,
            "color": "#49e6ff",
        },
        {
            "id": "blackout-001",
            "kind": "blackout",
            "number": "BLACK 01",
            "name": "Blackout",
            "start_frame": 6,
            "end_frame": 12,
        },
    ]
    project = store.create("With Blackout", payload)

    assert project["schema_version"] == 2
    assert project["camera_cuts"][0]["kind"] == "camera"
    assert project["camera_cuts"][0]["order"] == 0
    assert project["camera_cuts"][0]["color"] == "#49e6ff"
    assert project["camera_cuts"][1]["kind"] == "blackout"
    assert project["camera_cuts"][1]["camera_id"] == ""
    assert project["camera_cuts"][1]["color"] == "#111111"
    assert store.list()[0]["shot_count"] == 1
    assert store.list()[0]["blackout_count"] == 1

    overlapping = {**project, "camera_cuts": [dict(row) for row in project["camera_cuts"]]}
    overlapping["camera_cuts"][1]["start_frame"] = 5
    with pytest.raises(ValueError, match="must not overlap"):
        store.update(project["id"], overlapping, expected_revision=project["revision"])


def test_director_skeletal_actor_persists_root_motion_toggle(tmp_path: Path) -> None:
    store = DirectorProjectStore(tmp_path / "projects", tmp_path / "exports")
    payload = _project_payload()
    payload["scene_items"] = [
        {
            "id": "actor-main",
            "name": "Walker",
            "kind": "skeletal_actor",
            "source_asset_id": "character-main",
            "source_animation_id": "animation-main",
            "source_range": {"start_frame": 0, "end_frame": 12},
            "root_motion_enabled": True,
        }
    ]
    project = store.create("Root Motion", payload)
    assert project["scene_items"][0]["root_motion_enabled"] is True

    payload["scene_items"][0].pop("root_motion_enabled")
    legacy = store.create("Legacy Root Motion", payload)
    assert legacy["scene_items"][0]["root_motion_enabled"] is False


def test_director_export_checkpoints_and_reference_package(tmp_path: Path) -> None:
    store = DirectorProjectStore(tmp_path / "projects", tmp_path / "exports")
    project = store.create("Export", _project_payload())
    job = store.create_export(project["id"], {"kind": "reference_pack", "passes": ["rgb", "depth"]})
    png = b"\x89PNG\r\n\x1a\nplaceholder"

    store.save_frame(project["id"], job["id"], "rgb", 0, png)
    store.save_frame(project["id"], job["id"], "depth", 0, png)
    missing = store.missing_frames(project["id"], job["id"], "rgb")
    assert missing["completed_frames"] == 1
    assert missing["missing_frames"][0] == 1

    store.save_artifact(project["id"], job["id"], "camera-timeline.json", json.dumps({"shots": 1}).encode())
    package = store.package_export(project["id"], job["id"])
    with zipfile.ZipFile(package) as archive:
        assert "manifest.json" in archive.namelist()
        assert "camera-timeline.json" in archive.namelist()
        assert "control_passes/depth/00000000.png" in archive.namelist()

    resumable = store.create_export(project["id"], {"kind": "video", "passes": ["rgb"]})
    cancelled = store.cancel_export(project["id"], resumable["id"])
    resumed = store.resume_export(project["id"], resumable["id"])
    assert cancelled["state"] == "cancelled"
    assert resumed["state"] == "queued"
    assert resumed["encoding_attempt"] == 1


def test_director_routes_and_frontend_contract_are_registered() -> None:
    routes = {route.path for route in app.routes}
    assert "/api/director-projects" in routes
    assert "/api/director-projects/{project_id}/exports/{job_id}/frames/{pass_name}/{frame}" in routes
    assert "/api/director-projects/{project_id}/exports/{job_id}/download" in routes

    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    page = (static_root / "index.html").read_text(encoding="utf-8")
    script = (static_root / "director_studio.js").read_text(encoding="utf-8")
    css = (static_root / "director_studio.css").read_text(encoding="utf-8")
    i18n = (static_root / "studio_i18n.js").read_text(encoding="utf-8")

    assert page.index('data-tab="edit"') < page.index('data-tab="director"')
    assert 'id="directorCameraPreview"' in page
    assert 'id="directorCameraPreviewLabel"' in page
    assert 'id="directorCharacterSelect"' in page
    assert 'id="directorRootMotionEnabled"' in page
    assert 'id="directorReferencePackBtn"' in page
    assert 'id="directorAddBlackoutBtn"' in page
    assert 'item.character_id === characterId' in script
    assert 'source_asset_id: character.id' in script
    assert 'new THREE.Matrix4().set(...row)' in script
    assert 'gizmo.getRaycaster().layers.enable(HELPER_LAYER)' in script
    assert 'CAMERA_HELPER_FAR = 1.25' in script
    assert 'rebuildProjectScene({ frameActors: true })' in script
    assert "function applyActorRootMotion" in script
    assert "function makeRootMotionVisual" in script
    assert "function makeCameraNumberSprite" in script
    assert "function updateCameraSelectionVisuals" in script
    assert "SELECTED_CAMERA_COLOR = 0x2f7cff" in script
    assert "const helperColor = new THREE.Color(color)" in script
    assert "visual.helper?.setColors?.(helperColor, helperColor, helperColor, helperColor, helperColor)" in script
    assert "helper.material.depthTest = false" in script
    assert "function fittedViewport" in script
    assert "function renderMainViewport" in script
    assert "setScissorTest(true)" in script
    assert "function previewCamera" in script
    assert "function sequenceCamera" in script
    assert "function activeDirectorSegment" in script
    assert "function insertDirectorSegment" in script
    assert "function directorDropIndex" in script
    assert "function editDirectorSegment" in script
    assert "function uniqueSegmentColor" in script
    assert "shiftDirectorSegments(index, -removedLength)" in script
    assert 'kind: blackout ? "blackout" : "camera"' in script
    assert ".director-timeline-block.blackout" in css
    assert ".director-editor-modal .director-timeline-block" in css
    assert "state.playing ? sequenceCamera() : selectedPreviewCamera" in script
    assert ".director-camera-preview-label" in css
    assert ".director-outliner-row.active-camera" in css
    add_actor_source = script.split("async function addActor()", 1)[1].split("async function addAsset()", 1)[0]
    assert "frameActors()" not in add_actor_source
    assert "renderOutputBlob" in script
    assert "lookAtTarget" in script
    assert "runVideoExport" in script
    assert ".director-editor-modal button.primary" in css
    assert ".director-editor-body > .panel::before" in css
    assert '"导演仓库": "Director Warehouse"' in i18n
