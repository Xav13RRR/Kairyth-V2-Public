"""Small source contracts; real rendering/pointer acceptance is documented separately."""
from pathlib import Path
import re


STATIC = Path(__file__).resolve().parents[1] / "motionbuilder_app" / "static"


def test_camera_body_children_are_helper_only_after_attachment():
    script = (STATIC / "director_studio.js").read_text(encoding="utf-8")
    make_camera = script.split("function makeCameraVisual(", 1)[1].split("function paintCameraNumberSprite", 1)[0]
    assert make_camera.index("body.add(box, lens, numberSprite)") < make_camera.index(
        "body.traverse((object) => object.layers.set(HELPER_LAYER))"
    ) < make_camera.index("camera.add(body)")
    assert "runtime.raycaster.layers.enableAll()" in script
    assert "if (!state.lockDirectorView) runtime.renderCamera.layers.enable(HELPER_LAYER)" in script


def test_terrain_dialogs_share_editor_layer_above_sticky_navigation():
    css = (STATIC / "terrain_workspace.css").read_text(encoding="utf-8")
    rule = re.search(r"#terrainCreateModal,\s*#terrainEditorModal\s*\{([^}]+)\}", css)
    assert rule and re.search(r"z-index:\s*220\s*;", rule[1])


def test_held_font_is_not_requested_and_full_card_names_stay_in_flow():
    css = (STATIC / "studio_v5.css").read_text(encoding="utf-8")
    assert "Kairyth Oswald" not in css and "Oswald-Bold-Latin.woff2" not in css
    assert 'font-family: "Kairyth Geist"' in css
    assert ".asset-tech-card:has(> .asset-meta)" in css
    rule = css.split(".asset-tech-card > .asset-meta", 1)[1].split("}", 1)[0]
    assert "position: static" in rule and "margin-top: auto" in rule


def test_health_badge_separates_core_service_from_optional_readiness():
    script = (STATIC / "studio.js").read_text(encoding="utf-8")
    health = script.split("function renderHealthBadge()", 1)[1].split("async function health()", 1)[0]
    assert "healthSnapshot.service_ready" in health
    assert "healthSnapshot.rigging?.ready===false" in health
    assert "healthSnapshot.optional_backends" in health
    assert "rights_verified" not in health
    assert 'optionalMissing?"optional":"ok"' in health
    assert '"motionbuilder:languagechange",renderHealthBadge' in script
