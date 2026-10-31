from pathlib import Path
import inspect

from fastapi.testclient import TestClient

from motionbuilder_app.server import app


def test_public_routes_exist() -> None:
    paths = {route.path for route in app.routes}
    assert "/api/characters" in paths
    assert "/api/bind/{mesh_id}/auto" in paths
    assert "/api/job/{job_id}/ik/auto" in paths
    assert "/api/characters/{character_id}/ik/auto" in paths
    assert "/api/characters/{character_id}/exports/{file_format}" in paths
    assert "/api/motions" in paths
    assert "/api/motions/{motion_id}/edits" in paths
    assert "/api/motions/{motion_id}/exports/{file_format}" in paths
    assert "/api/motions/{motion_id}/cancel" in paths
    assert "/api/kimodo-motions/{motion_id}/cancel" in paths
    assert "/api/kimodo-motions/{motion_id}" in paths
    assert "/api/kimodo-motions/{motion_id}/preview-scene" in paths
    assert "/api/poses/from-motion" in paths
    assert "/api/animations" in paths
    assert "/api/retarget-previews" in paths
    assert "/api/retarget-previews/{preview_id}/commit" in paths
    assert "/api/additive-previews" in paths
    assert "/api/additive-previews/{preview_id}/commit" in paths
    assert "/api/animations/{animation_id}/exports/{file_format}" in paths
    assert "/api/compositions/{composition_id}/exports/{file_format}" in paths


def test_studio_page() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "KAIRYTH: Generative Motion Systems" in response.text
        assert 'id="motionUploadPreview"' in response.text
        assert '<option value="locked">锁定根骨骼</option>' in response.text
        assert 'id="timelineTrack"' in response.text
        assert 'id="gizmoTranslate"' in response.text
        assert 'id="blendInFrames"' in response.text
        assert 'id="blendOutFrames"' in response.text
        assert 'id="sourceVideo"' not in response.text
        assert 'id="retargetCanvas"' in response.text
        assert 'id="commitRetargetBtn"' in response.text
        assert 'value="text_to_motion"' in response.text
        assert 'id="kimodoPreviewInfo"' in response.text
        assert 'id="kimodoSeed" type="number"' in response.text
        assert 'id="kimodoConstraintPanel"' not in response.text
        assert 'id="kimodoTextGuidance"' not in response.text
        assert 'id="kimodoTransitionFrames"' not in response.text
        assert 'id="kimodoInitialHeading"' not in response.text
        assert 'id="kimodoConstraintNew"' not in response.text
        assert 'id="kimodoIKCreateControl"' not in response.text
        assert 'id="kimodoConstraintCapture"' not in response.text
        assert 'id="kimodoWaypointPoseTools"' in response.text
        assert 'id="kimodoWaypointPoseNode"' in response.text
        assert 'id="kimodoWaypointPoseTranslate"' in response.text
        assert 'id="kimodoWaypointPoseRotate"' in response.text
        assert 'id="kimodoWaypointPoseReset"' in response.text
        assert 'id="kimodoScenePanel"' in response.text
        assert 'id="animationPreviewModal"' in response.text
        assert 'id="additiveComposeBtn"' in response.text
        assert 'id="additiveBaseSpeed"' in response.text
        assert 'id="commitAdditiveBtn"' in response.text
        assert 'id="gizmoLocal"' in response.text
        assert 'id="gizmoWorld"' in response.text


def test_kimodo_waypoint_pose_editor_is_temporary_and_scene_compatible() -> None:
    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    studio = (static_root / "studio.js").read_text(encoding="utf-8")
    authoring = (static_root / "kimodo_authoring.js").read_text(encoding="utf-8")

    assert "point.pose_override=null" in studio
    assert "poseConstraints=point.pose_override?[point.pose_override]:libraryConstraints" in studio
    assert "onWaypointPoseChange:pose=>" in studio
    assert "function beginWaypointPose" in authoring
    assert "function canonicalWaypointPose" in authoring
    assert "function buildWaypointPoseTools" in authoring
    assert 'data.activeTool = "waypoint-pose"' in authoring
    assert "options.onWaypointPoseChange?.(null)" in authoring
    assert 'data.activeTool = "scene"' in authoring


def test_alignment_confirmation_follows_current_frontend_binding_steps() -> None:
    script = (Path(__file__).parents[1] / "motionbuilder_app" / "meshpreps" / "static" / "app.js").read_text(encoding="utf-8")
    assert "await startSkeletonBinding();" in script
    assert "setPanelVisible(bindStepSection, true);" in script
    assert "apiUrl(`/api/bind/${meshId}`)" in script
    assert "apiUrl(`/api/bind/${meshId}/auto`)" not in script
    assert 'st.phase === "skeleton" && st.state === "done"' in script
    assert "await acceptSkeletonAutomatically(progress);" in script
    assert "apiUrl(`/api/job/${jobId}/skeleton/accept`)" in script
    assert "showSomaEditPreview(data.preview);" in script
    assert "apiUrl(`/api/job/${jobId}/confirm`)" in script
    page = (Path(__file__).parents[1] / "motionbuilder_app" / "meshpreps" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="autoRiggingProgressBar"' in page

def test_two_bone_ik_frontend_and_asset_handoff_contract() -> None:
    root = Path(__file__).parents[1] / "motionbuilder_app"
    rig_page = (root / "meshpreps" / "static" / "index.html").read_text(encoding="utf-8")
    rig_script = (root / "meshpreps" / "static" / "app.js").read_text(encoding="utf-8")
    controls = (root / "meshpreps" / "static" / "ik_controls.js").read_text(encoding="utf-8")
    studio = (root / "static" / "studio.js").read_text(encoding="utf-8")

    assert 'id="autoIkBtn"' in rig_page
    assert 'id="ikSkeletonMode"' in rig_page
    assert 'id="ikControlMode"' in rig_page
    assert 'id="resetIkPreview"' in rig_page
    assert 'id="ikFullBodySolver"' in rig_page
    assert 'id="ikModeRotate"' in rig_page
    assert 'id="ikSpaceLocal"' in rig_page
    assert "new TwoBoneIKController" in rig_script
    assert "new FullBodyIKController" in rig_script
    assert "`/api/job/${jobId}/ik/auto`" in rig_script
    assert "`/api/characters/${ikSourceCharacterId}/ik/auto`" in rig_script
    assert 'profile.solver !== "TwoBoneIK"' in controls
    assert 'data-asset-action="auto-ik"' in studio
    assert "/rig-editor?mode=ik&character_id=" in studio
    assert 'api("/api/additive-previews"' in studio
    assert 'base_speed:+$("additiveBaseSpeed").value' in studio
    assert 'additive_speed:+$("additiveLayerSpeed").value' in studio


def test_video_generation_has_no_character_or_retarget_fields() -> None:
    route = next(route for route in app.routes if route.path == "/api/motions" and "POST" in route.methods)
    parameters = inspect.signature(route.endpoint).parameters
    assert "character_id" not in parameters
    assert "root_mode" in parameters


def test_video_upload_success_has_no_task_created_notice() -> None:
    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    studio = (static_root / "studio.js").read_text(encoding="utf-8")
    i18n = (static_root / "studio_i18n.js").read_text(encoding="utf-8")
    page = (static_root / "index.html").read_text(encoding="utf-8")

    assert "任务已创建" not in studio
    assert "Task created:" not in i18n
    assert 'textContent=renameError?`名称保存失败：${renameError}`:""' in studio
    assert "closeMotionVideoPreview({keepStatus:Boolean(renameError)})" in studio
    assert "studio_i18n.js?v=20-release-capabilities" in page
    assert "studio.js?v=110-release-capabilities" in page


def test_video_upload_button_matches_rigging_primary_style_when_enabled() -> None:
    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    page = (static_root / "index.html").read_text(encoding="utf-8")
    css = (static_root / "studio_v5.css").read_text(encoding="utf-8")

    assert 'id="motionVideoButton" class="primary research-release-unavailable"' in page
    assert "studio_v5.css?v=116-release-capabilities" in page
    assert "#motionVideoButton:disabled" in css
    english_rule = css.rsplit('html[data-language="en"] #motionVideoButton', 1)[1].split("}", 1)[0]
    assert "font-size: 10.5px !important;" in english_rule
    assert "white-space: nowrap !important;" in english_rule


def test_english_main_tabs_share_fixed_visual_metrics() -> None:
    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    css = (static_root / "studio_v5.css").read_text(encoding="utf-8")

    english_tabs_rule = css.rsplit(
        'html[data-language="en"] .tabs .tab,', 1
    )[1].split("}", 1)[0]
    assert "display: inline-flex;" in english_tabs_rule
    assert "align-items: center;" in english_tabs_rule
    assert "justify-content: center;" in english_tabs_rule
    assert "font-size: 9px !important;" in english_tabs_rule
    assert "overflow: hidden;" in english_tabs_rule
    assert "white-space: nowrap;" in english_tabs_rule
    assert 'html[data-language="en"] .tabs .tab:nth-child(8) { font-size:' not in css


def test_animation_filter_and_overwrite_labels_fit_fixed_controls() -> None:
    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    page = (static_root / "index.html").read_text(encoding="utf-8")
    i18n = (static_root / "studio_i18n.js").read_text(encoding="utf-8")

    assert 'data-animation-kind-filter="virtual">VIRTUALROOT</button>' in page
    assert "virtualroot动作" not in page
    assert '"覆盖原动作": "Overwrite Original"' in i18n
    assert '"覆盖原动作": "Overwrite Original Motion"' not in i18n
    assert "studio_i18n.js?v=20-release-capabilities" in page


def test_repository_card_kind_badges_translate_in_all_libraries() -> None:
    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    i18n = (static_root / "studio_i18n.js").read_text(encoding="utf-8")
    page = (static_root / "index.html").read_text(encoding="utf-8")

    for source, translated in (
        (".静态", ".Static"),
        (".角色", ".Character"),
        (".姿势", ".Pose"),
        (".普通", ".Standard"),
    ):
        assert f'"{source}": "{translated}"' in i18n
    assert "studio_i18n.js?v=20-release-capabilities" in page


def test_pose_from_motion_ui_and_bilingual_contract() -> None:
    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    page = (static_root / "index.html").read_text(encoding="utf-8")
    studio = (static_root / "studio.js").read_text(encoding="utf-8")
    i18n = (static_root / "studio_i18n.js").read_text(encoding="utf-8")

    assert page.index('id="poseImportExistingBtn"') < page.index('id="poseFromMotionBtn"')
    assert 'id="poseFromMotionModal"' in page
    assert 'id="poseFromMotionCanvas"' in page
    assert 'id="poseFromMotionTimeline"' in page
    assert 'id="poseFromMotionName"' in page
    assert 'id="exportPoseFromMotionBtn"' in page
    assert 'id="closePoseFromMotionModal"' not in page
    assert 'id="closePoseImportModal"' not in page
    assert page.index('id="renamePoseImportBtn"') < page.index('id="cancelPoseImportBtn"')
    assert 'kind==="video"?"/api/motions":"/api/kimodo-motions"' in studio
    assert 'api("/api/poses/from-motion"' in studio
    assert 'frame:player.frame' in studio
    assert '"poseFromMotionKind"' in i18n
    assert '"从动作提取姿势": "Extract Pose from Motion"' in i18n
    assert '"导出当前帧": "Export Current Frame"' in i18n
    assert "拖动时间轴选择要写入姿势库的帧。" not in studio
    assert "拖动时间轴选择要写入姿势库的帧。" not in i18n
    assert "已写入姿势库：" not in studio
    assert "Saved to the Pose Library:" not in i18n


def test_fixed_dropdown_translation_and_timeline_marker_visual_contracts() -> None:
    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    i18n = (static_root / "studio_i18n.js").read_text(encoding="utf-8")
    managed_selects = i18n.split("const MANAGED_OPTION_SELECTS", 1)[1].split("]);", 1)[0]
    assert '"orientationScope"' in managed_selects
    assert '"帧区间": "Frame Range"' in i18n

    css = (static_root / "studio_v5.css").read_text(encoding="utf-8")
    final_marker_rule = css.rsplit("#tab-edit .edit-marker::before", 1)[1].split("}", 1)[0]
    assert "content: none !important;" in final_marker_rule
    assert "display: none !important;" in final_marker_rule

    final_selected_rule = css.rsplit(
        "#tab-edit #attachmentList .attachment-row.selected,", 1
    )[1].split("}", 1)[0]
    assert "#tab-edit #segmentCurveList .edit-row.selected.segment-curve-selected" in final_selected_rule
    assert "#tab-edit #rootCurveSegmentList .edit-row.selected.root-curve-selected" in final_selected_rule
    assert "color: var(--aero-slate) !important;" in final_selected_rule
    assert "background: var(--aero-signal) !important;" in final_selected_rule

    selected_action_rule = css.rsplit(
        "#tab-edit #attachmentList .attachment-row.selected .actions button {", 1
    )[1].split("}", 1)[0]
    assert "color: var(--aero-slate) !important;" in selected_action_rule
    assert "background: transparent !important;" in selected_action_rule


def test_final_ik_and_model_library_preview_contracts() -> None:
    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    page = (static_root / "index.html").read_text(encoding="utf-8")
    studio = (static_root / "studio.js").read_text(encoding="utf-8")

    for element_id in (
        "modelPreviewTools",
        "modelPreviewFK",
        "modelPreviewIK",
        "modelPreviewTranslate",
        "modelPreviewRotate",
        "modelPreviewReset",
    ):
        assert f'id="{element_id}"' in page
    assert "const player={canvas,renderer,scene,camera,controls,gizmo" in studio
    assert "canvas=player?.canvas||player?.renderer?.domElement" in studio
    assert "prepareModelPreviewInteraction" in studio
    assert "clearModelPreviewInteraction" in studio


def test_final_timeline_dock_has_persistent_top_edge_resize() -> None:
    static_root = Path(__file__).parents[1] / "motionbuilder_app" / "static"
    page = (static_root / "index.html").read_text(encoding="utf-8")
    studio = (static_root / "studio.js").read_text(encoding="utf-8")
    css = (static_root / "studio_v5.css").read_text(encoding="utf-8")

    assert 'id="timelineDockResizeHandle"' in page
    assert 'class="timeline-dock-scroll"' in page
    assert "initFinalTimelineResize" in studio
    assert "FINAL_TIMELINE_HEIGHT_KEY" in studio
    assert 'addEventListener("pointerdown"' in studio
    assert "timeline-dock-resize-handle" in css
    assert "cursor: ns-resize" in css
