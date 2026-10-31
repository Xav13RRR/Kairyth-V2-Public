"""Source-level UI safety contracts; actual browser coverage is separate."""
from html.parser import HTMLParser
from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / "motionbuilder_app" / "static"


class Elements(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = {}
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids[attrs["id"]] = attrs


def test_only_new_research_inputs_default_to_disabled():
    elements = Elements((STATIC / "index.html").read_text(encoding="utf-8")).ids
    for name in ("poseImagePickBtn", "poseImage", "runPoseInferenceBtn", "motionVideoButton",
                 "motionVideo", "motionSubmitBtn", "motionStartInferenceBtn"):
        assert "disabled" in elements[name], name
    for name in ("poseImportExistingBtn", "poseExistingFile", "poseFromMotionBtn", "nativePlayBtn"):
        assert "disabled" not in elements[name], name
    for name in ("poseReleaseNotice", "motionReleaseNotice"):
        assert elements[name]["role"] == "status"
        assert elements[name]["aria-live"] == "polite"


def test_all_new_inference_handlers_guard_before_creating_uploads():
    script = (STATIC / "studio.js").read_text(encoding="utf-8")
    assert 'function openMotionVideoPreview(file){if(!guardResearchInference("gvhmr"' in script
    assert 'function showPoseImageImport(file){if(!guardResearchInference("hybrik"' in script
    video_submit = script.split('$("motionForm").onsubmit=', 1)[1].split("let poseImageUrl", 1)[0]
    assert video_submit.index('guardResearchInference("gvhmr"') < video_submit.index('api("/api/motions"')
    assert "finally{" in video_submit and "renderResearchAvailability()" in video_submit.split("finally{", 1)[1]
    pose_submit = script.split('$("runPoseInferenceBtn").onclick=', 1)[1].split('if($("poseImportExistingBtn"))', 1)[0]
    assert pose_submit.index('guardResearchInference("hybrik"') < pose_submit.index("new FormData()")
    gate = script.split("function researchFeatureAllowed(", 1)[1].split("function researchFeaturePresentation", 1)[0]
    assert 'profile_valid===true' in gate and 'feature?.enabled===true' in gate and 'feature.state==="enabled"' in gate
    assert 'healthSnapshot?.service_ready' in gate


def test_intentional_release_state_has_readable_experimental_notices():
    script = (STATIC / "studio.js").read_text(encoding="utf-8")
    translations = (STATIC / "studio_i18n.js").read_text(encoding="utf-8")
    css = (STATIC / "studio_v5.css").read_text(encoding="utf-8")
    assert '"disabled_experimental"' in script and '"disabled_configuration"' in script
    assert "实验性，后续会更新与HybrIK以及GVHMR的兼容包" in script
    assert "Compatibility packages for HybrIK and GVHMR are planned for a future update." in translations
    rule = css.split("#motionVideoButton.research-release-unavailable:disabled", 1)[1].split("}", 1)[0]
    assert "background: var(--accent)" in rule and "color: var(--bg-dark)" in rule and "opacity: 1" in rule
    assert '"motionbuilder:languagechange",renderResearchAvailability' in script
