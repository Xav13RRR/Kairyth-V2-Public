# SPDX-License-Identifier: Apache-2.0
# Original release capability policy; not an authorization/licensing detector.
"""Explicit artifact profile. Research environment variables cannot enable it."""
from __future__ import annotations
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = {"hybrik": "hybrik_image_inference", "gvhmr": "gvhmr_video_inference"}
EXPERIMENTAL_MESSAGE = "实验性，后续会更新与HybrIK以及GVHMR的兼容包"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError("duplicate profile field")
        result[key] = value
    return result


def capabilities(project: Path | None = None) -> dict:
    """Never infer a profile from .git, resource paths, env, or user rights."""
    marker = Path(project or PROJECT_ROOT) / "release-profile.json"
    profile = "unconfigured"
    try:
        if marker.stat().st_size > 4096: raise ValueError("oversized profile")
        data = json.loads(marker.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if (not isinstance(data, dict) or set(data) != {"schema", "profile"}
                or data["schema"] != "kairyth-release/v1"
                or data["profile"] not in {"local-research", "public-workbench"}):
            raise ValueError("invalid profile")
        profile = data["profile"]
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        pass
    enabled = profile == "local-research"
    valid = profile != "unconfigured"
    state = "enabled" if enabled else "disabled_experimental" if valid else "disabled_configuration"
    reason = "local_research" if enabled else "public_release_experimental" if valid else "release_profile_invalid"
    message_key = "research.local_research" if enabled else "research.compatibility_pending" if valid else "research.profile_unavailable"
    feature = dict(enabled=enabled, state=state, reason=reason, experimental=profile == "public-workbench", message_key=message_key)
    return dict(schema="kairyth-capabilities/v1", release_profile=profile, profile_valid=valid,
                features={key: dict(feature) for key in FEATURES.values()})


class InferenceUnavailableError(RuntimeError):
    def __init__(self, backend: str, caps: dict):
        self.capabilities = caps
        feature = caps["features"][FEATURES[backend]]
        self.detail = dict(code="release-unavailable", backend=backend, release_profile=caps["release_profile"],
            capability_state=feature["state"], reason=feature["reason"], message_key=feature["message_key"],
            message=EXPERIMENTAL_MESSAGE if feature["experimental"] else "Research inference release profile is unavailable")
        super().__init__(self.detail["message"])


def require_inference(backend: str) -> None:
    caps = capabilities()
    if not caps["features"][FEATURES[backend]]["enabled"]:
        raise InferenceUnavailableError(backend, caps)


def unavailable_backend_status(backend: str) -> dict | None:
    caps = capabilities()
    feature = caps["features"][FEATURES[backend]]
    if feature["enabled"]: return None
    error = InferenceUnavailableError(backend, caps)
    return dict(backend=backend, state="release-unavailable", ready=False, available=False, installed=False,
        code_installed=False, checkpoint_installed=False, smpl_model_installed=False, missing=[],
        release_profile=caps["release_profile"], capability_state=feature["state"], experimental=feature["experimental"],
        reason=feature["reason"], message_key=feature["message_key"], message=error.detail["message"],
        physical_check_performed=False, runtime_verified=False, rights_verified=False)
