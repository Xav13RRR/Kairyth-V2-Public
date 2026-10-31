"""V2 source/launcher isolation checks; never touch the original deployment."""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

from motionbuilder_app.meshpreps import env_util

ROOT = Path(__file__).resolve().parents[1]


def test_no_implicit_global_environment_fallback(tmp_path, monkeypatch):
    external = tmp_path / "old environment" / "python.exe"
    external.parent.mkdir()
    external.touch()
    monkeypatch.setenv("UNIRIG_PYTHON", str(external))
    monkeypatch.delenv("KAIRYTH_V2_ALLOW_EXTERNAL_ENV", raising=False)
    env_util._resolve_rig_python("UNIRIG_PYTHON", tmp_path / "missing.exe", "unirig")
    assert "UNIRIG_PYTHON" not in env_util.os.environ


def test_explicit_external_environment_requires_opt_in(tmp_path, monkeypatch):
    executable = tmp_path / "allowed environment" / "python.exe"
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.setenv("UNIRIG_PYTHON", str(executable))
    monkeypatch.setenv("KAIRYTH_V2_ALLOW_EXTERNAL_ENV", "1")
    env_util._resolve_rig_python("UNIRIG_PYTHON", tmp_path / "missing.exe", "unirig")
    assert env_util.os.environ["UNIRIG_PYTHON"] == str(executable)


def test_launcher_reserves_original_port_and_distribution():
    launch = (ROOT / "start_motionbuilder.ps1").read_text(encoding="utf-8")
    stop = (ROOT / "stop_motionbuilder.ps1").read_text(encoding="utf-8")
    assert "$Port = 7861" in launch
    assert "$Port -eq 7860" in launch
    assert "Kairyth-V2-GVHMR" in launch
    assert "$env:MOTIONBUILDER_WSL_DISTRO -eq 'Kairyth-GVHMR'" in launch
    assert "instance_root" in launch and "instance_id" in launch
    assert "instance_root" in stop and "instance_id" in stop
    assert "--terminate" not in stop
    assert "Stop-Process" not in stop


def test_health_identity_and_probe_reject_other_checkout(monkeypatch):
    from motionbuilder_app import server

    monkeypatch.setenv("KAIRYTH_V2_INSTANCE_ID", "synthetic-v2-identity")
    monkeypatch.setattr(server, "check_setup", lambda: (False, "not installed"))
    payload = server.health()
    assert Path(payload["instance_root"]).resolve() == ROOT.resolve()
    assert payload["instance_id"] == "synthetic-v2-identity"

    class Response(io.BytesIO):
        status = 200

    for root, identity, expected in [(str(ROOT), "synthetic-v2-identity", True),
                                     (str(ROOT), "different-instance", False),
                                     (str(ROOT.parent / "other-checkout"), "synthetic-v2-identity", False)]:
        with patch.object(server.urllib.request, "urlopen", return_value=Response(json.dumps({"instance_root": root, "instance_id": identity}).encode())):
            assert server._probe_meshpreps(7861) is expected
