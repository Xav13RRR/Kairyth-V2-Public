"""Launcher source/PowerShell argument checks, not a real server-launch test."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("start_motionbuilder.ps1", "install.ps1")


def test_launcher_keeps_old_instance_and_state_protections():
    launch = (ROOT / SCRIPTS[0]).read_text(encoding="utf-8")
    local = launch.split("if ($ReleaseProfile -eq 'local-research') {", 1)[1].split("} elseif", 1)[0]
    assert "$env:MOTIONBUILDER_WSL_DISTRO -eq 'Kairyth-GVHMR'" in local
    assert "Kairyth-V2-GVHMR" in local
    assert "$Port -eq 7860" in launch
    assert "@('data','runtime','third_party\\envs')" in launch
    assert "ReparsePoint" in launch
    assert "$health.instance_root -eq $Root" in launch and "$health.instance_id -eq $state.instance_id" in launch
    assert "unrecognized instance; no process was stopped" in launch
    for name in SCRIPTS:
        source = (ROOT / name).read_text(encoding="utf-8")
        assert "$ReleaseProfile = 'unconfigured'" in source
        assert "& $Python -I -B -c" in source and "runpy.run_path(sys.argv[1])['capabilities']()['release_profile']" in source
        assert "@ResourceArgs" in source and "Invoke-Expression" not in source


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="Requires actual PowerShell; no inference dependency involved")
@pytest.mark.parametrize("name", SCRIPTS)
def test_powershell_syntax_and_all_other_manifest_features_are_preserved(tmp_path, name):
    source = (ROOT / name).read_text(encoding="utf-8")
    snippet = source.split("# Profile-specific advisory arguments:", 1)[1].split("\n", 1)[1].split("# End profile-specific advisory arguments.", 1)[0]
    runner = tmp_path / "argument-test.ps1"
    runner.write_text("param([string]$Root,[string]$ReleaseProfile,[string]$Target)\n"
                      "$ErrorActionPreference='Stop'\n"
                      "$tokens=$null; $parseErrors=$null; [void][System.Management.Automation.Language.Parser]::ParseFile($Target,[ref]$tokens,[ref]$parseErrors); if($parseErrors.Count){throw 'PowerShell parse failure'}\n"
                      + snippet + "\nConvertTo-Json -InputObject @($ResourceArgs) -Compress\n", encoding="utf-8")
    inventory = tmp_path / "docs" / "resource-license-evidence.json"
    inventory.parent.mkdir()
    features = ["core", "ui", "hybrik", "gvhmr", "support", "kimodo-bvh-reference", "kimodo-g1-reference", "kimodo-smplx-reference", "kimodo-soma-layer"]
    inventory.write_text(json.dumps({"resources": [{"feature": x} for x in features]}), encoding="utf-8")
    for profile in ("local-research", "public-workbench", "unconfigured"):
        result = subprocess.run([shutil.which("pwsh"), "-NoProfile", "-NonInteractive", "-File", str(runner), str(tmp_path), profile, str(ROOT / name)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        arguments = json.loads(result.stdout.strip().splitlines()[-1])
        if profile == "local-research":
            assert arguments == ["--feature", "all", "--presence-only"]
        else:
            assert arguments[0] == "--presence-only"
            assert arguments[1::2] == ["--feature"] * (len(features) - 2)
            assert set(arguments[2::2]) == set(features) - {"hybrik", "gvhmr"}
