"""Parse actual workflow; prove checkout-time AND-presence gating, not job hashFiles."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/source-check.yml"


def test_workflow_parses_and_scope_gates_all_research_steps():
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    job = jobs["detector-boundary-cpu"]
    assert "if" not in job
    steps = job["steps"]
    assert steps[0]["uses"].startswith("actions/checkout@")
    assert steps[1]["id"] == "research_scope"
    assert " -and " in steps[1]["run"] and "NOT APPLICABLE" in steps[1]["run"]
    for step in steps[2:]:
        assert step["if"] == "steps.research_scope.outputs.available == 'true'"
    history = jobs["public-history-boundary"]
    assert not history.get("continue-on-error", False)
    assert "--profile public-workbench --history" in history["steps"][-1]["run"]


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="Actual Windows CI PowerShell condition test requires pwsh; not a model/integration skip")
@pytest.mark.parametrize("gvhmr,detector,expected", [(False,False,False),(True,False,False),(False,True,False),(True,True,True)])
def test_actual_presence_step_negative_and_positive_cases(tmp_path, gvhmr, detector, expected):
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    script = jobs["detector-boundary-cpu"]["steps"][1]["run"]
    for present, relative in ((gvhmr,"GVHMR/tools/demo/demo.py"),(detector,"detector_tool/cli.py")):
        if present:
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic presence fixture only")
    output = tmp_path / "github-output.txt"
    result = subprocess.run([shutil.which("pwsh"), "-NoProfile", "-NonInteractive", "-Command", script], cwd=tmp_path,
                            env={**os.environ, "GITHUB_OUTPUT": str(output)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").strip() == f"available={str(expected).lower()}"
    assert ("NOT APPLICABLE" in result.stdout) is not expected
