import json

from scripts.check_publication import content_issues, forbidden, held_resource_paths


def test_private_paths_rejected_even_if_force_added():
    for path in (
        "data/motions/task/manifest.json", "Data/uploads/person.mp4",
        "runtime/wsl/ext4.vhdx", "third_party/envs/rig/python.exe",
        "nested/.cache/huggingface/token", ".env.local", "nested/hf_token",
        "kimodo_runtime/src/kimodo/assets/demo/examples/example/motion.npz",
        "hybrik_runtime/src/HybrIK/hybrik/models/layers/smpl/lbs.py",
        "GVHMR/hmr4d/utils/body_model/utils.py",
        "GVHMR/hmr4d/utils/smplx_utils.py",
        "nested/.git/config", "nested/webui-staging-excluded/data.json",
        "weights/example.safetensors", "arbitrary/model.ckpt",
    ):
        assert forbidden(path), path
    for path in ("third_party/UniRig/src/data/dataset.py", "tests/test_motion_core.py", ".env.example"):
        assert not forbidden(path), path


def test_credential_diagnostic_does_not_disclose_value():
    fake_token = b"hf_" + b"A" * 30
    findings = content_issues(b"setting=" + fake_token, "config.txt")
    assert findings and "config.txt:1" in findings[0]
    assert fake_token.decode() not in "\n".join(findings)


def test_notebook_outputs_blocked_but_historical_outputs_can_be_secret_scanned():
    notebook = json.dumps({"cells": [{"cell_type": "code", "outputs": [{"text": "result"}], "execution_count": 1}]}).encode()
    assert content_issues(notebook, "demo.ipynb")
    assert not content_issues(notebook, "history/demo.ipynb", notebook=False)


def test_resource_publication_holds_are_separate_from_readiness():
    manifest = {"resources": [
        {"path": "held.npy", "publication": "held-notpublish", "required": False},
        {"path": "allowed.npy", "publication": "redistributable-under-upstream-apache"},
    ]}
    assert held_resource_paths(json.dumps(manifest).encode()) == {"held.npy"}
