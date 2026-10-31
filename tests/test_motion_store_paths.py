import json
from pathlib import Path

from motionbuilder_app import stores


def test_restore_rehomes_legacy_motion_paths(tmp_path, monkeypatch):
    motions_root = tmp_path / "motions"
    motion_dir = motions_root / "legacy-motion"
    motion_dir.mkdir(parents=True)
    (motion_dir / "input.mp4").write_bytes(b"video")
    pt_dir = motion_dir / "gvhmr" / "input"
    pt_dir.mkdir(parents=True)
    (pt_dir / "hmr4d_results.pt").write_bytes(b"result")
    old_root = Path("D:/example-legacy/MotionBuilder/data/motions/legacy-motion")
    manifest = motion_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "id": "legacy-motion",
                "status": "completed",
                "input_path": str(old_root / "input.mp4"),
                "pt_path": str(old_root / "gvhmr" / "input" / "hmr4d_results.pt"),
                "result_dir": str(old_root / "gvhmr" / "input"),
                "wsl_pid_file": str(old_root / "wsl-process.pid"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(stores, "MOTIONS_ROOT", motions_root)

    store = stores.MotionStore(None)
    job = store.get("legacy-motion")
    assert Path(job["input_path"]) == motion_dir / "input.mp4"
    assert Path(job["pt_path"]) == pt_dir / "hmr4d_results.pt"
    assert Path(job["result_dir"]) == pt_dir
    assert Path(job["wsl_pid_file"]) == motion_dir / "wsl-process.pid"
    assert "MotionBuilder" not in manifest.read_text(encoding="utf-8")
