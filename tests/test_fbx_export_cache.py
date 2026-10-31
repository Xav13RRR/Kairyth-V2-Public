"""Content/code-aware cache and transactional export tests, no Blender needed."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from motionbuilder_app import exporter


@pytest.fixture
def setup_export(tmp_path, monkeypatch):
    source, output = tmp_path / "motion.npz", tmp_path / "motion.fbx"
    script, logic = tmp_path / "blender_script.py", tmp_path / "packing.py"
    source.write_bytes(b"original motion")
    script.write_text("script version one", encoding="utf-8")
    logic.write_text("packing version one", encoding="utf-8")
    monkeypatch.setattr(exporter, "BLENDER_SCRIPT", script)
    monkeypatch.setattr(exporter, "EXPORTER_LOGIC", logic)
    monkeypatch.setattr(exporter, "find_blender", lambda: tmp_path / "blender.exe")
    monkeypatch.setattr(exporter, "_write_blender_bundle", lambda *a, **kw: {"bind_pose_only": kw["bind_pose_only"]})
    calls = []

    def run(command, **kwargs):
        meta = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        Path(meta["output_fbx"]).write_bytes(b"FBX output " + bytes([len(calls)]) * 200)
        calls.append(meta)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(exporter.subprocess, "run", run)
    return source, output, script, logic, calls


def test_legacy_cache_is_stale_even_with_future_mtime(setup_export):
    source, output, *rest = setup_export
    output.write_bytes(b"old bad FBX" * 20)
    os.utime(output, (2_000_000_000, 2_000_000_000))
    assert not exporter.is_fbx_export_current(source, output)


@pytest.mark.parametrize("bind_pose_only", [False, True])
def test_success_records_identity_and_cache_hit_without_running_blender(setup_export, bind_pose_only):
    source, output, _, _, calls = setup_export
    assert exporter.export_fbx(source, output, bind_pose_only=bind_pose_only) == output
    assert exporter.is_fbx_export_current(source, output, bind_pose_only=bind_pose_only)
    assert len(calls) == 1
    assert not exporter.is_fbx_export_current(source, output, bind_pose_only=not bind_pose_only)


@pytest.mark.parametrize("changed", ["source", "script", "logic", "output"])
def test_same_size_same_mtime_content_change_invalidates_cache(setup_export, changed):
    source, output, script, logic, _ = setup_export
    exporter.export_fbx(source, output)
    path = dict(source=source, output=output, script=script, logic=logic)[changed]
    original_stat = path.stat()
    content = path.read_bytes()
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert not exporter.is_fbx_export_current(source, output)


@pytest.mark.parametrize("content", ["not JSON", "[]", "{}", '{"identity":null,"output":null}'])
def test_malformed_marker_is_stale(setup_export, content):
    source, output, *_ = setup_export
    exporter.export_fbx(source, output)
    exporter._export_marker_path(output).write_text(content, encoding="utf-8")
    assert not exporter.is_fbx_export_current(source, output)


def test_failed_export_preserves_last_success_and_does_not_mark_changed_source(setup_export, monkeypatch):
    source, output, *_ = setup_export
    exporter.export_fbx(source, output)
    marker_path = exporter._export_marker_path(output)
    old_output, old_marker = output.read_bytes(), marker_path.read_bytes()
    source.write_bytes(b"changed motion")
    monkeypatch.setattr(exporter.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="test failure"))
    with pytest.raises(RuntimeError, match="test failure"):
        exporter.export_fbx(source, output)
    assert output.read_bytes() == old_output
    assert marker_path.read_bytes() == old_marker
    assert not exporter.is_fbx_export_current(source, output)


def test_initial_failure_leaves_no_success_marker(setup_export, monkeypatch):
    source, output, *_ = setup_export
    monkeypatch.setattr(exporter.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="test failure"))
    with pytest.raises(RuntimeError):
        exporter.export_fbx(source, output)
    assert not output.exists()
    assert not exporter._export_marker_path(output).exists()


def test_zero_exit_with_invalid_output_preserves_last_success(setup_export, monkeypatch):
    source, output, *_ = setup_export
    exporter.export_fbx(source, output)
    marker_path = exporter._export_marker_path(output)
    previous = output.read_bytes(), marker_path.read_bytes()

    def short_output(command, **kwargs):
        meta = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        Path(meta["output_fbx"]).write_bytes(b"short")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(exporter.subprocess, "run", short_output)
    with pytest.raises(RuntimeError, match="Blender"):
        exporter.export_fbx(source, output)
    assert output.read_bytes() == previous[0]
    assert marker_path.read_bytes() == previous[1]
    assert exporter.is_fbx_export_current(source, output)


@pytest.mark.parametrize("changed", ["source", "script", "logic"])
def test_identity_change_during_export_rejects_before_replacing_output(setup_export, monkeypatch, changed):
    source, output, script, logic, _ = setup_export
    exporter.export_fbx(source, output)
    previous = output.read_bytes(), exporter._export_marker_path(output).read_bytes()
    run = exporter.subprocess.run

    def changing_run(*a, **kw):
        result = run(*a, **kw)
        path = dict(source=source, script=script, logic=logic)[changed]
        path.write_bytes(path.read_bytes() + b" changed")
        return result

    monkeypatch.setattr(exporter.subprocess, "run", changing_run)
    with pytest.raises(RuntimeError, match="changed during export"):
        exporter.export_fbx(source, output)
    assert output.read_bytes() == previous[0]
    assert exporter._export_marker_path(output).read_bytes() == previous[1]
    assert not exporter.is_fbx_export_current(source, output)


def test_marker_replace_failure_cannot_validate_new_output_with_old_marker(setup_export, monkeypatch):
    source, output, *_ = setup_export
    exporter.export_fbx(source, output)
    marker_path = exporter._export_marker_path(output)
    old_marker = marker_path.read_bytes()
    replace = exporter.os.replace

    def fail_marker(source_path, target_path):
        if Path(target_path) == marker_path:
            raise OSError("marker replace failed")
        return replace(source_path, target_path)

    monkeypatch.setattr(exporter.os, "replace", fail_marker)
    with pytest.raises(OSError, match="marker replace failed"):
        exporter.export_fbx(source, output)
    assert marker_path.read_bytes() == old_marker
    assert not exporter.is_fbx_export_current(source, output)
