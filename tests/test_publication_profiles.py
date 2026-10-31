"""Public profile/package checks are separate from private hygiene and research tests."""
import json
from pathlib import Path
import stat
import zipfile

import pytest

from scripts.publication_policy import Policy, build_zip, check_artifact, check_inventory, digest, safe_path
from scripts import publication_policy as module


@pytest.fixture
def policy(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "publication-rules.json").write_text(json.dumps({"profiles": {"public-workbench": {
        "include_prefixes": ["motionbuilder_app/", "docs/"], "root_files": ["LICENSE", "release-profile.json"],
        "exclude_prefixes": ["GVHMR/"], "exclude_paths": ["motionbuilder_app/pose_runtime.py"],
        "required_files": ["LICENSE"]}}}), encoding="utf-8")
    (docs / "resource-license-evidence.json").write_text(json.dumps({"resources": [
        {"path": "motionbuilder_app/allowed.npy", "sha256": digest(b"approved"), "publication": "redistributable-under-upstream-apache"},
        {"path": "motionbuilder_app/held.svg", "sha256": digest(b"held"), "publication": "held-notpublish"}]}), encoding="utf-8")
    (tmp_path / "LICENSE").write_text("source license", encoding="utf-8")
    (tmp_path / "release-profile.json").write_text(json.dumps({"schema": "kairyth-release/v1", "profile": "local-research"}), encoding="utf-8")
    (tmp_path / "motionbuilder_app").mkdir()
    (tmp_path / "motionbuilder_app/core.py").write_text("value = 1\n", encoding="utf-8")
    return Policy(tmp_path, "public-workbench")


def inventory(policy):
    files = []
    for path in ("LICENSE", "motionbuilder_app/core.py", "release-profile.json"):
        data, origin = policy.materialize(path, policy.read(path))
        row = dict(path=path, sha256=digest(data), size=len(data))
        if origin is not None:
            row["origin"] = origin
        files.append(row)
    return dict(profile=policy.profile, policy_sha256=digest(policy.rules_bytes), files=files, errors=[])


@pytest.mark.parametrize("path", ["../outside", "a/../b", "a\\b", "a//b", "./a", "/root", "C:/root", ""])
def test_noncanonical_paths_are_rejected(path):
    assert not safe_path(path)


def test_profile_separates_research_and_exact_binary_evidence(policy):
    assert policy.reason("GVHMR/demo.py")
    assert policy.reason("motionbuilder_app/pose_runtime.py")
    assert policy.reason("motionbuilder_app/unknown.npy", b"unknown")
    assert policy.reason("motionbuilder_app/held.svg", b"held")
    assert policy.reason("motionbuilder_app/allowed.npy", b"wrong")
    assert policy.reason("motionbuilder_app/allowed.npy", b"approved") is None
    assert policy.reason("motionbuilder_app/core.py", b"binary\0")


def test_exact_inventory_roundtrip_and_no_overwrite(policy, tmp_path):
    plan = inventory(policy)
    output = tmp_path / "candidate.zip"
    assert check_inventory(policy, plan) == []
    build_zip(policy, plan, output)
    assert check_artifact(policy, output, plan) == []
    with pytest.raises(FileExistsError):
        build_zip(policy, plan, output)
    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(changed, "w") as archive:
        archive.writestr("LICENSE", "source license")
        archive.writestr("motionbuilder_app/core.py", "value = 2\n")
    assert any("fixed inventory" in issue for issue in check_artifact(policy, changed, plan))


def test_source_changed_after_inventory_blocks_build(policy, tmp_path):
    plan = inventory(policy)
    (tmp_path / "motionbuilder_app/core.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="not publishable"):
        build_zip(policy, plan, tmp_path / "must_not_build.zip")


@pytest.mark.parametrize("member", ["../escape.py", "motionbuilder_app/../escape.py", "motionbuilder_app/inner.zip"])
def test_unsafe_or_nested_archive_member_is_rejected(policy, tmp_path, member):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, b"payload")
    assert check_artifact(policy, path)


def test_zip_symlink_is_rejected_without_extracting(policy, tmp_path):
    path = tmp_path / "link.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("motionbuilder_app/link.py")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../../outside")
    assert any("link member" in issue for issue in check_artifact(policy, path))


def test_case_collisions_and_missing_required_source_are_rejected(policy, tmp_path):
    path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("motionbuilder_app/core.py", "a")
        archive.writestr("motionbuilder_app/CORE.py", "b")
    issues = check_artifact(policy, path)
    assert any("case-colliding" in issue for issue in issues)
    assert any("LICENSE" in issue for issue in issues)


def test_mock_windows_junction_parent_is_rejected_before_file_open(policy, monkeypatch):
    from types import SimpleNamespace
    target = policy.root / "motionbuilder_app"
    original = Path.lstat
    opened = []
    def lstat(path, *a, **kw):
        info = original(path, *a, **kw)
        if path == target:
            return SimpleNamespace(st_mode=info.st_mode, st_size=info.st_size, st_file_attributes=0x400)
        return info
    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "open", lambda *a, **kw: opened.append(a) or (_ for _ in ()).throw(AssertionError("must not read junction target")))
    with pytest.raises(ValueError, match="reparse"):
        policy.read("motionbuilder_app/core.py")
    assert not opened


def test_directory_junction_is_pruned_before_walk_recurses(policy, tmp_path, monkeypatch):
    from types import SimpleNamespace
    root = tmp_path / "artifact"
    root.mkdir()
    junction = root / "junction"
    junction.mkdir()
    original = Path.lstat
    def lstat(path, *a, **kw):
        info = original(path, *a, **kw)
        return SimpleNamespace(st_mode=info.st_mode, st_size=info.st_size, st_file_attributes=0x400) if path == junction else info
    def walk(*a, **kw):
        dirs = ["junction"]
        yield str(root), dirs, []
        assert not dirs, "must prune reparse directory before any descent"
    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(module.os, "walk", walk)
    assert any("directory reparse" in issue for issue in check_artifact(policy, root))


def test_oversized_source_stat_is_rejected_before_open(policy, monkeypatch):
    from types import SimpleNamespace
    target = policy.root / "motionbuilder_app/core.py"
    original = Path.lstat
    def lstat(path, *a, **kw):
        info = original(path, *a, **kw)
        return SimpleNamespace(st_mode=info.st_mode, st_size=module.MAX_FILE_BYTES + 1, st_file_attributes=0) if path == target else info
    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "open", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not read oversized file")))
    with pytest.raises(ValueError, match="bounded"):
        policy.read("motionbuilder_app/core.py")


def test_inventory_and_directory_apply_total_budget_before_next_read(policy, tmp_path, monkeypatch):
    plan = inventory(policy)
    monkeypatch.setattr(module, "MAX_TOTAL_BYTES", 20)
    assert any("bounded" in issue for issue in check_inventory(policy, plan))
    artifact = tmp_path / "artifact"
    (artifact / "motionbuilder_app").mkdir(parents=True)
    (artifact / "LICENSE").write_text("source license")
    (artifact / "motionbuilder_app/core.py").write_text("value = 1\n")
    assert any("bounded" in issue for issue in check_artifact(policy, artifact))


def test_cli_artifact_requires_explicit_inventory(policy, tmp_path):
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts/check_publication.py"
    result = subprocess.run([sys.executable, str(script), "--profile", "public-workbench", "--artifact", "candidate.zip"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2 and "--artifact requires --inventory" in result.stderr
    plan = inventory(policy)
    archive = tmp_path / "candidate.zip"
    build_zip(policy, plan, archive)
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps(plan))
    result = subprocess.run([sys.executable, str(script), "--profile", "public-workbench", "--artifact", str(archive), "--inventory", str(manifest)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_raw_snapshot_inventory_uses_fixed_paths_without_git(policy, tmp_path, monkeypatch):
    listing = tmp_path / "staged-paths.txt"
    listing.write_text("LICENSE\nmotionbuilder_app/core.py\nrelease-profile.json\n", encoding="utf-8")
    snapshot = Policy(tmp_path, "public-workbench", listing)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("snapshot must not enumerate parent Git")))
    result = module.make_inventory(snapshot)
    assert not result["errors"]
    assert [row["path"] for row in result["files"]] == ["LICENSE", "motionbuilder_app/core.py", "release-profile.json"]
    listing.write_text("../outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        Policy(tmp_path, "public-workbench", listing)


@pytest.mark.parametrize("arguments,message", [
    (["--profile", "public-workbench", "--source-paths", "paths.txt"], "--source-paths requires --source-root"),
    (["--profile", "public-workbench", "--source-root", ".", "--write-candidate-manifest", "out.json"], "snapshot inventory requires explicit --source-paths"),
    (["--profile", "public-workbench", "--source-root", ".", "--history"], "history checks use the actual repository"),
    (["--profile", "public-workbench", "--source-root", ".", "--worktree"], "snapshot inspection requires explicit --source-paths"),
    (["--source-root", "."], "requires an explicit publication profile"),
    (["--profile", "public-workbench", "--source-git-repo", "."], "requires both --source-git-repo and --source-git-ref"),
    (["--profile", "public-workbench", "--source-git-repo", ".", "--source-git-ref", "HEAD"], "requires a full immutable 40-hex commit"),
    (["--profile", "local-research", "--source-git-repo", ".", "--source-git-ref", "a"*40], "only used for public-workbench"),
])
def test_snapshot_cli_rejects_ambiguous_modes_before_reading_files(tmp_path, arguments, message):
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts/check_publication.py"
    result = subprocess.run([sys.executable, str(script), *arguments], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2 and message in result.stderr


def test_snapshot_cli_with_fixed_list_needs_no_git_or_recursive_fallback(policy, tmp_path):
    import subprocess
    import sys
    listing = tmp_path / "source-paths.txt"
    listing.write_text("LICENSE\nmotionbuilder_app/core.py\nrelease-profile.json\n", encoding="utf-8")
    output = tmp_path / "snapshot-inventory.json"
    script = Path(__file__).resolve().parents[1] / "scripts/check_publication.py"
    result = subprocess.run([sys.executable, str(script), "--profile", "public-workbench", "--source-root", str(tmp_path),
                             "--source-paths", str(listing), "--write-candidate-manifest", str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [row["path"] for row in json.loads(output.read_text())["files"]] == ["LICENSE", "motionbuilder_app/core.py", "release-profile.json"]
