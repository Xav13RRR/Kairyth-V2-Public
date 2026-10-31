"""One declared marker transform, never hidden inference enablement or false Git identity."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts import publication_policy as module
from scripts.publication_policy import Policy, RELEASE_MARKER, PUBLIC_MARKER_BYTES, parse_release_marker
from tests.test_publication_profiles import policy, inventory


def marker_row(plan):
    return next(row for row in plan["files"] if row["path"] == RELEASE_MARKER)


def test_local_template_is_unchanged_but_package_has_only_fixed_public_marker(policy, tmp_path):
    original = policy.read(RELEASE_MARKER)
    plan = inventory(policy)
    row = marker_row(plan)
    assert row["sha256"] == module.digest(PUBLIC_MARKER_BYTES)
    assert row["origin"]["source"]["utf8"].encode() == original
    assert row["origin"]["source"]["git_blob_oid"] is None
    assert row["origin"]["source"]["git_verification"] == "not-queried"
    assert row["origin"]["kind"] == "generated"
    assert str(tmp_path) not in json.dumps(plan)
    output = tmp_path / "public.zip"
    module.build_zip(policy, plan, output)
    assert policy.read(RELEASE_MARKER) == original
    with zipfile.ZipFile(output) as archive:
        assert archive.read(RELEASE_MARKER) == PUBLIC_MARKER_BYTES
    assert module.check_artifact(policy, output, plan) == []


def test_already_public_source_is_byte_identity_not_a_fake_transform(policy, tmp_path):
    source = b'{ "schema":"kairyth-release/v1", "profile":"public-workbench" }\n'
    (tmp_path / RELEASE_MARKER).write_bytes(source)
    plan = inventory(policy)
    row = marker_row(plan)
    assert row["origin"]["kind"] == "identity" and "transform" not in row["origin"]
    assert row["sha256"] == row["origin"]["source"]["sha256"] == module.digest(source)
    output = tmp_path / "public-identity.zip"
    module.build_zip(policy, plan, output)
    assert module.check_artifact(policy, output, plan) == []
    with zipfile.ZipFile(output) as archive:
        assert archive.read(RELEASE_MARKER) == source


@pytest.mark.parametrize("payload", [b'{}', b'[]', b'{"schema":"bad","profile":"public-workbench"}',
    b'{"schema":"kairyth-release/v1","profile":"anything"}',
    b'{"schema":"kairyth-release/v1","profile":"public-workbench","capabilities":{}}',
    b'{"schema":"kairyth-release/v1","profile":"public-workbench","profile":"local-research"}',
    b'[' * 1500 + b']' * 1500])
def test_unknown_duplicate_and_deep_marker_inputs_are_invalid(payload):
    with pytest.raises(ValueError):
        parse_release_marker(payload)


def test_forged_inventory_never_allows_local_public_artifact_or_omission(policy, tmp_path):
    plan = inventory(policy)
    forged = copy.deepcopy(plan)
    local = policy.read(RELEASE_MARKER)
    row = marker_row(forged)
    row["sha256"], row["size"] = module.digest(local), len(local)
    output = tmp_path / "forged-local.zip"
    with zipfile.ZipFile(output, "w") as archive:
        for entry in forged["files"]:
            archive.writestr(entry["path"], local if entry["path"] == RELEASE_MARKER else policy.read(entry["path"]))
    assert any("local-research" in issue for issue in module.check_artifact(policy, output, forged))
    omitted = {**plan, "files": [row for row in plan["files"] if row["path"] != RELEASE_MARKER]}
    missing = tmp_path / "missing.zip"
    with zipfile.ZipFile(missing, "w") as archive:
        for entry in omitted["files"]:
            archive.writestr(entry["path"], policy.read(entry["path"]))
    assert any("release-profile.json" in issue for issue in module.check_artifact(policy, missing, omitted))


@pytest.mark.parametrize("mutation", ["transform-id", "parameters", "template-path", "source-sha", "unqueried-oid", "fake-commit", "extra-origin"])
def test_origin_schema_and_only_fixed_transformation_are_enforced(policy, mutation):
    row = copy.deepcopy(marker_row(inventory(policy)))
    origin, source = row["origin"], row["origin"]["source"]
    if mutation == "transform-id": origin["transform"]["id"] = "arbitrary"
    if mutation == "parameters": origin["transform"]["parameters"]["profile"] = "local-research"
    if mutation == "template-path": source["path"] = "motionbuilder_app/core.py"
    if mutation == "source-sha": source["sha256"] = "0" * 64
    if mutation == "unqueried-oid": source["git_blob_oid"] = module.git_blob_oid(source["utf8"].encode())
    if mutation == "fake-commit": source.update(git_verification="exact-commit-blob", git_commit="HEAD", git_blob_oid=module.git_blob_oid(source["utf8"].encode()))
    if mutation == "extra-origin": origin["repository"] = "private-machine-path"
    assert module.marker_origin_issues(row, PUBLIC_MARKER_BYTES)


def test_no_arbitrary_generated_other_members(policy):
    plan = inventory(policy)
    next(row for row in plan["files"] if row["path"].endswith("core.py"))["origin"] = copy.deepcopy(marker_row(plan)["origin"])
    assert any("provenance changed" in issue for issue in module.check_inventory(policy, plan))


def git_mock(data, *, failure=None):
    oid = module.git_blob_oid(data)
    def query(command, **kwargs):
        args = command[4:]
        if failure == "repo": raise subprocess.CalledProcessError(128, command)
        if args[:2] == ["cat-file", "-t"]: return b"blob\n" if failure == "ref" else b"commit\n"
        if args[0] == "ls-tree":
            return b"" if failure == "missing" else f"{'120000' if failure == 'mode' else '100644'} blob {oid}\t{RELEASE_MARKER}\0".encode()
        if args[:2] == ["cat-file", "-s"]: return str(len(data)).encode()
        if args[:2] == ["cat-file", "blob"]: return b"different" if failure == "blob" else data
        raise AssertionError(args)
    return query


def test_explicit_git_query_witness_with_mocked_subprocess_has_no_repo_path(policy, monkeypatch):
    data = policy.read(RELEASE_MARKER)
    monkeypatch.setattr(module.subprocess, "check_output", git_mock(data))
    verified = Policy(policy.root, "public-workbench", git_repo=policy.root, git_ref="a" * 40)
    row = marker_row(inventory(verified))
    source = row["origin"]["source"]
    assert source["git_commit"] == "a" * 40 and source["git_blob_oid"] == module.git_blob_oid(data)
    assert source["git_verification"] == "exact-commit-blob"
    assert str(policy.root) not in json.dumps(row)
    assert module.marker_origin_issues(row, PUBLIC_MARKER_BYTES) == []


@pytest.mark.parametrize("failure", ["repo", "ref", "missing", "mode", "blob", "source"])
def test_wrong_git_repo_ref_blob_or_source_is_rejected(policy, monkeypatch, failure):
    data = policy.read(RELEASE_MARKER)
    monkeypatch.setattr(module.subprocess, "check_output", git_mock(data + b" " if failure == "source" else data, failure=failure))
    with pytest.raises(ValueError):
        Policy(policy.root, "public-workbench", git_repo=policy.root, git_ref="a" * 40)


def history_mock(markers):
    blobs = {module.git_blob_oid(data): data for data in markers.values() if data is not None}
    def query(command, **kwargs):
        args = command[4:]
        if args[0] == "ls-tree":
            data = markers[args[3]]
            return b"" if data is None else f"100644 blob {module.git_blob_oid(data)}\t{RELEASE_MARKER}\0".encode()
        if args[:2] == ["cat-file", "-s"]: return str(len(blobs[args[2]])).encode()
        if args[:2] == ["cat-file", "blob"]: return blobs[args[2]]
        raise AssertionError(args)
    return query


@pytest.mark.parametrize("history,expected", [(["public"], []), (["local", "public"], ["local-research"]), (["missing"], ["missing"]), (["invalid"], ["invalid"])])
def test_mock_history_reads_each_commit_blob_and_never_transforms_old_local_marker(policy, monkeypatch, history, expected):
    markers = {"public": PUBLIC_MARKER_BYTES, "local": policy.read(RELEASE_MARKER), "missing": None, "invalid": b"bad"}
    # Working tree may be public; historical local still blocks independently.
    (policy.root / RELEASE_MARKER).write_bytes(PUBLIC_MARKER_BYTES)
    monkeypatch.setattr(module.subprocess, "check_output", history_mock(markers))
    issues = module.history_release_issues(policy.root, history)
    assert len(issues) == len(expected)
    assert all(fragment in issue for fragment, issue in zip(expected, issues))


@pytest.mark.parametrize("state,oid_present,commit,valid", [
    ("not-queried",False,None,True), ("not-in-index",False,None,True),
    ("working-copy-differs-from-index",False,None,True), ("exact-index-blob",True,None,True),
    ("exact-commit-blob",True,"b"*40,True), ("exact-index-blob",True,"b"*40,False),
    ("exact-commit-blob",True,None,False), ("exact-commit-blob",False,"b"*40,False),
    ("not-queried",False,"b"*40,False), ("unrecognized",False,None,False),
])
def test_offline_git_witness_state_structure_not_a_git_presence_proof(policy, state, oid_present, commit, valid):
    row = copy.deepcopy(marker_row(inventory(policy)))
    source = row["origin"]["source"]
    source.update(git_verification=state, git_commit=commit,
                  git_blob_oid=module.git_blob_oid(source["utf8"].encode()) if oid_present else None)
    assert (module.marker_origin_issues(row, PUBLIC_MARKER_BYTES) == []) is valid


def test_direct_public_check_rejects_local_while_explicit_plan_allows_source_template(policy, tmp_path):
    listing = tmp_path / "paths.txt"
    listing.write_text("LICENSE\nmotionbuilder_app/core.py\nrelease-profile.json\n")
    script = Path(__file__).resolve().parents[1] / "scripts/check_publication.py"
    common = [sys.executable, str(script), "--profile", "public-workbench", "--source-root", str(tmp_path), "--source-paths", str(listing)]
    result = subprocess.run(common + ["--worktree"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1 and "local-research" in result.stdout
    result = subprocess.run(common + ["--write-candidate-manifest", str(tmp_path / "plan.json")], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
