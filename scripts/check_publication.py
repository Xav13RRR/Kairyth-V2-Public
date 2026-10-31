"""Read-only check of Git's index and optional reachable history; never prints secrets."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import PurePosixPath

EXCLUDED_ROOTS = (
    "hybrik_runtime/src/HybrIK/hybrik/models/layers/smpl/",
    "hybrik_runtime/src/HybrIK/hybrik/models/layers/smplx/",
    "GVHMR/hmr4d/utils/body_model/",
    "data/", "runtime/", "third_party/envs/", ".pnpm-store/", ".local-data/",
    ".agents/", "hybrik/", "model_files/", "GVHMR/inputs/checkpoints/",
    "GVHMR/outputs/", "GVHMR/inputs/webui/", "hybrik_runtime/model_files/",
    "hybrik_runtime/pretrained_models/", "hybrik_runtime/downloads/",
    "third_party/UniRig/experiments/", "third_party/SkinTokens/experiments/",
    "third_party/UniRig/lightning_logs/", "GVHMR/docs/example_video/",
    "GVHMR/third-party/DPVO/logs/", "hybrik_runtime/src/HybrIK/examples/",
    "third_party/UniRig/examples/", "third_party/SkinTokens/examples/",
    "kimodo_runtime/src/kimodo/assets/demo/examples/",
)
PRIVATE_PARTS = {".git", ".git-staging-excluded", "webui-staging-excluded", ".cache", ".mka_cache", ".kimodo", "__pycache__", ".pytest_cache", ".aws", ".ssh"}
SECRET = re.compile(
    rb"hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|"
    rb"github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|"
    rb"AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
ASSIGNMENT = re.compile(
    rb"""(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)\s*[=:]\s*["']([^"'\r\n]{16,})["']"""
)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def forbidden(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name.lower()
    return (
        path.lower() == "gvhmr/hmr4d/utils/smplx_utils.py"
        or
        path.lower().startswith(tuple(root.lower() for root in EXCLUDED_ROOTS))
        or any(part.lower() in PRIVATE_PARTS for part in p.parts)
        or (name.startswith(".env") and name != ".env.example")
        or name in {"hf_token", "token", "stored_tokens", "credentials.json"}
        or name.startswith("credentials.local.")
        or p.suffix.lower() in {".pem", ".key", ".log", ".pyc", ".pyo", ".vhdx",
                                ".ckpt", ".safetensors", ".onnx", ".pkl", ".pickle",
                                ".pt", ".pth", ".whl", ".exe", ".dll", ".pyd"}
    )


def content_issues(data: bytes, path: str, notebook: bool = True) -> list[str]:
    if b"\0" in data[:8192]:
        return []
    issues = []
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        issues.append(f"{path}: unresolved LFS pointer is not an ordinary-Git runtime resource")
    for match in SECRET.finditer(data):
        line = data.count(b"\n", 0, match.start()) + 1
        issues.append(f"{path}:{line}: credential signature (redacted)")
    for match in ASSIGNMENT.finditer(data):
        value = match[2].lower()
        if any(x in value for x in (b"example", b"your", b"token", b"password", b"test", b"placeholder", b"change", b"os.env")):
            continue
        issues.append(f"{path}: literal credential candidate (redacted)")
    if notebook and path.endswith(".ipynb"):
        try:
            cells = json.loads(data).get("cells", [])
            if any(c.get("outputs") or c.get("execution_count") is not None for c in cells):
                issues.append(f"{path}: notebook contains execution output/counts")
        except (ValueError, UnicodeError):
            issues.append(f"{path}: invalid notebook JSON")
    return issues


def held_resource_paths(data: bytes) -> set[str]:
    """Publication policy is independent from physical/readiness status."""
    manifest = json.loads(data)
    return {entry["path"] for entry in manifest["resources"]
            if entry.get("publication") == "held-notpublish"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Also scan all reachable Git blobs for credential patterns.")
    parser.add_argument("--worktree", action="store_true", help="Scan visible worktree candidates, without staging or modifying files.")
    parser.add_argument("--paths-from", help="UTF-8 newline allowlist to scan with --worktree, including deliberately retained upstream ignored source.")
    parser.add_argument("--profile", choices=("private-hygiene", "public-workbench", "standalone-detector", "local-research"), default="private-hygiene")
    parser.add_argument("--artifact", help="Independently inspect a source ZIP/directory under the selected public profile.")
    parser.add_argument("--inventory", help="Fixed source inventory required for independent --artifact byte/member verification.")
    parser.add_argument("--source-root", help="Explicit read-only raw-byte source snapshot root, e.g. a staged-blob snapshot; never copies the workspace recursively.")
    parser.add_argument("--source-paths", help="UTF-8 slash-normalized fixed path list for --source-root inventory generation; no Git lookup in the snapshot.")
    parser.add_argument("--source-git-repo", help="Explicit read-only Git repository for querying only the release marker's actual template blob; path is never recorded in inventories.")
    parser.add_argument("--source-git-ref", help="Full immutable 40-hex commit for --source-git-repo template provenance, not a revision expression.")
    parser.add_argument("--write-candidate-manifest", help="Write an explicit content-pinned source inventory; intentionally excluded research paths are recorded, not copied.")
    parser.add_argument("--build-from", help="Build only the exact content-pinned inventory, never a recursive workspace copy.")
    parser.add_argument("--output-zip", help="New ZIP path for --build-from; existing output is never overwritten.")
    parser.add_argument("--report-json", help="Full path-only/redacted report; no blob or credential values.")
    args = parser.parse_args()
    if args.profile != "private-hygiene":
        if bool(args.source_git_repo) != bool(args.source_git_ref):
            parser.error("explicit Git template provenance requires both --source-git-repo and --source-git-ref")
        if args.source_git_ref and not re.fullmatch(r"[0-9a-f]{40}", args.source_git_ref):
            parser.error("--source-git-ref requires a full immutable 40-hex commit")
        if args.source_git_ref and args.profile != "public-workbench":
            parser.error("Git template provenance is only used for public-workbench")
        if args.build_from and not args.output_zip:
            parser.error("--build-from requires --output-zip")
        if args.artifact and not args.inventory:
            parser.error("--artifact requires --inventory; boundary-only checking is not full artifact identity")
        if args.inventory and not args.artifact:
            parser.error("--inventory is for independent --artifact inspection")
        if args.output_zip and not args.build_from:
            parser.error("--output-zip requires --build-from")
        if args.paths_from:
            parser.error("profile modes use fixed inventories, not --paths-from")
        if args.source_paths and not args.source_root:
            parser.error("--source-paths requires --source-root")
        if args.source_root and args.write_candidate_manifest and not args.source_paths:
            parser.error("snapshot inventory requires explicit --source-paths")
        if args.source_root and not args.history and not (args.source_paths or args.build_from or args.artifact):
            parser.error("snapshot inspection requires explicit --source-paths; no parent Git fallback")
        if args.history and (args.source_root or args.source_paths):
            parser.error("history checks use the actual repository/refs, not a source snapshot")
        if sum(bool(x) for x in (args.artifact, args.build_from, args.write_candidate_manifest, args.history)) > 1:
            parser.error("artifact/build/inventory/history are independent scopes; run separately")
        try:
            from scripts.publication_policy import run
        except ModuleNotFoundError:
            from publication_policy import run
        return run(args)
    if args.artifact or args.inventory or args.source_root or args.source_paths or args.source_git_repo or args.source_git_ref or args.build_from or args.write_candidate_manifest or args.output_zip:
        parser.error("artifact/inventory/build requires an explicit publication profile")
    if args.worktree:
        from pathlib import Path
        manifest = Path("docs/resource-license-evidence.json")
        held = held_resource_paths(manifest.read_bytes())
        candidates = (Path(args.paths_from).read_text(encoding="utf-8-sig").splitlines()
                      if args.paths_from else git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0"))
        worktree_issues = []
        checked = 0
        for path in candidates:
            if not path:
                continue
            relative = PurePosixPath(path)
            if relative.is_absolute() or ".." in relative.parts or ":" in path or "\\" in path:
                worktree_issues.append("unsafe allowlist path (redacted)")
                continue
            if forbidden(path):
                worktree_issues.append(f"{path}: private/generated/held path must not be published")
            if path in held:
                worktree_issues.append(f"{path}: resource manifest holds this path from publication")
            item = Path(path)
            if item.is_file():
                checked += 1
                worktree_issues.extend(content_issues(item.read_bytes(), path))
        for issue in worktree_issues:
            print(issue)
        print(f"Worktree candidate check: {checked} files, {len(worktree_issues)} issues; index unchanged.")
        return 1 if worktree_issues else 0
    if args.paths_from:
        parser.error("--paths-from requires --worktree")
    indexed = []
    issues = []
    try:
        held = held_resource_paths(git("show", ":docs/resource-license-evidence.json"))
    except (subprocess.CalledProcessError, ValueError, KeyError):
        held = set()
        issues.append("resource manifest must be valid and staged before publication")
    for row in git("ls-files", "--stage", "-z").split(b"\0"):
        if not row:
            continue
        meta, raw_path = row.split(b"\t", 1)
        mode, oid, stage = meta.decode().split()
        path = raw_path.decode("utf-8")
        if stage != "0":
            issues.append(f"{path}: unresolved Git index conflict")
        if forbidden(path):
            issues.append(f"{path}: private/generated path must not be tracked")
        if path in held:
            issues.append(f"{path}: resource manifest holds this path from publication")
        if mode == "160000":
            continue
        indexed.append((oid, path, True))
    objects = list(indexed)
    if not indexed:
        print("Publication check: empty Git index; no candidate content was checked. Use --worktree before staging.")
        return 1
    if args.history:
        # Check names from every reachable change, including deleted files.
        # Scanning only blob contents would miss restricted historical paths.
        historical_paths = set(git("log", "--all", "--format=", "--name-only", "-z").split(b"\0"))
        for raw_path in historical_paths:
            path = raw_path.decode("utf-8").strip("\n\r")
            if path and (forbidden(path) or path in held):
                issues.append(f"history/path/{path}: private/generated/held path in reachable history")
        seen = {oid for oid, _, _ in objects}
        for row in git("rev-list", "--objects", "--all").splitlines():
            fields = row.decode("utf-8").split(" ", 1)
            if fields[0] not in seen:
                objects.append((fields[0], fields[1] if len(fields) > 1 else fields[0], False))
                seen.add(fields[0])
    proc = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    scanned = 0
    try:
        assert proc.stdin is not None and proc.stdout is not None
        for oid, path, current in objects:
            proc.stdin.write((oid + "\n").encode())
            proc.stdin.flush()
            header = proc.stdout.readline().decode().split()
            if len(header) != 3:
                issues.append(f"{path}: cannot read Git object")
                continue
            data = proc.stdout.read(int(header[2]))
            proc.stdout.read(1)
            if header[1] == "blob":
                scanned += 1
                label = path if current else f"history/{oid[:12]}/{path}"
                issues.extend(content_issues(data, label, notebook=current))
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    for issue in issues:
        print(issue)
    print(f"Publication check: {len(indexed)} indexed blobs, {scanned} scanned blobs, {len(issues)} issues.")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
