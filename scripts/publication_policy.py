"""Source-only profile, fixed-inventory ZIP builder and independent artifact check."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

try:
    from scripts.check_publication import content_issues, forbidden
except ModuleNotFoundError:
    from check_publication import content_issues, forbidden

TEXT_SUFFIXES = {".py", ".ps1", ".bat", ".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".js", ".mjs", ".cjs", ".css", ".html", ".sh", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".c", ".pxd", ".pyx", ".in", ".cmake", ".ipynb", ".xml", ".glsl"}
TEXT_NAMES = {"license", "notice", "notices", "copying", "dockerfile", "makefile", ".gitignore", ".gitattributes", ".env.example"}
ARCHIVES = {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}
APPROVED_RESOURCES = {"redistributable-under-upstream-apache", "retain-upstream-apache", "redistributable-under-ofl-1.1", "user-authorized-first-party"}
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
RELEASE_MARKER = "release-profile.json"
RELEASE_SCHEMA = "kairyth-release/v1"
RELEASE_TRANSFORM = "kairyth.release-profile.local-to-public/v1"
MAX_MARKER_BYTES = 4096
PUBLIC_MARKER_BYTES = (json.dumps({"schema": RELEASE_SCHEMA, "profile": "public-workbench"}, indent=2, sort_keys=True) + "\n").encode("utf-8")


def parse_release_marker(data: bytes) -> str:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate release marker field")
            result[key] = value
        return result
    if len(data) > MAX_MARKER_BYTES:
        raise ValueError("release marker exceeds size limit")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("invalid release marker JSON") from exc
    if not isinstance(value, dict) or set(value) != {"schema", "profile"} or value["schema"] != RELEASE_SCHEMA or value["profile"] not in ("local-research", "public-workbench"):
        raise ValueError("release marker requires exactly the known schema/profile fields")
    return value["profile"]


def git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def marker_origin_issues(row: dict, data: bytes) -> list[str]:
    """Offline verify the only allowed derivation; never trust inventory to enable inference."""
    try:
        origin = row["origin"]
        source = origin["source"]
        if set(source) != {"path", "utf8", "sha256", "size", "git_blob_oid", "git_commit", "git_verification"}:
            raise ValueError("unknown source identity fields")
        template = source["utf8"].encode("utf-8")
        profile = parse_release_marker(template)
        if source["path"] != RELEASE_MARKER or source["sha256"] != digest(template) or source["size"] != len(template):
            raise ValueError("template identity mismatch")
        oid = source.get("git_blob_oid")
        if oid is not None and (not isinstance(oid, str) or not re.fullmatch(r"[0-9a-f]{40}", oid) or oid != git_blob_oid(template)):
            raise ValueError("template blob address mismatch")
        commit, verification = source["git_commit"], source["git_verification"]
        if verification == "exact-commit-blob":
            if oid is None or not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError("invalid commit witness")
        elif verification == "exact-index-blob":
            if oid is None or commit is not None:
                raise ValueError("invalid index witness")
        elif verification in {"not-queried", "not-in-index", "working-copy-differs-from-index"}:
            if oid is not None or commit is not None:
                raise ValueError("unqueried identity cannot assert Git provenance")
        else:
            raise ValueError("unknown Git witness state")
        if origin["kind"] == "generated":
            if set(origin) != {"kind", "source", "transform"} or profile != "local-research" or origin["transform"] != {"id": RELEASE_TRANSFORM, "parameters": {"profile": "public-workbench"}} or data != PUBLIC_MARKER_BYTES:
                raise ValueError("undeclared/non-fixed release transform")
        elif origin["kind"] == "identity":
            if set(origin) != {"kind", "source"} or profile != "public-workbench" or data != template:
                raise ValueError("public marker identity changed")
        else:
            raise ValueError("unknown origin kind")
        if parse_release_marker(data) != "public-workbench":
            raise ValueError("public artifact marker must disable research inference")
    except (KeyError, TypeError, AttributeError, UnicodeError, ValueError, RecursionError):
        return ["release marker origin/transform identity is invalid"]
    return []


def is_reparse(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def reject_link_chain(path: Path) -> None:
    # Check ancestors before touching descendants: junctions are reparse points,
    # not necessarily is_symlink() on the supported Windows Python versions.
    for item in [*reversed(path.absolute().parents), path.absolute()]:
        if is_reparse(item):
            raise ValueError("reparse point/link is not permitted")


def bounded_read(path: Path, root: Path, remaining: int = MAX_TOTAL_BYTES) -> bytes:
    reject_link_chain(path)
    info = path.lstat()
    limit = min(MAX_FILE_BYTES, remaining)
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError("source/artifact exceeds bounded regular-file inspection size")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("source/artifact escaped inspection root")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("source/artifact grew beyond inspection size")
    return data


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(path: str) -> bool:
    p = PurePosixPath(path)
    return bool(path) and not p.is_absolute() and "\\" not in path and ":" not in path and all(x not in {"", ".", ".."} for x in path.split("/"))


class Policy:
    def __init__(self, root: Path, profile: str, source_paths: Path | None = None, *, git_repo: Path | None = None, git_ref: str | None = None):
        reject_link_chain(root.absolute())
        self.root = root.resolve()
        self.rules_bytes = bounded_read(self.root / "docs/publication-rules.json", self.root)
        self.rules = json.loads(self.rules_bytes)
        self.profile = profile
        self.spec = self.rules["profiles"][profile]
        self.git_repo, self.git_ref = git_repo, git_ref
        if git_ref is not None and (git_repo is None or not re.fullmatch(r"[0-9a-f]{40}", git_ref)):
            raise ValueError("Git provenance requires a repository and full immutable 40-hex commit")
        if git_repo is not None:
            reject_link_chain(git_repo.absolute())
        self.candidate_paths = None
        if source_paths is not None:
            listing = source_paths.absolute()
            self.candidate_paths = bounded_read(listing, listing.parent).decode("utf-8-sig").splitlines()
            if any(not safe_path(path) for path in self.candidate_paths) or len({path.casefold() for path in self.candidate_paths}) != len(self.candidate_paths):
                raise ValueError("explicit source list contains unsafe or duplicate paths")
        resources = json.loads(bounded_read(self.root / "docs/resource-license-evidence.json", self.root))["resources"]
        self.resources = {row["path"].casefold(): row for row in resources}
        gpl = root / "docs/gpl-source-evidence.json"
        self.gpl_paths = [row["path"] for row in json.loads(bounded_read(gpl, self.root))["files"]] if profile == "public-workbench" and gpl.is_file() else []
        if self.git_ref is not None:
            self.marker_git_identity(self.read(RELEASE_MARKER))

    def reason(self, path: str, data: bytes | None = None, *, artifact: bool = False) -> str | None:
        if not safe_path(path):
            return "unsafe/non-canonical path"
        if self.profile == "local-research":
            return "local-research is classification only, not a publication profile"
        folded = path.casefold()
        if forbidden(path):
            return "private/generated/held path"
        if any(folded.startswith(p.casefold()) for p in self.spec["exclude_prefixes"]) or folded in {p.casefold() for p in self.spec["exclude_paths"]}:
            return "excluded from this product profile; local copy preserved"
        if not (path in self.spec["root_files"] or any(folded.startswith(p.casefold()) for p in self.spec["include_prefixes"])):
            return "not in positive source roots/files"
        resource = self.resources.get(folded)
        if resource and resource.get("publication") == "held-notpublish":
            return "resource evidence holds this exact path"
        p = PurePosixPath(path)
        if p.suffix.lower() in ARCHIVES:
            return "nested archives are never source-package fallback"
        text = p.suffix.lower() in TEXT_SUFFIXES or p.name.lower() in TEXT_NAMES
        if not text:
            if not resource or resource.get("publication") not in APPROVED_RESOURCES or not resource.get("sha256"):
                return "binary/non-source file has no exact approved identity"
            if data is not None and digest(data) != resource["sha256"]:
                return "approved binary identity mismatch"
        elif data is not None and b"\0" in data:
            return "binary payload disguised as text source"
        if data is not None:
            if self.profile == "public-workbench" and path == RELEASE_MARKER:
                try:
                    profile = parse_release_marker(data)
                    if artifact and profile != "public-workbench":
                        return "public artifact cannot contain the local-research release marker"
                except ValueError:
                    return "invalid release marker"
            issues = content_issues(data, path)
            if issues:
                return "; ".join(issues)
        return None

    def read(self, path: str, remaining: int = MAX_TOTAL_BYTES) -> bytes:
        if not safe_path(path):
            raise ValueError("unsafe path")
        item = self.root / path
        return bounded_read(item, self.root, min(remaining, MAX_MARKER_BYTES) if path == RELEASE_MARKER else remaining)

    def marker_git_identity(self, data: bytes) -> dict:
        result = {"git_blob_oid": None, "git_commit": None, "git_verification": "not-queried"}
        if self.git_repo is None:
            return result
        def git(*args):
            try:
                return subprocess.check_output(["git", "--no-replace-objects", "-C", str(self.git_repo), *args], stderr=subprocess.PIPE)
            except (OSError, subprocess.CalledProcessError):
                raise ValueError("release template Git witness lookup failed") from None
        if self.git_ref is not None:
            if git("cat-file", "-t", self.git_ref).strip() != b"commit":
                raise ValueError("release template Git ref is not a commit")
            rows = git("ls-tree", "-z", "--full-tree", self.git_ref, "--", RELEASE_MARKER).split(b"\0")
            rows = [row for row in rows if row]
            if len(rows) != 1:
                raise ValueError("release template is absent from the specified commit")
            meta, name = rows[0].split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            if name.decode() != RELEASE_MARKER or kind != "blob" or mode not in {"100644", "100755"}:
                raise ValueError("release template is not a regular Git blob")
        else:
            rows = [row for row in git("ls-files", "--stage", "-z", "--", RELEASE_MARKER).split(b"\0") if row]
            if not rows:
                return {**result, "git_verification": "not-in-index"}
            if len(rows) != 1:
                raise ValueError("release template has unmerged index entries")
            meta, name = rows[0].split(b"\t", 1)
            mode, oid, stage = meta.decode().split()
            if stage != "0" or mode not in {"100644", "100755"} or name.decode() != RELEASE_MARKER:
                raise ValueError("release template has no regular stage-zero blob")
        if int(git("cat-file", "-s", oid)) > MAX_MARKER_BYTES:
            raise ValueError("queried release template exceeds size limit")
        if not re.fullmatch(r"[0-9a-f]{40}", oid):
            raise ValueError("invalid Git blob address")
        if git("cat-file", "blob", oid) != data or git_blob_oid(data) != oid:
            if self.git_ref is None:
                return {**result, "git_verification": "working-copy-differs-from-index"}
            raise ValueError("release template bytes differ from the specified Git commit")
        return {"git_blob_oid": oid, "git_commit": self.git_ref, "git_verification": "exact-commit-blob" if self.git_ref else "exact-index-blob"}

    def materialize(self, path: str, source: bytes) -> tuple[bytes, dict | None]:
        if self.profile != "public-workbench" or path != RELEASE_MARKER:
            return source, None
        profile = parse_release_marker(source)
        origin = {"kind": "generated" if profile == "local-research" else "identity",
                  "source": {"path": RELEASE_MARKER, "utf8": source.decode("utf-8"), "sha256": digest(source), "size": len(source), **self.marker_git_identity(source)}}
        if profile == "local-research":
            origin["transform"] = {"id": RELEASE_TRANSFORM, "parameters": {"profile": "public-workbench"}}
            return PUBLIC_MARKER_BYTES, origin
        return source, origin

    def required_issues(self, paths: set[str]) -> list[str]:
        required = self.spec.get("required_files", []) + self.gpl_paths + ([RELEASE_MARKER] if self.profile == "public-workbench" else [])
        return [f"missing required license/source notice: {p}" for p in dict.fromkeys(required) if p not in paths]


def candidates(root: Path) -> list[str]:
    return sorted(set(subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root).decode("utf-8").split("\0")) - {""})


def history_release_issues(root: Path, commits: list[str]) -> list[str]:
    """Read actual marker blobs for every reachable commit; never transform history."""
    issues = []
    for commit in commits:
        prefix = f"history/{commit}/{RELEASE_MARKER}"
        try:
            rows = [row for row in subprocess.check_output(["git", "--no-replace-objects", "-C", str(root), "ls-tree", "-z", "--full-tree", commit, "--", RELEASE_MARKER], stderr=subprocess.PIPE).split(b"\0") if row]
            if len(rows) != 1:
                issues.append(prefix + ": missing release marker")
                continue
            meta, name = rows[0].split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            if name.decode() != RELEASE_MARKER or kind != "blob" or mode not in {"100644", "100755"}:
                raise ValueError("not regular")
            size = int(subprocess.check_output(["git", "--no-replace-objects", "-C", str(root), "cat-file", "-s", oid], stderr=subprocess.PIPE))
            if size > MAX_MARKER_BYTES:
                raise ValueError("oversized")
            data = subprocess.check_output(["git", "--no-replace-objects", "-C", str(root), "cat-file", "blob", oid], stderr=subprocess.PIPE)
            if parse_release_marker(data) != "public-workbench":
                issues.append(prefix + ": local-research marker is not public-workbench history")
        except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError, RecursionError):
            issues.append(prefix + ": invalid/unreadable release marker")
    return issues


def make_inventory(policy: Policy) -> dict:
    files, excluded, errors = [], [], []
    total = 0
    for path in policy.candidate_paths if policy.candidate_paths is not None else candidates(policy.root):
        reason = policy.reason(path)
        if reason:
            excluded.append({"path": path, "reason": reason})
            continue
        try:
            source = policy.read(path, MAX_TOTAL_BYTES - total)
            total += len(source)
            reason = policy.reason(path, source)
            data, origin = policy.materialize(path, source)
        except (OSError, ValueError) as exc:
            reason = str(exc)
        if reason:
            errors.append({"path": path, "reason": reason})
        else:
            row = {"path": path, "sha256": digest(data), "size": len(data)}
            if origin is not None:
                row["origin"] = origin
            files.append(row)
    errors.extend({"path": "", "reason": msg} for msg in policy.required_issues({row["path"] for row in files}))
    return {"schema_version": 1, "profile": policy.profile, "policy_sha256": digest(policy.rules_bytes), "source_basis": "Explicit raw-byte source snapshot and fixed list; staged-blob equivalence is independently audited" if policy.candidate_paths is not None else "Git-visible working-copy bytes, not a claim of byte equivalence to Git blobs", "files": files, "excluded": excluded, "errors": errors, "history_clearance": "not implied; run --history separately"}


def check_inventory(policy: Policy, inventory: dict) -> list[str]:
    issues, seen = [], set()
    total = 0
    if inventory.get("profile") != policy.profile or inventory.get("policy_sha256") != digest(policy.rules_bytes) or inventory.get("errors"):
        issues.append("inventory profile/policy identity mismatch or recorded errors")
    for row in inventory.get("files", []):
        path = row["path"]
        if path.casefold() in seen:
            issues.append(f"duplicate inventory path: {path}")
        seen.add(path.casefold())
        try:
            source = policy.read(path, MAX_TOTAL_BYTES - total)
            total += len(source)
            data, origin = policy.materialize(path, source)
            reason = policy.reason(path, data, artifact=True)
            if row.get("origin") != origin:
                issues.append(f"{path}: source/generation provenance changed")
            if reason or row.get("sha256") != digest(data) or row.get("size") != len(data):
                issues.append(f"{path}: {reason or 'inventory content changed'}")
        except (OSError, ValueError) as exc:
            issues.append(f"{path}: {exc}")
    issues.extend(policy.required_issues({row["path"] for row in inventory.get("files", [])}))
    return issues


def build_zip(policy: Policy, inventory: dict, output: Path) -> None:
    issues = check_inventory(policy, inventory)
    if issues:
        raise ValueError("inventory not publishable: " + "; ".join(issues[:5]))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        total = 0
        for row in inventory["files"]:
            # Recheck each byte stream at writing time, not a recursive copy.
            source = policy.read(row["path"], MAX_TOTAL_BYTES - total)
            total += len(source)
            data, origin = policy.materialize(row["path"], source)
            if row.get("origin") != origin:
                raise ValueError("source/generation provenance changed while building")
            if digest(data) != row["sha256"]:
                raise ValueError("source changed while building; incomplete archive must not be released")
            info = zipfile.ZipInfo(row["path"], date_time=(2026, 10, 31, 0, 0, 0))
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)


def check_artifact(policy: Policy, artifact: Path, inventory: dict | None = None) -> list[str]:
    issues, seen, paths = [], set(), set()
    try:
        reject_link_chain(artifact.absolute())
    except (OSError, ValueError):
        return ["artifact root must not be a reparse point/link"]
    expected = {row["path"]: row for row in inventory["files"]} if inventory else {}
    if inventory and (inventory.get("profile") != policy.profile or inventory.get("policy_sha256") != digest(policy.rules_bytes) or inventory.get("errors")):
        issues.append("artifact inventory profile/policy mismatch or errors")
    def check(path, data):
        if path.casefold() in seen:
            issues.append(f"duplicate/case-colliding artifact path: {path}")
        seen.add(path.casefold())
        paths.add(path)
        reason = policy.reason(path, data, artifact=True)
        if reason:
            issues.append(f"{path}: {reason}")
        if inventory and (path not in expected or expected[path]["sha256"] != digest(data) or expected[path]["size"] != len(data)):
            issues.append(f"{path}: not the fixed inventory byte stream")
        if policy.profile == "public-workbench" and path == RELEASE_MARKER and inventory and path in expected:
            issues.extend(marker_origin_issues(expected[path], data))
        elif inventory and path in expected and expected[path].get("origin") is not None:
            issues.append(f"{path}: arbitrary generated members are not permitted")
    if artifact.is_dir():
        total = 0
        for folder, dirs, files in os.walk(artifact, followlinks=False):
            for name in list(dirs):
                item = Path(folder) / name
                if is_reparse(item):
                    issues.append("artifact contains directory reparse point/link")
                    dirs.remove(name)
            for name in files:
                item = Path(folder) / name
                if is_reparse(item):
                    issues.append("artifact contains file reparse point/link")
                else:
                    try:
                        data = bounded_read(item, artifact, MAX_TOTAL_BYTES - total)
                        total += len(data)
                        check(item.relative_to(artifact).as_posix(), data)
                    except (OSError, ValueError) as exc:
                        issues.append(str(exc))
    else:
        with zipfile.ZipFile(artifact) as archive:
            if archive.comment:
                issues.append("opaque archive comments are not part of the fixed source package")
            total = 0
            for info in archive.infolist():
                path = info.filename.rstrip("/") if info.is_dir() else info.filename
                mode = info.external_attr >> 16
                if info.extra or info.comment or info.flag_bits & 1:
                    issues.append("opaque/encrypted archive member metadata is not permitted")
                    continue
                allowed_modes = {0, stat.S_IFDIR} if info.is_dir() else {0, stat.S_IFREG}
                if not safe_path(path) or (mode & 0o170000) not in allowed_modes:
                    issues.append("archive contains unsafe path or non-regular/link member")
                    continue
                if info.is_dir():
                    continue
                total += info.file_size
                if info.file_size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                    issues.append("source archive exceeds bounded inspection size")
                    break
                check(path, archive.read(info))
    if inventory and set(expected) != paths:
        issues.append("artifact member set differs from fixed inventory")
    return issues + policy.required_issues(paths)


def run(args) -> int:
    root = Path(args.source_root).absolute() if args.source_root else Path.cwd()
    git_repo = Path(args.source_git_repo).absolute() if args.source_git_repo else (root if not args.source_root else None)
    policy = Policy(root, args.profile, Path(args.source_paths) if args.source_paths else None, git_repo=git_repo, git_ref=args.source_git_ref)
    report = {"profile": args.profile, "scope": "Product source/artifact boundary, not private hygiene or legal certification", "issues": []}
    if args.profile == "local-research":
        report["issues"] = ["local-research is classification only; no publication/build is permitted"]
    elif args.artifact:
        manifest = Path(args.inventory).absolute()
        inventory = json.loads(bounded_read(manifest, manifest.parent))
        report["issues"] = check_artifact(policy, Path(args.artifact), inventory)
    elif args.build_from:
        manifest = Path(args.build_from).absolute()
        inventory = json.loads(bounded_read(manifest, manifest.parent))
        build_zip(policy, inventory, Path(args.output_zip))
        report["issues"] = check_artifact(policy, Path(args.output_zip), inventory)
    elif args.write_candidate_manifest:
        inventory = make_inventory(policy)
        Path(args.write_candidate_manifest).write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
        report.update(candidate_files=len(inventory["files"]), excluded_files=len(inventory["excluded"]))
        report["issues"] = [f"{row['path']}: {row['reason']}" for row in inventory["errors"]]
    else:
        paths = policy.candidate_paths if policy.candidate_paths is not None else candidates(root)
        if args.history:
            commits = subprocess.check_output(["git", "rev-list", "--all"], cwd=root).decode().splitlines()
            report["history_commits"] = commits
            report["refs"] = subprocess.check_output(["git", "for-each-ref", "--format=%(refname) %(objectname)"], cwd=root).decode().splitlines()
            paths = sorted({p for commit in commits for p in subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "-z", commit], cwd=root).decode("utf-8").split("\0") if p})
            if policy.profile == "public-workbench":
                report["issues"].extend(history_release_issues(root, commits))
        for path in paths:
            reason = policy.reason(path)
            if not reason and not args.history:
                try:
                    reason = policy.reason(path, policy.read(path), artifact=True)
                except (OSError, ValueError) as exc:
                    reason = str(exc)
            if reason:
                report["issues"].append(f"{path}: {reason}")
        report["checked_paths"] = len(paths)
        if not args.history:
            report["issues"].extend(policy.required_issues(set(paths)))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for issue in report["issues"][:20]:
        print(issue)
    print(f"{args.profile}: {len(report['issues'])} issues; {'BLOCKED' if report['issues'] else 'PASS for this scope only'}; no Git changes.")
    return 1 if report["issues"] else 0
