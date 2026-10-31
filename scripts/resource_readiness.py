"""Offline resource evidence check. Does not download, deserialize or grant rights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "docs" / "resource-license-evidence.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_resources(
    root: Path,
    manifest: dict,
    *,
    features: set[str] | None = None,
    manual_paths: dict[str, str] | None = None,
    skeleton_root: Path | None = None,
    verify_hashes: bool = True,
) -> dict:
    """Check exact bytes separately from rights. Only explicit manual paths escape root."""
    root = root.resolve()
    if manual_paths is None:
        manual_paths = {}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("resources"), list):
        raise ValueError("Evidence manifest must contain a resources list")
    if not all(isinstance(entry, dict) and isinstance(entry.get("path"), str)
               for entry in manifest["resources"]):
        raise ValueError("Each resource must have a normalized relative path string")
    for entry in manifest["resources"]:
        if not all(isinstance(entry.get(key), str) for key in (
                "feature", "rights_status", "publication", "acquisition")):
            raise ValueError("Resource feature/rights/publication/acquisition must be strings")
        if not isinstance(entry.get("required"), bool):
            raise ValueError("Resource required must be a boolean")
        expected = entry.get("sha256")
        if expected is not None and (not isinstance(expected, str)
                                     or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None):
            raise ValueError("Resource sha256 must be null or exactly 64 hexadecimal characters")
    if not isinstance(manual_paths, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in manual_paths.items()):
        raise ValueError("Manual map must map manifest paths to absolute local path strings")
    known_paths = {entry["path"] for entry in manifest["resources"]}
    if set(manual_paths) - known_paths:
        raise ValueError("Manual map contains paths absent from the evidence manifest")
    rows = []
    for entry in manifest["resources"]:
        if features and "all" not in features and entry["feature"] not in features:
            continue
        relative = PurePosixPath(entry["path"])
        if (relative.is_absolute() or ".." in relative.parts or ":" in entry["path"]
                or "\\" in entry["path"] or not relative.parts
                or relative.as_posix() != entry["path"]
                or any(part in {".", ".."} for part in entry["path"].split("/"))):
            raise ValueError("Unsafe relative path in evidence manifest")
        path = root.joinpath(*relative.parts)
        origin = "repository"
        prefix = "motionbuilder_app/assets/skeletons/"
        if skeleton_root is not None and entry["path"].startswith(prefix):
            path = skeleton_root.joinpath(*PurePosixPath(entry["path"][len(prefix):]).parts)
            origin = "explicit-skeleton-root"
        if entry["path"] in manual_paths:
            path = Path(manual_paths[entry["path"]])
            if not path.is_absolute():
                raise ValueError("Manual map targets must be absolute local paths")
            origin = "manual-check-only"
        physical = "missing"
        integrity = "not-checked"
        error = None
        if path.is_file():
            physical = "present"
            expected = entry.get("sha256")
            if not expected:
                integrity = "unversioned"  # Do not read a large unpinned file unnecessarily.
            elif verify_hashes:
                try:
                    actual = file_sha256(path)
                    integrity = "match" if actual.lower() == expected.lower() else "mismatch"
                except OSError as exc:
                    integrity = "unreadable"
                    error = str(exc)
        row = {
            "path": entry["path"], "feature": entry["feature"],
            "required": entry["required"], "physical": physical,
            "integrity": integrity, "rights": entry["rights_status"],
            "publication": entry["publication"], "source": entry.get("source"),
            "action": entry["acquisition"], "resolved_path": str(path),
            "path_origin": origin,
        }
        if error:
            row["error"] = error
        rows.append(row)
    required = [row for row in rows if row["required"]]
    return {
        "scope": "Targeted evidence inventory only; not a full dependency/GPU or legal acceptance test",
        "offline": True,
        "verification_mode": "sha256" if verify_hashes else "presence-only",
        "physical_ready": bool(required) and all(row["physical"] == "present" for row in required),
        "integrity_ready": bool(required) and all(row["integrity"] == "match" for row in required),
        "rights": "individual-review-required; existence/hash-match never grants permission",
        "required_policy": "Conservative dependency inventory, not a hard runtime/UI enablement gate",
        "permission_granted_by_tool": False,
        "required_count": len(required),
        "missing_required_count": sum(row["physical"] != "present" for row in required),
        "resources": rows,
    }


def readiness_notice(report: dict) -> str:
    return (
        f"Resource check / 资源检查: {report['missing_required_count']}/"
        f"{report['required_count']} required assets missing / 必需资源缺失. "
        "Only affected features may be unavailable; the UI can still start. / "
        "仅相关功能可能不可用，界面仍可启动。\n"
        "Presence and SHA256 do not grant a license. No download or terms acceptance occurs. / "
        "文件存在及哈希匹配不代表获权；此检查不下载、不代接受协议。\n"
        "See / 详见 docs/RESOURCE_READINESS.md."
        f" Hash mode / 哈希模式: {report['verification_mode']}."
    )


def main(argv: list[str] | None = None) -> int:
    # Keep bilingual diagnostics readable when Windows redirects stdout to pipes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature", action="append", help="core, kimodo, hybrik, gvhmr, skintokens, unirig, ui, all")
    parser.add_argument("--manual-map", type=Path, help="JSON: manifest path -> absolute authorized local path; check only")
    parser.add_argument("--skeleton-root", type=Path, help="Core skeleton directory, equivalent to KAIRYTH_SKELETONS_ROOT")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--presence-only", action="store_true", help="Fast advisory startup check; hashes remain not-checked")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        manual = json.loads(args.manual_map.read_text(encoding="utf-8-sig")) if args.manual_map else {}
        if not isinstance(manual, dict) or not all(isinstance(value, str) for value in manual.values()):
            raise ValueError("Manual map must be a JSON object of local path strings")
        features = set(args.feature or ["core"])
        known = {entry["feature"] for entry in manifest["resources"]} | {"all"}
        if features - known:
            raise ValueError("Unknown feature: " + ", ".join(sorted(features - known)))
        external = args.skeleton_root
        if external is None and os.environ.get("KAIRYTH_SKELETONS_ROOT"):
            external = Path(os.environ["KAIRYTH_SKELETONS_ROOT"])
        report = inspect_resources(args.repo_root, manifest, features=features,
                                   manual_paths=manual, skeleton_root=external,
                                   verify_hashes=not args.presence_only)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(readiness_notice(report))
        for row in report["resources"]:
            if row["required"]:
                print(f"{row['path']}: physical={row['physical']}; hash={row['integrity']}; rights={row['rights']}")
                if row["physical"] == "missing":
                    print("  " + row["action"])
                    if row["source"]:
                        print("  Official reference (not opened): " + row["source"])
    # Advisory status for callers; do not turn this into a global UI launch gate.
    ready_for_check = report["physical_ready"] and (args.presence_only or report["integrity_ready"])
    return 0 if ready_for_check else 2


if __name__ == "__main__":
    raise SystemExit(main())
