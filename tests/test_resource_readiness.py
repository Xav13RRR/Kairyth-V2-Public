"""Synthetic, offline checks: no model package or original data is loaded."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_source(name, path, package=False):
    kwargs = {"submodule_search_locations": [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **kwargs)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load_source("v2_resource_checker", ROOT / "scripts/resource_readiness.py")


def fixture_manifest(payload=b"synthetic fixture, not a skeleton"):
    return {"resources": [{
        "path": "motionbuilder_app/assets/skeletons/somaskel30/joints.npy",
        "sha256": hashlib.sha256(payload).hexdigest(), "feature": "core",
        "required": True, "rights_status": "review-required",
        "publication": "held-notpublish", "source": "https://example.invalid/evidence",
        "acquisition": "Review rights before supplying exact existing data",
    }]}


class ResourceReadinessTests(unittest.TestCase):
    def test_missing_is_explicit_not_replaced(self):
        with tempfile.TemporaryDirectory(prefix="v2-resource-test-") as folder:
            report = checker.inspect_resources(Path(folder), fixture_manifest())
            self.assertFalse(report["physical_ready"])
            self.assertEqual(report["resources"][0]["physical"], "missing")
            self.assertEqual(report["resources"][0]["integrity"], "not-checked")
            self.assertEqual(list(Path(folder).iterdir()), [])
            self.assertIn("UI can still start", checker.readiness_notice(report))

    def test_manual_path_hash_and_rights_are_independent(self):
        with tempfile.TemporaryDirectory(prefix="v2-resource-test-") as folder:
            file = Path(folder) / "existing.dat"
            file.write_bytes(b"synthetic fixture, not a skeleton")
            manifest = fixture_manifest()
            report = checker.inspect_resources(Path(folder), manifest,
                manual_paths={manifest["resources"][0]["path"]: str(file)})
            self.assertTrue(report["physical_ready"])
            self.assertTrue(report["integrity_ready"])
            self.assertFalse(report["permission_granted_by_tool"])
            self.assertEqual(report["resources"][0]["rights"], "review-required")
            self.assertEqual(report["resources"][0]["path_origin"], "manual-check-only")

    def test_mismatch_is_not_ready(self):
        with tempfile.TemporaryDirectory(prefix="v2-resource-test-") as folder:
            file = Path(folder) / "different.dat"
            file.write_bytes(b"different")
            manifest = fixture_manifest()
            report = checker.inspect_resources(Path(folder), manifest,
                manual_paths={manifest["resources"][0]["path"]: str(file)})
            self.assertTrue(report["physical_ready"])
            self.assertFalse(report["integrity_ready"])
            self.assertEqual(report["resources"][0]["integrity"], "mismatch")

    def test_unknown_map_and_traversal_rejected(self):
        with self.assertRaises(ValueError):
            checker.inspect_resources(ROOT, fixture_manifest(), manual_paths={"unknown": "C:/unknown"})
        manifest = fixture_manifest()
        manifest["resources"][0]["path"] = "../outside"
        with self.assertRaises(ValueError):
            checker.inspect_resources(ROOT, manifest)
        for path in ("..\\outside", "a//b", "./a", "a/../b", "C:/outside", "/outside"):
            manifest["resources"][0]["path"] = path
            with self.assertRaises(ValueError):
                checker.inspect_resources(ROOT, manifest)

    def test_presence_check_never_reads_file_contents(self):
        with tempfile.TemporaryDirectory(prefix="v2-resource-test-") as folder:
            file = Path(folder) / "existing.dat"
            file.write_bytes(b"not a model")
            manifest = fixture_manifest()
            with patch.object(checker, "file_sha256", side_effect=AssertionError("must not hash")):
                report = checker.inspect_resources(Path(folder), manifest, verify_hashes=False,
                    manual_paths={manifest["resources"][0]["path"]: str(file)})
            self.assertTrue(report["physical_ready"])
            self.assertFalse(report["integrity_ready"])
            self.assertEqual(report["resources"][0]["integrity"], "not-checked")

    def test_unversioned_file_is_not_hashed(self):
        with tempfile.TemporaryDirectory(prefix="v2-resource-test-") as folder:
            file = Path(folder) / "existing.dat"
            file.write_bytes(b"not a model")
            manifest = fixture_manifest()
            manifest["resources"][0]["sha256"] = None
            with patch.object(checker, "file_sha256", side_effect=AssertionError("must not hash")):
                report = checker.inspect_resources(Path(folder), manifest,
                    manual_paths={manifest["resources"][0]["path"]: str(file)})
            self.assertEqual(report["resources"][0]["integrity"], "unversioned")
            self.assertFalse(report["integrity_ready"])

    def test_malformed_schema_rejected(self):
        for manifest in ({}, {"resources": "not-list"}, {"resources": [{}]}):
            with self.assertRaises(ValueError):
                checker.inspect_resources(ROOT, manifest)
        manifest = fixture_manifest()
        manifest["resources"][0]["sha256"] = "invalid"
        with self.assertRaises(ValueError):
            checker.inspect_resources(ROOT, manifest)

    def test_external_skeleton_root_is_check_only(self):
        with tempfile.TemporaryDirectory(prefix="v2-resource-test-") as folder:
            report = checker.inspect_resources(ROOT, fixture_manifest(), skeleton_root=Path(folder))
            row = report["resources"][0]
            self.assertEqual(Path(row["resolved_path"]), Path(folder) / "somaskel30/joints.npy")
            self.assertEqual(row["path_origin"], "explicit-skeleton-root")
            self.assertFalse(Path(row["resolved_path"]).exists())

    def test_core_resolver_gives_missing_resource_guidance(self):
        with tempfile.TemporaryDirectory(prefix="v2-assets-test-") as folder:
            with patch.dict("os.environ", {"KAIRYTH_SKELETONS_ROOT": folder}):
                assets = load_source("v2_test_assets", ROOT / "motionbuilder_app/assets.py")
            self.assertEqual(assets.skeleton_asset_path("somaskel30"), Path(folder) / "somaskel30")
            with self.assertRaisesRegex(FileNotFoundError, "RESOURCE_READINESS"):
                assets.skeleton_asset_path("somaskel30", "joints.npy")




if __name__ == "__main__":
    unittest.main()
