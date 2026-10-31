"""Pinned restored resource checks; no torch/pickle execution or old-install access."""
import hashlib
import json
from pathlib import Path
import pickletools
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_RESTORED_PATHS = {
    "docs/images/kairyth-showcase.png",
    "kimodo_runtime/src/kimodo/assets/skeletons/somaskel30/joints.p",
    "kimodo_runtime/src/kimodo/assets/skeletons/somaskel77/bvh_joints.p",
    "kimodo_runtime/src/kimodo/assets/skeletons/somaskel77/joints.p",
    "kimodo_runtime/src/kimodo/assets/skeletons/somaskel77/relaxed_hands_rest_pose.npy",
    "kimodo_runtime/src/kimodo/assets/skeletons/somaskel77/skin_standard.npz",
    "kimodo_runtime/src/kimodo/assets/skeletons/somaskel77/standard_t_pose_global_offsets_rots.p",
    "motionbuilder_app/assets/skeletons/somaskel30/joints.npy",
    "motionbuilder_app/static/ICON.png",
    "motionbuilder_app/static/ICON2.png",
    "motionbuilder_app/static/KMD_ICON.png",
    "motionbuilder_app/static/kairyth.png",
}
USER_CONFIRMED_PROJECT_SVGS = {
    "motionbuilder_app/static/kairyth_wordmark_clean.svg":
        "49facebd9d2f6f941c4f1436dd579cfcd6e4149c45b75730faf62d8e870b3c8e",
    "motionbuilder_app/static/grs_cube_vectorized_white.svg":
        "0620e7583272f16b85565ad5da59a60706b08fcdc06d183af860a17c3032cb1d",
}


def test_restored_assets_match_reviewed_hashes():
    manifest = json.loads((ROOT / "docs/resource-license-evidence.json").read_text(encoding="utf-8"))
    restored = [row for row in manifest["resources"] if row["publication"] in {
        "redistributable-under-upstream-apache", "user-authorized-first-party",
    }]
    assert len(ORIGINAL_RESTORED_PATHS) == 12
    expected_paths = ORIGINAL_RESTORED_PATHS | set(USER_CONFIRMED_PROJECT_SVGS)
    assert len(restored) == 14
    assert {row["path"] for row in restored} == expected_paths
    for row in restored:
        path = ROOT / row["path"]
        if row["path"] in USER_CONFIRMED_PROJECT_SVGS:
            assert row["sha256"] == USER_CONFIRMED_PROJECT_SVGS[row["path"]]
        assert path.is_file(), row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["path"]


def test_existing_numpy_joints_are_exact_official_float32_conversion():
    source = ROOT / "kimodo_runtime/src/kimodo/assets/skeletons/somaskel30/joints.p"
    target = ROOT / "motionbuilder_app/assets/skeletons/somaskel30/joints.npy"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "f8582822d4f7cdd7c9cbb5ee8003f07d3116155fc49be11ed125d373d13e4e8a"
    )
    with zipfile.ZipFile(source) as archive:
        # Inspect serialization metadata, never execute pickle GLOBAL/REDUCE.
        ops = list(pickletools.genops(archive.read("joints/data.pkl")))
        assert {arg for op, arg, _ in ops if op.name == "GLOBAL"} == {
            "torch._utils _rebuild_tensor_v2", "torch DoubleStorage", "collections OrderedDict",
        }
        assert [arg for op, arg, _ in ops if op.name == "BININT1"] == [90, 0, 30, 3, 3, 1]
        assert archive.read("joints/byteorder") == b"little"
        raw = archive.read("joints/data/0")
    assert len(raw) == 90 * 8
    official = np.frombuffer(raw, dtype="<f8").reshape(30, 3)
    existing = np.load(target, allow_pickle=False)
    assert existing.shape == (30, 3)
    assert existing.dtype == np.float32
    assert np.array_equal(official.astype(np.float32), existing)
    assert not np.array_equal(official, existing)  # Document rounding; do not claim identical dtypes.
    assert np.max(np.abs(official - existing)) <= 2.914e-8
