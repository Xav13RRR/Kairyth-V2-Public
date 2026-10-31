from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import motionbuilder_app.pose_store as pose_store_module


def test_pose_from_motion_extracts_exact_frame_without_mutating_source() -> None:
    original_root = pose_store_module.POSES_ROOT
    try:
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pose_store_module.POSES_ROOT = temp_root / "pose_assets"
            source = temp_root / "motion.npz"
            parents = np.asarray([-1, 0, 1], dtype=np.int64)
            names = np.asarray(["Root", "Spine", "Head"], dtype="U64")
            bind = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
            world = np.tile(np.eye(4, dtype=np.float32), (3, 3, 1, 1))
            world[0, :, 0, 3] = [0.0, 0.1, 0.2]
            world[1, :, 0, 3] = [1.0, 1.1, 1.2]
            world[2, :, 0, 3] = [2.0, 2.1, 2.2]
            np.savez_compressed(
                source,
                world_motion=world,
                target_bind=bind,
                parents=parents,
                names=names,
                fps=np.asarray([30.0], dtype=np.float32),
                skeleton_type=np.asarray(["TEST-SKELETON"], dtype="U32"),
            )
            source_before = source.read_bytes()

            store = pose_store_module.PoseStore()
            created = store.create_from_motion_bundle(
                source,
                source_kind="kimodo",
                source_id="motion-123",
                source_name="Walk",
                frame=1,
                name="Walk Contact Pose",
            )

            assert source.read_bytes() == source_before
            assert created["name"] == "Walk Contact Pose"
            assert created["source_motion_id"] == "motion-123"
            assert created["source_frame"] == 1
            with np.load(store.asset_path(created["id"]), allow_pickle=False) as data:
                assert data["world_motion"].shape == (1, 3, 4, 4)
                np.testing.assert_allclose(data["world_motion"][0], world[1])
                assert int(data["num_frames"][0]) == 1
                assert str(data["skeleton_type"][0]) == "TEST-SKELETON"
            assert store.constraints(created["id"])[0]["frame_indices"] == [0]
    finally:
        pose_store_module.POSES_ROOT = original_root
