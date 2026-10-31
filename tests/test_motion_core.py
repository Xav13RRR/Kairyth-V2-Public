from pathlib import Path
import threading

import numpy as np
from scipy.spatial.transform import Rotation

from motionbuilder_app.edit_tracks import EditOperation, EditProject
from motionbuilder_app.meshpreps.bind_math import BindingResult, build_bind_matrices
from motionbuilder_app.meshpreps.binding_io import save_binding
from motionbuilder_app.meshpreps.session import BindJob, JobPhase, JobState, SessionStore
from motionbuilder_app.motion_math import local_from_world, world_from_local
from motionbuilder_app.retarget import build_source_motion_bundle, retarget_extracted, retarget_source_bundle_fixed
from motionbuilder_app.rigging.soma_fit import _ensure_left_right, _split_left_right
from motionbuilder_app.skeleton import SOMASkeleton30
from motionbuilder_app import stores
from motionbuilder_app import library_stores
from motionbuilder_app.library_stores import AnimationStore, CompositionStore, RetargetPreviewStore, UpperLowerCompositePreviewStore
from motionbuilder_app.stores import CharacterStore, MotionStore
from motionbuilder_app.timeline import apply_ordered_cuts, blend_clip_boundary, segment_world
from motionbuilder_app.virtual_root import build_virtual_root


def _binding(path: Path) -> Path:
    skeleton = SOMASkeleton30(load=True)
    joints = skeleton.neutral_joints
    parents = skeleton.joint_parents
    vertices = np.array([[-.2, 0, 0], [.2, 0, 0], [0, 1, 0], [0, .5, .2]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    indices = np.zeros((4, 4), dtype=np.int64)
    weights = np.zeros((4, 4), dtype=np.float32)
    weights[:, 0] = 1
    result = BindingResult(vertices, faces, joints, parents, skeleton.bone_order_names, indices, weights, build_bind_matrices(joints, parents), None, "test")
    return save_binding(path, result)


def test_local_world_roundtrip() -> None:
    parents = np.array([-1, 0, 1], dtype=np.int64)
    local = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    local[:, :3, 3] = [[0, 1, 0], [0, .5, 0], [.2, 0, 0]]
    world = world_from_local(local, parents)
    np.testing.assert_allclose(local_from_world(world, parents), local, atol=1e-6)


def test_soma_left_right_convention_uses_positive_x() -> None:
    joints = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    left, right = _split_left_right(joints, 0.0, 1, 2)
    assert (left, right) == (2, 1)
    left, right = _ensure_left_right(joints, 0.0, 1, 2)
    assert (left, right) == (2, 1)

    skeleton = SOMASkeleton30(load=True)
    names = skeleton.bone_order_names
    rest = np.asarray(skeleton.neutral_joints)
    assert rest[names.index("LeftHand"), 0] > rest[names.index("RightHand"), 0]
    assert rest[names.index("LeftFoot"), 0] > rest[names.index("RightFoot"), 0]


def test_timeline_mirror_reflects_around_hips_and_swaps_left_right() -> None:
    names = ["Hips", "LeftHand", "RightHand"]
    world = np.repeat(np.eye(4, dtype=np.float32)[None, None], 3, axis=1)
    world[0, 0, :3, 3] = [1.0, 1.0, 0.0]
    world[0, 1, :3, 3] = [2.0, 1.2, 0.0]
    world[0, 2, :3, 3] = [0.6, 1.3, 0.0]
    mirrored = CompositionStore.mirror_world_clip_around_hip(world, names)
    np.testing.assert_allclose(mirrored[0, 0, :3, 3], [1.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(mirrored[0, 1, :3, 3], [1.4, 1.3, 0.0], atol=1e-6)
    np.testing.assert_allclose(mirrored[0, 2, :3, 3], [0.0, 1.2, 0.0], atol=1e-6)


def test_retarget_world_and_inplace(tmp_path: Path) -> None:
    binding = _binding(tmp_path / "binding.npz")
    source_rest = np.zeros((22, 3), dtype=np.float32)
    source_rest[7] = [-.1, -1, 0]
    source_rest[8] = [.1, -1, 0]
    np.savez_compressed(
        tmp_path / "source.npz", body_pose=np.zeros((3, 63), np.float32),
        global_orient=np.zeros((3, 3), np.float32), transl=np.array([[0, 0, 0], [1, .1, 0], [2, .2, 0]], np.float32),
        source_rest_joints=source_rest, fps=np.array([30], np.float32),
    )
    world_path = retarget_extracted(tmp_path / "source.npz", binding, tmp_path / "world.npz", root_mode="world")
    in_path = retarget_extracted(tmp_path / "source.npz", binding, tmp_path / "in.npz", root_mode="inplace")
    world = np.load(world_path)["world_motion"]
    inplace = np.load(in_path)["world_motion"]
    assert world.shape == (3, 30, 4, 4)
    assert world[-1, 0, 0, 3] > world[0, 0, 0, 3]
    expected_xz = np.repeat(inplace[0, 0, [0, 2], 3][None], 3, axis=0)
    np.testing.assert_allclose(inplace[:, 0, [0, 2], 3], expected_xz, atol=1e-6)


def test_keyframe_slerp_and_range(tmp_path: Path) -> None:
    parents = np.array([-1], dtype=np.int64)
    base = np.repeat(np.eye(4, dtype=np.float32)[None, None], 5, axis=0)
    project = EditProject(tmp_path / "edits.json", 5, 1)
    q0 = Rotation.from_euler("z", 0, degrees=True).as_quat().tolist()
    q1 = Rotation.from_euler("z", 90, degrees=True).as_quat().tolist()
    project.add(EditOperation.create(joint=0, scope="frame", start_frame=0, end_frame=0, blend_frames=0, translation=[0, 0, 0], rotation=q0))
    project.add(EditOperation.create(joint=0, scope="frame", start_frame=4, end_frame=4, blend_frames=0, translation=[4, 0, 0], rotation=q1))
    result = project.apply(base, parents)
    assert np.isclose(result[2, 0, 0, 3], 2.0)
    angle = Rotation.from_matrix(result[2, 0, :3, :3]).as_euler("xyz", degrees=True)[2]
    assert np.isclose(angle, 45, atol=1)


def test_single_frame_edit_obeys_blend_in_and_out(tmp_path: Path) -> None:
    parents = np.array([-1], dtype=np.int64)
    base = np.repeat(np.eye(4, dtype=np.float32)[None, None], 9, axis=0)
    project = EditProject(tmp_path / "frame_blend_edits.json", 9, 1)
    rotation = Rotation.from_euler("z", 90, degrees=True).as_quat().tolist()
    project.add(EditOperation.create(
        joint=0,
        scope="frame",
        start_frame=4,
        end_frame=4,
        blend_frames=0,
        blend_in_frames=2,
        blend_out_frames=3,
        translation=[1, 0, 0],
        rotation=rotation,
    ))

    result = project.apply(base, parents)
    expected_weights = np.array([0, 0, 0, .5, 1, .75, .25, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(result[:, 0, 0, 3], expected_weights, atol=1e-6)
    angles = Rotation.from_matrix(result[:, 0, :3, :3]).as_euler("xyz", degrees=True)[:, 2]
    np.testing.assert_allclose(angles, expected_weights * 90.0, atol=1e-4)


def test_composition_rename_and_commit_preserve_virtual_root_space(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(library_stores, "COMPOSITIONS_ROOT", tmp_path / "compositions")
    parents = np.asarray([-1], dtype=np.int64)
    local_world = np.repeat(np.eye(4, dtype=np.float32)[None, None], 3, axis=0)
    local_world[:, 0, 1, 3] = 1.0
    virtual_root = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    virtual_root[:, 0, 3] = [0.0, 1.0, 2.0]
    source_path = tmp_path / "source_virtual_root.npz"
    np.savez_compressed(
        source_path,
        world_motion=local_world,
        virtual_root_motion=virtual_root,
        root_motion=virtual_root[:, :3, 3] - virtual_root[[0], :3, 3],
        parents=parents,
        names=np.asarray(["Hips"], dtype="U32"),
        fps=np.asarray([30], dtype=np.float32),
        num_frames=np.asarray([3], dtype=np.int64),
    )

    class Characters:
        def get(self, character_id):
            return {"id": character_id}

    class Animations:
        characters = Characters()

        def __init__(self) -> None:
            self.saved = None

        def get(self, animation_id):
            return {"id": animation_id, "character_id": "character", "frames": 3, "fps": 30.0, "virtual_root": True}

        def effective_world(self, animation_id):
            data = np.load(source_path, allow_pickle=False)
            return data, np.asarray(data["world_motion"], dtype=np.float32)

        def root_motion(self, animation_id, data):
            return np.asarray(data["root_motion"], dtype=np.float32)

        def virtual_root_motion(self, data):
            return np.asarray(data["virtual_root_motion"], dtype=np.float32)

        def create_from_payload(self, character_id, name, payload, **kwargs):
            self.saved = {key: np.asarray(value).copy() for key, value in payload.items()}
            return {
                "id": "committed",
                "character_id": character_id,
                "name": name,
                "virtual_root": "virtual_root_motion" in payload,
                **kwargs,
            }

    animations = Animations()
    store = CompositionStore(animations)
    composition = store.create("character", "final composition")
    row = store.update(
        composition["id"],
        name=None,
        segments=[{
            "id": "segment-a",
            "animation_id": "source",
            "start_frame": 0,
            "end_frame": 2,
            "speed": 1.0,
            "root_motion": True,
            "display_name": "old name",
        }],
        cuts=[],
    )
    before = np.load(store.compile(composition["id"]), allow_pickle=False)["world_motion"].copy()
    renamed = store.rename_segments(composition["id"], [row["segments"][0]["id"]], "renamed clip")
    assert renamed["segments"][0]["display_name"] == "renamed clip"
    after = np.load(store.compile(composition["id"]), allow_pickle=False)["world_motion"].copy()
    np.testing.assert_allclose(after, before, atol=1e-6)

    project = EditProject(library_stores.COMPOSITIONS_ROOT / composition["id"] / "edits.json", 3, 1)
    project.add(EditOperation.create(
        joint=0,
        scope="all",
        start_frame=0,
        end_frame=2,
        blend_frames=0,
        translation=[0.0, 0.5, 0.0],
        rotation=[0.0, 0.0, 0.0, 1.0],
    ))
    expected_global = project.apply(after, parents)
    committed = store.commit_to_animation(composition["id"], "final baked")
    assert committed["virtual_root"] is True
    assert committed["motion_kind"] == "composition_baked"
    assert committed["hidden"] is False
    saved = animations.saved
    reconstructed_global = np.einsum(
        "fab,fjbc->fjac",
        saved["virtual_root_motion"],
        saved["world_motion"],
    )
    np.testing.assert_allclose(reconstructed_global, expected_global, atol=1e-5)


def test_weights_are_normalized(tmp_path: Path) -> None:
    data = np.load(_binding(tmp_path / "binding.npz"), allow_pickle=True)
    np.testing.assert_allclose(data["lbs_weights"].sum(axis=1), 1.0)


def test_character_rename_persists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(stores, "CHARACTERS_ROOT", tmp_path / "characters")
    character_store = CharacterStore()
    character = character_store.create("旧名称", _binding(tmp_path / "binding.npz"))
    renamed = character_store.rename(character["id"], "  新名称  ")
    assert renamed["name"] == "新名称"
    assert character_store.get(character["id"])["name"] == "新名称"
    assert character_store.export_path(character["id"], "fbx").name.endswith("_binding.fbx")


def test_camera_solver_failure_detection() -> None:
    job = {"log": ["File solver_two_view.py, line 105", "AttributeError: 'NoneType' object has no attribute 'matrix'"]}
    assert MotionStore._camera_solver_failed(job)
    assert not MotionStore._camera_solver_failed({"log": ["CUDA out of memory"]})


def test_auto_binding_advances_to_skin(monkeypatch) -> None:
    session_store = SessionStore()
    job = BindJob(job_id="auto-job", mesh_id="mesh")

    def finish_skeleton() -> None:
        job.phase = JobPhase.SKELETON
        job.state = JobState.DONE
        job.progress = 48

    skeleton_thread = threading.Thread(target=finish_skeleton)
    job._thread = skeleton_thread
    skeleton_thread.start()
    monkeypatch.setattr(session_store, "start_bind", lambda *args, **kwargs: job)

    def accept(job_id: str) -> BindJob:
        assert job_id == job.job_id
        job.phase = JobPhase.SOMA_EDIT
        return job

    def skin(job_id: str, *, timeout_s: float) -> BindJob:
        assert job_id == job.job_id
        job.phase = JobPhase.SKIN
        job.state = JobState.RUNNING
        return job

    monkeypatch.setattr(session_store, "accept_skeleton", accept)
    monkeypatch.setattr(session_store, "confirm_and_skin", skin)
    result = session_store.start_auto_bind("mesh")
    result._automation_thread.join(timeout=2)
    assert result.automatic
    assert result.phase == JobPhase.SKIN
    assert result.state == JobState.RUNNING


def test_fixed_length_retarget_keeps_hips_and_bone_offsets(tmp_path: Path) -> None:
    binding_path = _binding(tmp_path / "binding.npz")
    source_rest = np.zeros((22, 3), dtype=np.float32)
    for joint in range(1, 22):
        source_rest[joint] = source_rest[max(0, joint - 1)] + [0.0, 0.1, 0.0]
    body_pose = np.zeros((3, 63), dtype=np.float32)
    body_pose[1, 2] = np.pi / 4
    np.savez_compressed(
        tmp_path / "extracted.npz",
        body_pose=body_pose,
        global_orient=np.zeros((3, 3), dtype=np.float32),
        transl=np.array([[0, 0, 0], [2, 1, 3], [4, 2, 6]], dtype=np.float32),
        source_rest_joints=source_rest,
        fps=np.array([30], dtype=np.float32),
    )
    source_bundle = build_source_motion_bundle(tmp_path / "extracted.npz", tmp_path / "source.npz")
    output = retarget_source_bundle_fixed(source_bundle, binding_path, tmp_path / "retargeted.npz")
    data = np.load(output, allow_pickle=False)
    local = local_from_world(data["world_motion"], data["parents"])
    bind_local = local_from_world(data["target_bind"], data["parents"])
    expected_offsets = np.repeat(bind_local[None, :, :3, 3], len(local), axis=0)
    np.testing.assert_allclose(local[:, :, :3, 3], expected_offsets, atol=1e-6)
    expected_hips = np.repeat(data["target_bind"][None, 0, :3, 3], len(local), axis=0)
    np.testing.assert_allclose(data["world_motion"][:, 0, :3, 3], expected_hips, atol=1e-6)


def test_timeline_speed_reverse_and_ordered_cut() -> None:
    world = np.repeat(np.eye(4, dtype=np.float32)[None, None], 5, axis=0)
    world[:, 0, 0, 3] = np.arange(5)
    clip = segment_world(world, start_frame=0, end_frame=4, speed=2.0, reverse=True, source_fps=30, output_fps=30)
    assert len(clip) == 2
    assert np.isclose(clip[0, 0, 0, 3], 4)
    assert np.isclose(clip[-1, 0, 0, 3], 0)
    cut = apply_ordered_cuts(world, [{"start_frame": 1, "end_frame": 2}])
    np.testing.assert_allclose(cut[:, 0, 0, 3], [0, 2, 3, 4])


def test_locked_source_motion_keeps_root_position(tmp_path: Path) -> None:
    source_rest = np.zeros((22, 3), dtype=np.float32)
    np.savez_compressed(
        tmp_path / "source_extracted.npz",
        body_pose=np.zeros((4, 63), dtype=np.float32),
        global_orient=np.zeros((4, 3), dtype=np.float32),
        transl=np.arange(12, dtype=np.float32).reshape(4, 3),
        source_rest_joints=source_rest,
        fps=np.array([30], dtype=np.float32),
    )
    path = build_source_motion_bundle(tmp_path / "source_extracted.npz", tmp_path / "locked.npz", root_mode="locked")
    data = np.load(path, allow_pickle=False)
    expected = np.repeat(data["world_motion"][0:1, 0, :3, 3], 4, axis=0)
    np.testing.assert_allclose(data["world_motion"][:, 0, :3, 3], expected)


def test_separate_blend_in_and_out_frames(tmp_path: Path) -> None:
    parents = np.array([-1], dtype=np.int64)
    base = np.repeat(np.eye(4, dtype=np.float32)[None, None], 11, axis=0)
    project = EditProject(tmp_path / "blend_edits.json", 11, 1)
    project.add(EditOperation.create(
        joint=0,
        scope="range",
        start_frame=2,
        end_frame=8,
        blend_frames=0,
        blend_in_frames=2,
        blend_out_frames=3,
        translation=[1, 0, 0],
        rotation=[0, 0, 0, 1],
    ))
    x = project.apply(base, parents)[:, 0, 0, 3]
    np.testing.assert_allclose(x, [0, .5, 1, 1, 1, 1, 1, 1, 1, .75, .25], atol=1e-6)


def test_edit_blend_resize_and_applied_layer(tmp_path: Path) -> None:
    parents = np.array([-1], dtype=np.int64)
    base = np.repeat(np.eye(4, dtype=np.float32)[None, None], 20, axis=0)
    editable = EditProject(tmp_path / "edits.json", 20, 1)
    operation = editable.add(EditOperation.create(
        joint=0, scope="range", start_frame=7, end_frame=9, blend_frames=0,
        blend_in_frames=2, blend_out_frames=2,
        translation=[1, 0, 0], rotation=[0, 0, 0, 1],
    ))
    resized = editable.update_blend(operation.id, 5, 6)
    assert (resized.blend_in_frames, resized.blend_out_frames) == (5, 6)

    baked = EditProject(tmp_path / "baked_edits.json", 20, 1)
    baked.add(editable.take(operation.id))
    editable.clear()
    assert editable.operations == []
    x = baked.apply(base, parents)[:, 0, 0, 3]
    assert np.isclose(x[7], 1.0) and np.isclose(x[9], 1.0)
    assert 0 < x[4] < 1 and 0 < x[12] < 1


def test_clip_boundary_blends_full_start_to_end_range() -> None:
    left = np.repeat(np.eye(4, dtype=np.float32)[None, None], 4, axis=0)
    right = np.repeat(np.eye(4, dtype=np.float32)[None, None], 3, axis=0)
    left[:, 0, 0, 3] = [0.0, 1.0, 2.0, 3.0]
    right[:, 0, 0, 3] = [10.0, 11.0, 12.0]

    blended_left, blended_right = blend_clip_boundary(left, right, np.asarray([-1]), 2, 2)
    np.testing.assert_allclose(blended_left[-2:, 0, 0, 3], [2.0, 4.25], atol=1e-6)
    np.testing.assert_allclose(blended_right[:2, 0, 0, 3], [8.75, 11.0], atol=1e-6)
    assert np.all(np.diff(np.concatenate([blended_left[-2:, 0, 0, 3], blended_right[:2, 0, 0, 3]])) > 0)


def test_composition_virtual_root_stitch_cross_clip_delete_and_reorder(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(library_stores, "COMPOSITIONS_ROOT", tmp_path / "compositions")

    def write_motion(name: str, frames: int, offset: float) -> Path:
        world = np.repeat(np.eye(4, dtype=np.float32)[None, None], frames, axis=0)
        root = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
        root[:, 0, 3] = offset + np.arange(frames, dtype=np.float32)
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(path, world_motion=world, virtual_root_motion=root, parents=np.array([-1]), fps=np.array([30], dtype=np.float32))
        return path

    paths = {"first": write_motion("first", 200, 0), "second": write_motion("second", 100, 1000), "third": write_motion("third", 100, 2000)}

    class Characters:
        def get(self, character_id): return {"id": character_id}

    class Animations:
        characters = Characters()

        def get(self, animation_id):
            return {"id": animation_id, "character_id": "character", "frames": 200 if animation_id == "first" else 100, "fps": 30.0, "virtual_root": True}

        def effective_world(self, animation_id):
            data = np.load(paths[animation_id], allow_pickle=False)
            return data, np.asarray(data["world_motion"], dtype=np.float32)

        def root_motion(self, animation_id, data):
            root = np.asarray(data["virtual_root_motion"], dtype=np.float32)
            return root[:, :3, 3] - root[[0], :3, 3]

        def virtual_root_motion(self, data):
            return np.asarray(data["virtual_root_motion"], dtype=np.float32)

    store = CompositionStore(Animations())
    composition = store.create("character", "splice")
    segments = [
        {"id": "a", "animation_id": "first", "start_frame": 0, "end_frame": 199, "speed": 1, "root_motion": True},
        {"id": "b", "animation_id": "second", "start_frame": 0, "end_frame": 99, "speed": 1, "root_motion": True},
    ]
    initial = store.update(composition["id"], name=None, segments=segments, cuts=[])
    assert initial["transitions"] == []
    with_transition = store.add_transition(composition["id"], "a", "b")
    assert len(with_transition["transitions"]) == 1
    assert (with_transition["transitions"][0]["left_frames"], with_transition["transitions"][0]["right_frames"]) == (10, 5)
    with np.load(store.compile(composition["id"]), allow_pickle=False) as composed:
        joined = composed["world_motion"]
        roots = composed["virtual_root_motion"]
    assert len(joined) == 300
    assert len(roots) == 300
    assert roots[199, 0, 3] < roots[200, 0, 3]
    np.testing.assert_allclose(roots[190, 0, 3], 190.0, atol=1e-5)
    np.testing.assert_allclose(roots[204, 0, 3], 203.0, atol=1e-5)

    cut = store.delete_frames(composition["id"], 150, 250)
    assert cut["invalidated_transitions"] == 1
    assert cut["transitions"] == []
    assert [(s["animation_id"], s["start_frame"], s["end_frame"]) for s in cut["segments"]] == [("first", 0, 149), ("second", 50, 99)]
    assert all(not s["deleted_ranges"] for s in cut["segments"])
    with np.load(store.compile(composition["id"]), allow_pickle=False) as composed:
        joined = composed["world_motion"]
        roots = composed["virtual_root_motion"]
    assert len(joined) == 200
    assert len(roots) == 200
    np.testing.assert_allclose(roots[149, :3, 3], roots[150, :3, 3])

    reordered = store.update(composition["id"], name=None, segments=cut["segments"][::-1], cuts=[])
    joined = np.load(store.compile(composition["id"]), allow_pickle=False)["world_motion"]
    assert [segment["animation_id"] for segment in reordered["segments"]] == ["second", "first"]
    assert reordered["transitions"] == []
    explicit = store.add_transition(composition["id"], reordered["segments"][0]["id"], reordered["segments"][1]["id"])
    assert (explicit["transitions"][0]["left_frames"], explicit["transitions"][0]["right_frames"]) == (3, 8)
    applied = store.apply_transition(composition["id"], explicit["transitions"][0]["id"])
    assert applied["transitions"] == [] and len(applied["baked_transitions"]) == 1
    grouped_segments = [dict(segment, group_id="merged-test") for segment in applied["segments"]]
    grouped = store.update(composition["id"], name=None, segments=grouped_segments, cuts=[])
    assert [segment["group_id"] for segment in grouped["segments"]] == ["merged-test", "merged-test"]
    np.testing.assert_allclose(joined[49, 0, :3, 3], joined[50, 0, :3, 3])

    propagated = store.create("character", "propagated offset")
    propagated_segments = [
        {"id": "p0", "animation_id": "first", "start_frame": 0, "end_frame": 9, "speed": 1, "root_motion": True},
        {"id": "p1", "animation_id": "second", "start_frame": 0, "end_frame": 9, "speed": 1, "root_motion": True},
        {"id": "p2", "animation_id": "second", "start_frame": 10, "end_frame": 19, "speed": 1, "root_motion": True},
    ]
    store.update(propagated["id"], name=None, segments=propagated_segments, cuts=[])
    with np.load(store.compile(propagated["id"]), allow_pickle=False) as data:
        propagated_roots = data["virtual_root_motion"][:, 0, 3]
    np.testing.assert_allclose(propagated_roots[9:11], [9.0, 9.0], atol=1e-5)
    np.testing.assert_allclose(propagated_roots[10:21], np.arange(9, 20, dtype=np.float32), atol=1e-5)

    triple = store.create("character", "transition occupancy")
    triple_segments = [
        {"id": "t0", "animation_id": "first", "start_frame": 0, "end_frame": 19, "speed": 1, "root_motion": True},
        {"id": "t1", "animation_id": "second", "start_frame": 0, "end_frame": 19, "speed": 1, "root_motion": True},
        {"id": "t2", "animation_id": "third", "start_frame": 0, "end_frame": 19, "speed": 1, "root_motion": True},
    ]
    store.update(triple["id"], name=None, segments=triple_segments, cuts=[])
    first_transition = store.add_transition(triple["id"], "t0", "t1")["transitions"][0]
    both = store.add_transition(triple["id"], "t1", "t2")
    second_transition = next(item for item in both["transitions"] if item["id"] != first_transition["id"])
    store.update_transition(triple["id"], first_transition["id"], 3, 18)
    constrained = store.update_transition(triple["id"], second_transition["id"], 18, 3)
    incoming = next(item for item in constrained["transitions"] if item["id"] == first_transition["id"])
    outgoing = next(item for item in constrained["transitions"] if item["id"] == second_transition["id"])
    assert incoming["right_frames"] + outgoing["left_frames"] <= 19
    before_apply = np.load(store.compile(triple["id"]), allow_pickle=False)
    before_world = before_apply["world_motion"].copy()
    before_root = before_apply["virtual_root_motion"].copy()
    before_apply.close()
    store.apply_transition(triple["id"], second_transition["id"])
    with np.load(store.compile(triple["id"]), allow_pickle=False) as after_apply:
        np.testing.assert_allclose(after_apply["world_motion"], before_world, atol=1e-5)
        np.testing.assert_allclose(after_apply["virtual_root_motion"], before_root, atol=1e-5)


def test_upper_lower_composite_preview_uses_lower_root_and_upper_spine(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(library_stores, "UPPER_LOWER_PREVIEWS_ROOT", tmp_path / "upper_lower_previews")
    names = np.asarray(["Hips", "Spine1", "LeftLeg"], dtype="U32")
    parents = np.asarray([-1, 0, 0], dtype=np.int64)
    base = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    base[1, 1, 3] = 1.0
    base[2, 1, 3] = -1.0
    lower_world = np.repeat(base[None], 2, axis=0)
    upper_world = np.repeat(base[None], 2, axis=0)
    upper_world[:, 1, 1, 3] = 2.0
    root = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    root[:, 0, 3] = [0.0, 10.0]

    def write_motion(path: Path, world: np.ndarray, virtual_root: np.ndarray) -> None:
        np.savez_compressed(
            path,
            rest_vertices=np.zeros((3, 3), dtype=np.float32),
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
            lbs_indices=np.zeros((3, 4), dtype=np.int64),
            lbs_weights=np.pad(np.ones((3, 1), dtype=np.float32), ((0, 0), (0, 3))),
            target_bind=base,
            world_motion=world,
            virtual_root_motion=virtual_root,
            root_motion=virtual_root[:, :3, 3] - virtual_root[[0], :3, 3],
            parents=parents,
            names=names,
            fps=np.asarray([30], dtype=np.float32),
            num_frames=np.asarray([2], dtype=np.int64),
        )

    paths = {"upper": tmp_path / "upper.npz", "lower": tmp_path / "lower.npz"}
    write_motion(paths["upper"], upper_world, root)
    write_motion(paths["lower"], lower_world, root)

    class Characters:
        def get(self, character_id): return {"id": character_id}

    class Animations:
        characters = Characters()

        def get(self, animation_id):
            return {"id": animation_id, "character_id": "character", "frames": 2, "fps": 30.0, "virtual_root": True}

        def effective_world(self, animation_id):
            data = np.load(paths[animation_id], allow_pickle=False)
            return data, np.asarray(data["world_motion"], dtype=np.float32)

        def virtual_root_motion(self, data):
            return np.asarray(data["virtual_root_motion"], dtype=np.float32)

        def create_from_payload(self, *args, **kwargs):
            raise AssertionError("commit is not part of this test")

    preview_store = UpperLowerCompositePreviewStore(Animations())
    preview = preview_store.create("character", "upper", "lower", 1.0, "combo")
    with np.load(preview_store.bundle_path(preview["id"]), allow_pickle=False) as data:
        assert data["world_motion"].shape[0] == 30
        np.testing.assert_allclose(data["virtual_root_motion"][-1, 0, 3], 10.0, atol=1e-5)
        local = local_from_world(data["world_motion"], data["parents"])
        np.testing.assert_allclose(local[:, 1, 1, 3], 2.0, atol=1e-5)
        np.testing.assert_allclose(local[:, 2, 1, 3], -1.0, atol=1e-5)


def test_retarget_preview_requires_commit_and_segment_root_motion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(library_stores, "ANIMATIONS_ROOT", tmp_path / "animations")
    monkeypatch.setattr(library_stores, "COMPOSITIONS_ROOT", tmp_path / "compositions")
    monkeypatch.setattr(library_stores, "RETARGET_PREVIEWS_ROOT", tmp_path / "previews")
    binding = _binding(tmp_path / "binding.npz")
    source_rest = np.zeros((22, 3), dtype=np.float32)
    for joint in range(1, 22):
        source_rest[joint] = source_rest[joint - 1] + [0, .1, 0]
    np.savez_compressed(
        tmp_path / "extracted.npz",
        body_pose=np.zeros((3, 63), dtype=np.float32),
        global_orient=np.zeros((3, 3), dtype=np.float32),
        transl=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        source_rest_joints=source_rest,
        fps=np.array([30], dtype=np.float32),
    )
    source_bundle = build_source_motion_bundle(tmp_path / "extracted.npz", tmp_path / "source.npz")

    class Characters:
        def get(self, character_id): return {"id": character_id}
        def binding_path(self, character_id): return binding

    class Motions:
        def get(self, motion_id): return {"id": motion_id, "filename": "source.mp4"}
        def source_bundle_path(self, job): return source_bundle

    animations = AnimationStore(Characters(), Motions())
    previews = RetargetPreviewStore(Characters(), Motions(), animations)
    preview = previews.create("motion", "character", "draft")
    assert animations.list() == []
    animation = previews.commit(preview["id"])
    assert len(animations.list()) == 1
    assert not (tmp_path / "previews" / preview["id"]).exists()

    compositions = CompositionStore(animations)
    composition = compositions.create("character", "root test")
    base_segment = {"animation_id": animation["id"], "start_frame": 0, "end_frame": 2, "speed": 1, "reverse": False}
    compositions.update(composition["id"], name=None, segments=[{**base_segment, "root_motion": False}])
    fixed = np.load(compositions.compile(composition["id"]), allow_pickle=False)["world_motion"]
    compositions.update(composition["id"], name=None, segments=[{**base_segment, "root_motion": True}])
    moving = np.load(compositions.compile(composition["id"]), allow_pickle=False)["world_motion"]
    np.testing.assert_allclose(fixed[:, 0, 0, 3], fixed[0, 0, 0, 3])
    assert moving[-1, 0, 0, 3] > moving[0, 0, 0, 3]


def test_virtual_root_preserves_hip_local_motion_and_thresholds_jitter() -> None:
    parents = np.array([-1, 0], dtype=np.int64)
    names = ["Hips", "Chest"]
    target_bind = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    target_bind[0, :3, 3] = [0, 1, 0]
    target_bind[1, :3, 3] = [0, 1.5, 0]
    vertices = np.array([[0, 0, 0], [.1, .2, 0], [0, 1, 0]], dtype=np.float32)
    lbs_indices = np.zeros((3, 4), dtype=np.int64)
    lbs_weights = np.zeros((3, 4), dtype=np.float32)
    lbs_weights[:, 0] = 1.0
    world = np.repeat(target_bind[None], 6, axis=0)
    world[:, 0, 0, 3] = [0, .001, .002, 1.0, 2.0, 3.0]
    world[:, 1, 0, 3] = world[:, 0, 0, 3]
    world[:, 0, 1, 3] = [1.0, 1.1, .9, 1.2, .8, 1.0]
    result = build_virtual_root(world, parents, names, target_bind, rest_vertices=vertices, lbs_indices=lbs_indices, lbs_weights=lbs_weights, smoothing_frames=1, position_tolerance=.01)
    np.testing.assert_allclose(result.virtual_root_motion[:3, 0, 3], [0, 0, 0], atol=1e-6)
    assert result.virtual_root_motion[-1, 0, 3] > 2.9
    np.testing.assert_allclose(result.virtual_root_motion[:, 1, 3], world[:, 0, 1, 3] - 1.0, atol=1e-6)
    np.testing.assert_allclose(result.ground_heights, world[:, 0, 1, 3] - 1.0, atol=1e-6)
    np.testing.assert_allclose(result.hip_local_under_virtual_root[:, 1, 3], 1.0, atol=1e-6)
    np.testing.assert_allclose(result.preview_world_motion[:, 0, :3, 3], world[:, 0, :3, 3], atol=1e-5)


def test_virtual_root_preview_commit_and_composition_root_transform(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(library_stores, "ANIMATIONS_ROOT", tmp_path / "animations")
    monkeypatch.setattr(library_stores, "COMPOSITIONS_ROOT", tmp_path / "compositions")
    monkeypatch.setattr(library_stores, "VIRTUAL_ROOT_PREVIEWS_ROOT", tmp_path / "virtual_previews")
    binding_path = _binding(tmp_path / "binding.npz")

    class Characters:
        def get(self, character_id): return {"id": character_id}
        def binding_path(self, character_id): return binding_path

    class Motions:
        def get(self, motion_id): return {"id": motion_id, "filename": "source.mp4"}

    animations = AnimationStore(Characters(), Motions())
    source_id = "source_anim"
    source_root = tmp_path / "animations" / source_id
    source_root.mkdir(parents=True)
    with np.load(binding_path, allow_pickle=True) as binding:
        target_bind = binding["bind_matrices"]
        world = np.repeat(target_bind[None], 5, axis=0).astype(np.float32)
        world[:, 0, 0, 3] = np.linspace(0, 4, 5)
        world[:, :, 0, 3] += np.linspace(0, 4, 5)[:, None]
        world[:, :, 1, 3] += np.array([0.0, 0.05, 0.12, 0.03, 0.08], dtype=np.float32)[:, None]
        np.savez_compressed(
            source_root / "retargeted.npz",
            rest_vertices=binding["vertices"],
            faces=binding["faces"],
            lbs_indices=binding["lbs_indices"],
            lbs_weights=binding["lbs_weights"],
            target_bind=target_bind,
            world_motion=world,
            parents=binding["parents"],
            names=np.asarray([str(name) for name in binding["names"].tolist()], dtype="U64"),
            fps=np.array([30], dtype=np.float32),
            num_frames=np.array([5], dtype=np.int64),
        )
    library_stores._write_json(source_root / "manifest.json", {"id": source_id, "name": "walk", "character_id": "character", "bundle": "retargeted.npz", "frames": 5, "fps": 30.0, "created_at": 1, "updated_at": 1})

    previews = library_stores.VirtualRootPreviewStore(animations)
    preview = previews.create(source_id, "walk vr", smoothing_frames=1, position_tolerance=0)
    with np.load(previews.bundle_path(preview["id"]), allow_pickle=False) as preview_data:
        assert "virtual_root_ground_heights" in preview_data.files
        assert float(np.ptp(preview_data["virtual_root_motion"][:, 1, 3])) > 0.05
    committed = previews.commit(preview["id"])
    assert committed["virtual_root"] is True

    compositions = CompositionStore(animations)
    composition = compositions.create("character", "vr comp")
    segment = {"animation_id": committed["id"], "start_frame": 0, "end_frame": 4, "speed": 1, "reverse": False, "root_motion": True}
    compositions.update(composition["id"], name=None, segments=[segment], cuts=[])
    moving = np.load(compositions.compile(composition["id"]), allow_pickle=False)["world_motion"]
    assert moving[-1, 0, 0, 3] > moving[0, 0, 0, 3] + 3.5


def test_split_composition_at_current_frame(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(library_stores, "COMPOSITIONS_ROOT", tmp_path / "compositions")
    world = np.repeat(np.eye(4, dtype=np.float32)[None, None], 100, axis=0)
    root = np.repeat(np.eye(4, dtype=np.float32)[None], 100, axis=0)
    root[:, 0, 3] = np.arange(100, dtype=np.float32)
    path = tmp_path / "motion.npz"
    np.savez_compressed(path, world_motion=world, virtual_root_motion=root, parents=np.array([-1]), fps=np.array([30], dtype=np.float32))

    class Characters:
        def get(self, character_id): return {"id": character_id}

    class Animations:
        characters = Characters()

        def get(self, animation_id):
            return {"id": animation_id, "character_id": "character", "frames": 100, "fps": 30.0, "virtual_root": True}

        def effective_world(self, animation_id):
            data = np.load(path, allow_pickle=False)
            return data, np.asarray(data["world_motion"], dtype=np.float32)

        def root_motion(self, animation_id, data):
            return data["virtual_root_motion"][:, :3, 3] - data["virtual_root_motion"][[0], :3, 3]

        def virtual_root_motion(self, data):
            return np.asarray(data["virtual_root_motion"], dtype=np.float32)

    store = CompositionStore(Animations())
    composition = store.create("character", "split")
    row = store.update(composition["id"], name=None, segments=[{"id": "a", "animation_id": "motion", "start_frame": 0, "end_frame": 99, "speed": 1, "root_motion": True, "absolute_root": True}], cuts=[])
    assert len(row["segments"]) == 1
    split = store.split_at_frame(composition["id"], 40)
    assert [(s["start_frame"], s["end_frame"]) for s in split["segments"]] == [(0, 39), (40, 99)]
    with np.load(store.compile(composition["id"]), allow_pickle=False) as data:
        assert len(data["world_motion"]) == 100
        assert len(data["virtual_root_motion"]) == 100
        np.testing.assert_allclose(data["virtual_root_motion"][39, :3, 3], data["virtual_root_motion"][40, :3, 3], atol=1e-5)

    deleted_composition = store.create("character", "delete absolute root")
    store.update(deleted_composition["id"], name=None, segments=[{"id": "d", "animation_id": "motion", "start_frame": 0, "end_frame": 99, "speed": 1, "root_motion": True, "absolute_root": True}], cuts=[])
    deleted = store.delete_frames(deleted_composition["id"], 40, 60)
    assert len(deleted["segments"]) == 2
    assert deleted["segments"][1]["force_align_to_previous"] is True
    with np.load(store.compile(deleted_composition["id"]), allow_pickle=False) as data:
        assert len(data["virtual_root_motion"]) == 80
        np.testing.assert_allclose(data["virtual_root_motion"][39, :3, 3], data["virtual_root_motion"][40, :3, 3], atol=1e-5)
