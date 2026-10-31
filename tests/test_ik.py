from pathlib import Path
import json

import numpy as np

from motionbuilder_app import stores
from motionbuilder_app.ik import build_humanoid_ik_profile, solve_two_bone_positions
from motionbuilder_app.meshpreps.bind_math import BindingResult, build_bind_matrices
from motionbuilder_app.meshpreps.binding_io import binding_preview_dict, load_binding, save_binding
from motionbuilder_app.meshpreps.session import BindJob, JobPhase, JobState, SessionStore
from motionbuilder_app.skeleton import SOMASkeleton30
from motionbuilder_app.stores import CharacterStore


def _binding(path: Path) -> Path:
    skeleton = SOMASkeleton30(load=True)
    joints = np.asarray(skeleton.neutral_joints, dtype=np.float32)
    parents = np.asarray(skeleton.joint_parents, dtype=np.int64)
    vertices = np.array(
        [[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.2]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    indices = np.zeros((len(vertices), 4), dtype=np.int64)
    weights = np.zeros((len(vertices), 4), dtype=np.float32)
    weights[:, 0] = 1.0
    binding = BindingResult(
        vertices=vertices,
        faces=faces,
        joint_positions=joints,
        joint_parents=parents,
        joint_names=skeleton.bone_order_names,
        lbs_indices=indices,
        lbs_weights=weights,
        bind_matrices=build_bind_matrices(joints, parents),
        source_path=None,
        method="test",
    )
    return save_binding(path, binding)


def test_humanoid_profile_builds_only_four_limb_two_bone_chains(tmp_path: Path) -> None:
    binding = load_binding(_binding(tmp_path / "binding.npz"))
    profile = build_humanoid_ik_profile(binding)

    assert profile["version"] == 2
    assert profile["available_solvers"] == ["TwoBoneIK", "FullBodyIK"]
    assert profile["solver"] == "TwoBoneIK"
    assert profile["solver_stack"] == ["left_leg", "right_leg", "left_arm", "right_arm"]
    assert len(profile["chains"]) == 4
    assert all(chain["solver"] == "TwoBoneIK" for chain in profile["chains"])

    excluded = {"Jaw", "LeftEye", "RightEye"}
    assert set(profile["excluded_joints"]) == excluded
    chained_names = {name for chain in profile["chains"] for name in chain["joint_names"]}
    chained_indices = {index for chain in profile["chains"] for index in chain["joints"]}
    assert chained_names.isdisjoint(excluded)
    assert chained_indices.isdisjoint(profile["excluded_joint_indices"])
    full_body = profile["full_body"]
    assert full_body["solver"] == "FullBodyIK"
    assert full_body["root"] == "Hips"
    assert {item["id"] for item in full_body["effectors"]} == {
        "head", "left_hand", "right_hand", "left_foot", "right_foot"
    }
    assert set(full_body["excluded_joints"]) == excluded
    assert all(item["bone"] not in excluded for item in full_body["effectors"])


def test_ik_profile_roundtrips_inside_existing_binding(tmp_path: Path) -> None:
    path = _binding(tmp_path / "binding.npz")
    binding = load_binding(path)
    binding.ik_profile = build_humanoid_ik_profile(binding)
    save_binding(path, binding)

    restored = load_binding(path)
    assert restored.ik_profile == binding.ik_profile
    preview = binding_preview_dict(restored)
    assert preview["has_ik"] is True
    assert preview["ik_profile"]["solver"] == "TwoBoneIK"


def test_two_bone_solver_preserves_lengths_and_clamps_unreachable_target() -> None:
    start = np.array([0.0, 0.0, 0.0])
    middle = np.array([1.0, 0.0, 0.0])
    end = np.array([2.0, 0.0, 0.0])
    pole = np.array([0.0, 1.0, 0.0])

    reachable = solve_two_bone_positions(start, middle, end, np.array([1.0, 1.0, 0.0]), pole)
    np.testing.assert_allclose(reachable.end, [1.0, 1.0, 0.0], atol=2e-6)
    assert reachable.reached is True
    np.testing.assert_allclose(np.linalg.norm(reachable.middle - reachable.start), 1.0, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(reachable.end - reachable.middle), 1.0, atol=2e-6)

    unreachable = solve_two_bone_positions(start, middle, end, np.array([5.0, 0.0, 0.0]), pole)
    assert unreachable.reached is False
    assert unreachable.clamped_distance < 2.0
    np.testing.assert_allclose(np.linalg.norm(unreachable.middle - unreachable.start), 1.0, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(unreachable.end - unreachable.middle), 1.0, atol=2e-6)


def test_character_auto_ik_overwrites_same_asset_binding(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(stores, "CHARACTERS_ROOT", tmp_path / "characters")
    character_store = CharacterStore()
    character = character_store.create("SOMA", _binding(tmp_path / "source.npz"))
    character_id = character["id"]
    original_path = character_store.binding_path(character_id)

    assert character_store.get(character_id)["has_ik"] is False
    updated = character_store.auto_ik(character_id)

    assert character_store.binding_path(character_id) == original_path
    assert updated["id"] == character_id
    assert updated["version"] == character["version"]
    assert updated["has_ik"] is True
    assert load_binding(original_path).ik_profile["solver"] == "TwoBoneIK"


def test_completed_binding_job_can_generate_ik_in_place(tmp_path: Path) -> None:
    output_path = _binding(tmp_path / "job_binding.npz")
    session_store = SessionStore()
    job = BindJob(
        job_id="ik-job",
        mesh_id="mesh",
        state=JobState.DONE,
        phase=JobPhase.COMPLETE,
        output_path=output_path,
    )
    session_store._jobs[job.job_id] = job

    result = session_store.auto_ik(job.job_id)

    assert result["has_ik"] is True
    assert result["profile"]["solver"] == "TwoBoneIK"
    assert load_binding(output_path).ik_profile == result["profile"]


def test_legacy_asset_without_manifest_flag_detects_embedded_ik(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(stores, "CHARACTERS_ROOT", tmp_path / "characters")
    character_store = CharacterStore()
    character = character_store.create("Legacy", _binding(tmp_path / "legacy_source.npz"))
    character_id = character["id"]
    root = stores.CHARACTERS_ROOT / character_id
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("has_ik")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert character_store.get(character_id)["has_ik"] is False

    binding_path = character_store.binding_path(character_id)
    binding = load_binding(binding_path)
    binding.ik_profile = build_humanoid_ik_profile(binding)
    save_binding(binding_path, binding)

    assert character_store.get(character_id)["has_ik"] is True
