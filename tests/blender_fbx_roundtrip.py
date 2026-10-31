"""Explicit Blender integration harness, not collected by ordinary pytest.

Run with --factory-startup --disable-autoexec in an isolated acceptance root.
Only caller-provided synthetic/authorized local bundles are read.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def load_exporter(path):
    spec = importlib.util.spec_from_file_location("fbx_export_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_mesh(data, frame):
    world = np.repeat(np.eye(4)[None], len(data["names"]), axis=0)
    world[:, :3, :3] = data["global_rot_mats"][frame]
    world[:, :3, 3] = data["posed_joints"][frame]
    skin = world @ np.linalg.inv(data["bind_matrices"].astype(np.float64))
    verts = np.concatenate([data["vertices"], np.ones((len(data["vertices"]), 1))], axis=1)
    transformed = np.einsum("vkij,vj->vki", skin[data["lbs_indices"]], verts)
    return np.sum(transformed[..., :3] * data["lbs_weights"][..., None], axis=1)


def measure(data, label, output):
    arm = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    mesh = next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")
    names = [str(n)[:63] for n in data["names"]]
    expected_parents = {name: None if p < 0 else names[p] for name, p in zip(names, data["parents"])}
    actual_parents = {name: arm.data.bones[name].parent.name if arm.data.bones[name].parent else None for name in names}
    frames = list(range(len(data["posed_joints"])))
    rows, vertices, heads = [], [], []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_arm = arm.evaluated_get(depsgraph)
        evaluated_mesh = mesh.evaluated_get(depsgraph)
        evaluated = evaluated_mesh.to_mesh()
        verts = np.asarray([evaluated_mesh.matrix_world @ v.co for v in evaluated.vertices], dtype=np.float64)
        evaluated_mesh.to_mesh_clear()
        joints = np.asarray([(evaluated_arm.matrix_world @ evaluated_arm.pose.bones[name].matrix).translation for name in names])
        expected = expected_mesh(data, frame)
        pairs = np.flatnonzero(data["parents"] >= 0)
        source = data["posed_joints"][frame]
        source_lengths = np.linalg.norm(source[pairs] - source[data["parents"][pairs]], axis=1)
        actual_lengths = np.linalg.norm(joints[pairs] - joints[data["parents"][pairs]], axis=1)
        rows.append(dict(frame=frame, finite=bool(np.isfinite(verts).all() and np.isfinite(joints).all()),
                         mesh_vertex_count=len(verts), expected_vertex_count=len(expected),
                         mesh_max_lbs_error_m=float(np.max(np.linalg.norm(verts - expected, axis=1))),
                         mesh_rms_lbs_error_m=float(np.sqrt(np.mean((verts - expected) ** 2))),
                         joint_max_position_error_m=float(np.max(np.linalg.norm(joints - source, axis=1))),
                         hierarchy_segment_max_length_error_m=float(np.max(np.abs(actual_lengths - source_lengths))) if len(pairs) else 0.))
        vertices.append(verts)
        heads.append(joints)
    np.savez_compressed(output / f"{label}_evaluated.npz", frames=frames, vertices=vertices, joints=heads)
    return dict(armature_collection_order=list(arm.data.bones.keys()), input_order=names,
                mesh_parent_type=mesh.parent_type,
                connected_bones=[name for name in names if arm.data.bones[name].use_connect],
                hierarchy_matches=actual_parents == expected_parents,
                virtual_root_present="VirtualRoot" in arm.data.bones,
                checked_frame_count=len(frames),
                representative_frames=[rows[i] for i in sorted(set([0, len(rows) // 2, len(rows) - 1]))],
                frames=rows,
                pass_tolerance_m=2e-5,
                passed=actual_parents == expected_parents and all(row["finite"] and max(row["mesh_max_lbs_error_m"], row["joint_max_position_error_m"], row["hierarchy_segment_max_length_error_m"]) <= 2e-5 for row in rows))


def run_case(module, bundle, output, bind_only=False):
    with np.load(bundle, allow_pickle=True) as archive:
        data = {k: archive[k] for k in archive.files}
    names = [str(n) for n in data["names"]]
    module._clear_scene()
    arm = module._build_armature_from_bind(data["bind_matrices"], data["parents"], names)
    kwargs = dict(names=names) if "names" in inspect.signature(module._build_mesh).parameters else {}
    module._build_mesh(data["vertices"], data["faces"], arm, data["lbs_indices"], data["lbs_weights"], **kwargs)
    if not bind_only:
        args = [arm, names, data["posed_joints"], data["global_rot_mats"], float(data["fps"].flat[0])]
        if "bind_mats" in inspect.signature(module._apply_animation).parameters:
            args.append(data["bind_matrices"])
        module._apply_animation(*args)
    else:
        bpy.context.scene.frame_start = bpy.context.scene.frame_end = 0
    before = measure(data, "before", output)
    # Call the actual production main with injected file arguments, so the test
    # cannot silently drift from its export settings or bundle-loading path.
    meta = output / "export_meta.json"
    meta.write_text(json.dumps(dict(output_fbx=str(output / "animation.fbx"), bind_pose_only=bind_only)), encoding="utf-8")
    module._parse_args = lambda: (bundle, meta)
    module.main()
    module._clear_scene()
    bpy.ops.import_scene.fbx(filepath=str(output / "animation.fbx"), use_anim=True,
                             anim_offset=0., automatic_bone_orientation=False)
    after = measure(data, "roundtrip", output)
    return dict(bundle=str(bundle), bind_only=bind_only, before_export=before, fbx_roundtrip=after)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.output.mkdir(exist_ok=False)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    report = dict(blender_version=bpy.app.version_string, scope="Real Blender evaluated LBS and FBX export/import, not merely header validation", cases=[])
    sources = [("fixed", args.source)] + ([("baseline", args.baseline_source)] if args.baseline_source else [])
    for version, source in sources:
        module = load_exporter(source / "motionbuilder_app" / "blender_export_fbx.py")
        for case in plan:
            output = args.output / (version + "_" + case["name"])
            output.mkdir()
            result = run_case(module, Path(case["bundle"]), output, case.get("bind_only", False))
            result.update(version=version, name=case["name"])
            report["cases"].append(result)
            (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print("ROUNDTRIP", version, case["name"], result["before_export"]["passed"], result["fbx_roundtrip"]["passed"], flush=True)
    if any(not case[stage]["passed"] for case in report["cases"] if case["version"] == "fixed" for stage in ("before_export", "fbx_roundtrip")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
