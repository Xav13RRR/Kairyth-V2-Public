"""Blender headless script: export skinned SOMA mesh animation to FBX."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def _parse_args() -> tuple[Path, Path]:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("usage: blender --background --python blender_export_fbx.py -- bundle.npz meta.json")
    extra = argv[argv.index("--") + 1 :]
    if len(extra) < 2:
        raise SystemExit("expected npz path and meta json path")
    return Path(extra[0]), Path(extra[1])


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.armatures:
        bpy.data.armatures.remove(block)
    for block in bpy.data.actions:
        bpy.data.actions.remove(block)


def _matrix_from_numpy(m: np.ndarray) -> Matrix:
    return Matrix([list(row) for row in np.asarray(m, dtype=np.float64)])


def _build_armature_from_bind(
    bind_mats: np.ndarray,
    parents: np.ndarray,
    names: list[str],
) -> bpy.types.Object:
    bind_mats = np.asarray(bind_mats, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    n = int(bind_mats.shape[0])
    joints = bind_mats[:, :3, 3]

    arm_data = bpy.data.armatures.new("MotionBuilder_SOMA30")
    arm_obj = bpy.data.objects.new("MotionBuilder_SOMA30", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = arm_data.edit_bones
    bone_by_index: list = []
    for i, name in enumerate(names):
        bone = edit_bones.new(name[:63] or f"Bone_{i}")
        bone_by_index.append(bone)

    for i, bone in enumerate(bone_by_index):
        p = int(parents[i])
        head = Vector(joints[i])
        children = [j for j in range(n) if int(parents[j]) == i]
        if children:
            tail = Vector(joints[children[0]])
        else:
            tail = head + Vector(bind_mats[i, :3, :3] @ np.array([0.0, 0.05, 0.0]))
        if (tail - head).length < 1e-6:
            tail = head + Vector((0.0, 0.05, 0.0))
        bone.head = head
        bone.tail = tail
        if p >= 0:
            bone.parent = bone_by_index[p]
            bone.use_connect = False
        z_axis = Vector(bind_mats[i, :3, :3] @ np.array([0.0, 0.0, 1.0]))
        if z_axis.length > 1e-8:
            bone.align_roll(z_axis)

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def _build_armature(joints: np.ndarray, parents: np.ndarray, names: list[str]) -> bpy.types.Object:
    arm_data = bpy.data.armatures.new("MotionBuilder_SOMA30")
    arm_obj = bpy.data.objects.new("MotionBuilder_SOMA30", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = arm_data.edit_bones
    bone_objs: list = []
    for i, name in enumerate(names):
        bone = edit_bones.new(name[:63] or f"Bone_{i}")
        bone.head = Vector(joints[i])
        if int(parents[i]) >= 0:
            bone.parent = edit_bones[names[int(parents[i])][:63]]
        bone_objs.append(bone)

    for i, bone in enumerate(bone_objs):
        children = [j for j, p in enumerate(parents) if int(p) == i]
        if children:
            bone.tail = Vector(joints[children[0]])
        else:
            bone.tail = bone.head + Vector((0, 0.05, 0))

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def _build_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    arm_obj: bpy.types.Object,
    lbs_indices=None,
    lbs_weights=None,
    *,
    names: list[str] | None = None,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("CustomMesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    mesh_obj = bpy.data.objects.new("CustomMesh", mesh)
    bpy.context.collection.objects.link(mesh_obj)

    if lbs_indices is not None and lbs_weights is not None:
        lbs_indices = np.asarray(lbs_indices, dtype=np.int64)
        lbs_weights = np.asarray(lbs_weights, dtype=np.float32)
        # Blender may reorder its bone collection when parents are assigned.
        # Skin indices refer to the input joint order, never that collection.
        bone_names = [str(name)[:63] for name in names] if names is not None else [bone.name for bone in arm_obj.data.bones]
        for vi in range(len(vertices)):
            for k in range(lbs_indices.shape[1]):
                idx = int(lbs_indices[vi, k])
                w = float(lbs_weights[vi, k])
                if w < 1e-7 or not 0 <= idx < len(bone_names):
                    continue
                group = mesh_obj.vertex_groups.get(bone_names[idx])
                if group is None:
                    group = mesh_obj.vertex_groups.new(name=bone_names[idx])
                group.add([vi], w, "REPLACE")
    else:
        vgroup = mesh_obj.vertex_groups.new(name="SOMA")
        vgroup.add(list(range(len(vertices))), 1.0, "REPLACE")

    mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = arm_obj
    mesh_obj.parent = arm_obj
    # The modifier performs skinning. Legacy ARMATURE parenting would deform
    # again and is not supported by Blender's FBX exporter.
    mesh_obj.parent_type = "OBJECT"
    return mesh_obj


def _apply_animation(
    arm_obj: bpy.types.Object,
    names: list[str],
    posed_joints: np.ndarray,
    global_rot_mats: np.ndarray,
    fps: float,
    bind_mats: np.ndarray,
) -> None:
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.frame_start = 0
    scene.frame_end = max(0, posed_joints.shape[0] - 1)

    action = bpy.data.actions.new("GVHMR_Retargeted_Motion")
    arm_obj.animation_data_create()
    arm_obj.animation_data.action = action

    # Source skinning is W @ inverse(B); Blender skinning is P @ inverse(C),
    # where C is the actual edit-bone rest basis (its local Y follows the tail).
    # Therefore P = W @ inverse(B) @ C. Copying W directly loses this basis
    # correction even when its translation is in the correct last column.
    rest = {name: arm_obj.data.bones[name[:63]].matrix_local.copy() for name in names}
    correction = {
        name: _matrix_from_numpy(np.linalg.inv(bind_mats[j])) @ rest[name]
        for j, name in enumerate(names)
    }
    for name in names:
        arm_obj.pose.bones[name[:63]].rotation_mode = "QUATERNION"

    for fi in range(posed_joints.shape[0]):
        frame = fi
        desired = {}
        for j, name in enumerate(names):
            source_world = np.eye(4, dtype=np.float64)
            source_world[:3, :3] = global_rot_mats[fi, j]
            source_world[:3, 3] = posed_joints[fi, j]
            desired[name[:63]] = _matrix_from_numpy(source_world) @ correction[name]
        for name in names:
            bone = arm_obj.pose.bones[name[:63]]
            # Compute basis from the desired parent pose, not a partly updated
            # dependency graph. This also supports non-topological input order.
            if bone.parent is None:
                basis = rest[name].inverted() @ desired[bone.name]
            else:
                basis = (rest[name].inverted() @ bone.parent.bone.matrix_local
                         @ desired[bone.parent.name].inverted() @ desired[bone.name])
            bone.matrix_basis = basis
            bone.keyframe_insert(data_path="location", frame=frame)
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            bone.keyframe_insert(data_path="scale", frame=frame)


def main() -> None:
    npz_path, meta_path = _parse_args()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    out_fbx = Path(meta["output_fbx"])

    data = np.load(npz_path, allow_pickle=True)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int64)
    joints = np.asarray(data["joints"], dtype=np.float32)
    parents = np.asarray(data["parents"], dtype=np.int64)
    names = [str(n) for n in list(data["names"])]
    posed = np.asarray(data["posed_joints"], dtype=np.float32)
    rots = np.asarray(data["global_rot_mats"], dtype=np.float32)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    lbs_indices = data["lbs_indices"] if "lbs_indices" in data.files else None
    lbs_weights = data["lbs_weights"] if "lbs_weights" in data.files else None
    if lbs_indices is not None:
        lbs_indices = np.asarray(lbs_indices)
    if lbs_weights is not None:
        lbs_weights = np.asarray(lbs_weights)

    bind_pose_only = bool(meta.get("bind_pose_only", False))
    bind_mats = np.asarray(data["bind_matrices"], dtype=np.float32) if "bind_matrices" in data.files else None

    _clear_scene()
    if bind_pose_only and bind_mats is not None:
        arm_obj = _build_armature_from_bind(bind_mats, parents, names)
        _build_mesh(vertices, faces, arm_obj, lbs_indices, lbs_weights, names=names)
        bpy.ops.export_scene.fbx(
            filepath=str(out_fbx),
            use_selection=False,
            object_types={"ARMATURE", "MESH"},
            bake_anim=False,
            add_leaf_bones=False,
            apply_scale_options="FBX_SCALE_NONE",
            primary_bone_axis="X",
            secondary_bone_axis="Y",
        )
        return

    if bind_mats is None:
        # Legacy bundles carry joint origins but no source rest rotations.
        bind_mats = np.repeat(np.eye(4, dtype=np.float32)[None], len(joints), axis=0)
        bind_mats[:, :3, 3] = joints
    arm_obj = _build_armature_from_bind(bind_mats, parents, names)
    _build_mesh(vertices, faces, arm_obj, lbs_indices, lbs_weights, names=names)
    _apply_animation(arm_obj, names, posed, rots, fps, bind_mats)

    bpy.ops.export_scene.fbx(
        filepath=str(out_fbx),
        use_selection=False,
        object_types={"ARMATURE", "MESH"},
        bake_anim=True,
        bake_anim_use_all_bones=True,
        # Preserve every source frame; the default lossy curve simplification
        # moves intermediate poses even when first/middle/last happen to match.
        bake_anim_simplify_factor=0.0,
        add_leaf_bones=False,
        apply_scale_options="FBX_SCALE_NONE",
        # FBX uses X-along-bone here. Blender's importer otherwise infers a
        # connected Y-aligned child and discards animated child translation
        # (notably Hips under VirtualRoot), even with force_connect disabled.
        primary_bone_axis="X",
        secondary_bone_axis="Y",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
