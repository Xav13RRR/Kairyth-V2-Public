from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


def main() -> None:
    if "--" not in sys.argv:
        raise SystemExit("usage: blender --background --python blender_validate_fbx.py -- file.fbx")
    path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(path))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    actions = list(bpy.data.actions)
    action_ranges = [[float(action.frame_range[0]), float(action.frame_range[1])] for action in actions]
    result = {
        "file": str(path),
        "meshes": len(meshes),
        "vertices": sum(len(obj.data.vertices) for obj in meshes),
        "armatures": len(armatures),
        "bones": sum(len(obj.data.bones) for obj in armatures),
        "actions": len(actions),
        "action_ranges": action_ranges,
        "frame_start": int(bpy.context.scene.frame_start),
        "frame_end": int(bpy.context.scene.frame_end),
        "fps": int(bpy.context.scene.render.fps),
    }
    print("MOTIONBUILDER_FBX_VALIDATION=" + json.dumps(result, ensure_ascii=True))
    if result["meshes"] != 1 or result["armatures"] != 1 or result["bones"] != 30:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
