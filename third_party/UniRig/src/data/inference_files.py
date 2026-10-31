"""Inference file listing without Blender (bpy) dependency."""

from __future__ import annotations

import os
from typing import List, Tuple, Union


def get_files(
    data_name: str,
    input_dataset_dir: str,
    output_dataset_dir: str,
    inputs: Union[str, None] = None,
    require_suffix: List[str] = ["obj", "fbx", "FBX", "dae", "glb", "gltf", "vrm"],
    force_override: bool = False,
    warning: bool = True,
) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    if inputs is not None:
        vis: dict[str, bool] = {}
        for file in inputs.split(","):
            file = os.path.normpath(file.strip())
            stem_name = os.path.splitext(os.path.basename(file))[0]
            output_dir = os.path.join(output_dataset_dir, stem_name)
            raw_data_npz = os.path.join(output_dir, data_name)
            if not force_override and os.path.exists(raw_data_npz):
                continue
            if warning and output_dir in vis:
                print(
                    f"\033[33mWARNING: duplicate output directory: {output_dir}, "
                    "you need to rename prefix of files to avoid ambiguity\033[0m"
                )
            vis[output_dir] = True
            files.append((file, output_dir))
    else:
        vis = {}
        for root, _dirs, names in os.walk(input_dataset_dir):
            for file in names:
                if file.split(".")[-1] in require_suffix:
                    file_name = file.removeprefix("./")
                    file_name = ".".join(file_name.split(".")[:-1])
                    output_dir = os.path.join(
                        output_dataset_dir,
                        os.path.relpath(root, input_dataset_dir),
                        file_name,
                    )
                    raw_data_npz = os.path.join(output_dir, data_name)
                    if not force_override and os.path.exists(raw_data_npz):
                        continue
                    if warning and output_dir in vis:
                        print(
                            f"\033[33mWARNING: duplicate output directory: {output_dir}, "
                            "you need to rename prefix of files to avoid ambiguity\033[0m"
                        )
                    vis[output_dir] = True
                    files.append((os.path.join(root, file), output_dir))
    return files
