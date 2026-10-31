from __future__ import annotations

from pathlib import Path

import numpy as np

from motionbuilder_app.exporter import export_fbx
from motionbuilder_app.retarget import retarget_extracted
from tests.test_motion_core import _binding


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "validation"
    root.mkdir(parents=True, exist_ok=True)
    binding = _binding(root / "animated_binding.npz")
    source_rest = np.zeros((22, 3), dtype=np.float32)
    source_rest[7] = [-0.1, -1.0, 0.0]
    source_rest[8] = [0.1, -1.0, 0.0]
    body_pose = np.zeros((3, 63), dtype=np.float32)
    body_pose[1, 3 * (18 - 1) + 2] = 0.35
    np.savez_compressed(
        root / "animated_source.npz",
        body_pose=body_pose,
        global_orient=np.zeros((3, 3), dtype=np.float32),
        transl=np.array([[0, 0, 0], [0.1, 0.02, 0], [0.2, 0, 0]], dtype=np.float32),
        source_rest_joints=source_rest,
        fps=np.array([30], dtype=np.float32),
    )
    bundle = retarget_extracted(root / "animated_source.npz", binding, root / "animated_motion.npz")
    output = export_fbx(bundle, root / "animated_motion.fbx")
    print(output)


if __name__ == "__main__":
    main()
