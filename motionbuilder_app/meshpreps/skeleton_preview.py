from __future__ import annotations

from typing import Any


def skeleton_preview_dict(skel: dict) -> dict[str, Any]:
    from ..rigging.soma_fit import fit_soma30_positions

    fit = fit_soma30_positions(
        skel["joint_names"],
        skel["joint_positions"],
        skel.get("joint_parents"),
    )
    return {
        "kind": "skeleton",
        "has_skin": False,
        "vertices": skel["vertices"].tolist(),
        "faces": skel["faces"].tolist(),
        "joints": skel["joint_positions"].tolist(),
        "parents": skel["joint_parents"].tolist(),
        "names": skel["joint_names"],
        "n_verts": int(skel["vertices"].shape[0]),
        "n_joints": int(skel["joint_positions"].shape[0]),
        "soma_overlay": {
            "joints": fit.joints.tolist(),
            "parents": fit.parents.tolist(),
            "names": fit.names,
            "n_joints": len(fit.names),
        },
    }
