from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .kimodo_runtime import generate_kimodo_native_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KIMODO_MOTIONS_ROOT = PROJECT_ROOT / "data" / "kimodo_motions"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def run(job_id: str) -> dict[str, Any]:
    safe_id = Path(job_id).name
    root = (KIMODO_MOTIONS_ROOT / safe_id).resolve()
    if root.parent != KIMODO_MOTIONS_ROOT.resolve() or not root.is_dir():
        raise FileNotFoundError(f"Kimodo job not found: {safe_id}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    bundle = root / "native_kimodo.npz"
    generate_kimodo_native_bundle(
        bundle,
        prompts=list(manifest.get("prompts") or [manifest["prompt"]]),
        durations_sec=list(manifest.get("durations_sec") or [float(manifest["duration_sec"])]),
        seed=int(manifest["seed"]),
        diffusion_steps=int(manifest["diffusion_steps"]),
        model_name=str(manifest["model_name"]),
        constraints=list(manifest.get("constraints") or []),
    )
    with np.load(bundle, allow_pickle=False) as data:
        frames = int(data["world_motion"].shape[0])
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    result = {"bundle": bundle.name, "frames": frames, "fps": fps}
    _write_json(root / "worker_result.json", result)
    return result


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m motionbuilder_app.kimodo_worker <job_id>", file=sys.stderr)
        return 2
    try:
        run(sys.argv[1])
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
