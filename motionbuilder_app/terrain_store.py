from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .terrain_surface import HeightFieldSurface, heightfield_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TERRAINS_ROOT = PROJECT_ROOT / "data" / "terrains"
ALLOWED_RESOLUTIONS = {33, 65, 129, 257, 513}


def _default_transform() -> dict[str, Any]:
    return {"translation": [0.0, 0.0, 0.0], "yaw_degrees": 0.0, "uniform_scale": 1.0}


def _normalize_transform(value: dict[str, Any] | None) -> dict[str, Any]:
    source = value or {}
    translation = np.asarray(source.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    yaw = float(source.get("yaw_degrees", 0.0))
    scale = float(source.get("uniform_scale", 1.0))
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("terrain translation must contain three finite values")
    if float(np.max(np.abs(translation))) > 4096.0:
        raise ValueError("terrain translation exceeds supported range")
    if not np.isfinite(yaw):
        raise ValueError("terrain yaw must be finite")
    if not np.isfinite(scale) or not 0.01 <= scale <= 100.0:
        raise ValueError("terrain uniform scale must be between 0.01 and 100")
    return {
        "translation": translation.astype(float).tolist(),
        "yaw_degrees": ((yaw + 180.0) % 360.0) - 180.0,
        "uniform_scale": scale,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _height_hash(
    heights: np.ndarray,
    size_x: float,
    size_z: float,
    transform: dict[str, Any] | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(heights, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray([size_x, size_z], dtype="<f8").tobytes())
    digest.update(json.dumps(_normalize_transform(transform), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


class TerrainStore:
    def __init__(self) -> None:
        TERRAINS_ROOT.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in TERRAINS_ROOT.glob("*/manifest.json"):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        return sorted(rows, key=lambda row: row.get("updated_at", 0), reverse=True)

    def get(self, terrain_id: str) -> dict[str, Any]:
        path = TERRAINS_ROOT / Path(terrain_id).name / "manifest.json"
        if not path.is_file():
            raise KeyError(terrain_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def bundle_path(self, terrain_id: str) -> Path:
        row = self.get(terrain_id)
        path = TERRAINS_ROOT / Path(terrain_id).name / Path(str(row["bundle"])).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def load_heights(self, terrain_id: str) -> np.ndarray:
        with np.load(self.bundle_path(terrain_id), allow_pickle=False) as data:
            return np.asarray(data["heights"], dtype=np.float32).copy()

    def surface(self, terrain_id: str) -> HeightFieldSurface:
        row = self.get(terrain_id)
        transform = _normalize_transform(row.get("transform"))
        return HeightFieldSurface(
            self.load_heights(terrain_id),
            float(row["size_x"]),
            float(row["size_z"]),
            translation=transform["translation"],
            yaw_degrees=transform["yaw_degrees"],
            uniform_scale=transform["uniform_scale"],
        )

    def create(
        self,
        name: str,
        *,
        size_x: float = 8.0,
        size_z: float = 8.0,
        resolution: int = 129,
        initial_height: float = 0.0,
    ) -> dict[str, Any]:
        clean = str(name or "New Terrain").strip()[:80]
        if not clean:
            raise ValueError("terrain name cannot be empty")
        size_x = float(size_x)
        size_z = float(size_z)
        resolution = int(resolution)
        initial_height = float(initial_height)
        if not 0.5 <= size_x <= 4096.0 or not 0.5 <= size_z <= 4096.0:
            raise ValueError("terrain size must be between 0.5 and 4096 meters")
        if resolution not in ALLOWED_RESOLUTIONS:
            raise ValueError("unsupported terrain resolution")
        if not -1024.0 <= initial_height <= 1024.0:
            raise ValueError("initial terrain height is out of range")

        terrain_id = uuid.uuid4().hex[:12]
        root = TERRAINS_ROOT / terrain_id
        root.mkdir(parents=True, exist_ok=False)
        heights = np.full((resolution, resolution), initial_height, dtype=np.float32)
        bundle = root / "heightfield.npz"
        np.savez_compressed(bundle, heights=heights)
        now = time.time()
        row = {
            "id": terrain_id,
            "name": clean,
            "asset_kind": "terrain_heightfield",
            "version": 1,
            "revision": 1,
            "bundle": bundle.name,
            "size_x": size_x,
            "size_z": size_z,
            "resolution_x": resolution,
            "resolution_z": resolution,
            "min_height": initial_height,
            "max_height": initial_height,
            "up_axis": "Y",
            "units": "meter",
            "content_hash": _height_hash(heights, size_x, size_z, _default_transform()),
            "transform": _default_transform(),
            "layers": [{"id": "base", "name": "Base", "visible": True, "locked": False, "alpha": 1.0}],
            "created_at": now,
            "updated_at": now,
        }
        _write_json(root / "manifest.json", row)
        return row

    def rename(self, terrain_id: str, name: str) -> dict[str, Any]:
        clean = str(name or "").strip()[:80]
        if not clean:
            raise ValueError("terrain name cannot be empty")
        row = self.get(terrain_id)
        row.update(name=clean, updated_at=time.time())
        _write_json(TERRAINS_ROOT / Path(terrain_id).name / "manifest.json", row)
        return row

    def update_heights(
        self,
        terrain_id: str,
        heights: np.ndarray,
        expected_revision: int | None = None,
        transform: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = self.get(terrain_id)
        if expected_revision is not None and int(row.get("revision", 1)) != int(expected_revision):
            raise RuntimeError("terrain revision conflict")
        values = np.asarray(heights, dtype=np.float32)
        expected = (int(row["resolution_z"]), int(row["resolution_x"]))
        if values.shape != expected:
            raise ValueError(f"heightfield shape must be {expected}")
        if not np.isfinite(values).all():
            raise ValueError("heightfield contains non-finite values")
        if float(np.max(np.abs(values))) > 1024.0:
            raise ValueError("terrain height exceeds supported range")
        resolved_transform = _normalize_transform(transform if transform is not None else row.get("transform"))
        path = self.bundle_path(terrain_id)
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, heights=values)
        temp.replace(path)
        row.update(
            revision=int(row.get("revision", 1)) + 1,
            min_height=float(values.min()),
            max_height=float(values.max()),
            content_hash=_height_hash(values, float(row["size_x"]), float(row["size_z"]), resolved_transform),
            transform=resolved_transform,
            updated_at=time.time(),
        )
        _write_json(path.parent / "manifest.json", row)
        return row

    def scene(self, terrain_id: str) -> dict[str, Any]:
        row = self.get(terrain_id)
        heights = self.load_heights(terrain_id)
        vertices, faces = heightfield_mesh(heights, float(row["size_x"]), float(row["size_z"]))
        return {
            **row,
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
            "heights": heights.tolist(),
            "transform": _normalize_transform(row.get("transform")),
            "static_asset": True,
            "num_frames": 1,
            "fps": 30.0,
        }

    def copy_snapshot(self, terrain_id: str, destination: Path) -> dict[str, Any]:
        row = self.get(terrain_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.bundle_path(terrain_id), destination)
        return {
            "source_terrain_id": row["id"],
            "source_terrain_name": row["name"],
            "source_revision": int(row.get("revision", 1)),
            "snapshot": destination.name,
            "snapshot_hash": row.get("content_hash", ""),
            "size_x": float(row["size_x"]),
            "size_z": float(row["size_z"]),
            "resolution_x": int(row["resolution_x"]),
            "resolution_z": int(row["resolution_z"]),
            "transform": _normalize_transform(row.get("transform")),
            "display_by_default": True,
        }

    def delete(self, terrain_id: str) -> None:
        root = (TERRAINS_ROOT / Path(terrain_id).name).resolve()
        if root.parent != TERRAINS_ROOT.resolve() or not root.is_dir():
            raise KeyError(terrain_id)
        shutil.rmtree(root)
