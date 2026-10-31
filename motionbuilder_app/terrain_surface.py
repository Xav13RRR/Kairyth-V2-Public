from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceSamples:
    heights: np.ndarray
    normals: np.ndarray
    hits: np.ndarray


class PlaneSurface:
    """Infinite Y-up reference plane used when no terrain asset is selected."""

    kind = "plane"

    def __init__(self, height: float = 0.0) -> None:
        self.height = float(height)

    def sample_many(self, xz: np.ndarray) -> SurfaceSamples:
        points = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
        heights = np.full(len(points), self.height, dtype=np.float64)
        normals = np.repeat(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64), len(points), axis=0)
        return SurfaceSamples(heights=heights, normals=normals, hits=np.ones(len(points), dtype=bool))

    def sample(self, x: float, z: float) -> tuple[float, np.ndarray, bool]:
        sample = self.sample_many(np.asarray([[x, z]], dtype=np.float64))
        return float(sample.heights[0]), sample.normals[0].copy(), bool(sample.hits[0])


class HeightFieldSurface:
    """Finite regular-grid heightfield with an optional world actor transform."""

    kind = "heightfield"

    def __init__(
        self,
        heights: np.ndarray,
        size_x: float,
        size_z: float,
        *,
        translation: list[float] | tuple[float, float, float] | np.ndarray | None = None,
        yaw_degrees: float = 0.0,
        uniform_scale: float = 1.0,
    ) -> None:
        values = np.asarray(heights, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) < 2:
            raise ValueError("heightfield must have shape [rows, columns] with both dimensions >= 2")
        if not np.isfinite(values).all():
            raise ValueError("heightfield contains non-finite values")
        if float(size_x) <= 0.0 or float(size_z) <= 0.0:
            raise ValueError("terrain dimensions must be positive")
        self.heights = values
        self.size_x = float(size_x)
        self.size_z = float(size_z)
        self.rows, self.columns = values.shape
        self.cell_x = self.size_x / float(self.columns - 1)
        self.cell_z = self.size_z / float(self.rows - 1)
        self.translation = np.asarray(
            translation if translation is not None else [0.0, 0.0, 0.0], dtype=np.float64
        )
        if self.translation.shape != (3,) or not np.isfinite(self.translation).all():
            raise ValueError("terrain translation must contain three finite values")
        self.yaw_degrees = float(yaw_degrees)
        self.uniform_scale = float(uniform_scale)
        if not np.isfinite(self.yaw_degrees):
            raise ValueError("terrain yaw must be finite")
        if not np.isfinite(self.uniform_scale) or self.uniform_scale <= 0.0:
            raise ValueError("terrain uniform scale must be positive")
        yaw = np.deg2rad(self.yaw_degrees)
        self._yaw_cos = float(np.cos(yaw))
        self._yaw_sin = float(np.sin(yaw))

    def sample_many(self, xz: np.ndarray) -> SurfaceSamples:
        points = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
        world_x = points[:, 0] - self.translation[0]
        world_z = points[:, 1] - self.translation[2]
        local_x = (self._yaw_cos * world_x - self._yaw_sin * world_z) / self.uniform_scale
        local_z = (self._yaw_sin * world_x + self._yaw_cos * world_z) / self.uniform_scale
        gx = (local_x + self.size_x * 0.5) / self.cell_x
        gz = (local_z + self.size_z * 0.5) / self.cell_z
        hits = (gx >= 0.0) & (gx <= self.columns - 1) & (gz >= 0.0) & (gz <= self.rows - 1)

        safe_x = np.clip(gx, 0.0, self.columns - 1)
        safe_z = np.clip(gz, 0.0, self.rows - 1)
        x0 = np.floor(safe_x).astype(np.int64)
        z0 = np.floor(safe_z).astype(np.int64)
        x1 = np.minimum(x0 + 1, self.columns - 1)
        z1 = np.minimum(z0 + 1, self.rows - 1)
        tx = safe_x - x0
        tz = safe_z - z0

        h00 = self.heights[z0, x0]
        h10 = self.heights[z0, x1]
        h01 = self.heights[z1, x0]
        h11 = self.heights[z1, x1]
        h0 = h00 * (1.0 - tx) + h10 * tx
        h1 = h01 * (1.0 - tx) + h11 * tx
        sampled = h0 * (1.0 - tz) + h1 * tz

        xm = np.maximum(x0 - 1, 0)
        xp = np.minimum(x1 + 1, self.columns - 1)
        zm = np.maximum(z0 - 1, 0)
        zp = np.minimum(z1 + 1, self.rows - 1)
        dhdx = (self.heights[z0, xp] - self.heights[z0, xm]) / np.maximum((xp - xm) * self.cell_x, 1e-9)
        dhdz = (self.heights[zp, x0] - self.heights[zm, x0]) / np.maximum((zp - zm) * self.cell_z, 1e-9)
        normals = np.column_stack([-dhdx, np.ones(len(points), dtype=np.float64), -dhdz])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        local_normal_x = normals[:, 0].copy()
        local_normal_z = normals[:, 2].copy()
        normals[:, 0] = self._yaw_cos * local_normal_x + self._yaw_sin * local_normal_z
        normals[:, 2] = -self._yaw_sin * local_normal_x + self._yaw_cos * local_normal_z

        sampled = self.translation[1] + sampled.astype(np.float64) * self.uniform_scale
        sampled[~hits] = np.nan
        normals[~hits] = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        return SurfaceSamples(heights=sampled, normals=normals, hits=hits)

    def sample(self, x: float, z: float) -> tuple[float, np.ndarray, bool]:
        sample = self.sample_many(np.asarray([[x, z]], dtype=np.float64))
        return float(sample.heights[0]), sample.normals[0].copy(), bool(sample.hits[0])


def heightfield_mesh(heights: np.ndarray, size_x: float, size_z: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(heights, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("heightfield must have shape [rows, columns]")
    rows, columns = values.shape
    xs = np.linspace(-float(size_x) * 0.5, float(size_x) * 0.5, columns, dtype=np.float32)
    zs = np.linspace(-float(size_z) * 0.5, float(size_z) * 0.5, rows, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs)
    vertices = np.column_stack([grid_x.reshape(-1), values.reshape(-1), grid_z.reshape(-1)]).astype(np.float32)
    faces = np.empty(((rows - 1) * (columns - 1) * 2, 3), dtype=np.int64)
    cursor = 0
    for z in range(rows - 1):
        row = z * columns
        next_row = (z + 1) * columns
        for x in range(columns - 1):
            a, b = row + x, row + x + 1
            c, d = next_row + x, next_row + x + 1
            faces[cursor] = (a, c, b)
            faces[cursor + 1] = (b, c, d)
            cursor += 2
    return vertices, faces
