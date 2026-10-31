from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .terrain_store import TerrainStore


class TerrainCreateBody(BaseModel):
    name: str = Field(default="New Terrain", min_length=1, max_length=80)
    size_x: float = Field(default=8.0, ge=0.5, le=4096.0)
    size_z: float = Field(default=8.0, ge=0.5, le=4096.0)
    resolution: int = Field(default=129)
    initial_height: float = Field(default=0.0, ge=-1024.0, le=1024.0)


class TerrainRenameBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class TerrainTransformBody(BaseModel):
    translation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    yaw_degrees: float = Field(default=0.0)
    uniform_scale: float = Field(default=1.0, ge=0.01, le=100.0)


class TerrainHeightfieldBody(BaseModel):
    heights: list[list[float]]
    expected_revision: int | None = Field(default=None, ge=1)
    transform: TerrainTransformBody | None = None


def build_terrain_router(store: TerrainStore) -> APIRouter:
    router = APIRouter(prefix="/api/terrains", tags=["terrains"])

    @router.get("")
    def list_terrains() -> list[dict]:
        return store.list()

    @router.post("", status_code=201)
    def create_terrain(body: TerrainCreateBody) -> dict:
        try:
            return store.create(
                body.name,
                size_x=body.size_x,
                size_z=body.size_z,
                resolution=body.resolution,
                initial_height=body.initial_height,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.get("/{terrain_id}")
    def get_terrain(terrain_id: str) -> dict:
        try:
            return store.get(terrain_id)
        except KeyError as exc:
            raise HTTPException(404, "terrain not found") from exc

    @router.patch("/{terrain_id}")
    def rename_terrain(terrain_id: str, body: TerrainRenameBody) -> dict:
        try:
            return store.rename(terrain_id, body.name)
        except KeyError as exc:
            raise HTTPException(404, "terrain not found") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.get("/{terrain_id}/scene")
    def terrain_scene(terrain_id: str) -> dict:
        try:
            return store.scene(terrain_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(404, "terrain not found") from exc

    @router.put("/{terrain_id}/heightfield")
    def update_heightfield(terrain_id: str, body: TerrainHeightfieldBody) -> dict:
        try:
            return store.update_heights(
                terrain_id,
                np.asarray(body.heights, dtype=np.float32),
                expected_revision=body.expected_revision,
                transform=body.transform.model_dump() if body.transform is not None else None,
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(404, "terrain not found") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.delete("/{terrain_id}")
    def delete_terrain(terrain_id: str) -> dict:
        try:
            store.delete(terrain_id)
        except KeyError as exc:
            raise HTTPException(404, "terrain not found") from exc
        return {"ok": True}

    return router
