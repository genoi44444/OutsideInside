"""Level-centric architectural object tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import LineString, Polygon


class WallStatus(str, Enum):
    VIEWABLE = "VIEWABLE"
    UNVIEWABLE = "UNVIEWABLE"


@dataclass
class Opening:
    """A sanitized opening attached to exactly one wall and one level."""

    template_name: str
    x_local: float
    z_base_local: float
    width: float
    height: float
    source_label: str
    confidence: float


@dataclass
class PorchComponent:
    """Parametric porch/balcony component owned by a level."""

    wall_index: int
    x_center_local: float
    width: float
    depth: float
    z_floor: float
    source_confidence: float


@dataclass
class WallSegment:
    """Directed 2D wall segment with facade-local child assets."""

    line: LineString
    normal: np.ndarray
    wall_index: int
    status: WallStatus = WallStatus.VIEWABLE
    openings: list[Opening] = field(default_factory=list)
    facade_image: Path | None = None
    facade_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> float:
        return float(self.line.length)

    @property
    def start(self) -> np.ndarray:
        x, y = self.line.coords[0]
        return np.array([x, y], dtype=float)

    @property
    def end(self) -> np.ndarray:
        x, y = self.line.coords[-1]
        return np.array([x, y], dtype=float)

    @property
    def tangent(self) -> np.ndarray:
        direction = self.end - self.start
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError(f"Wall {self.wall_index} has zero length.")
        return direction / norm

    def point_at_local_x(self, x_local: float) -> np.ndarray:
        return self.start + self.tangent * x_local


@dataclass
class BuildingLevel:
    """A single floor/studio volume and all child components it owns."""

    level_index: int
    z_start: float
    z_end: float
    is_studio: bool = False
    walls: list[WallSegment] = field(default_factory=list)
    porches: list[PorchComponent] = field(default_factory=list)

    @property
    def height(self) -> float:
        return self.z_end - self.z_start


@dataclass
class BuildingAssembly:
    """Root assembly; no geometry floats outside its level hierarchy."""

    footprint: Polygon
    global_ceiling: float
    levels: list[BuildingLevel] = field(default_factory=list)
    crs: Any | None = None

    @property
    def centroid_xy(self) -> np.ndarray:
        centroid = self.footprint.centroid
        return np.array([centroid.x, centroid.y], dtype=float)
