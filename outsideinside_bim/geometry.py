"""Geometry utilities for wall explosion and facade coordinate mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Polygon

from .domain import WallSegment


@dataclass(frozen=True)
class FacadeFrame:
    """Mapping between facade pixels and metric wall-local coordinates."""

    wall: WallSegment
    image_width: int
    image_height: int
    facade_width_m: float
    facade_height_m: float
    z_ground: float
    z_roof: float

    def pixel_to_local(self, bbox_xywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert an image bbox into wall-local meters: x_center, z_center, width, height."""

        x_px, y_px, w_px, h_px = bbox_xywh
        x_center_m = (x_px + w_px / 2.0) / self.image_width * self.facade_width_m
        z_center_m = self.z_roof - (y_px + h_px / 2.0) / self.image_height * self.facade_height_m
        width_m = w_px / self.image_width * self.facade_width_m
        height_m = h_px / self.image_height * self.facade_height_m
        return x_center_m, z_center_m, width_m, height_m


def exterior_wall_segments(footprint: Polygon) -> list[LineString]:
    """Explode the exterior ring into directed wall segment lines."""

    coords = list(footprint.exterior.coords)
    return [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]


def outward_normal_for_line(line: LineString, centroid_xy: np.ndarray) -> np.ndarray:
    """Calculate the outward 2D normal by testing both side normals against the centroid."""

    p0 = np.array(line.coords[0], dtype=float)
    p1 = np.array(line.coords[-1], dtype=float)
    tangent = p1 - p0
    norm = np.linalg.norm(tangent)
    if norm == 0:
        raise ValueError("Cannot calculate normal for a zero-length line.")
    tangent = tangent / norm
    candidates = [
        np.array([-tangent[1], tangent[0]], dtype=float),
        np.array([tangent[1], -tangent[0]], dtype=float),
    ]
    midpoint = (p0 + p1) / 2.0
    to_centroid = centroid_xy - midpoint
    return min(candidates, key=lambda n: float(np.dot(n, to_centroid)))


def create_wall_segments(footprint: Polygon) -> list[WallSegment]:
    centroid_xy = np.array([footprint.centroid.x, footprint.centroid.y], dtype=float)
    walls: list[WallSegment] = []
    for index, line in enumerate(exterior_wall_segments(footprint)):
        normal = outward_normal_for_line(line, centroid_xy)
        walls.append(WallSegment(line=line, normal=normal, wall_index=index))
    return walls


def local_wall_axes(wall: WallSegment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return facade-local x, outward y, and vertical z unit axes in world coordinates."""

    tangent_2d = wall.tangent
    x_axis = np.array([tangent_2d[0], tangent_2d[1], 0.0], dtype=float)
    y_axis = np.array([wall.normal[0], wall.normal[1], 0.0], dtype=float)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    return x_axis, y_axis, z_axis
