"""Virtual orthographic facade camera generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

from .constants import (
    CAMERA_BACKOFF_METERS,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    FACADE_CORRIDOR_DEPTH,
)
from .domain import WallSegment


@dataclass(frozen=True)
class FacadeRender:
    wall_index: int
    image_path: Path
    image_rgb: np.ndarray
    camera_position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    clipping_range: tuple[float, float]


class FacadeRenderer:
    """Renders flat wall-facing RGB orthophotos from the unsegmented mesh."""

    def __init__(
        self,
        output_dir: Path,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        corridor_depth: float = FACADE_CORRIDOR_DEPTH,
    ) -> None:
        self.output_dir = output_dir
        self.image_width = image_width
        self.image_height = image_height
        self.corridor_depth = corridor_depth
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_wall(self, mesh: pv.DataSet, wall: WallSegment, ceiling_height: float) -> FacadeRender:
        center_xy = np.array(wall.line.interpolate(0.5, normalized=True).coords[0], dtype=float)
        normal = wall.normal / np.linalg.norm(wall.normal)
        camera_xy = center_xy + normal * CAMERA_BACKOFF_METERS
        facade_mid_z = ceiling_height / 2.0
        position = (float(camera_xy[0]), float(camera_xy[1]), facade_mid_z)
        focal_point = (float(center_xy[0]), float(center_xy[1]), facade_mid_z)

        clipped_mesh = self._corridor_clip(mesh, wall, ceiling_height)
        plotter = pv.Plotter(off_screen=True, window_size=(self.image_width, self.image_height))
        plotter.set_background("white")
        try:
            plotter.add_mesh(clipped_mesh, rgb=True, show_edges=False)
        except Exception:
            plotter.add_mesh(clipped_mesh, color="lightgray", show_edges=False)
        plotter.camera_position = [position, focal_point, (0.0, 0.0, 1.0)]
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = max(ceiling_height, 1.0) / 2.0
        clipping_range = (CAMERA_BACKOFF_METERS - self.corridor_depth, CAMERA_BACKOFF_METERS + self.corridor_depth)
        plotter.camera.clipping_range = clipping_range
        image = plotter.screenshot(return_img=True)
        plotter.close()

        image_rgb = np.asarray(image[:, :, :3], dtype=np.uint8)
        image_path = self.output_dir / f"facade_wall_{wall.wall_index:03d}.png"
        Image.fromarray(image_rgb).save(image_path)
        return FacadeRender(
            wall_index=wall.wall_index,
            image_path=image_path,
            image_rgb=image_rgb,
            camera_position=position,
            focal_point=focal_point,
            clipping_range=clipping_range,
        )

    def _corridor_clip(self, mesh: pv.DataSet, wall: WallSegment, ceiling_height: float) -> pv.DataSet:
        """Clip to a tight axis-aligned facade corridor in wall-local coordinates."""

        start = wall.start
        end = wall.end
        normal = wall.normal / np.linalg.norm(wall.normal)
        tangent = wall.tangent
        corners_2d = [
            start - tangent * 0.25 - normal * (self.corridor_depth / 2.0),
            end + tangent * 0.25 - normal * (self.corridor_depth / 2.0),
            end + tangent * 0.25 + normal * (self.corridor_depth / 2.0),
            start - tangent * 0.25 + normal * (self.corridor_depth / 2.0),
        ]
        z_min, z_max = -0.25, ceiling_height + 0.25
        points = np.array(
            [[x, y, z] for z in (z_min, z_max) for x, y in corners_2d],
            dtype=float,
        )
        box = pv.Box(bounds=(
            float(points[:, 0].min()),
            float(points[:, 0].max()),
            float(points[:, 1].min()),
            float(points[:, 1].max()),
            z_min,
            z_max,
        ))
        try:
            return mesh.clip_box(box, invert=False)
        except Exception:
            return mesh
