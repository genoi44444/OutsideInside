"""CadQuery parametric BIM generation and STEP export."""

from __future__ import annotations

from pathlib import Path

import cadquery as cq
import numpy as np

from .constants import BALCONY_SLAB_THICKNESS, REGULATORY_RAILING_HEIGHT, WALL_THICKNESS
from .domain import BuildingAssembly, BuildingLevel, PorchComponent, WallSegment, WallStatus


class CadQueryBimExporter:
    """Compiles the sanitized level tree into editable STEP solids."""

    def export_step(self, assembly: BuildingAssembly, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cq_assembly = cq.Assembly(name="OutsideInside_Parametric_BIM")

        for level in assembly.levels:
            level_compound = self._level_solid(level)
            cq_assembly.add(level_compound, name=f"Level_{level.level_index:02d}_Walls", color=cq.Color(0.72, 0.72, 0.68))
            for porch_index, porch in enumerate(level.porches):
                cq_assembly.add(
                    self._porch_solid(level, level.walls[porch.wall_index], porch),
                    name=f"Level_{level.level_index:02d}_Porch_{porch_index:02d}",
                    color=cq.Color(0.58, 0.58, 0.55),
                )
                cq_assembly.add(
                    self._guardrail_solid(level, level.walls[porch.wall_index], porch),
                    name=f"Level_{level.level_index:02d}_Guardrail_{porch_index:02d}",
                    color=cq.Color(0.35, 0.35, 0.35),
                )

        cq_assembly.save(str(output_path), exportType="STEP")
        return output_path

    def _level_solid(self, level: BuildingLevel) -> cq.Workplane:
        solids: list[cq.Workplane] = []
        for wall in level.walls:
            wall_solid = self._wall_solid(wall, level.z_start, level.height)
            if wall.status == WallStatus.VIEWABLE:
                for opening in wall.openings:
                    cutter = self._opening_cutter(wall, level.z_start, opening.x_local, opening.z_base_local, opening.width, opening.height)
                    wall_solid = wall_solid.cut(cutter)
            solids.append(wall_solid)
        if not solids:
            return cq.Workplane("XY")
        combined = solids[0]
        for solid in solids[1:]:
            combined = combined.union(solid)
        return combined

    @staticmethod
    def _wall_solid(wall: WallSegment, z_start: float, height: float) -> cq.Workplane:
        tangent = wall.tangent
        normal = wall.normal / np.linalg.norm(wall.normal)
        p0 = wall.start
        p1 = wall.end
        q0 = p0 - normal * WALL_THICKNESS
        q1 = p1 - normal * WALL_THICKNESS
        polygon = [(p0[0], p0[1]), (p1[0], p1[1]), (q1[0], q1[1]), (q0[0], q0[1])]
        return cq.Workplane("XY").polyline(polygon).close().extrude(height).translate((0.0, 0.0, z_start))

    @staticmethod
    def _opening_cutter(
        wall: WallSegment,
        z_level_start: float,
        x_local: float,
        z_base_local: float,
        width: float,
        height: float,
    ) -> cq.Workplane:
        plane = CadQueryBimExporter._wall_plane(wall, x_local, z_level_start + z_base_local + height / 2.0)
        return (
            cq.Workplane(plane)
            .box(width, WALL_THICKNESS * 3.0, height, centered=(True, True, True))
        )

    @staticmethod
    def _porch_solid(level: BuildingLevel, wall: WallSegment, porch: PorchComponent) -> cq.Workplane:
        z_center = level.z_start - BALCONY_SLAB_THICKNESS / 2.0
        plane = CadQueryBimExporter._wall_plane(wall, porch.x_center_local, z_center)
        return (
            cq.Workplane(plane)
            .center(0.0, porch.depth / 2.0)
            .box(porch.width, porch.depth, BALCONY_SLAB_THICKNESS, centered=(True, True, True))
        )

    @staticmethod
    def _guardrail_solid(level: BuildingLevel, wall: WallSegment, porch: PorchComponent) -> cq.Workplane:
        z_center = level.z_start + REGULATORY_RAILING_HEIGHT / 2.0
        rail_thickness = 0.12
        plane = CadQueryBimExporter._wall_plane(wall, porch.x_center_local, z_center)
        return (
            cq.Workplane(plane)
            .center(0.0, porch.depth)
            .box(porch.width, rail_thickness, REGULATORY_RAILING_HEIGHT, centered=(True, True, True))
        )

    @staticmethod
    def _wall_plane(wall: WallSegment, x_local: float, z_world: float) -> cq.Plane:
        point_xy = wall.point_at_local_x(x_local)
        tangent = wall.tangent
        normal = wall.normal / np.linalg.norm(wall.normal)
        generated_y = np.array([-tangent[1], tangent[0]], dtype=float)
        if float(np.dot(generated_y, normal)) < 0.0:
            tangent = -tangent
        return cq.Plane(
            origin=(float(point_xy[0]), float(point_xy[1]), float(z_world)),
            xDir=(float(tangent[0]), float(tangent[1]), 0.0),
            normal=(0.0, 0.0, 1.0),
        )
