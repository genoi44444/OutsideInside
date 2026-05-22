"""Architectural Rule Domination: sanitize perception into level-owned BIM assets."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from .constants import (
    BIM_TEMPLATES,
    DEFAULT_FLOOR_HEIGHT,
    DOOR_SILL_HEIGHT,
    HEIGHT_GUARDRAIL_TOLERANCE,
    MIN_PORCH_PROTRUSION,
    OCCLUSION_DENSITY_THRESHOLD,
    PRIVACY_SILL_HEIGHT,
    STANDARD_LINTEL_HEIGHT,
    STANDARD_SILL_HEIGHT,
    STUDIO_LEVEL_THRESHOLD,
    VERTICAL_STACK_THRESHOLD,
)
from .domain import BuildingAssembly, BuildingLevel, Opening, PorchComponent, WallSegment, WallStatus
from .geometry import FacadeFrame
from .perception import Detection, FacadeSemantics


@dataclass(frozen=True)
class WallPerceptionPacket:
    wall: WallSegment
    semantics: FacadeSemantics
    detections: list[Detection]


class ArchitecturalRulebook:
    """Transforms noisy image observations into strict parametric architecture."""

    def apply(self, assembly: BuildingAssembly, packets: list[WallPerceptionPacket]) -> BuildingAssembly:
        ceiling = self._guardrailed_ceiling(assembly.global_ceiling, packets)
        levels = self._deduce_levels(ceiling, packets)
        wall_templates = [packet.wall for packet in packets]
        assembly.levels = [
            BuildingLevel(
                level_index=level.level_index,
                z_start=level.z_start,
                z_end=level.z_end,
                is_studio=level.is_studio,
                walls=[self._clone_wall_structure(wall) for wall in wall_templates],
            )
            for level in levels
        ]
        base_stack_centers: dict[int, list[float]] = {}

        for packet in packets:
            if packet.semantics.texture_density < OCCLUSION_DENSITY_THRESHOLD:
                self._mark_unviewable(assembly, packet.wall.wall_index)
                continue
            frame = FacadeFrame(
                wall=packet.wall,
                image_width=packet.semantics.building_mask.shape[1],
                image_height=packet.semantics.building_mask.shape[0],
                facade_width_m=packet.wall.length,
                facade_height_m=ceiling,
                z_ground=0.0,
                z_roof=ceiling,
            )
            for detection in packet.detections:
                x_local, z_center, width_m, height_m = frame.pixel_to_local(detection.bbox_xywh)
                level = self._level_for_z(assembly.levels, z_center)
                if level is None:
                    continue
                level_wall = level.walls[packet.wall.wall_index]
                label = detection.label.lower()
                if "balcony" in label or "porch" in label:
                    self._add_porch(level, packet.wall.wall_index, x_local, max(width_m, 1.6), detection.confidence)
                    self._force_balcony_door(level_wall, x_local, detection.confidence)
                    continue
                if "door" in label:
                    self._add_opening(level_wall, "Door_Main", x_local, DOOR_SILL_HEIGHT, "door", detection.confidence)
                    continue
                if "window" in label:
                    self._add_window(level, level_wall, packet.wall.wall_index, x_local, width_m, height_m, z_center, detection)
                    if level.level_index == 0:
                        base_stack_centers.setdefault(packet.wall.wall_index, []).append(x_local)

        self._enforce_vertical_stacks(assembly, base_stack_centers)
        return assembly

    @staticmethod
    def _guardrailed_ceiling(shapefile_height: float, packets: list[WallPerceptionPacket]) -> float:
        image_heights = []
        for packet in packets:
            h_px = packet.semantics.building_mask.shape[0]
            pixel_span = max(1, packet.semantics.z_ground_pixel - packet.semantics.z_roof_pixel)
            image_heights.append(shapefile_height * pixel_span / h_px)
        median_image_height = float(np.median(image_heights)) if image_heights else shapefile_height
        deviation = abs(median_image_height - shapefile_height) / shapefile_height
        return shapefile_height if deviation > HEIGHT_GUARDRAIL_TOLERANCE else median_image_height

    def _deduce_levels(self, ceiling: float, packets: list[WallPerceptionPacket]) -> list[BuildingLevel]:
        z_centers: list[float] = []
        for packet in packets:
            frame = FacadeFrame(
                wall=packet.wall,
                image_width=packet.semantics.building_mask.shape[1],
                image_height=packet.semantics.building_mask.shape[0],
                facade_width_m=packet.wall.length,
                facade_height_m=ceiling,
                z_ground=0.0,
                z_roof=ceiling,
            )
            for det in packet.detections:
                if "window" in det.label.lower():
                    _, z_center, _, _ = frame.pixel_to_local(det.bbox_xywh)
                    z_centers.append(z_center)

        if not z_centers:
            return self._regular_levels(ceiling)
        clusters = self._cluster_1d(sorted(z_centers), tolerance=DEFAULT_FLOOR_HEIGHT * 0.35)
        if len(clusters) < 2:
            return self._regular_levels(ceiling)
        inferred_boundaries = [0.0]
        centers = [float(np.median(cluster)) for cluster in clusters]
        for a, b in zip(centers, centers[1:]):
            inferred_boundaries.append((a + b) / 2.0)
        inferred_boundaries.append(ceiling)
        levels: list[BuildingLevel] = []
        for i, (z0, z1) in enumerate(zip(inferred_boundaries, inferred_boundaries[1:])):
            if z1 - z0 < DEFAULT_FLOOR_HEIGHT * 0.55:
                continue
            levels.append(BuildingLevel(i, z0, z1, is_studio=(z1 - z0) > STUDIO_LEVEL_THRESHOLD))
        return levels or self._regular_levels(ceiling)

    @staticmethod
    def _regular_levels(ceiling: float) -> list[BuildingLevel]:
        count = max(1, ceil(ceiling / DEFAULT_FLOOR_HEIGHT))
        height = ceiling / count
        return [
            BuildingLevel(index, index * height, min(ceiling, (index + 1) * height), is_studio=height > STUDIO_LEVEL_THRESHOLD)
            for index in range(count)
        ]

    @staticmethod
    def _cluster_1d(values: list[float], tolerance: float) -> list[list[float]]:
        clusters: list[list[float]] = []
        for value in values:
            if not clusters or abs(np.median(clusters[-1]) - value) > tolerance:
                clusters.append([value])
            else:
                clusters[-1].append(value)
        return clusters

    @staticmethod
    def _clone_wall_structure(wall: WallSegment) -> WallSegment:
        return WallSegment(
            line=wall.line,
            normal=wall.normal.copy(),
            wall_index=wall.wall_index,
            status=wall.status,
            facade_image=wall.facade_image,
            facade_metadata=dict(wall.facade_metadata),
        )

    @staticmethod
    def _mark_unviewable(assembly: BuildingAssembly, wall_index: int) -> None:
        for level in assembly.levels:
            level.walls[wall_index].status = WallStatus.UNVIEWABLE
            level.walls[wall_index].openings.clear()

    def _add_window(
        self,
        level: BuildingLevel,
        wall: WallSegment,
        wall_index: int,
        x_local: float,
        width_m: float,
        height_m: float,
        z_center_abs: float,
        detection: Detection,
    ) -> None:
        z_center_local = z_center_abs - level.z_start
        area = width_m * height_m
        if area < 0.6 and z_center_local > 1.5:
            template_name = "Window_Bathroom"
            z_base = PRIVACY_SILL_HEIGHT
        elif level.is_studio and level.height >= BIM_TEMPLATES["Window_Studio"].height:
            template_name = "Window_Studio"
            z_base = max(0.2, STANDARD_LINTEL_HEIGHT - BIM_TEMPLATES[template_name].height)
        else:
            template_name = "Window_Standard"
            z_base = STANDARD_SILL_HEIGHT
        self._add_opening(wall, template_name, x_local, z_base, detection.label, detection.confidence)

    @staticmethod
    def _add_opening(wall: WallSegment, template_name: str, x_local: float, z_base: float, label: str, confidence: float) -> None:
        template = BIM_TEMPLATES[template_name]
        clamped_x = min(max(x_local, template.width / 2.0), max(template.width / 2.0, wall.length - template.width / 2.0))
        wall.openings.append(
            Opening(
                template_name=template_name,
                x_local=clamped_x,
                z_base_local=z_base,
                width=template.width,
                height=template.height,
                source_label=label,
                confidence=confidence,
            )
        )

    def _force_balcony_door(self, wall: WallSegment, x_local: float, confidence: float) -> None:
        self._add_opening(wall, "Door_Balcony", x_local, DOOR_SILL_HEIGHT, "forced_balcony_access", confidence)

    @staticmethod
    def _add_porch(level: BuildingLevel, wall_index: int, x_local: float, width: float, confidence: float) -> None:
        level.porches.append(
            PorchComponent(
                wall_index=wall_index,
                x_center_local=x_local,
                width=width,
                depth=max(MIN_PORCH_PROTRUSION, 1.2),
                z_floor=level.z_start,
                source_confidence=confidence,
            )
        )

    @staticmethod
    def _level_for_z(levels: list[BuildingLevel], z_abs: float) -> BuildingLevel | None:
        for level in levels:
            if level.z_start <= z_abs <= level.z_end:
                return level
        return None

    @staticmethod
    def _enforce_vertical_stacks(assembly: BuildingAssembly, base_stack_centers: dict[int, list[float]]) -> None:
        for level in assembly.levels:
            if level.level_index == 0:
                continue
            for wall in level.walls:
                lower_centers = base_stack_centers.get(wall.wall_index, [])
                for opening in wall.openings:
                    if not opening.template_name.startswith("Window"):
                        continue
                    nearest = min(lower_centers, key=lambda base_x: abs(base_x - opening.x_local), default=None)
                    if nearest is not None and abs(nearest - opening.x_local) <= VERTICAL_STACK_THRESHOLD:
                        opening.x_local = nearest
