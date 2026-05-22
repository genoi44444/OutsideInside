"""End-to-end reconstruction execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cad_export import CadQueryBimExporter
from .geometry import create_wall_segments
from .ingestion import IngestionService
from .perception import (
    ArchitecturalObjectDetector,
    DualEnginePerception,
    Sam31FallbackDetector,
    SemanticFacadeSegmenter,
)
from .rendering import FacadeRenderer
from .rulebook import ArchitecturalRulebook, WallPerceptionPacket


@dataclass(frozen=True)
class ReconstructionConfig:
    shapefile_path: Path
    tileset_path: Path
    output_step_path: Path
    output_image_dir: Path
    height_field: str = "height"
    target_crs: str | None = None
    feature_index: int = 0
    facade_segmenter_model: str | None = None
    detector_model: str | None = None
    sam31_model: str | None = None
    sam_device: str = "cuda"


class ReconstructionEngine:
    """Coordinates ingestion, facade rendering, AI perception, rules, and CAD export."""

    def __init__(self, config: ReconstructionConfig) -> None:
        self.config = config
        self.ingestion = IngestionService(height_field=config.height_field, target_crs=config.target_crs)
        self.renderer = FacadeRenderer(output_dir=config.output_image_dir)
        self.semantic_segmenter = SemanticFacadeSegmenter(model_path=config.facade_segmenter_model)
        self.perception = DualEnginePerception(
            ArchitecturalObjectDetector(model_path=config.detector_model),
            Sam31FallbackDetector(model_id=config.sam31_model, device=config.sam_device),
        )
        self.rulebook = ArchitecturalRulebook()
        self.exporter = CadQueryBimExporter()

    def run(self) -> Path:
        ingestion = self.ingestion.load(
            shapefile_path=self.config.shapefile_path,
            tileset_path=self.config.tileset_path,
            feature_index=self.config.feature_index,
        )
        assembly = ingestion.assembly
        walls = create_wall_segments(assembly.footprint)

        packets: list[WallPerceptionPacket] = []
        for wall in walls:
            render = self.renderer.render_wall(ingestion.mesh, wall, assembly.global_ceiling)
            wall.facade_image = render.image_path
            wall.facade_metadata = {
                "camera_position": render.camera_position,
                "focal_point": render.focal_point,
                "clipping_range": render.clipping_range,
                "metric_crs": ingestion.metric_crs.to_string(),
            }
            semantics = self.semantic_segmenter.segment(render.image_rgb, assembly.global_ceiling)
            detections = self.perception.detect(render.image_rgb, semantics)
            packets.append(WallPerceptionPacket(wall=wall, semantics=semantics, detections=detections))

        sanitized = self.rulebook.apply(assembly, packets)
        return self.exporter.export_step(sanitized, self.config.output_step_path)
