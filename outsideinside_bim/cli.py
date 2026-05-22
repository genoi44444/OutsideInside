"""Command-line entry point for the OutsideInside BIM reconstruction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import ReconstructionConfig, ReconstructionEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert fused textured 3D mesh + footprint SHP into a parametric STEP BIM.")
    parser.add_argument("--shp", required=True, type=Path, help="Input building footprint shapefile.")
    parser.add_argument("--mesh", required=True, type=Path, help="Input mesh/scene readable by PyVista.")
    parser.add_argument("--out-step", required=True, type=Path, help="Output STEP file path.")
    parser.add_argument("--out-images", default=Path("facade_renders"), type=Path, help="Directory for rendered facade images.")
    parser.add_argument("--height-field", default="height", help="Footprint attribute containing structural height.")
    parser.add_argument("--target-crs", default=None, help="Optional projected CRS, e.g. EPSG:32636.")
    parser.add_argument("--feature-index", default=0, type=int, help="Shapefile feature index to reconstruct.")
    parser.add_argument("--facade-seg-model", default=None, help="YOLO/SegFormer-exported facade segmentation checkpoint.")
    parser.add_argument("--detector-model", default=None, help="YOLOv8 architectural object detection checkpoint.")
    parser.add_argument("--sam31-model", default=None, help="SAM 3.1 model id/path exposing SAM3SemanticPredictor.")
    parser.add_argument("--sam-device", default="cuda", help="SAM execution device.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ReconstructionConfig(
        shapefile_path=args.shp,
        tileset_path=args.mesh,
        output_step_path=args.out_step,
        output_image_dir=args.out_images,
        height_field=args.height_field,
        target_crs=args.target_crs,
        feature_index=args.feature_index,
        facade_segmenter_model=args.facade_seg_model,
        detector_model=args.detector_model,
        sam31_model=args.sam31_model,
        sam_device=args.sam_device,
    )
    output = ReconstructionEngine(config).run()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
