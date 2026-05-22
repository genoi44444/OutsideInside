# OutsideInside BIM Reconstruction

Production-oriented Python pipeline for converting a monolithic, unclassified textured mesh plus a 2D building footprint shapefile into a level-nested parametric `.step` model.

The core assumption is architectural rule domination: the fused mesh is only a visual canvas. The footprint and height attribute define plan and envelope constraints, virtual orthographic cameras create facade evidence, AI perception proposes features, and strict BIM rules snap those observations into standardized elements.

## Run

```powershell
python -m outsideinside_bim.cli `
  --shp .\data\footprints.shp `
  --mesh .\data\scene.glb `
  --out-step .\out\building.step `
  --out-images .\out\facades `
  --height-field height `
  --target-crs EPSG:32636 `
  --facade-seg-model .\models\facade-seg.pt `
  --detector-model .\models\architectural-yolo.pt `
  --sam31-model .\models\sam3.1
```

## Notes

- `pyvista.read()` must be able to load the mesh path. If the source is raw Cesium 3D Tiles, preconvert it to glTF/OBJ/PLY/VTP/VTK before this stage.
- The SAM fallback expects a deployment exposing `SAM3SemanticPredictor` from either `transformers` or Meta's `sam3` package. If no SAM model is supplied, the primary YOLO detector still runs.
- If no detector checkpoints are supplied, the package still builds envelope geometry from the footprint and height attribute, but no openings will be inferred.
- All generated assets are owned by `BuildingAssembly -> BuildingLevel -> WallSegment/PorchComponent`; no opening or porch is created outside a level.

## Modules

- `constants.py`: global architectural rules and exact template dimensions.
- `domain.py`: level-centric object model.
- `ingestion.py`: GeoPandas/PyVista loading and CRS synchronization.
- `rendering.py`: orthographic facade camera and corridor clipping.
- `perception.py`: facade segmentation, YOLO object detection, SAM 3.1 fallback.
- `rulebook.py`: level deduction, snapping, occlusion filtering, balcony access, vertical stacking.
- `cad_export.py`: CadQuery wall, opening, slab, and guardrail generation.
- `engine.py`: end-to-end orchestrator.
