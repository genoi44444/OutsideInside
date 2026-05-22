"""Input loaders and CRS synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import open3d as o3d
import pyvista as pv
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon

from .domain import BuildingAssembly


@dataclass(frozen=True)
class IngestionResult:
    assembly: BuildingAssembly
    mesh: pv.DataSet
    open3d_mesh: o3d.geometry.TriangleMesh | None
    source_crs: Any
    metric_crs: CRS


class IngestionService:
    """Loads the shapefile footprint, height attribute, and monolithic 3D scene."""

    def __init__(self, height_field: str = "height", target_crs: str | None = None) -> None:
        self.height_field = height_field
        self.target_crs = CRS.from_user_input(target_crs) if target_crs else None

    def load(self, shapefile_path: Path, tileset_path: Path, feature_index: int = 0) -> IngestionResult:
        footprints = gpd.read_file(shapefile_path)
        if footprints.empty:
            raise ValueError(f"No features found in {shapefile_path}.")
        if self.height_field not in footprints.columns:
            raise KeyError(f"Missing required height attribute '{self.height_field}'.")

        source_crs = footprints.crs
        metric_crs = self.target_crs or self._best_metric_crs(footprints)
        footprints = footprints.to_crs(metric_crs)
        feature = footprints.iloc[feature_index]
        polygon = self._polygon_from_geometry(feature.geometry)
        global_ceiling = self._parse_height(feature[self.height_field])
        mesh = self._load_mesh(tileset_path)
        open3d_mesh = self._load_open3d_mesh_for_audit(tileset_path)

        assembly = BuildingAssembly(footprint=polygon, global_ceiling=global_ceiling, crs=metric_crs)
        return IngestionResult(assembly=assembly, mesh=mesh, open3d_mesh=open3d_mesh, source_crs=source_crs, metric_crs=metric_crs)

    @staticmethod
    def _best_metric_crs(footprints: gpd.GeoDataFrame) -> CRS:
        if footprints.crs and CRS.from_user_input(footprints.crs).is_projected:
            return CRS.from_user_input(footprints.crs)
        estimated = footprints.estimate_utm_crs()
        if estimated is None:
            raise ValueError("Could not infer a metric CRS. Pass --target-crs explicitly.")
        return CRS.from_user_input(estimated)

    @staticmethod
    def _polygon_from_geometry(geometry: Any) -> Polygon:
        if isinstance(geometry, Polygon):
            return geometry
        if isinstance(geometry, MultiPolygon):
            return max(geometry.geoms, key=lambda poly: poly.area)
        raise TypeError(f"Expected Polygon or MultiPolygon footprint, got {type(geometry)!r}.")

    @staticmethod
    def _parse_height(raw: Any) -> float:
        if raw is None:
            raise ValueError("Height attribute is null.")
        try:
            height = float(str(raw).strip().replace("m", ""))
        except ValueError as exc:
            raise ValueError(f"Height attribute is not numeric: {raw!r}") from exc
        if height <= 0:
            raise ValueError(f"Height must be positive, got {height}.")
        return height

    @staticmethod
    def _load_mesh(tileset_path: Path) -> pv.DataSet:
        """Load a textured scene PyVista can read.

        For 3D Tiles archives, preconvert the tileset into glTF/OBJ/PLY/VTP/VTK with
        Cesium tooling or py3dtiles before this pipeline stage.
        """

        if not tileset_path.exists():
            raise FileNotFoundError(tileset_path)
        return pv.read(tileset_path)

    @staticmethod
    def _load_open3d_mesh_for_audit(tileset_path: Path) -> o3d.geometry.TriangleMesh | None:
        """Load an Open3D copy when the format is supported for normals/bounds QA."""

        try:
            mesh = o3d.io.read_triangle_mesh(str(tileset_path), enable_post_processing=True)
            if mesh.is_empty():
                return None
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            return mesh
        except Exception:
            return None
