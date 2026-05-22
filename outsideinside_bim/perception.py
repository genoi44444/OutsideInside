"""AI perception adapters for facade masks, object detection, and SAM fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .constants import CONFIDENCE_THRESHOLD, OCCLUSION_DENSITY_THRESHOLD


@dataclass(frozen=True)
class Detection:
    label: str
    bbox_xywh: tuple[float, float, float, float]
    confidence: float
    source: str


@dataclass(frozen=True)
class FacadeSemantics:
    building_mask: np.ndarray
    z_ground_pixel: int
    z_roof_pixel: int
    image_height_m: float
    texture_density: float


class SemanticFacadeSegmenter:
    """Extracts building facade masks from an RGB facade orthophoto."""

    def __init__(self, model_path: str | None = None, facade_class_name: str = "building_facade") -> None:
        self.model_path = model_path
        self.facade_class_name = facade_class_name
        self._model = None
        if model_path:
            from ultralytics import YOLO

            self._model = YOLO(model_path)

    def segment(self, image_rgb: np.ndarray, fallback_height_m: float) -> FacadeSemantics:
        if self._model is None:
            mask = self._fallback_non_white_mask(image_rgb)
        else:
            mask = self._ultralytics_facade_mask(image_rgb)
            if not mask.any():
                mask = self._fallback_non_white_mask(image_rgb)

        ground = self._interpolated_bottom_boundary(mask)
        roof = self._top_boundary(mask)
        density = self._texture_density(image_rgb, mask)
        return FacadeSemantics(
            building_mask=mask,
            z_ground_pixel=ground,
            z_roof_pixel=roof,
            image_height_m=fallback_height_m,
            texture_density=density,
        )

    def _ultralytics_facade_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        results = self._model.predict(source=image_rgb, verbose=False)
        height, width = image_rgb.shape[:2]
        out = np.zeros((height, width), dtype=bool)
        for result in results:
            if result.masks is None:
                continue
            names = result.names
            classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []
            for mask_tensor, cls in zip(result.masks.data, classes):
                label = names.get(int(cls), str(cls))
                if label == self.facade_class_name:
                    resized = np.asarray(Image.fromarray(mask_tensor.cpu().numpy()).resize((width, height))) > 0.5
                    out |= resized
        return out

    @staticmethod
    def _fallback_non_white_mask(image_rgb: np.ndarray) -> np.ndarray:
        luminance = image_rgb.astype(float).mean(axis=2)
        chroma = image_rgb.astype(float).std(axis=2)
        return (luminance < 245) | (chroma > 8)

    @staticmethod
    def _interpolated_bottom_boundary(mask: np.ndarray) -> int:
        ys = []
        for x in range(mask.shape[1]):
            col = np.flatnonzero(mask[:, x])
            if col.size:
                ys.append(col.max())
        if not ys:
            return mask.shape[0] - 1
        return int(np.median(ys))

    @staticmethod
    def _top_boundary(mask: np.ndarray) -> int:
        ys = []
        for x in range(mask.shape[1]):
            col = np.flatnonzero(mask[:, x])
            if col.size:
                ys.append(col.min())
        if not ys:
            return 0
        return int(np.median(ys))

    @staticmethod
    def _texture_density(image_rgb: np.ndarray, mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        grayscale = image_rgb.astype(float).mean(axis=2)
        gx = np.abs(np.diff(grayscale, axis=1, append=grayscale[:, -1:]))
        gy = np.abs(np.diff(grayscale, axis=0, append=grayscale[-1:, :]))
        edges = (gx + gy) > 18.0
        return float(edges[mask].mean())


class ArchitecturalObjectDetector:
    """Primary task-specific YOLO detector for windows, doors, and balconies."""

    def __init__(self, model_path: str | None = None, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model = None
        if model_path:
            from ultralytics import YOLO

            self._model = YOLO(model_path)

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        if self._model is None:
            return []
        detections: list[Detection] = []
        for result in self._model.predict(source=image_rgb, conf=self.confidence_threshold, verbose=False):
            names = result.names
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy()
            for box, class_id, score in zip(xyxy, cls, conf):
                x1, y1, x2, y2 = map(float, box)
                detections.append(
                    Detection(
                        label=names.get(int(class_id), str(class_id)),
                        bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                        confidence=float(score),
                        source="ultralytics",
                    )
                )
        return detections


class Sam31FallbackDetector:
    """Generalized promptable concept segmentation fallback using SAM 3.1.

    The adapter expects a deployment exposing ``SAM3SemanticPredictor`` either
    from ``transformers`` or from Meta's ``sam3`` package. It keeps the rest of
    the BIM pipeline stable while model packaging evolves.
    """

    CONCEPTS = ("window", "exterior door", "cantilevered structural balcony")

    def __init__(self, model_id: str | None = None, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self._predictor = None
        if model_id:
            self._predictor = self._load_predictor(model_id, device)

    def detect(self, image_rgb: np.ndarray, concepts: Iterable[str] | None = None) -> list[Detection]:
        if self._predictor is None:
            return []
        image = Image.fromarray(image_rgb)
        detections: list[Detection] = []
        for concept in concepts or self.CONCEPTS:
            masks_scores = self._predict_concept(image, concept)
            for mask, score in masks_scores:
                if score < CONFIDENCE_THRESHOLD:
                    continue
                bbox = self._mask_to_bbox(mask)
                if bbox is None:
                    continue
                detections.append(Detection(label=concept, bbox_xywh=bbox, confidence=score, source="sam3.1"))
        return detections

    @staticmethod
    def _load_predictor(model_id: str, device: str):
        try:
            from transformers import SAM3SemanticPredictor  # type: ignore

            predictor = SAM3SemanticPredictor.from_pretrained(model_id)
            return predictor.to(device) if hasattr(predictor, "to") else predictor
        except Exception:
            from sam3 import SAM3SemanticPredictor  # type: ignore

            return SAM3SemanticPredictor.from_pretrained(model_id, device=device)

    def _predict_concept(self, image: Image.Image, concept: str) -> list[tuple[np.ndarray, float]]:
        result = self._predictor.predict(image=image, text=concept)
        masks = result.get("masks", [])
        scores = result.get("scores", [1.0] * len(masks))
        return [(np.asarray(mask, dtype=bool), float(score)) for mask, score in zip(masks, scores)]

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
        y, x = np.where(mask)
        if x.size == 0 or y.size == 0:
            return None
        x1, x2 = float(x.min()), float(x.max())
        y1, y2 = float(y.min()), float(y.max())
        return x1, y1, x2 - x1 + 1.0, y2 - y1 + 1.0


class DualEnginePerception:
    """Runs primary YOLO detection and falls back to SAM for weak/empty regions."""

    def __init__(self, primary: ArchitecturalObjectDetector, fallback: Sam31FallbackDetector) -> None:
        self.primary = primary
        self.fallback = fallback

    def detect(self, image_rgb: np.ndarray, semantics: FacadeSemantics) -> list[Detection]:
        if semantics.texture_density < OCCLUSION_DENSITY_THRESHOLD:
            return []
        detections = self.primary.detect(image_rgb)
        needs_fallback = not detections or any(det.confidence < CONFIDENCE_THRESHOLD for det in detections)
        if needs_fallback:
            detections.extend(self.fallback.detect(image_rgb))
        return self._deduplicate(detections)

    @staticmethod
    def _deduplicate(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
        ordered = sorted(detections, key=lambda det: det.confidence, reverse=True)
        kept: list[Detection] = []
        for det in ordered:
            if all(_bbox_iou(det.bbox_xywh, other.bbox_xywh) < iou_threshold for other in kept):
                kept.append(det)
        return kept


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / union
