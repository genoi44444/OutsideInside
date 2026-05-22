"""Global architectural rules and standardized BIM templates."""

from __future__ import annotations

from dataclasses import dataclass


# Vertical Structural Enforcements
DEFAULT_FLOOR_HEIGHT = 3.3
STUDIO_LEVEL_THRESHOLD = 5.0
REGULATORY_RAILING_HEIGHT = 1.05
WALL_THICKNESS = 0.3
BALCONY_SLAB_THICKNESS = 0.2

# Component Placement Rules
STANDARD_LINTEL_HEIGHT = 2.1
STANDARD_SILL_HEIGHT = 0.9
PRIVACY_SILL_HEIGHT = 1.5
DOOR_SILL_HEIGHT = 0.0

# Tolerance & Noise Gates
VERTICAL_STACK_THRESHOLD = 0.3
MIN_PORCH_PROTRUSION = 0.4
OCCLUSION_DENSITY_THRESHOLD = 0.05
CONFIDENCE_THRESHOLD = 0.65

# Rendering
CAMERA_BACKOFF_METERS = 5.0
FACADE_CORRIDOR_DEPTH = 1.0
DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1080
HEIGHT_GUARDRAIL_TOLERANCE = 0.10


@dataclass(frozen=True)
class BimTemplate:
    """Canonical template dimensions used by the architectural rulebook."""

    width: float
    height: float
    type: str


BIM_TEMPLATES: dict[str, BimTemplate] = {
    "Window_Standard": BimTemplate(width=1.2, height=1.2, type="window"),
    "Window_Bathroom": BimTemplate(width=0.6, height=0.6, type="privacy_window"),
    "Window_Studio": BimTemplate(width=1.2, height=3.0, type="double_height"),
    "Door_Main": BimTemplate(width=1.0, height=2.1, type="door"),
    "Door_Balcony": BimTemplate(width=1.4, height=2.1, type="balcony_door"),
}
