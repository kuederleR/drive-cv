"""
Data structures and enumerations for DriveCV.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
import numpy as np


class TrackLifecycle(Enum):
    """Lifecycle state of an object track."""
    TENTATIVE = auto()
    CONFIRMED = auto()
    LOST = auto()
    DELETED = auto()


class ADASAlertLevel(Enum):
    """Alert severity levels for ADAS features."""
    NONE = 0
    SAFE = 1
    CAUTION = 2
    WARNING = 3
    CRITICAL = 4


class LDWState(Enum):
    """Lane Departure Warning states."""
    NORMAL = auto()
    WARNING_LEFT = auto()
    WARNING_RIGHT = auto()


@dataclass
class BoundingBox:
    """Bounding box in [x, y, w, h] pixel coordinates."""
    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x

    @property
    def y1(self) -> float:
        return self.y

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h)

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def aspect_ratio(self) -> float:
        return self.w / max(1e-4, self.h)

    def as_xywh(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_int_xywh(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x)), int(round(self.y)), int(round(self.w)), int(round(self.h)))

    def as_int_xyxy(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))


@dataclass
class Detection:
    """Raw perception detection from neural network or sensor."""
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str = "vehicle"
    source: str = "yolopv2"


def _eval_lane_poly(coeffs: np.ndarray, y: float, y_bot: float, y_top: float) -> float:
    """x = a*yn^2 + b*yn + c with yn mapped from [y_bot, y_top]."""
    denom = y_top - y_bot
    yn = 0.0 if abs(denom) < 1e-3 else (float(y) - y_bot) / denom
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return a * yn * yn + b * yn + c


@dataclass
class LaneBoundaries:
    """Host ego-lane geometry boundaries and drivable corridor."""
    left_line: Optional[np.ndarray] = None   # [x_bot, x_top] at (y_bot, y_roi_top)
    right_line: Optional[np.ndarray] = None  # [x_bot, x_top]
    y_top: int = 0
    y_bot: int = 0
    y_roi_top: int = 0
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    lane_center_bottom: float = 0.0
    lane_width_bottom: float = 0.0
    vanish_x: Optional[float] = None
    vanish_y: Optional[float] = None
    drivable_polygon: Optional[np.ndarray] = None
    da_mask: Optional[np.ndarray] = None
    ll_mask: Optional[np.ndarray] = None
    left_type: str = "solid_yellow"
    right_type: str = "solid_white"
    left_color: str = "yellow"
    right_color: str = "white"
    left_pattern: str = "solid"
    right_pattern: str = "solid"
    left_poly: Optional[np.ndarray] = None       # [a, b, c] for x(yn)
    right_poly: Optional[np.ndarray] = None
    left_poly_px: Optional[np.ndarray] = None    # Nx2 image samples [[x, y], ...]
    right_poly_px: Optional[np.ndarray] = None
    left_poly_m: Optional[np.ndarray] = None     # Nx2 ground [[x_m, z_m], ...]
    right_poly_m: Optional[np.ndarray] = None
    curvature_1pm: float = 0.0                   # centerline d^2x/dz^2 (1/m)

    @property
    def is_valid(self) -> bool:
        return (self.left_confidence > 0.0 or self.right_confidence > 0.0)

    def x_at_side(self, side: str, y: float) -> Optional[float]:
        """Image x of one host line at row y (poly if present, else linear chord)."""
        y_top = float(self.y_roi_top if self.y_roi_top > 0 else self.y_top)
        y_bot = float(self.y_bot)
        denom = y_top - y_bot
        coeffs = self.left_poly if side == "left" else self.right_poly
        line = self.left_line if side == "left" else self.right_line
        if coeffs is not None and len(coeffs) >= 3:
            return _eval_lane_poly(coeffs, y, y_bot, y_top)
        if line is None:
            return None
        if abs(denom) < 1e-3:
            return float(line[0])
        t = (float(y) - y_bot) / denom
        return float(line[0] + t * (line[1] - line[0]))

    def x_bounds_at(self, y: float) -> Optional[Tuple[float, float]]:
        """Interpolates left/right lane x at image row y using poly or linear chords."""
        xl = self.x_at_side("left", y)
        xr = self.x_at_side("right", y)
        if xl is None or xr is None:
            return None
        if xr < xl + 8.0:
            return None
        return xl, xr

    def contains_contact(self, u: float, v: float, margin_frac: float = 0.12) -> bool:
        """True if ground-contact pixel (u, v) lies in the ego-lane trapezoid."""
        bounds = self.x_bounds_at(v)
        if bounds is None:
            return False
        y_top = float(self.y_roi_top if self.y_roi_top > 0 else self.y_top)
        if self.vanish_y is not None:
            y_top = min(y_top, float(self.vanish_y) + 20.0)
        if v < y_top or v > float(self.y_bot) + 4.0:
            return False
        xl, xr = bounds
        width = xr - xl
        pad = width * margin_frac
        return (xl - pad) <= u <= (xr + pad)


@dataclass
class Kinematics:
    """3D and 2D kinematic state of a tracked vehicle."""
    velocity_2d: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))  # [vx, vy] px/frame
    distance_m: float = 0.0               # Estimated longitudinal distance to target in meters
    lateral_offset_m: float = 0.0         # Estimated lateral offset from camera axis in meters
    rel_speed_mps: float = 0.0            # Relative closing speed (positive = approaching)
    rel_speed_kmh: float = 0.0            # Relative closing speed in km/h
    ttc_seconds: Optional[float] = None   # Time to collision in seconds
    is_lead_vehicle: bool = False         # Whether this vehicle is in ego corridor directly ahead


@dataclass
class Track:
    """Comprehensive state of an actively tracked object."""
    track_id: int
    bbox: BoundingBox
    lifecycle: TrackLifecycle = TrackLifecycle.TENTATIVE
    confidence: float = 1.0
    certainty: float = 1.0
    stability: float = 1.0
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    consecutive_misses: int = 0
    is_manual: bool = False
    class_name: str = "vehicle"
    color: Tuple[int, int, int] = (0, 255, 0)
    keypoints: Optional[np.ndarray] = None
    history: List[Tuple[int, int]] = field(default_factory=list)
    kinematics: Kinematics = field(default_factory=Kinematics)
    flash_frames: int = 0


@dataclass
class ADASAlert:
    """Aggregated ADAS warnings and telemetry for the current frame."""
    ldw_state: LDWState = LDWState.NORMAL
    ldw_offset_m: float = 0.0
    ldw_tlc_s: Optional[float] = None
    fcw_level: ADASAlertLevel = ADASAlertLevel.SAFE
    fcw_lead_track_id: Optional[int] = None
    fcw_lead_distance_m: Optional[float] = None
    fcw_lead_rel_speed_kmh: Optional[float] = None
    fcw_lead_ttc_s: Optional[float] = None
    warning_message: Optional[str] = None


@dataclass
class StageTimings:
    """Per-stage wall-clock costs in milliseconds for one processed frame."""
    decode_ms: float = 0.0
    resize_ms: float = 0.0
    track_ms: float = 0.0
    lanes_ms: float = 0.0
    adas_ms: float = 0.0
    vis_ms: float = 0.0
    total_ms: float = 0.0

    def format_hud(self) -> str:
        return (
            f"dec:{self.decode_ms:.1f} rsz:{self.resize_ms:.1f} "
            f"trk:{self.track_ms:.1f} ln:{self.lanes_ms:.1f} vis:{self.vis_ms:.1f}ms"
        )


@dataclass
class FrameData:
    """Frame container passing all perception, tracking, and telemetry through the pipeline."""
    frame_idx: int
    timestamp: float
    proc_frame: np.ndarray
    gray_frame: np.ndarray
    orig_frame: Optional[np.ndarray] = None
    tracks: List[Track] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    lanes: Optional[LaneBoundaries] = None
    adas: Optional[ADASAlert] = None
    fps: float = 0.0
    stage_ms: Optional[StageTimings] = None
