"""
Configuration models for DriveCV modules.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


# COCO class ids treated as vehicles for YOLOPv2's 80-class detection head.
COCO_VEHICLE_CLASS_IDS: Tuple[int, ...] = (2, 3, 5, 7)
COCO_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class DetectorConfig:
    """Configuration for deep neural detector (YOLOPv2)."""
    model_path: str = "weights/YOLOPv2.onnx"
    input_size: Tuple[int, int] = (640, 640)
    conf_thresh: float = 0.28
    iou_thresh: float = 0.45
    enabled: bool = True
    async_inference: bool = True
    interval_frames: int = 15
    crop_padding_ratio: float = 0.30  # Percentage buffer for targeted crop YOLO
    # None = letterbox the process frame directly to input_size (do not downscale then upscale).
    downsample_size: Optional[Tuple[int, int]] = None
    onnx_intra_op_threads: int = 4
    vehicle_class_ids: Tuple[int, ...] = COCO_VEHICLE_CLASS_IDS
    prefer_openvino: bool = True
    high_score_thresh: float = 0.50
    low_score_thresh: float = 0.30


@dataclass
class OpticalFlowConfig:
    """Configuration for sparse Lucas-Kanade optical flow."""
    max_corners: int = 40
    quality_level: float = 0.010
    min_distance: float = 4.5
    block_size: int = 5
    clahe_clip_limit: float = 3.5
    clahe_grid_size: Tuple[int, int] = (6, 6)
    clahe_luma_thresh: float = 80.0  # Skip CLAHE on well-lit crops
    win_size: Tuple[int, int] = (19, 19)
    max_level: int = 2
    max_iters: int = 20
    epsilon: float = 0.03


@dataclass
class TrackerConfig:
    """Configuration for multi-object tracking."""
    max_age: int = 20
    min_hits: int = 2
    iou_threshold: float = 0.22
    distance_threshold: float = 80.0
    certainty_decay_rate: float = 0.025
    stability_threshold: float = 0.80
    point_bbox_padding_pct: float = 0.12  # Unused for box geometry; kept for crop padding helpers
    enable_motion_crops: bool = True  # Motion ROIs request YOLO crops; they do not birth tracks
    lead_only: bool = True  # Track the single ego-lane lead vehicle for FCW
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)


@dataclass
class LaneConfig:
    """Configuration for classical host lane line and drivable corridor estimation."""
    y_top_ratio: float = 0.58
    y_bot_ratio: float = 0.95
    hood_mask_enabled: bool = True
    hood_height_ratio: float = 0.20  # Fraction of image height at bottom occupied by hood (0.0 to 0.40)
    ema_alpha: float = 0.20
    path_width_ratio: float = 0.80
    canny_low: int = 40
    canny_high: int = 120
    hough_threshold: int = 15
    hough_min_line_length: int = 15
    hough_max_line_gap: int = 25


@dataclass
class CameraConfig:
    """Camera intrinsic and extrinsic parameters for monocular 3D ADAS estimation."""
    focal_length_px: float = 1150.0
    camera_height_m: float = 1.25
    camera_pitch_rad: float = 0.0
    horizon_y_ratio: float = 0.50
    lane_width_m: float = 3.70
    vehicle_height_m: float = 1.50


@dataclass
class ADASConfig:
    """Configuration for ADAS safety modules (LDW & FCW)."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    ldw_offset_threshold_m: float = 0.45
    ldw_tlc_threshold_s: float = 1.8
    ldw_cooldown_frames: int = 25
    fcw_safe_ttc_s: float = 3.0
    fcw_caution_ttc_s: float = 2.2
    fcw_warning_ttc_s: float = 1.5
    fcw_critical_ttc_s: float = 1.0
    fcw_corridor_width_m: float = 2.4
    fcw_in_lane_margin_frac: float = 0.12


@dataclass
class VisualizerConfig:
    """Configuration for HUD and visual presentation."""
    show_hud: bool = True
    show_path: bool = True
    show_boxes: bool = True
    show_vectors: bool = True
    show_points: bool = True
    show_seg_masks: bool = False
    show_adas_badges: bool = True
    show_stage_timings: bool = True
    drivable_color: Tuple[int, int, int] = (255, 180, 0)   # Sleek Electric Cyan (BGR)
    lane_color: Tuple[int, int, int] = (0, 255, 255)       # Yellow
    lead_color_safe: Tuple[int, int, int] = (0, 255, 0)
    lead_color_caution: Tuple[int, int, int] = (0, 215, 255)
    lead_color_warning: Tuple[int, int, int] = (0, 140, 255)
    lead_color_critical: Tuple[int, int, int] = (0, 0, 255)


@dataclass
class PipelineConfig:
    """Master pipeline configuration combining all module configs."""
    width: int = 960
    height: int = 540
    fps: float = 25.0
    opencv_num_threads: int = 4
    use_ffmpeg_scale: bool = True
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    lane: LaneConfig = field(default_factory=LaneConfig)
    adas: ADASConfig = field(default_factory=ADASConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
