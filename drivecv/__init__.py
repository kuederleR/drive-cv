"""
DriveCV: High-Performance Autonomous Driving Perception & Tracking Suite
Combining Classical Computer Vision with Deep Learning for Real-Time ADAS.
"""

__version__ = "1.0.0"

from drivecv.config import (
    ADASConfig,
    CameraConfig,
    DetectorConfig,
    LaneConfig,
    OpticalFlowConfig,
    PipelineConfig,
    TrackerConfig,
    VisualizerConfig,
)
from drivecv.pipeline import ADASPipeline
from drivecv.types import (
    ADASAlert,
    ADASAlertLevel,
    BoundingBox,
    Detection,
    FrameData,
    Kinematics,
    LaneBoundaries,
    LDWState,
    StageTimings,
    Track,
    TrackLifecycle,
)

__all__ = [
    "ADASConfig",
    "CameraConfig",
    "DetectorConfig",
    "LaneConfig",
    "OpticalFlowConfig",
    "PipelineConfig",
    "TrackerConfig",
    "VisualizerConfig",
    "ADASPipeline",
    "ADASAlert",
    "ADASAlertLevel",
    "BoundingBox",
    "Detection",
    "FrameData",
    "Kinematics",
    "StageTimings",
    "LaneBoundaries",
    "LDWState",
    "Track",
    "TrackLifecycle",
]
