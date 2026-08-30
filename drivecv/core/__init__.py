"""
DriveCV Core Math and Geometry Utilities.
"""

from drivecv.core.geometry import CameraGeometry, RangeKalman
from drivecv.core.math_utils import (
    compute_iou,
    compute_iou_matrix,
    compute_center_distance,
    exponential_moving_average,
    clip_scalar,
)

__all__ = [
    "CameraGeometry",
    "RangeKalman",
    "compute_iou",
    "compute_iou_matrix",
    "compute_center_distance",
    "exponential_moving_average",
    "clip_scalar",
]
