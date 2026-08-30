"""
Mathematical and geometric utility functions.
"""

from typing import List, Sequence, Tuple, Union
import numpy as np
from drivecv.types import BoundingBox


def _xywh(bb: Union[BoundingBox, Sequence[float], np.ndarray]) -> Tuple[float, float, float, float]:
    if isinstance(bb, BoundingBox):
        return bb.as_xywh()
    return float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])


def _intersection_areas(
    bb1: Union[BoundingBox, Sequence[float], np.ndarray],
    bb2: Union[BoundingBox, Sequence[float], np.ndarray],
) -> Tuple[float, float, float]:
    x1, y1, w1, h1 = _xywh(bb1)
    x2, y2, w2, h2 = _xywh(bb2)

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    intersection = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    a1 = max(0.0, w1) * max(0.0, h1)
    a2 = max(0.0, w2) * max(0.0, h2)
    return intersection, a1, a2


def compute_iou(
    bb1: Union[BoundingBox, Sequence[float], np.ndarray],
    bb2: Union[BoundingBox, Sequence[float], np.ndarray],
) -> float:
    """True Intersection over Union between two [x, y, w, h] boxes."""
    intersection, a1, a2 = _intersection_areas(bb1, bb2)
    union = a1 + a2 - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def compute_iom(
    bb1: Union[BoundingBox, Sequence[float], np.ndarray],
    bb2: Union[BoundingBox, Sequence[float], np.ndarray],
) -> float:
    """Intersection over the smaller box area (containment)."""
    intersection, a1, a2 = _intersection_areas(bb1, bb2)
    min_area = min(a1, a2)
    if min_area <= 0.0:
        return 0.0
    return intersection / min_area


def compute_iou_matrix(
    boxes1: List[BoundingBox],
    boxes2: List[BoundingBox],
) -> np.ndarray:
    """Computes IoU cost matrix between two lists of bounding boxes."""
    n1 = len(boxes1)
    n2 = len(boxes2)
    iou_matrix = np.zeros((n1, n2), dtype=np.float32)

    for i in range(n1):
        for j in range(n2):
            iou_matrix[i, j] = compute_iou(boxes1[i], boxes2[j])

    return iou_matrix


def compute_center_distance(
    bb1: Union[BoundingBox, Sequence[float]],
    bb2: Union[BoundingBox, Sequence[float]],
) -> float:
    """Computes Euclidean distance between centers of two bounding boxes."""
    if isinstance(bb1, BoundingBox):
        cx1, cy1 = bb1.center
    else:
        cx1, cy1 = bb1[0] + bb1[2] / 2.0, bb1[1] + bb1[3] / 2.0

    if isinstance(bb2, BoundingBox):
        cx2, cy2 = bb2.center
    else:
        cx2, cy2 = bb2[0] + bb2[2] / 2.0, bb2[1] + bb2[3] / 2.0

    return float(np.hypot(cx1 - cx2, cy1 - cy2))


def exponential_moving_average(
    current_val: Union[float, np.ndarray],
    target_val: Union[float, np.ndarray],
    alpha: float = 0.20,
) -> Union[float, np.ndarray]:
    """Applies exponential moving average smoothing."""
    return (1.0 - alpha) * current_val + alpha * target_val


def clip_scalar(val: float, min_val: float, max_val: float) -> float:
    """Clamps a floating point scalar between min_val and max_val."""
    return max(min_val, min(max_val, float(val)))
