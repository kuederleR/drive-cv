"""
Bipartite association between active tracks and incoming detections.
ByteTrack-style two-stage matching: high-score detections first, then low-score.
Uses Hungarian algorithm with true IoU and spatial distance gating.
"""

from typing import List, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from drivecv.core.math_utils import compute_center_distance, compute_iou
from drivecv.types import BoundingBox, Detection


def _match_subset(
    track_indices: List[int],
    det_indices: List[int],
    track_boxes: List[BoundingBox],
    detections: List[Detection],
    iou_threshold: float,
    distance_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if not track_indices or not det_indices:
        return [], list(track_indices), list(det_indices)

    cost_matrix = np.zeros((len(track_indices), len(det_indices)), dtype=np.float32)
    for i, t_idx in enumerate(track_indices):
        t_box = track_boxes[t_idx]
        for j, d_idx in enumerate(det_indices):
            det_box = detections[d_idx].bbox
            iou = compute_iou(t_box, det_box)
            dist = compute_center_distance(t_box, det_box)
            if dist > distance_threshold and iou < 0.10:
                cost_matrix[i, j] = 10.0 + dist
            else:
                cost_matrix[i, j] = (1.0 - iou) + (dist / max(1.0, distance_threshold)) * 0.20

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int]] = []
    unmatched_tracks: Set[int] = set(track_indices)
    unmatched_dets: Set[int] = set(det_indices)

    for r, c in zip(row_ind, col_ind):
        t_idx = track_indices[r]
        d_idx = det_indices[c]
        iou = compute_iou(track_boxes[t_idx], detections[d_idx].bbox)
        dist = compute_center_distance(track_boxes[t_idx], detections[d_idx].bbox)
        if iou >= iou_threshold or (dist <= distance_threshold * 0.60 and iou >= 0.12):
            matches.append((t_idx, d_idx))
            unmatched_tracks.discard(t_idx)
            unmatched_dets.discard(d_idx)

    return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


def associate_detections_to_tracks(
    track_boxes: List[BoundingBox],
    detections: List[Detection],
    iou_threshold: float = 0.25,
    distance_threshold: float = 80.0,
    high_score_thresh: float = 0.50,
    low_score_thresh: float = 0.30,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Associates predicted track bounding boxes to new detections.

    High-score detections are matched first (can birth unmatched tracks).
    Remaining tracks are then matched to low-score detections (no birth).

    Returns:
        matches: List of tuples (track_idx, detection_idx)
        unmatched_tracks: track indices with no matching detection
        unmatched_detections: high-score detection indices with no matching track
            (low-score unmatched detections are omitted so they do not spawn tracks)
    """
    if len(track_boxes) == 0:
        unmatched_high = [
            i for i, d in enumerate(detections) if d.confidence >= high_score_thresh
        ]
        if not unmatched_high:
            # Fall back: if no det crosses the high gate, allow all above low_score to birth.
            unmatched_high = [
                i for i, d in enumerate(detections) if d.confidence >= low_score_thresh
            ]
        return [], [], unmatched_high

    if len(detections) == 0:
        return [], list(range(len(track_boxes))), []

    high_dets = [i for i, d in enumerate(detections) if d.confidence >= high_score_thresh]
    low_dets = [
        i for i, d in enumerate(detections) if low_score_thresh <= d.confidence < high_score_thresh
    ]
    # Detections between conf_thresh and low_score_thresh are ignored.

    track_ids = list(range(len(track_boxes)))
    matches_h, unmatched_tracks, unmatched_high = _match_subset(
        track_ids, high_dets, track_boxes, detections, iou_threshold, distance_threshold
    )
    matches_l, unmatched_tracks, _unmatched_low = _match_subset(
        unmatched_tracks, low_dets, track_boxes, detections, iou_threshold, distance_threshold
    )

    return matches_h + matches_l, unmatched_tracks, unmatched_high
