"""
Master Multi-Object Tracker:
- Kalman box state with batched Lucas-Kanade velocity measurements.
- Motion attention ROIs request YOLO crops; they do not birth tracks.
- ByteTrack-style association with neural detections only.
"""

from typing import List, Optional, Tuple
import numpy as np
from drivecv.config import TrackerConfig
from drivecv.core.math_utils import compute_iom, compute_iou
from drivecv.perception.motion_detector import FastMotionAttentionGrid
from drivecv.perception.optical_flow import SparseOpticalFlowTracker
from drivecv.tracking.association import associate_detections_to_tracks
from drivecv.tracking.track import TrackObject
from drivecv.types import BoundingBox, Detection, Track, TrackLifecycle


class MultiObjectTracker:
    """
    Orchestrates:
    1. Batched optical flow (one pyramid pair per frame).
    2. Kalman box updates from flow and YOLO.
    3. Motion-grid crop hints (no track birth).
    4. Track birth from unmatched high-score neural detections only.
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.flow_tracker = SparseOpticalFlowTracker(self.config.optical_flow)
        self.attention_grid = FastMotionAttentionGrid()
        self.tracks: List[TrackObject] = []
        self.next_track_id: int = 1
        self.newly_discovered_rois: List[BoundingBox] = []

    def add_manual_track(self, bbox: BoundingBox, gray_frame: Optional[np.ndarray] = None) -> Track:
        """Manually locks a target vehicle bounding box with 1.0 certainty."""
        det = Detection(
            bbox=bbox,
            confidence=1.0,
            class_id=2,
            class_name="car",
            source="manual",
        )
        track = TrackObject(
            track_id=self.next_track_id,
            initial_detection=det,
            config=self.config,
            flow_tracker=self.flow_tracker,
            gray_frame=gray_frame,
        )
        track.is_manual = True
        track.lifecycle = TrackLifecycle.CONFIRMED
        track.certainty = 1.0
        self.tracks.append(track)
        self.next_track_id += 1
        return track.to_track_data()

    def _update_tracks_with_batched_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray):
        for track in self.tracks:
            if track.keypoints is None or len(track.keypoints) < 5:
                track.extract_features(prev_gray, max_corners=35)

        all_pts: List[np.ndarray] = []
        spans: List[Tuple[int, int, int]] = []
        for i, track in enumerate(self.tracks):
            if track.keypoints is not None and len(track.keypoints) > 0:
                start = len(all_pts)
                all_pts.append(track.keypoints)
                spans.append((i, start, start + len(track.keypoints)))

        if not all_pts:
            for track in self.tracks:
                track.coast(curr_gray)
            return

        stacked = np.vstack(all_pts)
        p0, p1, valid = self.flow_tracker.track_batched(prev_gray, curr_gray, stacked)
        if p0 is None or p1 is None or valid is None:
            for track in self.tracks:
                track.coast(curr_gray)
            return

        flowed = set()
        for i, start, end in spans:
            track = self.tracks[i]
            mask = valid[start:end]
            if np.sum(mask) >= 3:
                track.apply_flow_correspondences(p0[start:end][mask], p1[start:end][mask], curr_gray)
                flowed.add(i)
            else:
                track.coast(curr_gray)
                flowed.add(i)

        for i, track in enumerate(self.tracks):
            if i not in flowed:
                track.coast(curr_gray)

    def update(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        curr_bgr_frame: np.ndarray,
        detections: Optional[List[Detection]] = None,
        lanes=None,
        dt: float = 0.04,
        high_score_thresh: float = 0.50,
        low_score_thresh: float = 0.30,
    ) -> List[Track]:
        """
        Executes one tracking cycle:
        1. Advance track counters.
        2. Batched optical flow + Kalman box update.
        3. Motion attention crop hints (no birth).
        4. Associate and fuse incoming YOLO detections.
        5. Prune lost tracks and deduplicate.
        """
        self.newly_discovered_rois.clear()
        del lanes, dt
        h_img, w_img = curr_gray.shape[:2]

        for track in self.tracks:
            track.predict()

        if self.tracks:
            self._update_tracks_with_batched_flow(prev_gray, curr_gray)

        if self.config.enable_motion_crops:
            motion_rois = self.attention_grid.update_and_get_attention_rois(curr_gray)
            for m_roi in motion_rois:
                covered = any(compute_iou(t.bbox, m_roi) > 0.20 for t in self.tracks)
                if not covered:
                    self.newly_discovered_rois.append(m_roi)

        if detections is not None and len(detections) > 0:
            track_boxes = [t.bbox for t in self.tracks]
            matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
                track_boxes=track_boxes,
                detections=detections,
                iou_threshold=self.config.iou_threshold,
                distance_threshold=self.config.distance_threshold,
                high_score_thresh=high_score_thresh,
                low_score_thresh=low_score_thresh,
            )

            for t_idx, d_idx in matches:
                self.tracks[t_idx].update_with_detection(detections[d_idx], gray_frame=curr_gray)

            for d_idx in unmatched_dets:
                det = detections[d_idx]
                if det.confidence >= low_score_thresh and det.bbox.w >= 12 and det.bbox.h >= 12:
                    new_track = TrackObject(
                        track_id=self.next_track_id,
                        initial_detection=det,
                        config=self.config,
                        flow_tracker=self.flow_tracker,
                        gray_frame=curr_gray,
                    )
                    self.tracks.append(new_track)
                    self.next_track_id += 1

        self.tracks = [t for t in self.tracks if t.lifecycle != TrackLifecycle.DELETED]
        self._deduplicate_tracks()
        return [t.to_track_data() for t in self.tracks]

    def get_priority_crop_target(self, frame_w: int, frame_h: int, pad_ratio: float = 0.30) -> Optional[BoundingBox]:
        """Padded crop for a motion hint or unverified / lead track."""
        if self.newly_discovered_rois:
            roi = self.newly_discovered_rois[0]
            pad_w = max(24.0, roi.w * pad_ratio)
            pad_h = max(20.0, roi.h * pad_ratio)
            x1 = max(0.0, roi.x - pad_w)
            y1 = max(0.0, roi.y - pad_h)
            x2 = min(float(frame_w), roi.x + roi.w + pad_w)
            y2 = min(float(frame_h), roi.y + roi.h + pad_h)
            return BoundingBox(x=x1, y=y1, w=max(16.0, x2 - x1), h=max(16.0, y2 - y1))

        unverified = [t for t in self.tracks if t.certainty < 0.70]
        if unverified:
            unverified.sort(key=lambda t: (t.certainty, -t.bbox.area))
            return unverified[0].get_crop_bbox(frame_w, frame_h, pad_ratio=pad_ratio)

        if self.tracks:
            target = min(
                self.tracks,
                key=lambda t: t.kinematics.distance_m if t.kinematics.distance_m > 0 else 999.0,
            )
            return target.get_crop_bbox(frame_w, frame_h, pad_ratio=pad_ratio)

        return None

    def _deduplicate_tracks(self):
        """Merges overlapping duplicate boxes on the same physical vehicle."""
        if len(self.tracks) < 2:
            return

        self.tracks.sort(
            key=lambda t: (
                1 if t.lifecycle == TrackLifecycle.CONFIRMED else 0,
                t.certainty,
                t.bbox.area,
            ),
            reverse=True,
        )

        keep: List[TrackObject] = []
        for track in self.tracks:
            is_dup = False
            for existing in keep:
                iou = compute_iou(existing.bbox, track.bbox)
                iom = compute_iom(existing.bbox, track.bbox)
                if iou > 0.45 or iom > 0.75:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(track)

        self.tracks = keep

    def clear(self):
        """Clears all active tracks."""
        self.tracks.clear()
