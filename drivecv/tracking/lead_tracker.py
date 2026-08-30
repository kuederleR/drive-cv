"""
Single ego-lane lead-vehicle tracker for LDW/FCW.

Only the closest vehicle whose ground-contact point lies in the measured
ego-lane trapezoid is locked. Range is estimated from apparent lane width
at that contact row and filtered with a 1D Kalman filter.
"""

from typing import List, Optional, Tuple
import numpy as np
from drivecv.config import CameraConfig, TrackerConfig
from drivecv.core.geometry import CameraGeometry, RangeKalman
from drivecv.core.math_utils import compute_iou
from drivecv.perception.optical_flow import SparseOpticalFlowTracker
from drivecv.tracking.track import TrackObject
from drivecv.types import BoundingBox, Detection, LaneBoundaries, Track, TrackLifecycle


class LeadVehicleTracker:
    """Locks and ranges the single in-lane lead vehicle."""

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        camera_geom: Optional[CameraGeometry] = None,
        in_lane_margin_frac: float = 0.12,
    ):
        self.config = config or TrackerConfig()
        self.camera_geom = camera_geom or CameraGeometry(
            CameraConfig(), img_width=960, img_height=540
        )
        self.in_lane_margin_frac = in_lane_margin_frac
        self.flow_tracker = SparseOpticalFlowTracker(self.config.optical_flow)
        self.lead: Optional[TrackObject] = None
        self.range_kf: Optional[RangeKalman] = None
        self.next_track_id: int = 1
        self.out_of_lane_frames: int = 0
        self.newly_discovered_rois: List[BoundingBox] = []

    @property
    def tracks(self) -> List[TrackObject]:
        return [self.lead] if self.lead is not None else []

    def clear(self):
        self.lead = None
        self.range_kf = None
        self.out_of_lane_frames = 0

    def add_manual_track(self, bbox: BoundingBox, gray_frame: Optional[np.ndarray] = None) -> Track:
        det = Detection(
            bbox=bbox,
            confidence=1.0,
            class_id=2,
            class_name="car",
            source="manual",
        )
        self._lock(det, gray_frame)
        if self.lead is not None:
            self.lead.is_manual = True
            self.lead.lifecycle = TrackLifecycle.CONFIRMED
            self.lead.certainty = 1.0
            return self.lead.to_track_data()
        return Track(track_id=0, bbox=bbox)

    def _in_lane(self, bbox: BoundingBox, lanes: Optional[LaneBoundaries]) -> bool:
        if lanes is None or not lanes.is_valid:
            # Without lanes, accept boxes in the lower-central image (weak prior).
            cx, cy_bot = bbox.bottom_center
            return (0.28 * self.camera_geom.img_width <= cx <= 0.72 * self.camera_geom.img_width) and (
                cy_bot > 0.45 * self.camera_geom.img_height
            )
        u, v = bbox.bottom_center
        return lanes.contains_contact(u, v, margin_frac=self.in_lane_margin_frac)

    def _range_of(self, bbox: BoundingBox, lanes: Optional[LaneBoundaries]) -> Tuple[float, float]:
        return self.camera_geom.estimate_range_from_lanes(bbox, lanes)

    def _lock(
        self,
        det: Detection,
        gray_frame: Optional[np.ndarray],
        lanes: Optional[LaneBoundaries] = None,
    ):
        self.lead = TrackObject(
            track_id=self.next_track_id,
            initial_detection=det,
            config=self.config,
            flow_tracker=self.flow_tracker,
            gray_frame=gray_frame,
        )
        self.lead.lifecycle = TrackLifecycle.CONFIRMED
        self.next_track_id += 1
        self.out_of_lane_frames = 0
        z, _ = self._range_of(det.bbox, lanes)
        self.range_kf = RangeKalman(z)

    def _select_lead_detection(
        self,
        detections: List[Detection],
        lanes: Optional[LaneBoundaries],
    ) -> Optional[Detection]:
        best: Optional[Detection] = None
        best_z = 1e9
        for det in detections:
            if det.bbox.w < 14 or det.bbox.h < 14:
                continue
            if not self._in_lane(det.bbox, lanes):
                continue
            z, _ = self._range_of(det.bbox, lanes)
            # Closest in-lane vehicle (smallest Z). Ignore bumper/hood ghosts < 3 m.
            if 3.0 <= z < best_z:
                best_z = z
                best = det
        return best

    def _apply_range(self, lanes: Optional[LaneBoundaries], dt: float):
        if self.lead is None:
            return
        z_meas, lat = self._range_of(self.lead.bbox, lanes)
        if self.range_kf is None:
            self.range_kf = RangeKalman(z_meas)
        else:
            self.range_kf.predict(dt)
            self.range_kf.update(z_meas, r=5.0)
        z = self.range_kf.z
        closing = -self.range_kf.vz  # positive = approaching
        self.lead.kinematics.distance_m = float(z)
        self.lead.kinematics.lateral_offset_m = float(lat)
        self.lead.kinematics.rel_speed_mps = float(closing)
        self.lead.kinematics.rel_speed_kmh = float(closing * 3.6)
        self.lead.kinematics.is_lead_vehicle = True
        if closing > 0.4 and z > 1.5:
            self.lead.kinematics.ttc_seconds = float(z / closing)
        else:
            self.lead.kinematics.ttc_seconds = None

    def update(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        curr_bgr_frame: np.ndarray,
        detections: Optional[List[Detection]] = None,
        lanes: Optional[LaneBoundaries] = None,
        dt: float = 0.04,
        high_score_thresh: float = 0.50,
        low_score_thresh: float = 0.30,
    ) -> List[Track]:
        del curr_bgr_frame, high_score_thresh, low_score_thresh
        self.newly_discovered_rois.clear()
        self.camera_geom.calibrate_from_lanes(lanes)
        dets = detections or []

        if self.lead is not None:
            self.lead.predict()
            if self.lead.keypoints is None or len(self.lead.keypoints) < 5:
                self.lead.extract_features(prev_gray, max_corners=35)
            if self.lead.keypoints is not None and len(self.lead.keypoints) > 0:
                p0, p1, valid = self.flow_tracker.track_batched(
                    prev_gray, curr_gray, self.lead.keypoints
                )
                if p0 is not None and valid is not None and np.sum(valid) >= 3:
                    self.lead.apply_flow_correspondences(p0[valid], p1[valid], curr_gray)
                else:
                    self.lead.coast(curr_gray)
            else:
                self.lead.coast(curr_gray)

            in_lane = self._in_lane(self.lead.bbox, lanes)
            if in_lane:
                self.out_of_lane_frames = 0
            else:
                self.out_of_lane_frames += 1

            matched = False
            in_lane_dets = [d for d in dets if self._in_lane(d.bbox, lanes)]
            if in_lane_dets:
                # Prefer IoU match to the locked box; otherwise take the closest in-lane car
                # (handles cut-ins that should become the new lead).
                best_iou = 0.0
                best_det: Optional[Detection] = None
                for d in in_lane_dets:
                    iou = compute_iou(self.lead.bbox, d.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_det = d
                closest = self._select_lead_detection(in_lane_dets, lanes)
                if best_det is not None and best_iou >= 0.18:
                    self.lead.update_with_detection(best_det, gray_frame=curr_gray)
                    matched = True
                elif closest is not None:
                    z_new, _ = self._range_of(closest.bbox, lanes)
                    z_old = self.lead.kinematics.distance_m or 99.0
                    # Cut-in: a closer in-lane vehicle with little overlap
                    if z_new + 2.0 < z_old and best_iou < 0.18:
                        self._lock(closest, curr_gray, lanes)
                        matched = True

            if self.lead.lifecycle == TrackLifecycle.DELETED or self.out_of_lane_frames > 10:
                self.clear()
            elif not matched and self.lead.consecutive_misses > self.config.max_age:
                self.clear()

        if self.lead is None:
            candidate = self._select_lead_detection(dets, lanes)
            if candidate is not None:
                self._lock(candidate, curr_gray, lanes)

        if self.lead is not None:
            self._apply_range(lanes, dt)
            return [self.lead.to_track_data()]
        return []

    def get_lane_crop(
        self,
        lanes: Optional[LaneBoundaries],
        frame_w: int,
        frame_h: int,
        pad_ratio: float = 0.20,
    ) -> Optional[BoundingBox]:
        """YOLO crop: locked lead box, else the ego-lane trapezoid."""
        if lanes is None:
            if self.lead is not None:
                return self.lead.get_crop_bbox(frame_w, frame_h, pad_ratio=0.45)
            return None
        lead_box = self.lead.bbox if self.lead is not None else None
        return self.camera_geom.lane_search_crop(
            lanes, frame_w, frame_h, lead_bbox=lead_box, pad_ratio=pad_ratio
        )

    def get_priority_crop_target(
        self, frame_w: int, frame_h: int, pad_ratio: float = 0.30
    ) -> Optional[BoundingBox]:
        if self.lead is not None:
            return self.lead.get_crop_bbox(frame_w, frame_h, pad_ratio=max(pad_ratio, 0.40))
        return None
