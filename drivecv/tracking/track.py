"""
Individual tracked object representation.
Kalman filter owns box geometry; optical flow measures velocity and scale only.
"""

from typing import List, Optional, Tuple
import numpy as np
from drivecv.config import TrackerConfig
from drivecv.perception.optical_flow import SparseOpticalFlowTracker
from drivecv.tracking.kalman import KalmanBoxTracker
from drivecv.types import BoundingBox, Detection, Kinematics, Track, TrackLifecycle

PALETTE = [
    (0, 255, 0),      # Bright Lime Green
    (0, 215, 255),    # Gold / Amber
    (255, 0, 255),    # Vivid Magenta
    (0, 255, 255),    # Bright Cyan
    (50, 100, 255),   # Neon Coral / Orange
    (255, 160, 0),    # Vivid Sky Blue
    (180, 105, 255),  # Hot Pink
    (0, 255, 128),    # Spring Green
    (255, 0, 128),    # Violet / Purple
    (0, 140, 255),    # Deep Orange
]


class TrackObject:
    """
    Tracks a physical vehicle with a 7-state Kalman box:
    - YOLO detections are box measurements.
    - Lucas-Kanade flow updates velocity and weak scale, never box extents from keypoints.
    - Lifecycle states: TENTATIVE -> CONFIRMED -> LOST -> DELETED.
    """

    def __init__(
        self,
        track_id: int,
        initial_detection: Detection,
        config: Optional[TrackerConfig] = None,
        flow_tracker: Optional[SparseOpticalFlowTracker] = None,
        gray_frame: Optional[np.ndarray] = None,
    ):
        self.track_id = track_id
        self.config = config or TrackerConfig()
        self.flow_tracker = flow_tracker or SparseOpticalFlowTracker(self.config.optical_flow)

        self.bbox: BoundingBox = initial_detection.bbox
        self.kalman = KalmanBoxTracker(self.bbox)
        self.lifecycle = TrackLifecycle.TENTATIVE
        self.confidence = float(initial_detection.confidence)
        self.certainty = 1.0 if self.confidence > 0.40 else 0.50
        self.stability = 1.0
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.consecutive_misses = 0
        self.is_manual = False
        self.class_name = initial_detection.class_name
        self.class_id = initial_detection.class_id
        self.color = PALETTE[(track_id - 1) % len(PALETTE)]
        self.keypoints: Optional[np.ndarray] = None
        self.history: List[Tuple[int, int]] = []
        self.kinematics = Kinematics()
        self.flash_frames = 6
        self._frame_wh: Optional[Tuple[int, int]] = None

        if gray_frame is not None:
            self._frame_wh = (gray_frame.shape[1], gray_frame.shape[0])
            self.extract_features(gray_frame)

        cx, cy = self.bbox.center
        self.history.append((int(cx), int(cy)))

    def extract_features(self, gray_frame: np.ndarray, max_corners: Optional[int] = None):
        """Extracts optical flow keypoints within current bounding box."""
        self.keypoints = self.flow_tracker.extract_features(gray_frame, self.bbox, max_corners=max_corners)

    def predict(self):
        """Advances track time and decay counters (Kalman predict happens on coast)."""
        self.age += 1
        self.time_since_update += 1
        if self.flash_frames > 0:
            self.flash_frames -= 1

    def _commit_bbox(self, bbox: BoundingBox, frame_wh: Optional[Tuple[int, int]] = None):
        wh = frame_wh or self._frame_wh
        if wh is not None:
            w_img, h_img = wh
            x = max(0.0, min(float(w_img - 2), bbox.x))
            y = max(0.0, min(float(h_img - 2), bbox.y))
            bw = max(8.0, min(float(w_img) - x, bbox.w))
            bh = max(8.0, min(float(h_img) - y, bbox.h))
            self.bbox = BoundingBox(x=x, y=y, w=bw, h=bh)
        else:
            self.bbox = bbox
        vx, vy = self.kalman.get_velocity()
        self.kinematics.velocity_2d = np.array([vx, vy], dtype=np.float32)

    def update_with_detection(self, detection: Detection, gray_frame: Optional[np.ndarray] = None):
        """Kalman measurement update from a verified neural detection."""
        self.kalman.update_detection(detection.bbox, conf=float(detection.confidence))
        if gray_frame is not None:
            self._frame_wh = (gray_frame.shape[1], gray_frame.shape[0])
        self._commit_bbox(self.kalman.get_state_bbox())
        self.confidence = float(detection.confidence)
        self.class_name = detection.class_name
        self.class_id = detection.class_id
        self.certainty = 1.0
        self.stability = 1.0
        self.hits += 1
        self.time_since_update = 0
        self.consecutive_misses = 0
        self.flash_frames = 6

        if self.lifecycle == TrackLifecycle.TENTATIVE and self.hits >= self.config.min_hits:
            self.lifecycle = TrackLifecycle.CONFIRMED
        elif self.lifecycle == TrackLifecycle.LOST:
            self.lifecycle = TrackLifecycle.CONFIRMED

        cx, cy = self.bbox.center
        self.history.append((int(cx), int(cy)))
        if len(self.history) > 30:
            self.history.pop(0)

        if gray_frame is not None:
            self.extract_features(gray_frame, max_corners=40)

    def apply_flow_correspondences(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        curr_gray: np.ndarray,
    ):
        """Applies batched LK inliers as a Kalman flow update. Does not bind bbox to points."""
        self._frame_wh = (curr_gray.shape[1], curr_gray.shape[0])
        vel, scale_ratio, inlier_ratio = self.flow_tracker.estimate_motion_and_scale(p0, p1)
        self.kalman.update_optical_flow(float(vel[0]), float(vel[1]), scale_ratio)
        self._commit_bbox(self.kalman.get_state_bbox())
        self.keypoints = p1

        retention = float(len(p1)) / max(1.0, float(len(p0)))
        self.stability = max(0.0, min(1.0, float(retention * inlier_ratio)))

        if self.stability < self.config.stability_threshold:
            decay = self.config.certainty_decay_rate * (
                (self.config.stability_threshold - self.stability) / 0.8
            )
            self.certainty = max(0.0, self.certainty - decay)

        self.consecutive_misses = 0
        cx, cy = self.bbox.center
        self.history.append((int(cx), int(cy)))
        if len(self.history) > 30:
            self.history.pop(0)

        if self.keypoints is None or len(self.keypoints) < 10:
            supp_pts = self.flow_tracker.extract_features(curr_gray, self.bbox, max_corners=25)
            if supp_pts is not None and len(supp_pts) > 0:
                if self.keypoints is not None and len(self.keypoints) > 0:
                    self.keypoints = np.vstack([self.keypoints, supp_pts])
                else:
                    self.keypoints = supp_pts

    def coast(self, curr_gray: Optional[np.ndarray] = None):
        """No valid flow: Kalman constant-velocity predict and decay certainty."""
        if curr_gray is not None:
            self._frame_wh = (curr_gray.shape[1], curr_gray.shape[0])
        self._commit_bbox(self.kalman.predict())
        self.consecutive_misses += 1
        self.certainty = max(0.0, self.certainty - self.config.certainty_decay_rate * 2.0)
        self._check_loss()

    def update_with_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ):
        """Single-track fallback used by tests; MOT uses batched LK instead."""
        if self.keypoints is None or len(self.keypoints) < 5:
            self.extract_features(prev_gray, max_corners=35)

        if self.keypoints is None or len(self.keypoints) == 0:
            self.coast(curr_gray)
            return

        p0, p1 = self.flow_tracker.track_points_forward_backward(prev_gray, curr_gray, self.keypoints)
        if p0 is not None and p1 is not None and len(p1) >= 3:
            self.apply_flow_correspondences(p0, p1, curr_gray)
        else:
            self.coast(curr_gray)

    def get_crop_bbox(self, frame_w: int, frame_h: int, pad_ratio: float = 0.30) -> BoundingBox:
        """Padded crop around the Kalman box for targeted YOLO."""
        bx, by, bw, bh = self.bbox.as_xywh()
        pad_x = max(20.0, bw * pad_ratio)
        pad_y = max(20.0, bh * pad_ratio)

        x1 = max(0.0, bx - pad_x)
        y1 = max(0.0, by - pad_y)
        x2 = min(float(frame_w), bx + bw + pad_x)
        y2 = min(float(frame_h), by + bh + pad_y)

        return BoundingBox(x=x1, y=y1, w=max(16.0, x2 - x1), h=max(16.0, y2 - y1))

    def _check_loss(self):
        if self.lifecycle == TrackLifecycle.CONFIRMED and self.consecutive_misses > 6:
            self.lifecycle = TrackLifecycle.LOST
        elif self.lifecycle == TrackLifecycle.TENTATIVE and self.consecutive_misses > 3:
            self.lifecycle = TrackLifecycle.DELETED
        elif self.lifecycle == TrackLifecycle.LOST and self.consecutive_misses > self.config.max_age:
            self.lifecycle = TrackLifecycle.DELETED

    def to_track_data(self) -> Track:
        """Exports Track dataclass representation."""
        return Track(
            track_id=self.track_id,
            bbox=self.bbox,
            lifecycle=self.lifecycle,
            confidence=self.confidence,
            certainty=self.certainty,
            stability=self.stability,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            consecutive_misses=self.consecutive_misses,
            is_manual=self.is_manual,
            class_name=self.class_name,
            color=self.color,
            keypoints=self.keypoints.copy() if self.keypoints is not None else None,
            history=list(self.history),
            kinematics=self.kinematics,
            flash_frames=self.flash_frames,
        )
