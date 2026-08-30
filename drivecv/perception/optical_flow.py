"""
Sparse Lucas-Kanade Optical Flow with Adaptive CLAHE and Forward-Backward Error Checking.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from drivecv.config import OpticalFlowConfig
from drivecv.types import BoundingBox


class SparseOpticalFlowTracker:
    """
    Robust vehicle silhouette keypoint extractor and LK optical flow engine:
    - Adaptive CLAHE only on dark crops.
    - Batched forward-backward LK so image pyramids are built twice per frame, not per track.
    - Computes inlier median motion vector and spatial scale ratio.
    """

    def __init__(self, config: Optional[OpticalFlowConfig] = None):
        self.config = config or OpticalFlowConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )
        self.lk_params = dict(
            winSize=self.config.win_size,
            maxLevel=self.config.max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.config.max_iters,
                self.config.epsilon,
            ),
        )

    def extract_features(
        self,
        gray_frame: np.ndarray,
        bbox: BoundingBox,
        max_corners: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Extracts high-quality corner keypoints within a vehicle bounding box.
        Applies local CLAHE only when the crop is dark.
        """
        x, y, w, h = bbox.as_int_xywh()
        h_img, w_img = gray_frame.shape[:2]

        x1 = max(0, min(w_img - 2, x))
        y1 = max(0, min(h_img - 2, y))
        x2 = max(x1 + 2, min(w_img, x + w))
        y2 = max(y1 + 2, min(h_img, y + h))

        if x2 - x1 < 8 or y2 - y1 < 8:
            return None

        crop = gray_frame[y1:y2, x1:x2]
        mean_lum = float(np.mean(crop))
        if mean_lum < self.config.clahe_luma_thresh:
            enhanced_crop = self.clahe.apply(crop)
        else:
            enhanced_crop = crop

        dyn_quality = max(0.003, min(0.020, self.config.quality_level * (mean_lum / 100.0)))
        num_corners = max_corners or self.config.max_corners

        pts = cv2.goodFeaturesToTrack(
            enhanced_crop,
            maxCorners=num_corners,
            qualityLevel=dyn_quality,
            minDistance=self.config.min_distance,
            blockSize=self.config.block_size,
        )

        if pts is None or len(pts) == 0:
            return None

        pts_global = pts.reshape(-1, 2).astype(np.float32) + np.array([x1, y1], dtype=np.float32)
        return pts_global

    def track_points_forward_backward(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        points: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Bidirectional LK for a single point set. Prefer track_batched in the MOT loop."""
        p0, p1, valid = self.track_batched(prev_gray, curr_gray, points)
        if p0 is None or valid is None or np.sum(valid) < 2:
            return None, None
        return p0[valid], p1[valid]

    def track_batched(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        points: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        One forward + one backward PyrLK for all points (all tracks).
        Returns (p0, p1, valid_mask) with the original length, or (None, None, None).
        """
        if points is None or len(points) == 0:
            return None, None, None

        p0 = np.ascontiguousarray(points.reshape(-1, 1, 2).astype(np.float32))

        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
        if p1 is None or st1 is None:
            return None, None, None

        p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1, None, **self.lk_params)
        if p0_back is None or st2 is None:
            return None, None, None

        st1 = st1.ravel() == 1
        st2 = st2.ravel() == 1
        fb_dist = np.linalg.norm(p0.reshape(-1, 2) - p0_back.reshape(-1, 2), axis=1)
        valid = st1 & st2 & (fb_dist <= 2.0)
        return p0.reshape(-1, 2), p1.reshape(-1, 2), valid

    @staticmethod
    def estimate_motion_and_scale(
        p0: np.ndarray,
        p1: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Computes:
        - Median displacement vector [dx, dy]
        - Scale ratio (p1_spread / p0_spread)
        - Inlier ratio
        """
        disp = p1 - p0
        med_dx = float(np.median(disp[:, 0]))
        med_dy = float(np.median(disp[:, 1]))
        med_vel = np.array([med_dx, med_dy], dtype=np.float32)

        vel_diffs = np.linalg.norm(disp - med_vel, axis=1)
        inliers = vel_diffs <= 2.5
        inlier_ratio = float(np.sum(inliers)) / float(len(p0))

        n = len(p0)
        if n < 3:
            scale_ratio = 1.0
        else:
            m = min(n, 15)
            d0 = np.linalg.norm(p0[:m, None, :] - p0[None, :m, :], axis=2)
            d1 = np.linalg.norm(p1[:m, None, :] - p1[None, :m, :], axis=2)
            pair_mask = np.triu(d0 > 6.0, k=1)
            if np.any(pair_mask):
                scale_ratio = float(np.median((d1 / np.maximum(d0, 1e-6))[pair_mask]))
            else:
                scale_ratio = 1.0

        scale_ratio = max(0.92, min(1.08, scale_ratio))
        return med_vel, scale_ratio, inlier_ratio
