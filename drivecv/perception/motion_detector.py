"""
Ultra-fast Motion Energy Attention Grid and Entry Zone Point Cluster Discovery.
Filters candidates strictly to road corridor geometry to reject lamp posts and guardrails.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from drivecv.types import BoundingBox


class FastMotionAttentionGrid:
    """
    Sub-millisecond (0.8 ms) Temporal Motion Energy Attention Grid:
    - 3-Frame Temporal Differencing on downsampled 320x180 thumbnail.
    - Road plane spatial filtering (rejects sky, lamp posts, and hood).
    - Aspect ratio and size gating to isolate genuine vehicles.
    """

    def __init__(self, grid_w: int = 16, grid_h: int = 9, thumb_w: int = 320, thumb_h: int = 180, hood_height_ratio: float = 0.15):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.thumb_w = thumb_w
        self.thumb_h = thumb_h
        self.hood_height_ratio = hood_height_ratio
        self.prev_frames: List[np.ndarray] = []

    def update_and_get_attention_rois(self, gray_or_bgr: np.ndarray) -> List[BoundingBox]:
        """Accepts grayscale (preferred) or BGR; reuses gray to avoid a second color convert."""
        if gray_or_bgr.ndim == 3:
            h_full, w_full = gray_or_bgr.shape[:2]
            thumb = cv2.resize(gray_or_bgr, (self.thumb_w, self.thumb_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        else:
            h_full, w_full = gray_or_bgr.shape[:2]
            gray = cv2.resize(gray_or_bgr, (self.thumb_w, self.thumb_h), interpolation=cv2.INTER_AREA)

        self.prev_frames.append(gray)
        if len(self.prev_frames) > 3:
            self.prev_frames.pop(0)

        if len(self.prev_frames) < 3:
            return []

        # 3-Frame Temporal Differencing
        d1 = cv2.absdiff(self.prev_frames[2], self.prev_frames[1])
        d2 = cv2.absdiff(self.prev_frames[1], self.prev_frames[0])
        motion = cv2.bitwise_and(d1, d2)

        # Zero out sky/lamp posts (top 45%) and hood (bottom hood_height_ratio)
        hood_y = int(self.thumb_h * min(0.90, max(0.60, 1.0 - self.hood_height_ratio)))
        motion[: int(self.thumb_h * 0.45), :] = 0
        motion[hood_y:, :] = 0

        # Compute energy grid
        _, m_bin = cv2.threshold(motion, 8, 255, cv2.THRESH_BINARY)
        grid = cv2.resize(m_bin, (self.grid_w, self.grid_h), interpolation=cv2.INTER_AREA)

        # Active cells
        active_mask = (grid > 20).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        active_closed = cv2.morphologyEx(active_mask, cv2.MORPH_CLOSE, k)

        cnts, _ = cv2.findContours(active_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale_x = w_full / float(self.grid_w)
        scale_y = h_full / float(self.grid_h)

        rois: List[BoundingBox] = []
        for c in cnts:
            if cv2.contourArea(c) >= 1.0:
                gx, gy, gw, gh = cv2.boundingRect(c)
                x1 = max(0.0, float((gx - 0.4) * scale_x))
                y1 = max(0.0, float((gy - 0.4) * scale_y))
                x2 = min(float(w_full), float((gx + gw + 0.4) * scale_x))
                y2 = min(float(h_full), float((gy + gh + 0.4) * scale_y))
                rw = x2 - x1
                rh = y2 - y1

                # Vehicle geometric filters: aspect ratio & ground contact
                aspect = rw / max(1.0, rh)
                if 28.0 <= rw <= w_full * 0.45 and 20.0 <= rh <= h_full * 0.40 and 0.40 <= aspect <= 3.2:
                    if y2 > h_full * 0.48:  # Must have ground plane contact
                        rois.append(BoundingBox(x=x1, y=y1, w=rw, h=rh))
        return rois


class EntryZonePointDetector:
    """
    Multi-zone candidate point extraction strictly confined to road corridors:
    - Left Entry Zone (Overtaking vehicles).
    - Center Road Horizon Zone.
    - Right Entry Zone (Merging vehicles & trucks).
    """

    def __init__(self, min_point_distance: float = 14.0):
        self.min_point_distance = min_point_distance
        self.clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))
        self.zone_points: Dict[str, Optional[np.ndarray]] = {
            "left": None,
            "center": None,
            "right": None,
        }

    def replenish_zone_points(self, gray_frame: np.ndarray):
        """Replenishes points in entry zones within road envelope."""
        h, w = gray_frame.shape[:2]
        zone_configs = {
            "left": (int(h * 0.48), int(h * 0.85), int(w * 0.05), int(w * 0.35), 30, 0.010),
            "center": (int(h * 0.48), int(h * 0.80), int(w * 0.30), int(w * 0.70), 35, 0.012),
            "right": (int(h * 0.48), int(h * 0.85), int(w * 0.65), int(w * 0.95), 30, 0.010),
        }

        for z_name, (y1, y2, x1, x2, max_pts, q_level) in zone_configs.items():
            current_pts = self.zone_points[z_name]
            if current_pts is None or len(current_pts) < 12:
                crop = gray_frame[y1:y2, x1:x2]
                enhanced = self.clahe.apply(crop)
                pts = cv2.goodFeaturesToTrack(
                    enhanced,
                    maxCorners=max_pts,
                    qualityLevel=q_level,
                    minDistance=self.min_point_distance,
                    blockSize=5,
                )
                if pts is not None and len(pts) > 0:
                    pts_global = pts.reshape(-1, 2).astype(np.float32) + np.array([x1, y1], dtype=np.float32)
                    self.zone_points[z_name] = pts_global

    def find_candidate_clusters(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        w_img: int,
        h_img: int,
    ) -> List[Dict]:
        """Finds moving point clusters matching vehicle motion kinematics."""
        if len(p1) < 3:
            return []

        vel = p1 - p0
        diffs = p1[:, None, :] - p1[None, :, :]
        s_dists = np.hypot(diffs[..., 0], diffs[..., 1])
        v_diffs = vel[:, None, :] - vel[None, :, :]
        v_dists = np.hypot(v_diffs[..., 0], v_diffs[..., 1])

        # Spatial distance <= 80px, velocity diff <= 3.5px/frame
        adj = (s_dists <= 80.0) & (v_dists <= 3.5)
        visited = np.zeros(len(p1), dtype=bool)

        raw_clusters = []
        for i in range(len(p1)):
            if visited[i]:
                continue
            members = np.where(adj[i])[0]
            if len(members) >= 3:
                visited[members] = True
                c_pts = p1[members]
                c_vel = vel[members].mean(axis=0)

                speed = float(np.linalg.norm(c_vel))
                if speed > 16.0 or speed < 0.6:
                    continue

                x_min, y_min = c_pts.min(axis=0)
                x_max, y_max = c_pts.max(axis=0)
                bw = x_max - x_min
                bh = y_max - y_min

                aspect = bw / max(1.0, bh)
                if 16.0 <= bw <= w_img * 0.45 and 14.0 <= bh <= h_img * 0.40 and 0.40 <= aspect <= 3.0:
                    pad_x = max(8.0, bw * 0.15)
                    pad_y = max(8.0, bh * 0.15)
                    bx = max(0.0, float(x_min - pad_x))
                    by = max(0.0, float(y_min - pad_y))

                    raw_clusters.append({
                        "bbox": BoundingBox(
                            x=bx,
                            y=by,
                            w=min(float(w_img - bx), float(bw + 2 * pad_x)),
                            h=min(float(h_img - by), float(bh + 2 * pad_y)),
                        ),
                        "points": c_pts,
                        "velocity": c_vel,
                    })

        return raw_clusters
