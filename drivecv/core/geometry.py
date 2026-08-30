"""
Camera geometry and monocular 3D distance estimation on the ego-lane ground plane.
"""

import math
from typing import List, Optional, Tuple
import numpy as np
from drivecv.config import CameraConfig
from drivecv.types import BoundingBox, LaneBoundaries


class RangeKalman:
    """
    1D constant-velocity filter on longitudinal range.
    State: [Z meters, vz m/s] with vz > 0 meaning the target is receding.
    Closing speed for FCW is -vz.
    """

    def __init__(self, z0: float):
        self.z = float(max(1.5, z0))
        self.vz = 0.0
        self.P = np.diag([36.0, 25.0]).astype(np.float32)

    def predict(self, dt: float):
        dt = float(max(1e-3, dt))
        self.z = max(1.5, min(150.0, self.z + self.vz * dt))
        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
        q = np.array(
            [[0.8 * dt * dt, 0.4 * dt], [0.4 * dt, 3.0 * dt]],
            dtype=np.float32,
        )
        self.P = f @ self.P @ f.T + q

    def update(self, z_meas: float, r: float = 6.0):
        z_meas = float(max(1.5, min(150.0, z_meas)))
        y = z_meas - self.z
        s = float(self.P[0, 0] + r)
        if s <= 1e-6:
            return
        k = self.P[:, 0] / s
        self.z = max(1.5, min(150.0, self.z + float(k[0]) * y))
        self.vz = float(self.vz + float(k[1]) * y)
        i_kh = np.eye(2, dtype=np.float32)
        i_kh[0, 0] -= float(k[0])
        i_kh[1, 0] -= float(k[1])
        self.P = i_kh @ self.P


class CameraGeometry:
    """
    Pinhole + flat-road model, calibrated each frame from ego-lane vanishing point
    and apparent lane width (known real width, default 3.7 m).
    """

    def __init__(self, config: CameraConfig, img_width: int = 1280, img_height: int = 720):
        self.config = config
        self.img_width = img_width
        self.img_height = img_height
        self.fy = config.focal_length_px * (img_height / 720.0)
        self.fx = self.fy
        self.cx = img_width / 2.0
        self.cy = img_height / 2.0
        self.cam_height = config.camera_height_m
        self.pitch = config.camera_pitch_rad
        self.horizon_y = config.horizon_y_ratio * img_height
        self.lane_width_m = config.lane_width_m

    def update_resolution(self, width: int, height: int):
        """Updates internal resolution and scales focal parameters."""
        self.img_width = width
        self.img_height = height
        self.fy = self.config.focal_length_px * (height / 720.0)
        self.fx = self.fy
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.horizon_y = self.config.horizon_y_ratio * height

    def calibrate_from_lanes(self, lanes: Optional[LaneBoundaries]):
        """Sets horizon from the lane vanishing point when available."""
        if lanes is not None and lanes.vanish_y is not None and lanes.is_valid:
            self.horizon_y = float(lanes.vanish_y)

    def estimate_distance_to_contact_point(self, u: float, v: float) -> Optional[float]:
        """Ground-plane range from contact pixel using horizon and camera height."""
        dy = v - self.horizon_y
        if dy <= 1.0:
            return None
        distance_z = (self.fy * self.cam_height) / dy
        if not math.isfinite(distance_z):
            return None
        return max(1.5, min(150.0, float(distance_z)))

    def estimate_lateral_offset(self, u: float, distance_z: float) -> float:
        """Lateral X relative to the camera optical axis."""
        return float(((u - self.cx) * distance_z) / self.fx)

    def image_to_ground(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """
        Projects an image pixel onto the flat-road ground plane.
        Returns (x_m, z_m) with +x right of the camera axis and +z forward, or None.
        """
        z_m = self.estimate_distance_to_contact_point(u, v)
        if z_m is None:
            return None
        x_m = self.estimate_lateral_offset(u, z_m)
        return float(x_m), float(z_m)

    def _poly_px_to_m(self, poly_px: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if poly_px is None:
            return None
        arr = np.asarray(poly_px, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return None
        out: List[Tuple[float, float]] = []
        for u, v in arr:
            ground = self.image_to_ground(float(u), float(v))
            if ground is not None:
                out.append(ground)
        if len(out) < 2:
            return None
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def _centerline_curvature_1pm(
        left_m: Optional[np.ndarray],
        right_m: Optional[np.ndarray],
    ) -> float:
        """Fits x(z) = A z^2 + B z + C on the centerline; returns 2A (1/m)."""
        if left_m is None or right_m is None:
            return 0.0
        z = left_m[:, 1]
        if z.size < 3:
            return 0.0
        xr = np.interp(z, right_m[:, 1], right_m[:, 0])
        xc = 0.5 * (left_m[:, 0] + xr)
        order = np.argsort(z)
        z = z[order]
        xc = xc[order]
        if float(z[-1] - z[0]) < 3.0:
            return 0.0
        try:
            a, _b, _c = np.polyfit(z.astype(np.float64), xc.astype(np.float64), 2)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0
        if not math.isfinite(a):
            return 0.0
        return float(2.0 * a)

    def project_lane_boundaries(self, lanes: Optional[LaneBoundaries]) -> None:
        """
        Fills left_poly_m / right_poly_m / curvature_1pm on `lanes`.

        Points are shifted so the nearest sample origin is the ego-lane center
        (+x right, +z forward, meters), matching HUD / LDW convention.
        """
        if lanes is None:
            return
        left_m = self._poly_px_to_m(lanes.left_poly_px)
        right_m = self._poly_px_to_m(lanes.right_poly_px)
        if left_m is not None and right_m is not None:
            x0 = 0.5 * (float(left_m[0, 0]) + float(right_m[0, 0]))
            left_m = left_m.copy()
            right_m = right_m.copy()
            left_m[:, 0] -= x0
            right_m[:, 0] -= x0
        elif left_m is not None:
            left_m = left_m.copy()
            left_m[:, 0] -= float(left_m[0, 0]) + 0.5 * self.lane_width_m
        elif right_m is not None:
            right_m = right_m.copy()
            right_m[:, 0] -= float(right_m[0, 0]) - 0.5 * self.lane_width_m
        lanes.left_poly_m = left_m
        lanes.right_poly_m = right_m
        lanes.curvature_1pm = self._centerline_curvature_1pm(left_m, right_m)

    def estimate_range_from_lanes(
        self,
        bbox: BoundingBox,
        lanes: Optional[LaneBoundaries],
    ) -> Tuple[float, float]:
        """
        Returns (distance_z_m, lane_lateral_m).

        Primary cue: apparent ego-lane width at the bbox ground-contact row
        Z = fy * W_lane / W_px(v). Secondary: pinhole using the lane vanishing point.
        Lateral is relative to the lane center, not the image center.
        """
        u_bot, v_bot = bbox.bottom_center
        z_width: Optional[float] = None
        z_ground: Optional[float] = None
        lat_lane = 0.0

        if lanes is not None and lanes.is_valid:
            bounds = lanes.x_bounds_at(v_bot)
            if bounds is not None:
                xl, xr = bounds
                w_px = max(8.0, xr - xl)
                z_width = (self.fy * self.lane_width_m) / w_px
                z_width = max(1.5, min(150.0, float(z_width)))
                lane_cx = 0.5 * (xl + xr)
                lat_lane = ((u_bot - lane_cx) / w_px) * self.lane_width_m

            if lanes.vanish_y is not None:
                z_ground = self.estimate_distance_to_contact_point(u_bot, v_bot)

        if z_width is not None and z_ground is not None:
            ratio = z_width / max(1e-3, z_ground)
            if 0.45 <= ratio <= 2.2:
                dist_z = 0.65 * z_width + 0.35 * z_ground
            else:
                dist_z = z_width
        elif z_width is not None:
            dist_z = z_width
        elif z_ground is not None:
            dist_z = z_ground
        else:
            dist_z, lat_cam = self.estimate_bbox_3d_position(bbox)
            return dist_z, lat_cam

        return float(dist_z), float(lat_lane)

    def estimate_bbox_3d_position(self, bbox: BoundingBox) -> Tuple[float, float]:
        """Fallback (no lanes): ground-plane pinhole + height prior."""
        u_bot, v_bot = bbox.bottom_center
        v_clamped = max(self.horizon_y + 10.0, v_bot)
        dist_z = self.estimate_distance_to_contact_point(u_bot, v_clamped)

        if dist_z is None:
            ref_height_px = 65.0 * (self.img_height / 720.0)
            dist_z = max(3.0, min(120.0, (20.0 * ref_height_px) / max(5.0, bbox.h)))

        lat_x = self.estimate_lateral_offset(u_bot, dist_z)
        return dist_z, lat_x

    def lane_search_crop(
        self,
        lanes: LaneBoundaries,
        frame_w: int,
        frame_h: int,
        lead_bbox: Optional[BoundingBox] = None,
        pad_ratio: float = 0.20,
    ) -> Optional[BoundingBox]:
        """
        Crop covering the ego-lane trapezoid, or a padded box around a locked lead.
        YOLO runs on this crop only — one in-lane vehicle, not the whole scene.
        """
        if lead_bbox is not None and lead_bbox.w >= 16 and lead_bbox.h >= 16:
            pad_x = max(24.0, lead_bbox.w * 0.45)
            pad_y = max(20.0, lead_bbox.h * 0.35)
            x1 = max(0.0, lead_bbox.x - pad_x)
            y1 = max(0.0, lead_bbox.y - pad_y)
            x2 = min(float(frame_w), lead_bbox.x + lead_bbox.w + pad_x)
            y2 = min(float(frame_h), lead_bbox.y + lead_bbox.h + pad_y)
            return BoundingBox(x=x1, y=y1, w=max(16.0, x2 - x1), h=max(16.0, y2 - y1))

        if not lanes.is_valid or lanes.left_line is None or lanes.right_line is None:
            return None

        y_bot = float(lanes.y_bot)
        y_top = float(lanes.y_roi_top if lanes.y_roi_top > 0 else lanes.y_top)
        if lanes.vanish_y is not None:
            y_top = min(y_top, float(lanes.vanish_y) + 24.0)
        # Include vehicle body above the road ROI
        y_top = max(0.0, y_top - 0.22 * (y_bot - y_top))
        y_bot = min(float(frame_h), y_bot + 8.0)

        b_top = lanes.x_bounds_at(y_top + 4.0)
        b_bot = lanes.x_bounds_at(min(y_bot - 2.0, y_bot))
        if b_top is None and b_bot is None:
            return None
        xs = []
        if b_top is not None:
            xs.extend(b_top)
        if b_bot is not None:
            xs.extend(b_bot)
        pad = max(16.0, (max(xs) - min(xs)) * pad_ratio)
        x1 = max(0.0, min(xs) - pad)
        x2 = min(float(frame_w), max(xs) + pad)
        y1 = max(0.0, y_top)
        y2 = min(float(frame_h), y_bot)
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None
        return BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
