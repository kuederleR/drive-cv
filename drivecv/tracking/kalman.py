"""
Kalman Filter for 2D Bounding Box Kinematic State Estimation.
Tracks [x_center, y_center, scale_area, aspect_ratio, vx, vy, vs].
"""

import math
from typing import Optional, Tuple
import numpy as np
from drivecv.types import BoundingBox


class KalmanBoxTracker:
    """
    2D Constant Velocity Kalman Filter:
    - State: x = [u, v, s, r, u_dot, v_dot, s_dot]^T
    - Measurement: z = [u, v, s, r]^T
    - Intermediate Flow Update: applies delta (dx, dy, ds) directly to state with flow covariance.
    """

    def __init__(self, bbox: BoundingBox):
        # 7-state vector, 4-measurement vector
        self.dim_x = 7
        self.dim_z = 4

        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self._init_state(bbox)

        # State transition matrix F
        self.F = np.eye(self.dim_x, dtype=np.float32)
        self.F[0, 4] = 1.0  # u += u_dot
        self.F[1, 5] = 1.0  # v += v_dot
        self.F[2, 6] = 1.0  # s += s_dot

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Covariance matrices
        self.P = np.diag([10.0, 10.0, 50.0, 10.0, 100.0, 100.0, 100.0]).astype(np.float32)
        self.Q = np.diag([1.0, 1.0, 4.0, 0.1, 2.0, 2.0, 4.0]).astype(np.float32)
        self.R = np.diag([2.0, 2.0, 10.0, 1.0]).astype(np.float32)

    def _init_state(self, bbox: BoundingBox):
        cx, cy = bbox.center
        s = max(1.0, bbox.area)
        r = max(0.1, bbox.aspect_ratio)
        self.x[0, 0] = cx
        self.x[1, 0] = cy
        self.x[2, 0] = s
        self.x[3, 0] = r
        self.x[4, 0] = 0.0
        self.x[5, 0] = 0.0
        self.x[6, 0] = 0.0

    def predict(self) -> BoundingBox:
        """Advances state vector and covariance by 1 time step."""
        # Enforce positive area
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.get_state_bbox()

    def update_detection(self, bbox: BoundingBox, conf: float = 1.0):
        """Standard Kalman measurement update from a neural detection."""
        cx, cy = bbox.center
        s = max(1.0, bbox.area)
        r = max(0.1, bbox.aspect_ratio)
        z = np.array([[cx], [cy], [s], [r]], dtype=np.float32)

        # Scale measurement noise R inversely with detection confidence
        scale_r = 1.0 / max(0.2, conf)
        R_scaled = self.R * scale_r

        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + R_scaled
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, R_scaled), K.T)

    def update_optical_flow(self, dx: float, dy: float, scale_factor: float):
        """
        Intermediate kinematic update directly from optical flow displacement.
        Blends optical flow velocity into Kalman state.
        """
        self.x[0, 0] += float(dx)
        self.x[1, 0] += float(dy)
        self.x[2, 0] = max(10.0, float(self.x[2, 0] * (scale_factor ** 2)))

        # Update velocity estimates with exponential smoothing
        self.x[4, 0] = 0.65 * self.x[4, 0] + 0.35 * float(dx)
        self.x[5, 0] = 0.65 * self.x[5, 0] + 0.35 * float(dy)

    def get_state_bbox(self) -> BoundingBox:
        """Converts internal [u, v, s, r] state into BoundingBox [x, y, w, h]."""
        u = float(self.x[0, 0])
        v = float(self.x[1, 0])
        s = max(1.0, float(self.x[2, 0]))
        r = max(0.1, float(self.x[3, 0]))

        w = math.sqrt(s * r)
        h = s / max(1e-4, w)

        x = u - w / 2.0
        y = v - h / 2.0

        return BoundingBox(x=x, y=y, w=w, h=h)

    def get_velocity(self) -> Tuple[float, float]:
        """Returns current velocity [vx, vy] in pixels/frame."""
        return float(self.x[4, 0]), float(self.x[5, 0])
