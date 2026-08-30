"""
Lane Departure Warning (LDW) System.
Computes host vehicle lateral deviation, departure rate, and Time-to-Lane-Crossing (TLC).
"""

from typing import Optional, Tuple
import numpy as np
from drivecv.config import ADASConfig
from drivecv.types import LaneBoundaries, LDWState


class LaneDepartureWarning:
    """
    Evaluates host vehicle position within detected lane boundaries:
    - Lateral offset from lane center in meters.
    - Departure velocity and Time-to-Lane-Crossing (TLC).
    - Supports Left & Right Wheel Edge Calibration (calibrates vehicle width & camera mount bias).
    - Generates directional warnings (WARNING_LEFT, WARNING_RIGHT, NORMAL).
    """

    def __init__(self, config: Optional[ADASConfig] = None):
        self.config = config or ADASConfig()
        self.prev_offset_m: Optional[float] = None
        self.cooldown_counter: int = 0
        self.current_state: LDWState = LDWState.NORMAL

        # Calibrated vehicle wheel edge positions relative to camera center (default: 1.90m wide sedan)
        self.calibrated_left_m: float = -0.95
        self.calibrated_right_m: float = 0.95
        self.is_calibrating: bool = False
        self.calibration_side: Optional[str] = None
        self.calibration_frames_target: int = 75  # ~3 seconds at 25 fps
        self.calibration_samples: list = []

    def start_calibration(self, side: str):
        """Starts 3-second recording of lane position when vehicle wheel edge is on the lane line."""
        if side in ("left", "right"):
            self.is_calibrating = True
            self.calibration_side = side
            self.calibration_samples = []

    def reset_calibration(self):
        """Resets calibration to standard 1.90m vehicle width defaults."""
        self.calibrated_left_m = -0.95
        self.calibrated_right_m = 0.95
        self.is_calibrating = False
        self.calibration_side = None
        self.calibration_samples = []

    def get_calibration_dict(self) -> dict:
        """Returns calibration status for WebSocket telemetry payload."""
        progress = (
            len(self.calibration_samples) / max(1, self.calibration_frames_target)
            if self.is_calibrating
            else 0.0
        )
        vehicle_width = self.calibrated_right_m - self.calibrated_left_m
        camera_bias = (self.calibrated_left_m + self.calibrated_right_m) / 2.0
        return {
            "is_calibrating": self.is_calibrating,
            "calibration_side": self.calibration_side,
            "calibration_progress": round(min(1.0, progress), 2),
            "calibrated_left_m": round(self.calibrated_left_m, 2),
            "calibrated_right_m": round(self.calibrated_right_m, 2),
            "vehicle_width_m": round(vehicle_width, 2),
            "camera_bias_m": round(camera_bias, 2),
        }

    def update(
        self,
        lanes: Optional[LaneBoundaries],
        frame_width: int,
        dt: float = 0.04,
    ) -> Tuple[LDWState, float, Optional[float]]:
        """
        Updates LDW state.
        Returns:
            (state, lateral_offset_m, tlc_seconds)
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        if lanes is None or not lanes.is_valid or lanes.lane_width_bottom <= 20.0:
            return LDWState.NORMAL, 0.0, None

        # Camera optical center in image coordinates is vehicle ego center
        ego_x = frame_width / 2.0
        lane_center_x = lanes.lane_center_bottom
        lane_width_px = max(1.0, lanes.lane_width_bottom)

        # Lateral offset of vehicle center from lane center in meters
        offset_px = ego_x - lane_center_x
        standard_lane_width_m = 3.70
        offset_m = (offset_px / lane_width_px) * standard_lane_width_m

        # Detected left and right line positions relative to camera center in meters
        left_line_x = lane_center_x - (lane_width_px / 2.0)
        right_line_x = lane_center_x + (lane_width_px / 2.0)

        left_line_m = (left_line_x - ego_x) / lane_width_px * standard_lane_width_m
        right_line_m = (right_line_x - ego_x) / lane_width_px * standard_lane_width_m

        # Handle active calibration sampling (when wheel is on line, sample relative line position)
        if self.is_calibrating and self.calibration_side:
            if self.calibration_side == "left":
                self.calibration_samples.append(left_line_m)
            elif self.calibration_side == "right":
                self.calibration_samples.append(right_line_m)

            if len(self.calibration_samples) >= self.calibration_frames_target:
                avg_val = float(np.mean(self.calibration_samples))
                if self.calibration_side == "left":
                    # Clamp left wheel edge to valid physical bounds [-1.35m, -0.65m]
                    self.calibrated_left_m = max(-1.35, min(-0.65, -abs(avg_val)))
                    print(f"[CALIBRATION] Left wheel edge calibrated at {self.calibrated_left_m:.2f} m")
                elif self.calibration_side == "right":
                    # Clamp right wheel edge to valid physical bounds [+0.65m, +1.35m]
                    self.calibrated_right_m = max(0.65, min(1.35, abs(avg_val)))
                    print(f"[CALIBRATION] Right wheel edge calibrated at {self.calibrated_right_m:.2f} m")
                self.is_calibrating = False
                self.calibration_side = None

        # Current 3D position of vehicle wheel edges relative to lane center
        left_wheel_pos_m = offset_m + self.calibrated_left_m
        right_wheel_pos_m = offset_m + self.calibrated_right_m

        # Clearance from vehicle wheel edges to detected lane lines
        dist_to_left_line = left_wheel_pos_m - left_line_m
        dist_to_right_line = right_line_m - right_wheel_pos_m

        # Compute lateral velocity
        tlc: Optional[float] = None
        if self.prev_offset_m is not None and dt > 0.0:
            v_lat = (offset_m - self.prev_offset_m) / dt  # m/s

            if offset_m > 0 and v_lat > 0.05:  # Moving further right
                tlc = max(0.0, dist_to_right_line) / v_lat
            elif offset_m < 0 and v_lat < -0.05:  # Moving further left
                tlc = max(0.0, dist_to_left_line) / abs(v_lat)

        self.prev_offset_m = offset_m

        # Warning threshold logic: warning fires when wheel distance to lane line is < ldw_offset_threshold_m
        thresh_m = self.config.ldw_offset_threshold_m  # default 0.45m
        tlc_thresh = self.config.ldw_tlc_threshold_s

        new_state = LDWState.NORMAL
        if dist_to_right_line <= thresh_m or (tlc is not None and 0.0 < tlc <= tlc_thresh and offset_m > 0):
            new_state = LDWState.WARNING_RIGHT
        elif dist_to_left_line <= thresh_m or (tlc is not None and 0.0 < tlc <= tlc_thresh and offset_m < 0):
            new_state = LDWState.WARNING_LEFT

        self.current_state = new_state
        return self.current_state, float(offset_m), tlc





