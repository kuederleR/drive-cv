"""
Panoptic Scene Visualizer: AR Drivable Path, Vehicle 3D Bounding Boxes, and ADAS Badges.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from drivecv.config import VisualizerConfig
from drivecv.types import ADASAlertLevel, FrameData, LaneBoundaries, Track, TrackLifecycle


def certainty_to_color(certainty: float) -> Tuple[int, int, int]:
    """Maps certainty [0.0, 1.0] to BGR color gradient: Red -> Yellow -> Green."""
    certainty = max(0.0, min(1.0, float(certainty)))
    if certainty >= 0.5:
        t = (certainty - 0.5) * 2.0
        return (0, 255, int(255 * (1.0 - t)))
    else:
        t = certainty * 2.0
        return (0, int(255 * t), 255)


class PanopticVisualizer:
    """
    Renders rich augmented driving visualization:
    - Host lane AR drivable path (Electric Cyan) extending safely to lead vehicle.
    - Lane boundary lines.
    - Vehicle bounding boxes with certainty colors and white flash on neural lock.
    - ADAS badges (Distance, Closing Speed, TTC, Collision warning rings).
    - Trajectory trail and motion velocity vectors.
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config or VisualizerConfig()

    def render(
        self,
        frame: np.ndarray,
        frame_data: FrameData,
        vis_mode: str = "ALL",
    ) -> np.ndarray:
        """Renders complete scene overlay on frame."""
        vis_frame = frame.copy()

        # 1. Neural Segmentation Overlay (if enabled and available)
        debug_mode = str(getattr(self.config, "lane_debug", "off") or "off").lower()
        show_masks = self.config.show_seg_masks or debug_mode in ("masks", "all")
        if show_masks and frame_data.lanes is not None:
            self._draw_neural_masks(vis_frame, frame_data.lanes)

        # 1b. Lane tracker debug (Canny / ridge search / measurements)
        if debug_mode not in ("", "off", "none") and frame_data.lanes is not None:
            self._draw_lane_debug(vis_frame, frame_data.lanes, debug_mode)

        # 2. AR Drivable Path & Non-Crossing Boundary Lines
        if self.config.show_path and frame_data.lanes is not None:
            self._draw_drivable_path(vis_frame, frame_data.lanes)

        # 3. Tracked Objects & ADAS Badges
        if self.config.show_boxes:
            for track in frame_data.tracks:
                if vis_mode == "MINIMAL" and not track.kinematics.is_lead_vehicle:
                    continue
                if vis_mode == "DET_ONLY" and track.certainty < 0.30:
                    continue

                self._draw_track(
                    vis_frame,
                    track,
                    show_vectors=(vis_mode in ["ALL", "DET_ONLY"]),
                    show_points=(vis_mode == "ALL"),
                )

        return vis_frame

    def _draw_neural_masks(self, frame: np.ndarray, lanes: LaneBoundaries):
        overlay = frame.copy()
        if lanes.da_mask is not None and np.any(lanes.da_mask == 1):
            overlay[lanes.da_mask == 1] = (255, 180, 0)
        if lanes.ll_mask is not None and np.any(lanes.ll_mask == 1):
            # Highlight YOLOP neural lane lines in bright red
            overlay[lanes.ll_mask == 1] = (0, 0, 255)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    def _draw_lane_debug(self, frame: np.ndarray, lanes: LaneBoundaries, mode: str):
        """Overlays Canny, search bands, and ridge measurements on the camera feed."""
        h, w = frame.shape[:2]
        show_canny = mode in ("canny", "all")
        show_ridge = mode in ("ridge", "all", "canny")

        if show_canny and lanes.debug_canny is not None and lanes.debug_canny.shape[:2] == (h, w):
            overlay = frame.copy()
            overlay[lanes.debug_canny > 0] = (0, 255, 80)
            cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

        def draw_bands(bands, color):
            if bands is None or len(bands) == 0:
                return
            for x_pred, y, half in bands:
                yi = int(round(y))
                if yi < 0 or yi >= h:
                    continue
                x1 = int(round(float(x_pred) - float(half)))
                x2 = int(round(float(x_pred) + float(half)))
                cv2.line(frame, (max(0, x1), yi), (min(w - 1, x2), yi), color, 1, cv2.LINE_AA)
                cv2.circle(frame, (int(round(float(x_pred))), yi), 2, color, -1, cv2.LINE_AA)

        def draw_meas(pts, default_color):
            src_colors = {
                0: (0, 255, 255),   # yellow chroma
                1: (0, 180, 255),   # intensity ridge
                2: (80, 220, 255),  # matched filter
                3: (0, 255, 80),    # Canny pair
                4: (0, 0, 255),     # YOLO ll_mask
                5: (255, 255, 255), # Hough
            }
            if pts is None or len(pts) == 0:
                return
            arr = np.asarray(pts)
            for row in arr:
                x, y = float(row[0]), float(row[1])
                color = default_color
                if row.size >= 3:
                    color = src_colors.get(int(row[2]), default_color)
                cv2.drawMarker(
                    frame,
                    (int(round(x)), int(round(y))),
                    color,
                    cv2.MARKER_CROSS,
                    8,
                    1,
                    cv2.LINE_AA,
                )

        if show_ridge:
            draw_bands(lanes.debug_left_bands, (255, 180, 80))
            draw_bands(lanes.debug_right_bands, (80, 180, 255))
            draw_meas(lanes.debug_left_meas, (0, 255, 255))
            draw_meas(lanes.debug_right_meas, (255, 0, 255))
            if lanes.left_poly_px is not None and len(lanes.left_poly_px) >= 2:
                cv2.polylines(frame, [lanes.left_poly_px.astype(np.int32)], False, (0, 220, 255), 1, cv2.LINE_AA)
            if lanes.right_poly_px is not None and len(lanes.right_poly_px) >= 2:
                cv2.polylines(frame, [lanes.right_poly_px.astype(np.int32)], False, (255, 80, 255), 1, cv2.LINE_AA)

        label = f"LANE DBG: {mode.upper()}"
        cv2.putText(frame, label, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)

    def _draw_drivable_path(self, frame: np.ndarray, lanes: LaneBoundaries):
        if lanes.drivable_polygon is not None and len(lanes.drivable_polygon) >= 6:
            h, w = frame.shape[:2]
            y_top = lanes.y_top
            y_bot = lanes.y_bot

            if y_bot > y_top:
                road_slice = frame[y_top:y_bot, :].copy()
                poly_rel = lanes.drivable_polygon.copy()
                poly_rel[:, 1] -= y_top
                cv2.fillPoly(road_slice, [poly_rel], self.config.drivable_color)
                cv2.addWeighted(road_slice, 0.35, frame[y_top:y_bot, :], 0.65, 0, frame[y_top:y_bot, :])

                # Draw non-crossing boundary lines along the polygon edges
                half = len(lanes.drivable_polygon) // 2
                pts_left = lanes.drivable_polygon[:half]
                pts_right = np.flipud(lanes.drivable_polygon[half:])

                if lanes.left_line is not None:
                    cv2.polylines(frame, [pts_left], False, self.config.lane_color, 2, cv2.LINE_AA)
                if lanes.right_line is not None:
                    cv2.polylines(frame, [pts_right], False, self.config.lane_color, 2, cv2.LINE_AA)

                # Hood cutoff line
                cv2.line(frame, (0, y_bot), (w, y_bot), (80, 80, 80), 1, cv2.LINE_AA)

    def _draw_track(
        self,
        frame: np.ndarray,
        track: Track,
        show_vectors: bool = True,
        show_points: bool = False,
    ):
        x, y, w, h = track.bbox.as_int_xywh()
        if w <= 0 or h <= 0:
            return

        cx, cy = int(x + w / 2.0), int(y + h / 2.0)
        box_color = (255, 255, 255) if track.flash_frames > 0 else certainty_to_color(track.certainty)

        # Draw Bounding Box with rounded corners / corner brackets
        thickness = 2 if track.kinematics.is_lead_vehicle else 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, thickness, cv2.LINE_AA)

        # Center dot
        area = w * h
        dot_r = max(3, min(8, int(round(5.0 * (area / 12000.0) ** 0.4))))
        cv2.circle(frame, (cx, cy), dot_r, box_color, -1, cv2.LINE_AA)

        # Motion trail history
        if len(track.history) > 1:
            pts = np.array(track.history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, box_color, 1, cv2.LINE_AA)

        # Velocity vector
        if show_vectors and np.linalg.norm(track.kinematics.velocity_2d) > 0.3:
            vx, vy = track.kinematics.velocity_2d
            end_x = int(cx + vx * 4.0)
            end_y = int(cy + vy * 4.0)
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 255, 255), 1, tipLength=0.3)

        # Keypoints
        if show_points and track.keypoints is not None:
            for pt in track.keypoints:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

        # ADAS Badge above bounding box
        if self.config.show_adas_badges:
            badge_parts = [f"ID:{track.track_id}"]
            if track.kinematics.distance_m > 0:
                badge_parts.append(f"{track.kinematics.distance_m:.1f}m")
            if abs(track.kinematics.rel_speed_kmh) >= 1.0:
                badge_parts.append(f"{track.kinematics.rel_speed_kmh:+.0f}km/h")
            if track.kinematics.ttc_seconds is not None:
                badge_parts.append(f"TTC:{track.kinematics.ttc_seconds:.1f}s")

            badge_text = " ".join(badge_parts)
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)

            bx1 = max(0, x)
            by1 = max(0, y - th - 6)
            bx2 = min(frame.shape[1], bx1 + tw + 6)
            by2 = y

            if bx2 > bx1 and by2 > by1:
                badge_slice = frame[by1:by2, bx1:bx2].copy()
                bg_color = (0, 0, 180) if track.kinematics.is_lead_vehicle and track.kinematics.ttc_seconds and track.kinematics.ttc_seconds < 2.0 else (30, 30, 30)
                cv2.rectangle(badge_slice, (0, 0), (bx2 - bx1, by2 - by1), bg_color, -1)
                cv2.addWeighted(badge_slice, 0.80, frame[by1:by2, bx1:bx2], 0.20, 0, frame[by1:by2, bx1:bx2])
                cv2.putText(frame, badge_text, (bx1 + 3, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
