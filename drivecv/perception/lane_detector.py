"""
High-speed host-lane tracker: mask-seeded quadratic Kalman with Canny
narrow-band updates and symmetric Hough cold-start (target <2 ms / frame).
"""

from typing import List, Optional
import cv2
import numpy as np
from drivecv.config import LaneConfig
from drivecv.perception.lane_fit import (
    SideKalman,
    eval_quadratic,
    extract_host_lane_points,
    gate_points,
    hough_lane_points,
    occlude_tracks,
    refine_lane_points,
    sample_quadratic,
)
from drivecv.perception.lane_type_detector import LaneTypeDetector
from drivecv.types import LaneBoundaries, Track


class ClassicalLaneDetector:
    """
    Host ego-lane tracker:
    - YOLOPv2 ll/da masks seed dashed / cold-start association only.
    - Locked tracks snap to the paint ridge (intensity peak / Canny edge-pair
      midpoint), not the nearest Canny edge — that nearest-edge walk is what
      drifted the line in or out of the lane.
    - Tracked vehicle boxes are blanked from masks and edges.
    - Symmetric Hough fallback for cold start.
    """

    def __init__(self, config: Optional[LaneConfig] = None):
        self.config = config or LaneConfig()
        max_a = float(getattr(self.config, "max_poly_a", 48.0))
        max_jump = float(getattr(self.config, "max_jump_px", 28.0))
        self.left = SideKalman(max_poly_a=max_a, max_jump_px=max_jump)
        self.right = SideKalman(max_poly_a=max_a, max_jump_px=max_jump)
        self.last_ll_mask: Optional[np.ndarray] = None
        self.last_da_mask: Optional[np.ndarray] = None
        self.type_detector = LaneTypeDetector()
        # Legacy aliases used by older tests / debug
        self.left_line_ema: Optional[np.ndarray] = None
        self.right_line_ema: Optional[np.ndarray] = None
        self.left_confidence: float = 0.0
        self.right_confidence: float = 0.0

    def _empty(
        self,
        y_top: int,
        y_bot: int,
        w: int,
        da_mask: Optional[np.ndarray],
        ll_mask: Optional[np.ndarray],
    ) -> LaneBoundaries:
        return LaneBoundaries(
            left_line=None,
            right_line=None,
            y_top=y_top,
            y_bot=y_bot,
            y_roi_top=y_top,
            left_confidence=0.0,
            right_confidence=0.0,
            lane_center_bottom=w / 2.0,
            lane_width_bottom=0.38 * w,
            drivable_polygon=None,
            da_mask=da_mask,
            ll_mask=ll_mask,
            left_type="solid_yellow",
            right_type="solid_white",
            left_color="yellow",
            right_color="white",
            left_pattern="solid",
            right_pattern="solid",
        )

    def update(
        self,
        curr_gray: np.ndarray,
        tracked_objects: Optional[List[Track]] = None,
        da_mask: Optional[np.ndarray] = None,
        ll_mask: Optional[np.ndarray] = None,
        curr_bgr: Optional[np.ndarray] = None,
    ) -> LaneBoundaries:
        """Updates host lane boundary estimates and drivable corridor."""
        h, w = curr_gray.shape[:2]
        if getattr(self.config, "hood_mask_enabled", True):
            hood_ratio = getattr(self.config, "hood_height_ratio", 0.15)
            effective_y_bot_ratio = min(self.config.y_bot_ratio, max(0.40, 1.0 - hood_ratio))
        else:
            effective_y_bot_ratio = self.config.y_bot_ratio

        y_bot = int(h * effective_y_bot_ratio)
        y_top = int(h * min(self.config.y_top_ratio, effective_y_bot_ratio - 0.08))
        y_top = max(0, y_top)

        if da_mask is not None and getattr(self.config, "hood_mask_enabled", True):
            hood_cutoff = int(h * (1.0 - getattr(self.config, "hood_height_ratio", 0.15)))
            da_mask = da_mask.copy()
            da_mask[hood_cutoff:, :] = 0
        if ll_mask is not None and getattr(self.config, "hood_mask_enabled", True):
            hood_cutoff = int(h * (1.0 - getattr(self.config, "hood_height_ratio", 0.15)))
            ll_mask = ll_mask.copy()
            ll_mask[hood_cutoff:, :] = 0

        if ll_mask is not None:
            self.last_ll_mask = ll_mask.copy()
        if da_mask is not None:
            self.last_da_mask = da_mask.copy()

        active_ll = ll_mask if ll_mask is not None else self.last_ll_mask
        active_da = da_mask if da_mask is not None else self.last_da_mask
        occlude_pad = int(getattr(self.config, "vehicle_occlude_pad", 14))
        if tracked_objects:
            active_ll = occlude_tracks(active_ll, tracked_objects, pad=occlude_pad)
            active_da = occlude_tracks(active_da, tracked_objects, pad=occlude_pad)

        self.left.predict()
        self.right.predict()

        n_rows = int(getattr(self.config, "n_sample_rows", 24))
        y_samples = np.linspace(float(y_bot), float(y_top), n_rows)

        def pred_left(y: float):
            return self.left.eval_x(y, float(y_bot), float(y_top))

        def pred_right(y: float):
            return self.right.eval_x(y, float(y_bot), float(y_top))

        mask_left = []
        mask_right = []
        if active_ll is not None and active_ll.shape[0] == h and active_ll.shape[1] == w:
            mask_left, mask_right = extract_host_lane_points(
                ll_mask=active_ll,
                da_mask=active_da if (active_da is not None and active_da.shape[:2] == (h, w)) else None,
                y_top=y_top,
                y_bot=y_bot,
                n_rows=n_rows,
                mask_width=int(getattr(self.config, "mask_width", 320)),
                pred_left=pred_left,
                pred_right=pred_right,
            )

        road = curr_gray[y_top:y_bot, :]
        road_h, road_w = road.shape
        scale_w = 640.0 / max(1, road_w)
        small_h = max(10, int(road_h * scale_w))
        small_road = cv2.resize(road, (640, small_h), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(small_road, (5, 5), 0)
        edges = cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)
        if tracked_objects:
            edges = occlude_tracks(
                edges,
                tracked_objects,
                x_scale=scale_w,
                y_scale=scale_w,
                y0=float(y_top),
                pad=occlude_pad,
            )

        band = float(getattr(self.config, "search_band_px", 18.0))
        mask_gate = float(getattr(self.config, "mask_gate_px", 16.0))
        band_left = []
        band_right = []
        if self.left.x is not None:
            pred_xs = np.array([pred_left(y) or 0.0 for y in y_samples], dtype=np.float32)
            band_left = refine_lane_points(
                curr_gray, edges, y_top, scale_w, y_samples, pred_xs, band, bgr=curr_bgr
            )
        if self.right.x is not None:
            pred_xs = np.array([pred_right(y) or 0.0 for y in y_samples], dtype=np.float32)
            band_right = refine_lane_points(
                curr_gray, edges, y_top, scale_w, y_samples, pred_xs, band, bgr=curr_bgr
            )

        if self.left.valid:
            mask_left = gate_points(mask_left, pred_left, mask_gate)
        if self.right.valid:
            mask_right = gate_points(mask_right, pred_right, mask_gate)

        min_fit = int(getattr(self.config, "min_fit_points", 2))
        # Ridge = paint center; gated YOLO mask centroid is also paint center. Fuse both.
        left_pts = list(band_left) + list(mask_left)
        right_pts = list(band_right) + list(mask_right)

        need_hough = (not self.left.valid and len(left_pts) < min_fit) or (
            not self.right.valid and len(right_pts) < min_fit
        )
        if need_hough:
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=self.config.hough_threshold,
                minLineLength=self.config.hough_min_line_length,
                maxLineGap=self.config.hough_max_line_gap,
            )
            h_left, h_right = hough_lane_points(
                lines=lines,
                scale=scale_w,
                y_top=y_top,
                y_bot=y_bot,
                y_roi_top=y_top,
                img_w=w,
                min_length=float(self.config.hough_min_line_length),
                y_samples=y_samples,
            )
            if not self.left.valid and len(left_pts) < min_fit:
                left_pts.extend(h_left)
            if not self.right.valid and len(right_pts) < min_fit:
                right_pts.extend(h_right)

        if self.left.valid:
            left_pts = gate_points(left_pts, pred_left, max(band, mask_gate))
        if self.right.valid:
            right_pts = gate_points(right_pts, pred_right, max(band, mask_gate))

        self.left.update_points(left_pts, float(y_bot), float(y_top), min_points=min_fit)
        self.right.update_points(right_pts, float(y_bot), float(y_top), min_points=min_fit)

        self.left_confidence = self.left.confidence
        self.right_confidence = self.right.confidence
        self.left_line_ema = (
            np.array(
                [
                    eval_quadratic(self.left.x, float(y_bot), float(y_bot), float(y_top)),
                    eval_quadratic(self.left.x, float(y_top), float(y_bot), float(y_top)),
                ],
                dtype=np.float32,
            )
            if self.left.valid
            else None
        )
        self.right_line_ema = (
            np.array(
                [
                    eval_quadratic(self.right.x, float(y_bot), float(y_bot), float(y_top)),
                    eval_quadratic(self.right.x, float(y_top), float(y_bot), float(y_top)),
                ],
                dtype=np.float32,
            )
            if self.right.valid
            else None
        )

        left_valid = self.left.valid
        right_valid = self.right.valid
        if not left_valid and not right_valid:
            return self._empty(y_top, y_bot, w, da_mask, ll_mask)

        left_bot = self.left.eval_x(float(y_bot), float(y_bot), float(y_top)) if left_valid else None
        left_top = self.left.eval_x(float(y_top), float(y_bot), float(y_top)) if left_valid else None
        right_bot = self.right.eval_x(float(y_bot), float(y_bot), float(y_top)) if right_valid else None
        right_top = self.right.eval_x(float(y_top), float(y_bot), float(y_top)) if right_valid else None

        # 1. Vanishing point from linear chords (horizon cue)
        if left_valid and right_valid and left_bot is not None and right_bot is not None:
            dx_l = left_top - left_bot
            dx_r = right_top - right_bot
            denom = dx_l - dx_r
            if abs(denom) > 1e-4:
                t_cross = (right_bot - left_bot) / denom
                y_cross = y_bot + t_cross * (y_top - y_bot)
                x_cross = left_bot + t_cross * (left_top - left_bot)
            else:
                y_cross = -9999.0
                x_cross = w / 2.0
        else:
            y_cross = -9999.0
            x_cross = w / 2.0

        lead_obj = None
        min_y = 9999.0
        if tracked_objects is not None:
            for obj in tracked_objects:
                obj_cx = obj.bbox.x + obj.bbox.w / 2.0
                obj_bottom = obj.bbox.y + obj.bbox.h
                if 0.30 * w <= obj_cx <= 0.70 * w and obj_bottom > y_top * 0.8:
                    if obj_bottom < min_y:
                        min_y = obj_bottom
                        lead_obj = obj

        if lead_obj is not None:
            target_y_bottom = float(lead_obj.bbox.y + lead_obj.bbox.h)
            y_target = max(float(y_top), min(float(y_bot - 10), target_y_bottom))
        else:
            y_target = float(y_top)

        if y_cross < y_bot:
            y_target = max(y_target, y_cross + 25.0)
        y_target = max(float(y_top), min(float(y_bot - 15), y_target))

        num_pts = 25
        y_vals = np.linspace(y_bot, y_target, num_pts)

        def eval_side(tracker: SideKalman, yv: np.ndarray, fallback_from: Optional[np.ndarray], sign: float):
            if tracker.valid and tracker.x is not None:
                return np.array(
                    [eval_quadratic(tracker.x, float(y), float(y_bot), float(y_top)) for y in yv],
                    dtype=np.float32,
                )
            if fallback_from is not None:
                return fallback_from + sign * 0.38 * w
            return np.full(yv.shape, w / 2.0 + sign * 0.19 * w, dtype=np.float32)

        if left_valid and right_valid:
            raw_left_x = eval_side(self.left, y_vals, None, -1.0)
            raw_right_x = eval_side(self.right, y_vals, None, 1.0)
            lane_center_bottom = 0.5 * (left_bot + right_bot)
            lane_width_bottom = float(right_bot - left_bot)
        elif left_valid:
            raw_left_x = eval_side(self.left, y_vals, None, -1.0)
            raw_right_x = raw_left_x + 0.38 * w
            lane_center_bottom = float(left_bot) + 0.19 * w
            lane_width_bottom = 0.38 * w
        else:
            raw_right_x = eval_side(self.right, y_vals, None, 1.0)
            raw_left_x = raw_right_x - 0.38 * w
            lane_center_bottom = float(right_bot) - 0.19 * w
            lane_width_bottom = 0.38 * w

        left_x = np.zeros(num_pts, dtype=np.float32)
        right_x = np.zeros(num_pts, dtype=np.float32)
        for k in range(num_pts):
            if raw_right_x[k] < raw_left_x[k] + 28.0:
                mid = (raw_left_x[k] + raw_right_x[k]) / 2.0
                left_x[k] = mid - 14.0
                right_x[k] = mid + 14.0
            else:
                left_x[k] = raw_left_x[k]
                right_x[k] = raw_right_x[k]

        pts_left = np.vstack([left_x, y_vals]).T.astype(np.int32)
        pts_right = np.vstack([right_x, y_vals]).T.astype(np.int32)
        path_poly = np.vstack([pts_left, np.flipud(pts_right)])

        n_poly = int(getattr(self.config, "n_poly_samples", 12))
        left_poly = self.left.x.copy() if left_valid else None
        right_poly = self.right.x.copy() if right_valid else None
        left_poly_px = (
            sample_quadratic(left_poly, float(y_bot), float(y_top), n=n_poly, y_end=y_target)
            if left_poly is not None
            else None
        )
        right_poly_px = (
            sample_quadratic(right_poly, float(y_bot), float(y_top), n=n_poly, y_end=y_target)
            if right_poly is not None
            else None
        )

        out_left_line = (
            np.array([left_bot, left_top], dtype=np.float32) if left_valid else None
        )
        out_right_line = (
            np.array([right_bot, right_top], dtype=np.float32) if right_valid else None
        )

        frame_bgr = curr_bgr if curr_bgr is not None else cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
        left_info = self.type_detector.analyze_line(
            frame_bgr=frame_bgr,
            line=out_left_line,
            y_bot=y_bot,
            y_top=int(y_top),
            default_color="yellow",
            default_pattern="solid",
            side="left",
        )
        right_info = self.type_detector.analyze_line(
            frame_bgr=frame_bgr,
            line=out_right_line,
            y_bot=y_bot,
            y_top=int(y_top),
            default_color="white",
            default_pattern="solid",
            side="right",
        )

        return LaneBoundaries(
            left_line=out_left_line,
            right_line=out_right_line,
            y_top=int(y_target),
            y_bot=y_bot,
            y_roi_top=int(y_top),
            left_confidence=float(self.left.confidence / SideKalman.CONF_MAX),
            right_confidence=float(self.right.confidence / SideKalman.CONF_MAX),
            lane_center_bottom=float(lane_center_bottom),
            lane_width_bottom=float(lane_width_bottom),
            vanish_x=float(x_cross) if y_cross > -1000 else None,
            vanish_y=float(y_cross) if y_cross > -1000 else None,
            drivable_polygon=path_poly,
            da_mask=da_mask,
            ll_mask=ll_mask,
            left_type=left_info["type"],
            right_type=right_info["type"],
            left_color=left_info["color"],
            right_color=right_info["color"],
            left_pattern=left_info["pattern"],
            right_pattern=right_info["pattern"],
            left_poly=left_poly,
            right_poly=right_poly,
            left_poly_px=left_poly_px,
            right_poly_px=right_poly_px,
        )
