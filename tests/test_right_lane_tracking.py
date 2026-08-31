"""
Unit tests for host right lane tracking: Hough fallback, dashed coasting,
mask extraction, quadratic curves, and unseen-right = None.
"""

import unittest
import cv2
import numpy as np
from drivecv.config import LaneConfig
from drivecv.perception.lane_detector import ClassicalLaneDetector
from drivecv.perception.lane_fit import (
    extract_host_lane_points,
    fit_quadratic,
    eval_quadratic,
    intensity_ridge_x,
)
from drivecv.types import BoundingBox, LaneBoundaries, Track


def _blank(h=540, w=960, value=50):
    return np.full((h, w), value, dtype=np.uint8)


def _draw_left(gray):
    cv2.line(gray, (240, 500), (430, 310), 255, thickness=4)


def _draw_right(gray):
    cv2.line(gray, (720, 500), (550, 310), 255, thickness=4)


def _draw_dashed_right(gray, period=18, on=8):
    p0 = np.array([720.0, 500.0])
    p1 = np.array([550.0, 310.0])
    length = float(np.linalg.norm(p1 - p0))
    direction = (p1 - p0) / max(1e-3, length)
    s = 0.0
    while s < length:
        a = p0 + direction * s
        b = p0 + direction * min(length, s + on)
        cv2.line(gray, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 255, thickness=4)
        s += period


class TestRightLaneTracking(unittest.TestCase):
    def test_right_lane_positioning_relative_to_left(self):
        config = LaneConfig(y_bot_ratio=0.95, y_top_ratio=0.58)
        detector = ClassicalLaneDetector(config=config)

        gray = _blank()
        _draw_left(gray)
        _draw_right(gray)

        lanes = detector.update(curr_gray=gray)

        self.assertIsNotNone(lanes.left_line)
        self.assertIsNotNone(lanes.right_line)

        left_bot = lanes.left_line[0]
        right_bot = lanes.right_line[0]

        self.assertGreater(right_bot, 500.0)
        self.assertLess(left_bot, 480.0)

    def test_right_lane_unseen_returns_none(self):
        config = LaneConfig(y_bot_ratio=0.95, y_top_ratio=0.58)
        detector = ClassicalLaneDetector(config=config)

        gray = _blank()
        _draw_left(gray)

        lanes = detector.update(curr_gray=gray)

        self.assertIsNotNone(lanes.left_line)
        self.assertIsNone(lanes.right_line)
        self.assertIsNone(lanes.right_poly_px)

    def test_dashed_right_persists_through_gaps(self):
        config = LaneConfig(y_bot_ratio=0.95, y_top_ratio=0.58)
        detector = ClassicalLaneDetector(config=config)

        lanes = None
        for _ in range(6):
            gray = _blank()
            _draw_left(gray)
            _draw_dashed_right(gray)
            lanes = detector.update(curr_gray=gray)

        self.assertIsNotNone(lanes.right_line)
        self.assertGreater(float(lanes.right_line[0]), 500.0)

        for _ in range(10):
            gray = _blank()
            _draw_left(gray)
            lanes = detector.update(curr_gray=gray)

        self.assertIsNotNone(lanes.right_line, "right lane should coast through dashed gaps")
        self.assertGreater(float(lanes.right_confidence), 0.0)

    def test_mask_extraction_assigns_da_edges(self):
        h, w = 540, 960
        ll = np.zeros((h, w), dtype=np.uint8)
        da = np.zeros((h, w), dtype=np.uint8)
        for y in range(320, 500):
            t = (y - 320) / 180.0
            xl = int(430 + t * (240 - 430))
            xr = int(550 + t * (720 - 550))
            ll[y, max(0, xl - 2) : min(w, xl + 3)] = 1
            ll[y, max(0, xr - 2) : min(w, xr + 3)] = 1
            da[y, max(0, xl) : min(w, xr + 1)] = 1

        left_pts, right_pts = extract_host_lane_points(
            ll_mask=ll, da_mask=da, y_top=320, y_bot=490, n_rows=16
        )
        self.assertGreater(len(left_pts), 4)
        self.assertGreater(len(right_pts), 4)
        left_mean = float(np.mean([p[0] for p in left_pts]))
        right_mean = float(np.mean([p[0] for p in right_pts]))
        self.assertLess(left_mean, 480.0)
        self.assertGreater(right_mean, 500.0)

    def test_detector_uses_lane_masks(self):
        config = LaneConfig(y_bot_ratio=0.95, y_top_ratio=0.58, hood_mask_enabled=False)
        detector = ClassicalLaneDetector(config=config)
        h, w = 540, 960
        gray = _blank()
        ll = np.zeros((h, w), dtype=np.uint8)
        da = np.zeros((h, w), dtype=np.uint8)
        for y in range(310, 513):
            t = (y - 310) / 203.0
            xl = int(430 + t * (240 - 430))
            xr = int(550 + t * (720 - 550))
            ll[y, max(0, xl - 3) : min(w, xl + 4)] = 1
            ll[y, max(0, xr - 3) : min(w, xr + 4)] = 1
            da[y, max(0, xl) : min(w, xr + 1)] = 1

        lanes = detector.update(curr_gray=gray, ll_mask=ll, da_mask=da)
        self.assertIsNotNone(lanes.left_line)
        self.assertIsNotNone(lanes.right_line)
        self.assertGreater(float(lanes.right_line[0]), 500.0)
        self.assertIsNotNone(lanes.left_poly_px)
        self.assertIsNotNone(lanes.right_poly_px)

    def test_quadratic_curve_fit(self):
        ys = np.linspace(400.0, 250.0, 12)
        # Right-bending curve: x increases faster than linear toward the bottom
        xs = 520.0 + 0.35 * (ys - 250.0) + 0.0012 * (ys - 250.0) ** 2
        coeffs = fit_quadratic(xs, ys, y_bot=400.0, y_top=250.0)
        self.assertIsNotNone(coeffs)
        mid_y = 325.0
        x_mid = eval_quadratic(coeffs, mid_y, 400.0, 250.0)
        x_bot = eval_quadratic(coeffs, 400.0, 400.0, 250.0)
        x_top = eval_quadratic(coeffs, 250.0, 400.0, 250.0)
        chord_mid = 0.5 * (x_bot + x_top)
        self.assertGreater(abs(x_mid - chord_mid), 2.0)

    def test_x_bounds_at_uses_poly(self):
        y_bot, y_top = 500.0, 300.0
        # x = 80*yn^2 + 120*yn + 200  -> 200 at bottom, 280 at mid, 400 at top
        left_poly = np.array([80.0, 120.0, 200.0], dtype=np.float32)
        right_poly = np.array([-80.0, -160.0, 760.0], dtype=np.float32)
        lanes = LaneBoundaries(
            left_line=np.array([200.0, 400.0], dtype=np.float32),
            right_line=np.array([760.0, 520.0], dtype=np.float32),
            y_bot=int(y_bot),
            y_roi_top=int(y_top),
            y_top=int(y_top),
            left_confidence=1.0,
            right_confidence=1.0,
            left_poly=left_poly,
            right_poly=right_poly,
        )
        bounds = lanes.x_bounds_at(400.0)
        self.assertIsNotNone(bounds)
        xl, xr = bounds
        self.assertAlmostEqual(xl, 280.0, delta=1.0)
        # Linear chord of left_line would be 300 at mid-row
        self.assertNotAlmostEqual(xl, 300.0, delta=5.0)

    def test_published_line_follows_yolo_mask(self):
        """With YOLO ll/da masks, the published polyline must stay on the mask blobs."""
        config = LaneConfig(y_bot_ratio=0.95, y_top_ratio=0.58, hood_mask_enabled=False)
        detector = ClassicalLaneDetector(config=config)
        h, w = 540, 960
        gray = _blank()
        ll = np.zeros((h, w), dtype=np.uint8)
        da = np.zeros((h, w), dtype=np.uint8)
        for y in range(310, 513):
            t = (y - 310) / 203.0
            xl = int(430 + t * (240 - 430))
            xr = int(550 + t * (720 - 550))
            ll[y, max(0, xl - 3) : min(w, xl + 4)] = 1
            ll[y, max(0, xr - 3) : min(w, xr + 4)] = 1
            da[y, max(0, xl) : min(w, xr + 1)] = 1

        lanes = None
        for _ in range(4):
            lanes = detector.update(curr_gray=gray, ll_mask=ll, da_mask=da)
        self.assertIsNotNone(lanes.left_poly_px)
        self.assertIsNotNone(lanes.right_poly_px)

        def true_xl(y):
            t = (float(y) - 310.0) / 203.0
            return 430.0 + t * (240.0 - 430.0)

        errs = []
        for x, y in lanes.left_poly_px:
            if 320.0 <= y <= 500.0:
                errs.append(abs(float(x) - true_xl(y)))
        self.assertGreater(len(errs), 4)
        self.assertLess(float(np.median(errs)), 10.0)

    def test_cv_refine_stays_inside_yolo_segment(self):
        """Bright glare outside the YOLO blob must not pull the fit off the mask."""
        config = LaneConfig(y_bot_ratio=0.95, y_top_ratio=0.58, hood_mask_enabled=False)
        detector = ClassicalLaneDetector(config=config)
        h, w = 540, 960
        gray = _blank()
        # Strong vertical glare well inside the lane, away from the left paint
        gray[:, 360:380] = 255
        ll = np.zeros((h, w), dtype=np.uint8)
        da = np.zeros((h, w), dtype=np.uint8)
        for y in range(310, 513):
            t = (y - 310) / 203.0
            xl = int(430 + t * (240 - 430))
            xr = int(550 + t * (720 - 550))
            ll[y, max(0, xl - 3) : min(w, xl + 4)] = 1
            ll[y, max(0, xr - 3) : min(w, xr + 4)] = 1
            da[y, max(0, xl) : min(w, xr + 1)] = 1
            gray[y, max(0, xl - 2) : min(w, xl + 3)] = 220

        lanes = None
        for _ in range(5):
            lanes = detector.update(curr_gray=gray, ll_mask=ll, da_mask=da)
        self.assertIsNotNone(lanes.left_line)
        self.assertLess(float(lanes.left_line[0]), 320.0)
        self.assertGreater(float(lanes.left_line[0]), 180.0)

    def test_da_edges_track_when_ll_mask_empty(self):
        """Drivable-area left/right edges must still produce host lines (ll_mask is often sparse)."""
        config = LaneConfig(y_bot_ratio=0.95, y_top_ratio=0.58, hood_mask_enabled=False)
        detector = ClassicalLaneDetector(config=config)
        h, w = 540, 960
        gray = _blank()
        ll = np.zeros((h, w), dtype=np.uint8)
        da = np.zeros((h, w), dtype=np.uint8)
        for y in range(310, 513):
            t = (y - 310) / 203.0
            xl = int(430 + t * (240 - 430))
            xr = int(550 + t * (720 - 550))
            da[y, max(0, xl) : min(w, xr + 1)] = 1

        lanes = None
        for _ in range(4):
            lanes = detector.update(curr_gray=gray, ll_mask=ll, da_mask=da)
        self.assertIsNotNone(lanes.left_line)
        self.assertIsNotNone(lanes.right_line)
        self.assertLess(float(lanes.left_line[0]), 320.0)
        self.assertGreater(float(lanes.right_line[0]), 500.0)
        self.assertGreater(float(lanes.left_confidence), 0.10)

    def test_hood_mask_clips_y_bot(self):
        config = LaneConfig(y_bot_ratio=0.95, hood_mask_enabled=True, hood_height_ratio=0.20)
        detector = ClassicalLaneDetector(config=config)
        lanes = detector.update(curr_gray=_blank())
        self.assertEqual(lanes.y_bot, 432)

    def test_hood_mask_disabled_uses_y_bot_ratio(self):
        config = LaneConfig(y_bot_ratio=0.95, hood_mask_enabled=False, hood_height_ratio=0.20)
        detector = ClassicalLaneDetector(config=config)
        lanes = detector.update(curr_gray=_blank())
        self.assertEqual(lanes.y_bot, 513)

    def test_left_lane_stable_when_car_and_yolo_blob(self):
        """Passing-car contours + YOLO ll/da edges must not yank a locked left line."""
        config = LaneConfig(y_bot_ratio=0.95, y_top_ratio=0.58)
        detector = ClassicalLaneDetector(config=config)
        lanes = None
        for _ in range(6):
            gray = _blank()
            _draw_left(gray)
            _draw_right(gray)
            lanes = detector.update(curr_gray=gray)
        self.assertIsNotNone(lanes.left_line)
        locked_bot = float(lanes.left_line[0])

        h, w = 540, 960
        car = Track(track_id=7, bbox=BoundingBox(x=70.0, y=270.0, w=130.0, h=210.0))
        for _ in range(8):
            gray = _blank()
            _draw_left(gray)
            _draw_right(gray)
            cv2.rectangle(gray, (70, 270), (200, 480), 230, thickness=-1)
            ll = np.zeros((h, w), dtype=np.uint8)
            da = np.zeros((h, w), dtype=np.uint8)
            for y in range(310, 500):
                t = (y - 310) / 190.0
                xl = int(430 + t * (240 - 430))
                xr = int(550 + t * (720 - 550))
                ll[y, max(0, xl - 2) : min(w, xl + 3)] = 1
                ll[y, max(0, xr - 2) : min(w, xr + 3)] = 1
                da[y, max(0, xl) : min(w, xr + 1)] = 1
            # False YOLO paint on the car's left body, well off the real line
            ll[280:470, 85:115] = 1
            da[280:470, 90:720] = 1
            lanes = detector.update(
                curr_gray=gray,
                ll_mask=ll,
                da_mask=da,
                tracked_objects=[car],
            )

        self.assertIsNotNone(lanes.left_line)
        self.assertLess(abs(float(lanes.left_line[0]) - locked_bot), 28.0)
        # Must not have snapped onto the car (~x=100)
        self.assertGreater(float(lanes.left_line[0]), 200.0)

    def test_ridge_expands_when_wide_paint_clips_window(self):
        """Near-field yellow is wider than a 12px band; ridge must expand to the true center."""
        gray = np.full((120, 400), 40, dtype=np.uint8)
        gray[100, 80:130] = 220  # 50px marking, center x=104.5
        x = intensity_ridge_x(gray, y=100.0, x_pred=118.0, band_px=12.0, max_band_px=64.0)
        self.assertIsNotNone(x)
        self.assertAlmostEqual(x, 104.5, delta=4.0)

    def test_intensity_ridge_is_paint_center_not_edge(self):
        gray = np.full((80, 200), 40, dtype=np.uint8)
        gray[40, 80:100] = 220  # 20px-wide marking, center x=89.5
        x = intensity_ridge_x(gray, y=40.0, x_pred=90.0, band_px=24.0)
        self.assertIsNotNone(x)
        self.assertAlmostEqual(x, 89.5, delta=2.0)

    def test_left_lane_does_not_walk_off_thick_paint(self):
        """Nearest-edge Canny would walk inward on a wide marking; ridge must stay put."""
        config = LaneConfig(y_bot_ratio=0.95, y_top_ratio=0.58)
        detector = ClassicalLaneDetector(config=config)
        lanes = None
        for _ in range(8):
            gray = _blank()
            cv2.line(gray, (240, 500), (430, 310), 255, thickness=12)
            _draw_right(gray)
            lanes = detector.update(curr_gray=gray)
        self.assertIsNotNone(lanes.left_line)
        locked = float(lanes.left_line[0])
        for _ in range(40):
            gray = _blank()
            cv2.line(gray, (240, 500), (430, 310), 255, thickness=12)
            _draw_right(gray)
            lanes = detector.update(curr_gray=gray)
        self.assertIsNotNone(lanes.left_line)
        self.assertLess(abs(float(lanes.left_line[0]) - locked), 10.0)


if __name__ == "__main__":
    unittest.main()
