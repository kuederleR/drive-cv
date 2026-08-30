"""
Unit tests for host right lane line tracking and spatial separation in ClassicalLaneDetector.
"""

import unittest
import cv2
import numpy as np
from drivecv.config import LaneConfig
from drivecv.perception.lane_detector import ClassicalLaneDetector


class TestRightLaneTracking(unittest.TestCase):
    def test_right_lane_positioning_relative_to_left(self):
        config = LaneConfig(y_bot_ratio=0.95, y_top_ratio=0.58)
        detector = ClassicalLaneDetector(config=config)

        w, h = 960, 540
        gray = np.full((h, w), 50, dtype=np.uint8)

        # Draw a clear left lane line (yellow/white line on left side)
        cv2.line(gray, (240, 500), (430, 310), 255, thickness=4)
        # Draw a line in the middle of the lane (center noise artifact)
        cv2.line(gray, (480, 500), (490, 310), 255, thickness=4)

        lanes = detector.update(curr_gray=gray)

        self.assertIsNotNone(lanes.left_line)
        self.assertIsNotNone(lanes.right_line)

        left_bot = lanes.left_line[0]
        right_bot = lanes.right_line[0]

        # Center of frame is 480. Right lane line bottom must be well to the right of frame center (> 500)
        self.assertGreater(right_bot, 500.0)

        # Check reference EMA at y_ref_bot (513) maintains true host lane width (0.38 * w)
        self.assertIsNotNone(detector.left_line_ema)
        self.assertIsNotNone(detector.right_line_ema)
        ema_width = detector.right_line_ema[0] - detector.left_line_ema[0]
        self.assertAlmostEqual(ema_width, 0.38 * w, delta=40.0)

    def test_right_lane_fallback_when_unseen(self):
        config = LaneConfig(y_bot_ratio=0.95, y_top_ratio=0.58)
        detector = ClassicalLaneDetector(config=config)

        w, h = 960, 540
        gray = np.full((h, w), 50, dtype=np.uint8)

        # Draw only left line
        cv2.line(gray, (240, 500), (430, 310), 255, thickness=4)

        lanes = detector.update(curr_gray=gray)

        self.assertIsNotNone(detector.left_line_ema)
        self.assertIsNotNone(detector.right_line_ema)

        # At reference bottom (y_ref_bot), right line EMA must fallback using parallel projection (left + 0.38 * w)
        ema_left_bot = detector.left_line_ema[0]
        ema_right_bot = detector.right_line_ema[0]
        self.assertAlmostEqual(ema_right_bot, ema_left_bot + 0.38 * w, delta=5.0)


if __name__ == "__main__":
    unittest.main()
