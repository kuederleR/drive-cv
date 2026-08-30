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
        # Draw a real right lane line on the right side of the road
        cv2.line(gray, (720, 500), (550, 310), 255, thickness=4)

        lanes = detector.update(curr_gray=gray)

        self.assertIsNotNone(lanes.left_line)
        self.assertIsNotNone(lanes.right_line)

        left_bot = lanes.left_line[0]
        right_bot = lanes.right_line[0]

        # Center of frame is 480. Right lane line bottom must be well to the right of frame center (> 500)
        self.assertGreater(right_bot, 500.0)

    def test_right_lane_unseen_returns_none(self):
        config = LaneConfig(y_bot_ratio=0.95, y_top_ratio=0.58)
        detector = ClassicalLaneDetector(config=config)

        w, h = 960, 540
        gray = np.full((h, w), 50, dtype=np.uint8)

        # Draw only left line
        cv2.line(gray, (240, 500), (430, 310), 255, thickness=4)

        lanes = detector.update(curr_gray=gray)

        self.assertIsNotNone(lanes.left_line)
        # When right line is not detected, right_line must be None (no right line shown)
        self.assertIsNone(lanes.right_line)


if __name__ == "__main__":
    unittest.main()
