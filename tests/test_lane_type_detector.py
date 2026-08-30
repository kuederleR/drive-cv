"""
Unit tests for LaneTypeDetector (color, pattern, and double line detection).
"""

import unittest
import numpy as np
import cv2
from drivecv.perception.lane_type_detector import LaneTypeDetector
from drivecv.types import LaneBoundaries


class TestLaneTypeDetector(unittest.TestCase):

    def setUp(self):
        self.detector = LaneTypeDetector(history_size=5)

    def test_solid_white_line_detection(self):
        # Create a test frame with a solid white vertical line
        frame = np.full((300, 400, 3), 40, dtype=np.uint8)  # Dark asphalt background
        cv2.line(frame, (100, 290), (100, 100), (240, 240, 240), thickness=6)  # White line

        line = np.array([100.0, 100.0], dtype=np.float32)
        res = self.detector.analyze_line(frame, line, y_bot=290, y_top=100, side="right")

        self.assertEqual(res["color"], "white")
        self.assertEqual(res["pattern"], "solid")
        self.assertEqual(res["type"], "solid_white")

    def test_solid_yellow_line_detection(self):
        # Create a test frame with a solid yellow vertical line
        frame = np.full((300, 400, 3), 40, dtype=np.uint8)
        # BGR yellow (0, 215, 255)
        cv2.line(frame, (100, 290), (100, 100), (0, 215, 255), thickness=6)

        line = np.array([100.0, 100.0], dtype=np.float32)
        res = self.detector.analyze_line(frame, line, y_bot=290, y_top=100, side="left")

        self.assertEqual(res["color"], "yellow")
        self.assertEqual(res["pattern"], "solid")
        self.assertEqual(res["type"], "solid_yellow")

    def test_dashed_white_line_detection(self):
        # Create a test frame with a dashed white line
        frame = np.full((300, 400, 3), 40, dtype=np.uint8)
        for y in range(100, 290, 30):
            cv2.line(frame, (100, y + 15), (100, y), (240, 240, 240), thickness=6)

        line = np.array([100.0, 100.0], dtype=np.float32)
        res = self.detector.analyze_line(frame, line, y_bot=290, y_top=100, side="right")

        self.assertEqual(res["color"], "white")
        self.assertEqual(res["pattern"], "dashed")
        self.assertEqual(res["type"], "dashed_white")

    def test_double_yellow_line_detection(self):
        # Create a test frame with double parallel yellow lines
        frame = np.full((300, 400, 3), 40, dtype=np.uint8)
        cv2.line(frame, (94, 290), (94, 100), (0, 215, 255), thickness=4)
        cv2.line(frame, (106, 290), (106, 100), (0, 215, 255), thickness=4)

        line = np.array([100.0, 100.0], dtype=np.float32)
        res = self.detector.analyze_line(frame, line, y_bot=290, y_top=100, side="left")

        self.assertEqual(res["color"], "yellow")
        self.assertEqual(res["pattern"], "double")
        self.assertEqual(res["type"], "double_yellow")


if __name__ == "__main__":
    unittest.main()
