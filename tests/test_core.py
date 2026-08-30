"""
Unit tests for DriveCV core geometry and math utilities using unittest.
"""

import unittest
import numpy as np
from drivecv.config import CameraConfig
from drivecv.core.geometry import CameraGeometry, RangeKalman
from drivecv.core.math_utils import compute_iom, compute_iou, compute_iou_matrix
from drivecv.types import BoundingBox, LaneBoundaries


class TestCore(unittest.TestCase):
    def test_iou_calculation(self):
        box1 = BoundingBox(x=10, y=10, w=100, h=100)
        box2 = BoundingBox(x=10, y=10, w=100, h=100)
        # Identical boxes -> IoU == 1.0
        self.assertAlmostEqual(compute_iou(box1, box2), 1.0, places=2)

        # Disjoint boxes -> IoU == 0.0
        box3 = BoundingBox(x=200, y=200, w=50, h=50)
        self.assertEqual(compute_iou(box1, box3), 0.0)

        # Half overlap
        box4 = BoundingBox(x=60, y=10, w=100, h=100)
        iou = compute_iou(box1, box4)
        self.assertTrue(0.2 < iou < 0.6)

        # Nested fragment must use true IoU (not a containment boost)
        big = BoundingBox(x=0, y=0, w=100, h=100)
        small = BoundingBox(x=10, y=10, w=20, h=20)
        self.assertLess(compute_iou(big, small), 0.10)
        self.assertAlmostEqual(compute_iom(big, small), 1.0, places=2)

    def test_iou_matrix(self):
        boxes1 = [BoundingBox(0, 0, 50, 50), BoundingBox(100, 100, 50, 50)]
        boxes2 = [BoundingBox(0, 0, 50, 50), BoundingBox(200, 200, 50, 50)]
        mat = compute_iou_matrix(boxes1, boxes2)
        self.assertEqual(mat.shape, (2, 2))
        self.assertAlmostEqual(float(mat[0, 0]), 1.0, places=2)
        self.assertEqual(float(mat[0, 1]), 0.0)

    def test_camera_geometry(self):
        config = CameraConfig(focal_length_px=1150.0, camera_height_m=1.25, camera_pitch_rad=0.0)
        geom = CameraGeometry(config, img_width=1280, img_height=720)

        # Point at bottom of image (u=640, v=700) -> very close vehicle
        dist = geom.estimate_distance_to_contact_point(640, 700)
        self.assertIsNotNone(dist)
        self.assertTrue(2.0 <= dist <= 10.0)

        # Point near horizon (v=380) -> distant vehicle
        dist_far = geom.estimate_distance_to_contact_point(640, 380)
        self.assertIsNotNone(dist_far)
        self.assertTrue(dist_far > dist)

        # Lateral offset test
        lat_center = geom.estimate_lateral_offset(640, 20.0)
        self.assertAlmostEqual(lat_center, 0.0, places=2)

        lat_right = geom.estimate_lateral_offset(800, 20.0)
        self.assertTrue(lat_right > 0.0)

    def _synthetic_lanes(self) -> LaneBoundaries:
        return LaneBoundaries(
            left_line=np.array([200.0, 450.0], dtype=np.float32),
            right_line=np.array([760.0, 510.0], dtype=np.float32),
            y_bot=680,
            y_roi_top=500,
            y_top=500,
            vanish_x=480.0,
            vanish_y=320.0,
            left_confidence=1.0,
            right_confidence=1.0,
            lane_width_bottom=560.0,
            lane_center_bottom=480.0,
        )

    def test_lane_contains_contact(self):
        lanes = self._synthetic_lanes()
        self.assertTrue(lanes.contains_contact(480.0, 650.0))
        self.assertFalse(lanes.contains_contact(40.0, 650.0))
        self.assertFalse(lanes.contains_contact(480.0, 200.0))

    def test_range_from_lane_width(self):
        config = CameraConfig(focal_length_px=1150.0, camera_height_m=1.25, lane_width_m=3.7)
        geom = CameraGeometry(config, img_width=1280, img_height=720)
        lanes = self._synthetic_lanes()
        geom.calibrate_from_lanes(lanes)

        near = BoundingBox(x=420, y=560, w=120, h=110)  # contact ~670
        far = BoundingBox(x=440, y=500, w=80, h=90)     # contact ~590
        z_near, lat_near = geom.estimate_range_from_lanes(near, lanes)
        z_far, _ = geom.estimate_range_from_lanes(far, lanes)
        self.assertGreater(z_far, z_near)
        self.assertTrue(4.0 < z_near < 25.0)
        self.assertLess(abs(lat_near), 1.8)

    def test_range_kalman_smooths(self):
        kf = RangeKalman(20.0)
        for z in [19.5, 18.8, 19.2, 18.4, 17.9]:
            kf.predict(0.04)
            kf.update(z, r=4.0)
        self.assertTrue(17.0 < kf.z < 21.0)
        self.assertLess(kf.vz, 0.0)  # approaching (Z decreasing)


if __name__ == "__main__":
    unittest.main()
