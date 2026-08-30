"""
Unit tests for DriveCV ADAS Modules: LDW & FCW using unittest.
"""

import unittest
import numpy as np
from drivecv.adas.fcw import ForwardCollisionWarning
from drivecv.adas.ldw import LaneDepartureWarning
from drivecv.config import ADASConfig
from drivecv.types import ADASAlertLevel, BoundingBox, LaneBoundaries, LDWState, Track, TrackLifecycle


class TestADAS(unittest.TestCase):
    def test_ldw_calculations(self):
        ldw = LaneDepartureWarning()

        # Normal centered lane: ego camera center at 640px, lane center at 640px
        lanes_centered = LaneBoundaries(
            lane_center_bottom=640.0,
            lane_width_bottom=400.0,
            left_confidence=1.0,
            right_confidence=1.0,
        )
        state, offset_m, tlc = ldw.update(lanes_centered, frame_width=1280)
        self.assertEqual(state, LDWState.NORMAL)
        self.assertAlmostEqual(offset_m, 0.0, delta=0.05)

        # Lane shifted right (vehicle drifting left)
        lanes_shifted_right = LaneBoundaries(
            lane_center_bottom=750.0,
            lane_width_bottom=400.0,
            left_confidence=1.0,
            right_confidence=1.0,
        )
        state_l, offset_l, _ = ldw.update(lanes_shifted_right, frame_width=1280)
        self.assertEqual(state_l, LDWState.WARNING_LEFT)
        self.assertTrue(offset_l < -0.45)

    def test_fcw_collision_warning(self):
        fcw = ForwardCollisionWarning()

        # Create approaching lead vehicle track in ego corridor
        track = Track(
            track_id=1,
            bbox=BoundingBox(x=590, y=450, w=100, h=80),
            lifecycle=TrackLifecycle.CONFIRMED,
            certainty=1.0,
        )

        # Frame 1: Initial observation
        alert1, lead, dist1, speed1, ttc1 = fcw.update([track], None, timestamp=0.0, dt=0.04)

        # Frame 2: Rapidly approaching vehicle (moved lower down on screen)
        track.bbox = BoundingBox(x=590, y=550, w=130, h=100)
        alert2, lead2, dist2, speed2, ttc2 = fcw.update([track], None, timestamp=0.04, dt=0.04)

        self.assertIsNotNone(lead2)
        self.assertTrue(lead2.kinematics.is_lead_vehicle)
        self.assertTrue(dist2 < dist1)
        self.assertTrue(speed2 > 0)  # Closing speed > 0
        self.assertIn(alert2, [ADASAlertLevel.WARNING, ADASAlertLevel.CRITICAL])


if __name__ == "__main__":
    unittest.main()
