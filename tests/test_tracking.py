"""
Unit tests for DriveCV Kalman Filter and Track Lifecycle using unittest.
"""

import unittest
import numpy as np
from drivecv.config import TrackerConfig
from drivecv.tracking.association import associate_detections_to_tracks
from drivecv.tracking.kalman import KalmanBoxTracker
from drivecv.tracking.lead_tracker import LeadVehicleTracker
from drivecv.tracking.multi_tracker import MultiObjectTracker
from drivecv.tracking.track import TrackObject
from drivecv.types import BoundingBox, Detection, LaneBoundaries, TrackLifecycle


class TestTracking(unittest.TestCase):
    def test_kalman_prediction_and_update(self):
        init_box = BoundingBox(x=100, y=200, w=80, h=60)
        tracker = KalmanBoxTracker(init_box)

        box = tracker.get_state_bbox()
        self.assertAlmostEqual(box.x, 100.0, delta=1.0)
        self.assertAlmostEqual(box.y, 200.0, delta=1.0)

        for _ in range(5):
            tracker.predict()
            tracker.update_detection(BoundingBox(x=box.x + 5, y=box.y + 2, w=80, h=60))
            box = tracker.get_state_bbox()

        vx, vy = tracker.get_velocity()
        self.assertTrue(vx > 0.0)

    def test_track_object_owns_kalman(self):
        det = Detection(bbox=BoundingBox(100, 100, 80, 60), confidence=0.9, class_id=2, class_name="car")
        track = TrackObject(track_id=1, initial_detection=det)
        self.assertIsInstance(track.kalman, KalmanBoxTracker)
        self.assertAlmostEqual(track.bbox.w, 80.0, delta=1.0)

    def test_track_association(self):
        track_boxes = [
            BoundingBox(100, 100, 50, 50),
            BoundingBox(300, 300, 50, 50),
        ]
        detections = [
            Detection(bbox=BoundingBox(102, 101, 50, 50), confidence=0.9, class_id=2),
            Detection(bbox=BoundingBox(500, 500, 50, 50), confidence=0.8, class_id=2),
        ]

        matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
            track_boxes=track_boxes,
            detections=detections,
            iou_threshold=0.25,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], (0, 0))
        self.assertIn(1, unmatched_tracks)
        self.assertIn(1, unmatched_dets)

    def test_low_score_unmatched_does_not_birth_when_tracks_exist(self):
        track_boxes = [BoundingBox(100, 100, 50, 50)]
        detections = [
            Detection(bbox=BoundingBox(400, 400, 40, 40), confidence=0.35, class_id=2),
        ]
        matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
            track_boxes=track_boxes,
            detections=detections,
            high_score_thresh=0.50,
            low_score_thresh=0.30,
        )
        self.assertEqual(matches, [])
        self.assertEqual(unmatched_tracks, [0])
        self.assertEqual(unmatched_dets, [])

    def test_multi_tracker_lifecycle(self):
        config = TrackerConfig(min_hits=2, max_age=5)
        tracker = MultiObjectTracker(config)

        prev_gray = np.zeros((480, 640), dtype=np.uint8)
        curr_gray = np.zeros((480, 640), dtype=np.uint8)
        curr_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        det1 = Detection(bbox=BoundingBox(100, 100, 60, 50), confidence=0.85, class_id=2)
        tracks = tracker.update(prev_gray, curr_gray, curr_bgr, [det1])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].lifecycle, TrackLifecycle.TENTATIVE)
        self.assertIsInstance(tracker.tracks[0].kalman, KalmanBoxTracker)

        det2 = Detection(bbox=BoundingBox(102, 100, 60, 50), confidence=0.90, class_id=2)
        tracks = tracker.update(prev_gray, curr_gray, curr_bgr, [det2])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].lifecycle, TrackLifecycle.CONFIRMED)

    def test_optical_flow_does_not_shrink_box_to_points(self):
        config = TrackerConfig(min_hits=1)
        tracker = MultiObjectTracker(config)
        gray = np.zeros((480, 640), dtype=np.uint8)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        det = Detection(
            bbox=BoundingBox(200, 200, 120, 80),
            confidence=0.92,
            class_id=2,
            class_name="car",
        )
        tracker.update(gray, gray, bgr, [det])
        self.assertEqual(len(tracker.tracks), 1)
        track = tracker.tracks[0]
        orig_w, orig_h = track.bbox.w, track.bbox.h
        track.keypoints = np.array(
            [[205.0, 205.0], [208.0, 206.0], [210.0, 208.0], [207.0, 210.0], [212.0, 205.0]],
            dtype=np.float32,
        )
        tracker.update(gray, gray, bgr, None)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertGreater(tracker.tracks[0].bbox.w, orig_w * 0.70)
        self.assertGreater(tracker.tracks[0].bbox.h, orig_h * 0.70)

    def test_motion_does_not_birth_tracks(self):
        tracker = MultiObjectTracker()
        prev = np.zeros((180, 320), dtype=np.uint8)
        curr = np.zeros((180, 320), dtype=np.uint8)
        curr[100:140, 80:140] = 255
        bgr = np.zeros((180, 320, 3), dtype=np.uint8)
        for _ in range(4):
            tracks = tracker.update(prev, curr, bgr, detections=None)
            prev = curr
        self.assertEqual(len(tracks), 0)

    def test_lead_tracker_ignores_out_of_lane(self):
        lanes = LaneBoundaries(
            left_line=np.array([200.0, 450.0], dtype=np.float32),
            right_line=np.array([760.0, 510.0], dtype=np.float32),
            y_bot=680,
            y_roi_top=500,
            y_top=500,
            vanish_y=320.0,
            left_confidence=1.0,
            right_confidence=1.0,
            lane_width_bottom=560.0,
            lane_center_bottom=480.0,
        )
        tracker = LeadVehicleTracker()
        gray = np.zeros((720, 1280), dtype=np.uint8)
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        outside = Detection(
            bbox=BoundingBox(x=20, y=500, w=80, h=70),
            confidence=0.95,
            class_id=2,
        )
        tracks = tracker.update(gray, gray, bgr, [outside], lanes=lanes, dt=0.04)
        self.assertEqual(len(tracks), 0)

        inside = Detection(
            bbox=BoundingBox(x=420, y=540, w=120, h=120),
            confidence=0.92,
            class_id=2,
        )
        tracks = tracker.update(gray, gray, bgr, [inside], lanes=lanes, dt=0.04)
        self.assertEqual(len(tracks), 1)
        self.assertTrue(tracks[0].kinematics.is_lead_vehicle)
        self.assertGreater(tracks[0].kinematics.distance_m, 3.0)

        # A second in-lane detection farther away must not steal the lock via MOT spawn
        farther = Detection(
            bbox=BoundingBox(x=450, y=500, w=70, h=70),
            confidence=0.90,
            class_id=2,
        )
        tracks = tracker.update(gray, gray, bgr, [inside, farther], lanes=lanes, dt=0.04)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)


if __name__ == "__main__":
    unittest.main()
