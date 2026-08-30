"""
Unit tests for DriveCV Telemetry Serialization and Web Application Server.
"""

import unittest
import numpy as np
from drivecv.config import PipelineConfig
from drivecv.pipeline import ADASPipeline
from drivecv.types import (
    ADASAlert,
    ADASAlertLevel,
    BoundingBox,
    FrameData,
    Kinematics,
    LaneBoundaries,
    LDWState,
    StageTimings,
    Track,
)
from drivecv.web.server import ADASWebServer


class TestTelemetrySerialization(unittest.TestCase):
    def test_telemetry_dict_export(self):
        lanes = LaneBoundaries(
            left_line=np.array([200.0, 400.0]),
            right_line=np.array([760.0, 560.0]),
            y_top=250,
            y_bot=540,
            y_roi_top=250,
            left_confidence=0.9,
            right_confidence=0.85,
            lane_center_bottom=480.0,
            lane_width_bottom=560.0,
            vanish_x=480.0,
            vanish_y=250.0,
        )

        track = Track(
            track_id=1,
            bbox=BoundingBox(x=450.0, y=280.0, w=80.0, h=70.0),
            class_name="car",
            kinematics=Kinematics(
                distance_m=24.5,
                lateral_offset_m=0.15,
                rel_speed_kmh=-3.2,
                ttc_seconds=4.5,
                is_lead_vehicle=True,
            ),
        )

        adas = ADASAlert(
            ldw_state=LDWState.NORMAL,
            ldw_offset_m=-0.08,
            ldw_tlc_s=8.0,
            fcw_level=ADASAlertLevel.SAFE,
            fcw_lead_track_id=1,
            fcw_lead_distance_m=24.5,
            fcw_lead_rel_speed_kmh=-3.2,
            fcw_lead_ttc_s=4.5,
            warning_message=None,
        )

        frame_data = FrameData(
            frame_idx=42,
            timestamp=1.68,
            proc_frame=np.zeros((540, 960, 3), dtype=np.uint8),
            gray_frame=np.zeros((540, 960), dtype=np.uint8),
            tracks=[track],
            lanes=lanes,
            adas=adas,
            fps=25.0,
            stage_ms=StageTimings(total_ms=16.5),
        )

        telemetry = ADASPipeline.get_telemetry_dict(frame_data)

        self.assertEqual(telemetry["frame_idx"], 42)
        self.assertAlmostEqual(telemetry["timestamp"], 1.68)
        self.assertIsNotNone(telemetry["lanes"])
        self.assertTrue(telemetry["lanes"]["is_valid"])
        self.assertEqual(len(telemetry["tracks"]), 1)
        self.assertEqual(telemetry["tracks"][0]["track_id"], 1)
        self.assertTrue(telemetry["tracks"][0]["is_lead"])
        self.assertEqual(telemetry["adas"]["ldw_state"], "NORMAL")
        self.assertEqual(telemetry["adas"]["fcw_level"], "SAFE")


class TestWebServerRoutes(unittest.TestCase):
    def setUp(self):
        config = PipelineConfig()
        self.server = ADASWebServer(config=config, host="127.0.0.1", port=5099, ws_port=5100)
        self.client = self.server.app.test_client()

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"DriveCV 3D Ego-Lane & ADAS HUD", response.data)

    def test_api_control_route(self):
        response = self.client.post("/api/control", json={"action": "pause"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.server._is_paused)

        response = self.client.post("/api/control", json={"action": "play"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.server._is_paused)

    def test_api_calibrate_route(self):
        response = self.client.post("/api/calibrate", json={"action": "start", "side": "left"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["calibration"]["is_calibrating"])
        self.assertEqual(data["calibration"]["calibration_side"], "left")

    def test_api_source_route(self):
        response_get = self.client.get("/api/source")
        self.assertEqual(response_get.status_code, 200)
        data_get = response_get.get_json()
        self.assertEqual(data_get["status"], "ok")
        self.assertEqual(data_get["active_source"], "camera")

        response_post = self.client.post("/api/source", json={"source": "video"})
        self.assertEqual(response_post.status_code, 200)
        data_post = response_post.get_json()
        self.assertEqual(data_post["status"], "ok")
        self.assertEqual(data_post["requested_source"], "video")
        self.assertEqual(self.server._pending_source_change, "video")


if __name__ == "__main__":
    unittest.main()
