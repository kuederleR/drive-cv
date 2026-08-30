"""
Unit tests for Hood Masking in DriveCV perception and web server API.
"""

import unittest
import numpy as np
from drivecv.config import LaneConfig, PipelineConfig
from drivecv.perception.lane_detector import ClassicalLaneDetector
from drivecv.perception.motion_detector import FastMotionAttentionGrid
from drivecv.web.server import ADASWebServer


class TestHoodMask(unittest.TestCase):
    """Tests hood mask configuration, ROI clipping, and web API handling."""

    def test_lane_config_defaults(self):
        config = LaneConfig()
        self.assertFalse(config.hood_mask_enabled)
        self.assertAlmostEqual(config.hood_height_ratio, 0.20)

    def test_classical_lane_detector_hood_mask(self):
        config = LaneConfig(y_bot_ratio=0.95, hood_mask_enabled=True, hood_height_ratio=0.20)
        detector = ClassicalLaneDetector(config=config)
        
        # 960x540 blank frame
        gray = np.zeros((540, 960), dtype=np.uint8)
        lanes = detector.update(curr_gray=gray)
        
        # Effective y_bot should be at 540 * (1.0 - 0.20) = 432
        self.assertEqual(lanes.y_bot, 432)

    def test_classical_lane_detector_disabled_hood_mask(self):
        config = LaneConfig(y_bot_ratio=0.95, hood_mask_enabled=False, hood_height_ratio=0.20)
        detector = ClassicalLaneDetector(config=config)
        
        gray = np.zeros((540, 960), dtype=np.uint8)
        lanes = detector.update(curr_gray=gray)
        
        # Effective y_bot should be 540 * 0.95 = 513
        self.assertEqual(lanes.y_bot, 513)

    def test_fast_motion_attention_grid_hood_cutoff(self):
        grid = FastMotionAttentionGrid(hood_height_ratio=0.25)
        self.assertAlmostEqual(grid.hood_height_ratio, 0.25)

    def test_web_server_hood_mask_api(self):
        config = PipelineConfig()
        server = ADASWebServer(config=config)
        client = server.app.test_client()

        # GET /api/hood_mask
        res_get = client.get("/api/hood_mask")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.get_json()
        self.assertFalse(data_get["enabled"])
        self.assertAlmostEqual(data_get["height_ratio"], 0.20)

        # POST /api/hood_mask (update parameters)
        res_post = client.post("/api/hood_mask", json={"enabled": True, "height_ratio": 0.25})
        self.assertEqual(res_post.status_code, 200)
        data_post = res_post.get_json()
        self.assertTrue(data_post["enabled"])
        self.assertAlmostEqual(data_post["height_ratio"], 0.25)

        # Verify PipelineConfig updated
        self.assertTrue(server.config.lane.hood_mask_enabled)
        self.assertAlmostEqual(server.config.lane.hood_height_ratio, 0.25)


if __name__ == "__main__":
    unittest.main()
