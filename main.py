#!/usr/bin/env python3
"""
DriveCV Main CLI Application
High-speed Autonomous Driving Perception, Multi-Object Tracking & ADAS Suite.
"""

import argparse
import glob
import os
import sys
from typing import Optional

# Ensure local libs directory is included in sys.path
_LIBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
if os.path.exists(_LIBS_DIR) and _LIBS_DIR not in sys.path:
    sys.path.insert(0, _LIBS_DIR)

from drivecv.config import PipelineConfig
from drivecv.pipeline import ADASPipeline


def find_default_video() -> Optional[str]:
    """Finds first mp4 video in the current directory."""
    mp4_files = sorted(glob.glob("*.mp4"))
    return mp4_files[0] if mp4_files else None


def main():
    parser = argparse.ArgumentParser(
        description="DriveCV: High-Performance Autonomous Driving Perception, Tracking & ADAS Suite."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (default: auto-detects first .mp4).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/YOLOPv2.onnx",
        help="Path to YOLOPv2 ONNX model weights (default: weights/YOLOPv2.onnx).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Processing width (default: 960).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=540,
        help="Processing height (default: 540).",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.28,
        help="YOLOPv2 detection confidence threshold (default: 0.28).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Cadence for periodic background neural detection in frames (default: 15).",
    )
    parser.add_argument(
        "--show-seg",
        action="store_true",
        help="Overlay YOLOPv2 neural drivable area and lane segmentation masks.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output video file path to record tracked video (e.g. output.mp4).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process before exiting.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface for web application server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for web application server (default: 5000).",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=None,
        help="Optional WebSocket telemetry port (default: port + 1).",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable web server and run in traditional CLI mode.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="camera",
        choices=["camera", "video"],
        help="Default input source mode: 'camera' for live USB UVC video feed or 'video' for demo video (default: camera).",
    )
    parser.add_argument(
        "--camera-device",
        type=str,
        default="0",
        help="USB UVC camera device index or path (default: 0 or /dev/video0).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without local OpenCV GUI display windows for batch processing or benchmarks.",
    )

    args = parser.parse_args()

    if args.camera_device:
        os.environ["CAMERA_DEVICE"] = args.camera_device

    video_path = args.video or find_default_video()
    if not video_path or not os.path.exists(video_path):
        # Create dummy string if video not found so camera can still run
        video_path = "12838618_3840_2160_25fps.mp4"
        if not os.path.exists(video_path):
            print(f"[WARNING] Demo video file not found at '{video_path}'. Demo video fallback may fail.")

    # Build Configuration
    config = PipelineConfig(
        width=args.width,
        height=args.height,
    )
    config.detector.model_path = args.model
    config.detector.conf_thresh = args.conf_thresh
    config.detector.interval_frames = args.interval
    config.visualizer.show_seg_masks = args.show_seg

    if args.no_web:
        # Traditional CLI Execution
        pipeline = ADASPipeline(
            config=config,
            headless=args.headless,
            output_path=args.output,
        )
        src_path = args.camera_device if args.source == "camera" else video_path
        pipeline.run(video_path=src_path, max_frames=args.max_frames)
    else:
        # Serve Mobile 3D HUD Web Application
        from drivecv.web import ADASWebServer

        ws_port = args.ws_port or (args.port + 1)
        server = ADASWebServer(
            config=config,
            host=args.host,
            port=args.port,
            ws_port=ws_port,
            output_path=args.output,
            default_source=args.source,
        )
        server.run(video_path=video_path, max_frames=args.max_frames)



if __name__ == "__main__":
    main()
