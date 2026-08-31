"""
Interactive GUI player loop and keyboard control handler.
"""

import sys
import time
from typing import Callable, Optional, Tuple
import cv2
import numpy as np
from drivecv.types import BoundingBox, FrameData


class InteractivePlayer:
    """
    Manages OpenCV display window, keyboard event loop, and user ROI selection.
    """

    def __init__(
        self,
        window_name: str = "DriveCV: Real-Time Perception & ADAS Suite",
        headless: bool = False,
        target_fps: float = 25.0,
    ):
        self.window_name = window_name
        self.headless = headless
        self.target_fps = target_fps
        self.is_paused: bool = False
        self.step_single_frame: bool = False
        self.auto_schedule: bool = True
        self.vis_mode: str = "DET_ONLY"
        self.vis_modes = ["DET_ONLY", "ALL", "MINIMAL"]

        if not self.headless:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def select_roi(self, display_frame: np.ndarray) -> Optional[BoundingBox]:
        """Prompts user to select a region of interest with the mouse."""
        if self.headless:
            return None

        print("\n[INFO] Drag mouse to select target vehicle. Press ENTER/SPACE to confirm, or 'c' to cancel.")
        roi = cv2.selectROI(self.window_name, display_frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi
        if w > 10 and h > 10:
            print(f"[INFO] Selected ROI: ({x}, {y}, {w}, {h})")
            return BoundingBox(x=float(x), y=float(y), w=float(w), h=float(h))
        else:
            print("[INFO] Selection cancelled.")
            return None

    def handle_keys(
        self,
        frame_start_time: float,
        on_trigger_yolo: Optional[Callable[[], None]] = None,
        on_clear_tracks: Optional[Callable[[], None]] = None,
        on_select_roi: Optional[Callable[[], None]] = None,
        on_cycle_lane_debug: Optional[Callable[[], None]] = None,
    ) -> bool:
        """
        Handles keyboard input and regulates playback timing.
        Returns False if user requested exit, True otherwise.
        """
        if self.headless:
            return True

        target_frame_time = 1.0 / max(1.0, self.target_fps)
        elapsed = time.time() - frame_start_time
        wait_ms = 0 if self.is_paused else max(1, int((target_frame_time - elapsed) * 1000.0))

        key = cv2.waitKey(wait_ms) & 0xFF

        if key in [ord("q"), 27]:  # 'q' or ESC
            print("[INFO] User requested exit.")
            return False
        elif key == ord(" "):
            self.is_paused = not self.is_paused
            print(f"[INFO] {'PAUSED' if self.is_paused else 'RESUMED'}")
        elif key == ord("t"):
            if on_trigger_yolo:
                on_trigger_yolo()
        elif key == ord("a"):
            self.auto_schedule = not self.auto_schedule
            print(f"[INFO] Auto-Scheduler: {'ENABLED' if self.auto_schedule else 'DISABLED'}")
        elif key in [ord("s"), ord("r")]:
            if on_select_roi:
                on_select_roi()
        elif key == ord("v"):
            idx = (self.vis_modes.index(self.vis_mode) + 1) % len(self.vis_modes)
            self.vis_mode = self.vis_modes[idx]
            print(f"[INFO] Visualization Mode: {self.vis_mode}")
        elif key == ord("c"):
            if on_clear_tracks:
                on_clear_tracks()
        elif key == ord("d"):
            if self.is_paused:
                self.step_single_frame = True
        elif key == ord("l"):
            if on_cycle_lane_debug:
                on_cycle_lane_debug()

        return True

    def show(self, display_frame: np.ndarray):
        """Displays frame in OpenCV window if not headless."""
        if not self.headless:
            cv2.imshow(self.window_name, display_frame)

    def close(self):
        """Destroys OpenCV windows."""
        if not self.headless:
            cv2.destroyAllWindows()
