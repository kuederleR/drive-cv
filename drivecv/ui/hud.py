"""
High-Performance Telemetric HUD and ADAS Banner Overlays.
"""

from typing import Optional
import cv2
import numpy as np
from drivecv.config import VisualizerConfig
from drivecv.types import ADASAlert, ADASAlertLevel, FrameData, LDWState


class HUDOverlay:
    """
    Renders telemetry HUD, status bars, and high-visibility ADAS safety alert banners.
    Uses slice-based alpha blending for sub-millisecond drawing performance.
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config or VisualizerConfig()

    def draw(
        self,
        frame: np.ndarray,
        frame_data: FrameData,
        total_frames: int,
        is_paused: bool = False,
        auto_schedule: bool = True,
        vis_mode: str = "ALL",
    ):
        """Draws top HUD, ADAS alert banner, and bottom controls."""
        h, w = frame.shape[:2]

        # 1. Top Telemetry Bar Slice
        bar_h = 48 if (self.config.show_stage_timings and frame_data.stage_ms is not None) else 40
        top_slice = frame[0:bar_h, :].copy()
        cv2.rectangle(top_slice, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.addWeighted(top_slice, 0.82, frame[0:bar_h, :], 0.18, 0, frame[0:bar_h, :])

        # Status badge
        status_str = "PAUSED" if is_paused else "PLAYING"
        status_color = (0, 165, 255) if is_paused else (0, 255, 0)
        cv2.putText(frame, f"[{status_str}]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2, cv2.LINE_AA)

        # Telemetry metrics
        num_tracks = len(frame_data.tracks)
        num_confirmed = sum(1 for t in frame_data.tracks if t.certainty >= 0.70)
        sched_str = "AUTO" if auto_schedule else "MANUAL"

        hud_text = (
            f"Frame: {frame_data.frame_idx}/{total_frames} | FPS: {frame_data.fps:.1f} | "
            f"Vehicles: {num_tracks} (Locked: {num_confirmed}) | "
            f"Vis: [{vis_mode}] | Sched: [{sched_str}]"
        )
        cv2.putText(frame, hud_text, (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        if self.config.show_stage_timings and frame_data.stage_ms is not None:
            cv2.putText(
                frame,
                frame_data.stage_ms.format_hud(),
                (100, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (180, 220, 255),
                1,
                cv2.LINE_AA,
            )

        # 2. ADAS Warning Center Banner
        if frame_data.adas is not None and frame_data.adas.warning_message:
            adas = frame_data.adas
            banner_bg = (0, 0, 200) if adas.fcw_level == ADASAlertLevel.CRITICAL else (0, 140, 255)
            text = f" ! {adas.warning_message} ! "

            font_scale = 0.65
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            bx1 = int((w - tw) / 2.0 - 15)
            by1 = 50
            bx2 = int((w + tw) / 2.0 + 15)
            by2 = 50 + th + 16

            bx1 = max(0, bx1)
            bx2 = min(w, bx2)
            banner_slice = frame[by1:by2, bx1:bx2].copy()
            cv2.rectangle(banner_slice, (0, 0), (bx2 - bx1, by2 - by1), banner_bg, -1)
            cv2.addWeighted(banner_slice, 0.85, frame[by1:by2, bx1:bx2], 0.15, 0, frame[by1:by2, bx1:bx2])
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(
                frame,
                text,
                (bx1 + 15, by1 + th + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # 3. Bottom Controls Bar Slice
        bot_slice = frame[h - 30 : h, :].copy()
        cv2.rectangle(bot_slice, (0, 0), (w, 30), (20, 20, 20), -1)
        cv2.addWeighted(bot_slice, 0.82, frame[h - 30 : h, :], 0.18, 0, frame[h - 30 : h, :])

        controls_text = (
            "[SPACE]: Play/Pause | [t]: Trigger YOLO | [a]: Auto-Sched | "
            "[s]: Select ROI | [v]: Vis Mode | [c]: Clear | [d]: Step | [q]: Quit"
        )
        cv2.putText(frame, controls_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
