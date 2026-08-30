"""
Lane Line Type Classification Module for DriveCV ADAS.
Analyzes spatial pixel intensities, HSV color spectrum, and transverse profiles
along detected lane boundaries to classify line type:
- Colors: 'white', 'yellow'
- Patterns: 'solid', 'dashed', 'double'
- Composite types: 'solid_white', 'dashed_white', 'solid_yellow', 'dashed_yellow', 'double_yellow', 'double_white', 'solid_dashed_yellow'
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class LaneTypeDetector:
    """
    Analyzes spatial pixel intensities and color spectrum along lane boundaries
    to classify lane line color, pattern, and structure with exponential hysteresis smoothing.
    """

    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.left_conf: Dict[str, float] = {}
        self.right_conf: Dict[str, float] = {}
        self.active_left: Optional[str] = None
        self.active_right: Optional[str] = None

    def analyze_line(
        self,
        frame_bgr: np.ndarray,
        line: Optional[np.ndarray],  # [x_bot, x_top]
        y_bot: int,
        y_top: int,
        default_color: str = "white",
        default_pattern: str = "solid",
        side: str = "left",
    ) -> Dict[str, str]:
        """
        Analyzes a single lane line boundary using self-correcting adaptive peak tracking.
        Returns dict with keys: 'type', 'color', 'pattern'.
        """
        if line is None or y_bot <= y_top:
            fallback_type = f"{default_pattern}_{default_color}" if default_pattern != "double" else f"double_{default_color}"
            return {
                "type": fallback_type,
                "color": default_color,
                "pattern": default_pattern,
            }

        h, w = frame_bgr.shape[:2]
        x_bot, x_top = float(line[0]), float(line[1])

        num_samples = 30
        y_samples = np.linspace(y_bot, y_top, num_samples, dtype=np.int32)
        dx_est = (x_top - x_bot) / float(max(1, num_samples - 1))

        x_curr = x_bot
        search_r = max(20, int(w * 0.035))

        paint_presence: List[int] = []
        hsv_paint_rows: List[np.ndarray] = []
        double_peaks = 0

        for y in y_samples:
            if y < 0 or y >= h:
                paint_presence.append(0)
                x_curr += dx_est
                continue

            x1 = max(0, int(x_curr) - search_r)
            x2 = min(w, int(x_curr) + search_r + 1)
            if x2 - x1 < 6:
                paint_presence.append(0)
                x_curr += dx_est
                continue

            row_bgr = frame_bgr[y : y + 1, x1:x2][0]
            row_gray = cv2.cvtColor(row_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY)[0]
            row_hsv = cv2.cvtColor(row_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0]

            v_vals = row_hsv[:, 2]
            max_idx = int(np.argmax(v_vals))
            max_v = int(v_vals[max_idx])
            min_v = int(np.min(v_vals))
            contrast = max_v - min_v

            is_paint = (max_v >= 155 and contrast >= 30)
            paint_presence.append(1 if is_paint else 0)

            if is_paint:
                x_curr = x1 + max_idx
                bright_mask = v_vals >= (max_v - 20)
                hsv_paint_rows.append(row_hsv[bright_mask])
            else:
                x_curr += dx_est

            # Double peak detection
            if len(row_gray) >= 12:
                smoothed = np.convolve(row_gray, np.array([0.2, 0.6, 0.2]), mode="same")
                raw_peaks = []
                for k in range(1, len(smoothed) - 1):
                    if (
                        smoothed[k] > min_v + 0.45 * contrast
                        and smoothed[k] >= smoothed[k - 1]
                        and smoothed[k] >= smoothed[k + 1]
                    ):
                        raw_peaks.append((k, float(smoothed[k])))
                
                # Cluster peaks within 3 pixels
                clustered_peaks: List[List[Tuple[int, float]]] = []
                for pk in raw_peaks:
                    if not clustered_peaks:
                        clustered_peaks.append([pk])
                    else:
                        if pk[0] - clustered_peaks[-1][-1][0] <= 3:
                            clustered_peaks[-1].append(pk)
                        else:
                            clustered_peaks.append([pk])

                peaks = []
                for cluster in clustered_peaks:
                    avg_k = int(round(np.mean([p[0] for p in cluster])))
                    max_v_peak = max(p[1] for p in cluster)
                    peaks.append((avg_k, max_v_peak))

                if len(peaks) >= 2:
                    p1, p2 = peaks[0][0], peaks[1][0]
                    if 5 <= abs(p2 - p1) <= 18:
                        dip = float(np.min(smoothed[min(p1, p2) : max(p1, p2) + 1]))
                        if (min(peaks[0][1], peaks[1][1]) - dip) > 0.25 * contrast:
                            double_peaks += 1

        # Color decision
        if len(hsv_paint_rows) >= 5:
            arr_hsv = np.vstack(hsv_paint_rows)
            med_h = float(np.median(arr_hsv[:, 0]))
            med_s = float(np.median(arr_hsv[:, 1]))
            if 10 <= med_h <= 36 and med_s >= 55:
                color = "yellow"
            else:
                color = "white"
        else:
            color = default_color

        # Median filter smoothing on paint presence (window = 3)
        pres_arr = np.array(paint_presence, dtype=np.uint8)
        smoothed_pres = cv2.medianBlur(pres_arr.reshape(-1, 1), 3).flatten()

        gap_runs = []
        dash_runs = []
        curr_run = 1
        for i in range(1, len(smoothed_pres)):
            if smoothed_pres[i] == smoothed_pres[i - 1]:
                curr_run += 1
            else:
                if smoothed_pres[i - 1] == 0:
                    gap_runs.append(curr_run)
                else:
                    dash_runs.append(curr_run)
                curr_run = 1
        if smoothed_pres[-1] == 0:
            gap_runs.append(curr_run)
        else:
            dash_runs.append(curr_run)

        transitions = len(gap_runs) + len(dash_runs) - 1
        max_gap = max(gap_runs) if gap_runs else 0
        fill_ratio = float(np.mean(smoothed_pres))
        double_ratio = double_peaks / float(max(1, num_samples))

        if double_ratio >= 0.35:
            pattern = "double"
        elif fill_ratio >= 0.94 or max_gap <= 1:
            pattern = "solid"
        elif max_gap >= 2 and transitions >= 2:
            pattern = "dashed"
        elif fill_ratio <= 0.65:
            pattern = "dashed"
        else:
            pattern = "solid"

        raw_type = f"double_{color}" if pattern == "double" else f"{pattern}_{color}"

        # Exponential Hysteresis Confidence Filter
        conf_dict = self.left_conf if side == "left" else self.right_conf
        active_type = self.active_left if side == "left" else self.active_right

        if active_type is None:
            if side == "left":
                self.active_left = raw_type
                self.left_conf = {raw_type: 2.0}
            else:
                self.active_right = raw_type
                self.right_conf = {raw_type: 2.0}
            final_type = raw_type
        else:
            for k in list(conf_dict.keys()):
                conf_dict[k] *= 0.82
            conf_dict[raw_type] = conf_dict.get(raw_type, 0.0) + 0.35

            top_cand = max(conf_dict, key=conf_dict.get)  # type: ignore
            if conf_dict[top_cand] > conf_dict.get(active_type, 0.0) + 0.40:
                if side == "left":
                    self.active_left = top_cand
                else:
                    self.active_right = top_cand

            final_type = self.active_left if side == "left" else self.active_right  # type: ignore

        parts = final_type.split("_")
        final_pattern = parts[0]
        final_color = parts[1] if len(parts) > 1 else color

        return {
            "type": final_type,
            "color": final_color,
            "pattern": final_pattern,
        }
