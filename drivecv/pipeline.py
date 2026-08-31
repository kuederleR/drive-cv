"""
End-to-end Autonomous Driving Perception, Tracking, and ADAS Pipeline.
"""

import os
import shutil
import subprocess
import time
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

# Suppress verbose OpenCV C++ library log messages (e.g. Corrupt JPEG data)
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass
from drivecv.adas.adas_manager import ADASManager
from drivecv.config import PipelineConfig
from drivecv.perception.async_detector import AsyncPerceptionWorker
from drivecv.perception.lane_detector import ClassicalLaneDetector
from drivecv.perception.yolopv2 import YOLOPv2Perception
from drivecv.tracking.multi_tracker import MultiObjectTracker
from drivecv.tracking.lead_tracker import LeadVehicleTracker
from drivecv.types import BoundingBox, Detection, FrameData, StageTimings, Track
from drivecv.ui.hud import HUDOverlay
from drivecv.ui.player import InteractivePlayer
from drivecv.ui.visualizer import PanopticVisualizer


def is_camera_source(source: Union[int, str]) -> bool:
    """Checks if input source represents a live camera device."""
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        if source.isdigit():
            return True
        if source.startswith("/dev/video"):
            return True
        if source.lower() in ("camera", "webcam", "live"):
            return True
    return False


class ScaledVideoCapture:
    """
    Decodes video file scaled to process resolution via FFmpeg or OpenCV,
    or captures live frames from a USB UVC camera device (/dev/video0).
    """

    def __init__(self, source: Union[int, str], width: int, height: int, use_ffmpeg: bool = True):
        self.width = width
        self.height = height
        self.source = source
        self._proc: Optional[subprocess.Popen] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._use_ffmpeg = False
        self.frame_nbytes = width * height * 3
        self.is_live = is_camera_source(source)

        if self.is_live:
            dev: Union[int, str] = 0
            if isinstance(source, int):
                dev = source
            elif isinstance(source, str):
                if source.isdigit():
                    dev = int(source)
                elif source.lower() in ("camera", "webcam", "live"):
                    env_dev = os.environ.get("CAMERA_DEVICE", "0")
                    dev = int(env_dev) if env_dev.isdigit() else env_dev
                else:
                    dev = source

            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if isinstance(dev, str) and dev.startswith("/dev/video") else cv2.VideoCapture(dev)
            if not cap.isOpened():
                raise RuntimeError(f"[ERROR] Unable to open camera device: '{source}' (device={dev})")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.orig_w = w if w > 0 else width
            self.orig_h = h if h > 0 else height

            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps) if (fps and 0 < fps <= 120) else 30.0
            self.frame_count = -1
            self._cap = cap
        else:
            path = str(source)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"[ERROR] Unable to open video file: '{path}'")
            self.orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            self.orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps) if (fps and 0 < fps <= 120) else 25.0
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._cap = cap

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            return False, None

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return True, frame

    def release(self):
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=0.5)
            except Exception:
                pass
            self._proc = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
            if self.is_live:
                time.sleep(0.15)  # Allow V4L2 camera kernel driver to release handle


class ADASPipeline:
    """
    Main High-Performance Driving Vision Pipeline:
    - Fuses Classical Lane Detection with YOLOPv2 Multi-Task Perception.
    - Kalman box tracker + batched sparse optical flow.
    - Real-Time ADAS Engine: Lane Departure Warning (LDW) & Forward Collision Warning (FCW).
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        headless: bool = False,
        output_path: Optional[str] = None,
    ):
        self.config = config or PipelineConfig()
        self.headless = headless
        self.output_path = output_path

        cv2.setNumThreads(max(1, int(self.config.opencv_num_threads)))

        self.async_worker: Optional[AsyncPerceptionWorker] = None
        if self.config.detector.enabled and os.path.exists(self.config.detector.model_path):
            detector = YOLOPv2Perception(self.config.detector)
            self.async_worker = AsyncPerceptionWorker(self.config.detector, detector=detector)
            print(f"[INFO] Initialized YOLOPv2 Perception Worker from '{self.config.detector.model_path}'.")
        else:
            print("[WARNING] Running in Classical CV mode (YOLOPv2 model weights not found or disabled).")

        self.lane_detector = ClassicalLaneDetector(self.config.lane)
        self.adas = ADASManager(self.config.adas)
        self.adas.update_resolution(self.config.width, self.config.height)
        if self.config.tracker.lead_only:
            self.tracker = LeadVehicleTracker(
                self.config.tracker,
                camera_geom=self.adas.camera_geom,
                in_lane_margin_frac=self.config.adas.fcw_in_lane_margin_frac,
            )
            print("[INFO] Lead-vehicle tracker: ego-lane FCW lock (single target).")
        else:
            self.tracker = MultiObjectTracker(self.config.tracker)

        self.visualizer = PanopticVisualizer(self.config.visualizer)
        self.hud = HUDOverlay(self.config.visualizer)
        self.player = InteractivePlayer(
            headless=headless,
            target_fps=self.config.fps,
        )

        self.video_writer: Optional[cv2.VideoWriter] = None
        self.frames_since_yolo: int = 0
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_time: Optional[float] = None
        self.recent_fps_history: List[float] = []
        self.last_decode_ms: float = 0.0
        self._timing_log_interval: int = 30
        self._last_tracks: List[Track] = []

    def _compute_padded_crop(self, bbox: BoundingBox, frame_w: int, frame_h: int) -> BoundingBox:
        pad_ratio = self.config.detector.crop_padding_ratio
        pad_w = max(24.0, bbox.w * pad_ratio)
        pad_h = max(20.0, bbox.h * pad_ratio)
        x1 = max(0.0, bbox.x - pad_w)
        y1 = max(0.0, bbox.y - pad_h)
        x2 = min(float(frame_w), bbox.x + bbox.w + pad_w)
        y2 = min(float(frame_h), bbox.y + bbox.h + pad_h)
        return BoundingBox(x=x1, y=y1, w=max(16.0, x2 - x1), h=max(16.0, y2 - y1))

    def trigger_manual_yolo(self, frame: np.ndarray, target_bbox: Optional[BoundingBox] = None):
        """Forces an asynchronous YOLOPv2 pass on target ROI crop or full frame."""
        if self.async_worker is not None:
            self.async_worker.submit_frame(frame, target_bbox)
            self.frames_since_yolo = 0
            print("[INFO] Dispatched on-demand YOLOPv2 perception pass!")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> Tuple[np.ndarray, FrameData]:
        """
        Executes a single pipeline cycle on an input frame:
        Perception -> Tracking -> ADAS.
        """
        timings = StageTimings(decode_ms=self.last_decode_ms)
        t0 = time.perf_counter()

        if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
            proc_frame = cv2.resize(
                frame, (self.config.width, self.config.height), interpolation=cv2.INTER_AREA
            )
        else:
            proc_frame = frame
        curr_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        timings.resize_ms = (time.perf_counter() - t0) * 1000.0

        dt = (
            (timestamp - self.prev_time)
            if (self.prev_time is not None and timestamp > self.prev_time)
            else (1.0 / self.config.fps)
        )
        self.prev_time = timestamp
        self.frames_since_yolo += 1

        if self.prev_gray is None:
            self.prev_gray = curr_gray

        fresh_detections: Optional[List[Detection]] = None
        da_mask: Optional[np.ndarray] = None
        ll_mask: Optional[np.ndarray] = None

        if self.async_worker is not None:
            worker_res = self.async_worker.fetch_results()
            if worker_res is not None:
                fresh_detections, da_mask, ll_mask = worker_res

        t2 = time.perf_counter()
        lanes = self.lane_detector.update(
            curr_gray=curr_gray,
            tracked_objects=self._last_tracks,
            da_mask=da_mask,
            ll_mask=ll_mask,
            curr_bgr=proc_frame,
        )
        timings.lanes_ms = (time.perf_counter() - t2) * 1000.0

        t1 = time.perf_counter()
        tracks = self.tracker.update(
            prev_gray=self.prev_gray,
            curr_gray=curr_gray,
            curr_bgr_frame=proc_frame,
            detections=fresh_detections,
            lanes=lanes,
            dt=dt,
            high_score_thresh=self.config.detector.high_score_thresh,
            low_score_thresh=self.config.detector.low_score_thresh,
        )
        self._last_tracks = tracks
        timings.track_ms = (time.perf_counter() - t1) * 1000.0

        if (
            self.player.auto_schedule
            and self.async_worker is not None
            and not self.async_worker.is_busy
            and self.frames_since_yolo >= self.config.detector.interval_frames
        ):
            self.async_worker.submit_frame(proc_frame, roi_crop=None)
            self.frames_since_yolo = 0

        t3 = time.perf_counter()
        adas_alert = self.adas.process(
            tracks=tracks,
            lanes=lanes,
            frame_width=self.config.width,
            frame_height=self.config.height,
            timestamp=timestamp,
            dt=dt,
        )
        self.adas.camera_geom.project_lane_boundaries(lanes)
        timings.adas_ms = (time.perf_counter() - t3) * 1000.0

        frame_data = FrameData(
            frame_idx=frame_idx,
            timestamp=timestamp,
            proc_frame=proc_frame,
            gray_frame=curr_gray,
            orig_frame=None,
            tracks=tracks,
            detections=fresh_detections or [],
            lanes=lanes,
            adas=adas_alert,
            fps=0.0,
            stage_ms=timings,
        )

        self.prev_gray = curr_gray
        return proc_frame, frame_data

    @staticmethod
    def get_telemetry_dict(frame_data: FrameData, adas_manager=None, pipeline_config=None) -> dict:
        """Serializes FrameData to a JSON-compatible dictionary for web clients."""
        lanes_dict = None
        if frame_data.lanes is not None:
            l = frame_data.lanes
            if l.left_poly_m is None and l.right_poly_m is None and (
                l.left_poly_px is not None or l.right_poly_px is not None
            ):
                geom = None
                if adas_manager is not None and getattr(adas_manager, "camera_geom", None) is not None:
                    geom = adas_manager.camera_geom
                else:
                    from drivecv.config import CameraConfig
                    from drivecv.core.geometry import CameraGeometry
                    geom = CameraGeometry(CameraConfig())
                    if pipeline_config is not None:
                        geom.update_resolution(pipeline_config.width, pipeline_config.height)
                    else:
                        geom.update_resolution(960, 540)
                    geom.calibrate_from_lanes(l)
                geom.project_lane_boundaries(l)

            def _poly_m_list(arr):
                if arr is None:
                    return None
                return [[float(p[0]), float(p[1])] for p in np.asarray(arr)]

            lanes_dict = {
                "is_valid": bool(l.is_valid),
                "left_line": [float(x) for x in l.left_line] if l.left_line is not None else None,
                "right_line": [float(x) for x in l.right_line] if l.right_line is not None else None,
                "y_top": int(l.y_top),
                "y_bot": int(l.y_bot),
                "y_roi_top": int(l.y_roi_top),
                "left_confidence": float(l.left_confidence),
                "right_confidence": float(l.right_confidence),
                "lane_center_bottom": float(l.lane_center_bottom),
                "lane_width_bottom": float(l.lane_width_bottom),
                "vanish_x": float(l.vanish_x) if l.vanish_x is not None else None,
                "vanish_y": float(l.vanish_y) if l.vanish_y is not None else None,
                "left_type": getattr(l, "left_type", "solid_yellow"),
                "right_type": getattr(l, "right_type", "solid_white"),
                "left_color": getattr(l, "left_color", "yellow"),
                "right_color": getattr(l, "right_color", "white"),
                "left_pattern": getattr(l, "left_pattern", "solid"),
                "right_pattern": getattr(l, "right_pattern", "solid"),
                "left_poly_m": _poly_m_list(getattr(l, "left_poly_m", None)),
                "right_poly_m": _poly_m_list(getattr(l, "right_poly_m", None)),
                "curvature_1pm": float(getattr(l, "curvature_1pm", 0.0) or 0.0),
            }

        tracks_list = []
        for t in frame_data.tracks:
            kin = t.kinematics
            tracks_list.append({
                "track_id": int(t.track_id),
                "class_name": str(t.class_name),
                "bbox": [float(t.bbox.x), float(t.bbox.y), float(t.bbox.w), float(t.bbox.h)],
                "confidence": float(t.confidence),
                "distance_m": float(kin.distance_m),
                "lateral_offset_m": float(kin.lateral_offset_m),
                "rel_speed_kmh": float(kin.rel_speed_kmh),
                "ttc_s": float(kin.ttc_seconds) if kin.ttc_seconds is not None else None,
                "is_lead": bool(kin.is_lead_vehicle),
            })

        adas_dict = None
        if frame_data.adas is not None:
            a = frame_data.adas
            calib = (
                adas_manager.ldw.get_calibration_dict()
                if adas_manager is not None
                else {
                    "is_calibrating": False,
                    "calibration_side": None,
                    "calibration_progress": 0.0,
                    "calibrated_left_m": -0.95,
                    "calibrated_right_m": 0.95,
                    "vehicle_width_m": 1.90,
                    "camera_bias_m": 0.0,
                }
            )
            adas_dict = {
                "ldw_state": a.ldw_state.name,
                "ldw_offset_m": float(a.ldw_offset_m),
                "ldw_tlc_s": float(a.ldw_tlc_s) if a.ldw_tlc_s is not None else None,
                "fcw_level": a.fcw_level.name,
                "fcw_lead_track_id": int(a.fcw_lead_track_id) if a.fcw_lead_track_id is not None else None,
                "fcw_lead_distance_m": float(a.fcw_lead_distance_m) if a.fcw_lead_distance_m is not None else None,
                "fcw_lead_rel_speed_kmh": float(a.fcw_lead_rel_speed_kmh) if a.fcw_lead_rel_speed_kmh is not None else None,
                "fcw_lead_ttc_s": float(a.fcw_lead_ttc_s) if a.fcw_lead_ttc_s is not None else None,
                "warning_message": a.warning_message,
                "calibration": calib,
            }


        stage_ms = None
        if frame_data.stage_ms is not None:
            st = frame_data.stage_ms
            stage_ms = {
                "decode_ms": float(st.decode_ms),
                "resize_ms": float(st.resize_ms),
                "track_ms": float(st.track_ms),
                "lanes_ms": float(st.lanes_ms),
                "adas_ms": float(st.adas_ms),
                "vis_ms": float(st.vis_ms),
                "total_ms": float(st.total_ms),
            }

        hood_mask_dict = None
        if pipeline_config is not None and hasattr(pipeline_config, "lane"):
            hood_mask_dict = {
                "enabled": bool(getattr(pipeline_config.lane, "hood_mask_enabled", True)),
                "height_ratio": float(getattr(pipeline_config.lane, "hood_height_ratio", 0.15)),
            }

        return {
            "frame_idx": int(frame_data.frame_idx),
            "timestamp": float(frame_data.timestamp),
            "fps": float(frame_data.fps),
            "lanes": lanes_dict,
            "tracks": tracks_list,
            "adas": adas_dict,
            "stage_ms": stage_ms,
            "hood_mask": hood_mask_dict,
        }


    def run(self, video_path: str, max_frames: Optional[int] = None):
        """Runs the complete pipeline on a video file."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"[ERROR] Video file not found at '{video_path}'.")

        cap = ScaledVideoCapture(
            video_path,
            self.config.width,
            self.config.height,
            use_ffmpeg=self.config.use_ffmpeg_scale,
        )
        orig_w, orig_h = cap.orig_w, cap.orig_h
        video_fps = cap.fps
        total_frames = cap.frame_count

        self.config.fps = video_fps
        self.player.target_fps = video_fps

        print(f"\n[INFO] Loaded Video: {video_path}")
        print(f"[INFO] Source: {orig_w}x{orig_h} @ {video_fps:.1f} FPS | Total: {total_frames} frames")
        print(f"[INFO] Target Processing: {self.config.width}x{self.config.height}")
        print(f"[INFO] Decode: {'FFmpeg scaled' if cap._use_ffmpeg else 'OpenCV'} | OpenCV threads={self.config.opencv_num_threads}")

        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, video_fps, (self.config.width, self.config.height)
            )
            print(f"[INFO] Recording output to '{self.output_path}'")

        frame_idx = 1
        t_dec = time.perf_counter()
        ret, frame = cap.read()
        self.last_decode_ms = (time.perf_counter() - t_dec) * 1000.0
        if not ret:
            print("[ERROR] Failed to read first frame.")
            cap.release()
            return

        latest_proc_frame: Optional[np.ndarray] = None
        latest_frame_data: Optional[FrameData] = None

        while True:
            t_start_wall = time.time()
            t_start = time.perf_counter()
            did_process = False

            if not self.player.is_paused or self.player.step_single_frame or self.headless:
                timestamp = frame_idx / max(1.0, video_fps)
                latest_proc_frame, latest_frame_data = self.process_frame(frame, frame_idx, timestamp)
                self.player.step_single_frame = False
                did_process = True

            if latest_proc_frame is not None and latest_frame_data is not None:
                t_vis = time.perf_counter()
                vis_frame = self.visualizer.render(
                    latest_proc_frame,
                    latest_frame_data,
                    vis_mode=self.player.vis_mode,
                )
                if self.config.visualizer.show_hud:
                    self.hud.draw(
                        vis_frame,
                        latest_frame_data,
                        total_frames=total_frames,
                        is_paused=self.player.is_paused,
                        auto_schedule=self.player.auto_schedule,
                        vis_mode=self.player.vis_mode,
                    )
                vis_ms = (time.perf_counter() - t_vis) * 1000.0
                if latest_frame_data.stage_ms is not None:
                    latest_frame_data.stage_ms.vis_ms = vis_ms
                    latest_frame_data.stage_ms.total_ms = (time.perf_counter() - t_start) * 1000.0

                if did_process:
                    dt_total = time.perf_counter() - t_start
                    self.recent_fps_history.append(dt_total)
                    if len(self.recent_fps_history) > 20:
                        self.recent_fps_history.pop(0)
                    latest_frame_data.fps = len(self.recent_fps_history) / max(
                        1e-4, sum(self.recent_fps_history)
                    )
                    if self.headless and frame_idx % self._timing_log_interval == 0:
                        st = latest_frame_data.stage_ms
                        extra = f" | {st.format_hud()}" if st is not None else ""
                        lead_txt = ""
                        if latest_frame_data.tracks:
                            lead = latest_frame_data.tracks[0]
                            lead_txt = (
                                f" lead={lead.kinematics.distance_m:.1f}m"
                                f" {lead.kinematics.rel_speed_kmh:+.0f}km/h"
                            )
                        print(
                            f"[TIMING] frame={frame_idx} fps={latest_frame_data.fps:.1f}{extra} "
                            f"tracks={len(latest_frame_data.tracks)}{lead_txt}"
                        )

                if self.video_writer is not None:
                    self.video_writer.write(vis_frame)

                self.player.show(vis_frame)

            if max_frames and frame_idx >= max_frames:
                print(f"[INFO] Reached max frame limit ({max_frames}). Exiting...")
                break

            def _trigger_yolo():
                if latest_proc_frame is not None:
                    if self.tracker.tracks:
                        target = self.tracker.tracks[0]
                        crop_roi = self._compute_padded_crop(
                            target.bbox, self.config.width, self.config.height
                        )
                        self.trigger_manual_yolo(latest_proc_frame, target_bbox=crop_roi)
                    else:
                        self.trigger_manual_yolo(latest_proc_frame, target_bbox=None)

            def _clear():
                self.tracker.clear()
                print("[INFO] Cleared all tracked objects.")

            def _select_roi():
                if latest_proc_frame is not None and latest_frame_data is not None:
                    roi = self.player.select_roi(latest_proc_frame)
                    if roi is not None:
                        self.tracker.add_manual_track(roi, latest_frame_data.gray_frame)

            def _cycle_lane_debug():
                modes = ["off", "canny", "ridge", "masks", "all"]
                cur = str(getattr(self.config.visualizer, "lane_debug", "off") or "off")
                nxt = modes[(modes.index(cur) + 1) % len(modes)] if cur in modes else "canny"
                self.config.visualizer.lane_debug = nxt
                print(f"[INFO] Lane debug view: {nxt}")

            keep_running = self.player.handle_keys(
                frame_start_time=t_start_wall,
                on_trigger_yolo=_trigger_yolo,
                on_clear_tracks=_clear,
                on_select_roi=_select_roi,
                on_cycle_lane_debug=_cycle_lane_debug,
            )
            if not keep_running:
                break

            if not self.player.is_paused or self.player.step_single_frame or self.headless:
                t_dec = time.perf_counter()
                ret, frame = cap.read()
                self.last_decode_ms = (time.perf_counter() - t_dec) * 1000.0
                if not ret:
                    print("[INFO] End of video reached.")
                    break
                frame_idx += 1

        if self.async_worker is not None:
            self.async_worker.stop()

        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        self.player.close()
