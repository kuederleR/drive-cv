"""
Thread-safe asynchronous perception worker for YOLOPv2.
Allows non-blocking neural inference while the main tracker runs every frame.
"""

import queue
import threading
import time
from typing import List, Optional, Tuple
import numpy as np
from drivecv.config import DetectorConfig
from drivecv.perception.yolopv2 import YOLOPv2Perception
from drivecv.types import BoundingBox, Detection


class AsyncPerceptionWorker:
    """
    Background worker thread running YOLOPv2 multi-task inference.
    Global passes letterbox the process frame to model input size (no 320→640 upscale).
    Crop submits copy only the ROI.
    """

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        detector: Optional[YOLOPv2Perception] = None,
    ):
        self.config = config or DetectorConfig()
        self.detector = detector if detector is not None else YOLOPv2Perception(self.config)

        self.input_queue: queue.Queue = queue.Queue(maxsize=1)
        self.result_queue: queue.Queue = queue.Queue(maxsize=1)
        self.running: bool = True
        self._busy_lock = threading.Lock()
        self._busy: bool = False

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    @property
    def is_busy(self) -> bool:
        with self._busy_lock:
            return self._busy

    def _set_busy(self, value: bool):
        with self._busy_lock:
            self._busy = value

    def _worker_loop(self):
        while self.running:
            try:
                task = self.input_queue.get(timeout=0.04)
            except queue.Empty:
                continue

            frame, roi_offset = task
            self._set_busy(True)
            try:
                dets, da_mask, ll_mask = self.detector.infer(frame)
                if roi_offset is not None:
                    ox, oy = roi_offset
                    remapped = []
                    for d in dets:
                        remapped.append(
                            Detection(
                                bbox=BoundingBox(
                                    x=d.bbox.x + ox,
                                    y=d.bbox.y + oy,
                                    w=d.bbox.w,
                                    h=d.bbox.h,
                                ),
                                confidence=d.confidence,
                                class_id=d.class_id,
                                class_name=d.class_name,
                                source="yolopv2_crop",
                            )
                        )
                    self._push_result((remapped, da_mask, ll_mask))
                else:
                    for d in dets:
                        d.source = "yolopv2_global"
                    self._push_result((dets, da_mask, ll_mask))
            except Exception as e:
                print(f"[ASYNC DETECTOR ERROR]: {e}")
            finally:
                self._set_busy(False)

    def _push_result(self, result_tuple):
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        self.result_queue.put(result_tuple)

    def submit_frame(self, frame: np.ndarray, roi_crop: Optional[BoundingBox] = None) -> bool:
        """Submits a owned copy of the global frame or ROI crop. Returns True if accepted."""
        if self.is_busy or not self.input_queue.empty():
            return False
        try:
            if roi_crop is not None:
                h_f, w_f = frame.shape[:2]
                rx, ry, rw, rh = roi_crop.as_int_xywh()
                rx = max(0, min(w_f - 16, rx))
                ry = max(0, min(h_f - 16, ry))
                rw = max(16, min(w_f - rx, rw))
                rh = max(16, min(h_f - ry, rh))
                crop = np.ascontiguousarray(frame[ry : ry + rh, rx : rx + rw].copy())
                self.input_queue.put_nowait((crop, (float(rx), float(ry))))
            else:
                self.input_queue.put_nowait((np.ascontiguousarray(frame.copy()), None))
            return True
        except queue.Full:
            return False

    def fetch_results(self) -> Optional[Tuple[List[Detection], Optional[np.ndarray], Optional[np.ndarray]]]:
        """Fetches completed inference results if available without blocking."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stops background thread safely."""
        self.running = False
        start_wait = time.time()
        while self.is_busy and time.time() - start_wait < 1.5:
            time.sleep(0.02)
        if self.thread.is_alive():
            self.thread.join(timeout=1.5)
