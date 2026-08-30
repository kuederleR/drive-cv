"""
YOLOPv2 ONNX multi-task perception engine.
Decodes multi-class object detections, drivable area segmentation, and lane line masks.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Optional, Tuple
from drivecv.config import COCO_CLASS_NAMES, DetectorConfig
from drivecv.perception.runtime import create_ort_session
from drivecv.types import BoundingBox, Detection


class YOLOPv2Perception:
    """
    High-Performance YOLOPv2 ONNX multi-task inference engine:
    1. Object Detection (filtered to vehicle classes)
    2. Drivable Area Segmentation
    3. Lane Line Segmentation
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        model_path = self.config.model_path

        if not os.path.exists(model_path):
            try:
                from scripts.download_weights import download_weights
                download_weights(model_path)
            except Exception as err:
                print(f"[WARNING] Weight auto-download attempt failed: {err}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[ERROR] YOLOPv2 model weights not found at '{model_path}'. "
                "Run 'python scripts/download_weights.py' or operate in Classical CV mode."
            )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = max(1, int(self.config.onnx_intra_op_threads))
        sess_options.inter_op_num_threads = 1

        self.session, self.providers = create_ort_session(
            model_path, sess_options, prefer_openvino=self.config.prefer_openvino
        )
        print(f"[INFO] YOLOPv2 ORT providers: {self.providers} (threads={self.config.onnx_intra_op_threads})")
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = self.config.input_size
        self.conf_thresh = self.config.conf_thresh
        self.iou_thresh = self.config.iou_thresh
        self.strides = [8, 16, 32]
        self.vehicle_class_ids = set(self.config.vehicle_class_ids)
        self._grid_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    def _get_grid(self, nx: int, ny: int, stride: int) -> np.ndarray:
        key = (nx, ny, stride)
        cached = self._grid_cache.get(key)
        if cached is None:
            xv, yv = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
            cached = np.stack((xv, yv), 2).reshape(1, 1, ny, nx, 2)
            self._grid_cache[key] = cached
        return cached

    def _letterbox(
        self,
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Pads and resizes image preserving aspect ratio."""
        shape = img.shape[:2]  # [h, w]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2.0
        dh /= 2.0

        interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
        if shape[::-1] != new_unpad:
            resized = cv2.resize(img, new_unpad, interpolation=interp)
        else:
            resized = img

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        if padded.shape[0] != new_shape[0] or padded.shape[1] != new_shape[1]:
            padded = cv2.resize(padded, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
        return padded, r, (dw, dh)

    def _decode_detections(
        self,
        det_maps: List[np.ndarray],
        anchor_grids: List[np.ndarray],
        ratio: float,
        pad: Tuple[float, float],
        orig_shape: Tuple[int, int],
    ) -> List[Detection]:
        """Decodes anchor grid detection tensors and executes non-maximum suppression (NMS)."""
        z = []
        for i in range(3):
            bs, _, ny, nx = det_maps[i].shape
            d = det_maps[i].reshape(bs, 3, 85, ny, nx).transpose(0, 1, 3, 4, 2)
            d = self._sigmoid(d)

            grid = self._get_grid(nx, ny, self.strides[i])
            d[..., 0:2] = (d[..., 0:2] * 2.0 - 0.5 + grid) * self.strides[i]
            d[..., 2:4] = (d[..., 2:4] * 2.0) ** 2 * anchor_grids[i]
            z.append(d.reshape(bs, -1, 85))

        pred = np.concatenate(z, 1)[0]
        boxes_cxcywh = pred[:, :4]
        obj_conf = pred[:, 4]
        class_scores = pred[:, 5:]

        scores = obj_conf[:, None] * class_scores
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = max_scores > self.conf_thresh
        if self.vehicle_class_ids:
            mask = mask & np.isin(class_ids, list(self.vehicle_class_ids))
        if not np.any(mask):
            return []

        filtered_boxes = boxes_cxcywh[mask]
        filtered_scores = max_scores[mask]
        filtered_class_ids = class_ids[mask]

        dw, dh = pad
        orig_h, orig_w = orig_shape

        x_min = (filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2.0 - dw) / ratio
        y_min = (filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2.0 - dh) / ratio
        box_w = filtered_boxes[:, 2] / ratio
        box_h = filtered_boxes[:, 3] / ratio

        x_min = np.clip(x_min, 0.0, float(orig_w - 1))
        y_min = np.clip(y_min, 0.0, float(orig_h - 1))
        box_w = np.clip(box_w, 1.0, float(orig_w) - x_min)
        box_h = np.clip(box_h, 1.0, float(orig_h) - y_min)

        boxes_arr = np.stack([x_min, y_min, box_w, box_h], axis=1)
        nms_boxes = boxes_arr.astype(np.float32).tolist()
        scores_list = filtered_scores.astype(np.float32).tolist()

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            scores_list,
            self.conf_thresh,
            self.iou_thresh,
        )

        results = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                bx, by, bw, bh = boxes_arr[idx]
                if bw >= 10 and bh >= 10:
                    cid = int(filtered_class_ids[idx])
                    results.append(
                        Detection(
                            bbox=BoundingBox(x=float(bx), y=float(by), w=float(bw), h=float(bh)),
                            confidence=float(filtered_scores[idx]),
                            class_id=cid,
                            class_name=COCO_CLASS_NAMES.get(cid, "vehicle"),
                            source="yolopv2",
                        )
                    )

        return results

    def _decode_segmentation(
        self,
        da_raw: np.ndarray,
        ll_raw: np.ndarray,
        pad: Tuple[float, float],
        orig_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decodes drivable area and lane line segmentation probability maps."""
        dw, dh = pad
        orig_h, orig_w = orig_shape
        inp_h, inp_w = self.input_size

        top_crop = int(round(dh))
        bot_crop = max(top_crop + 1, inp_h - int(round(dh)))
        left_crop = int(round(dw))
        right_crop = max(left_crop + 1, inp_w - int(round(dw)))

        da_mask_raw = np.argmax(da_raw[0], axis=0).astype(np.uint8)
        da_crop = da_mask_raw[top_crop:bot_crop, left_crop:right_crop]
        da_mask = cv2.resize(da_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        ll_mask_raw = (ll_raw[0][0] > 0.5).astype(np.uint8)
        ll_crop = ll_mask_raw[top_crop:bot_crop, left_crop:right_crop]
        ll_mask = cv2.resize(ll_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return da_mask, ll_mask

    def infer(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[Detection], np.ndarray, np.ndarray]:
        """
        Runs full inference on frame.
        Returns (detections, drivable_area_mask, lane_line_mask).
        """
        orig_shape = frame.shape[:2]
        padded_img, ratio, pad = self._letterbox(frame, new_shape=self.input_size)

        input_tensor = (
            padded_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        input_tensor = np.ascontiguousarray(np.expand_dims(input_tensor, axis=0))

        outputs = self.session.run(None, {self.input_name: input_tensor})

        det_maps = outputs[0]
        anchor_grids = [outputs[1], outputs[2], outputs[3]]
        detections = self._decode_detections(det_maps, anchor_grids, ratio, pad, orig_shape)

        da_raw = outputs[4]
        ll_raw = outputs[5]
        da_mask, ll_mask = self._decode_segmentation(da_raw, ll_raw, pad, orig_shape)

        return detections, da_mask, ll_mask
