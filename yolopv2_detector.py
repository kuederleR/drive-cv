#!/usr/bin/env python3
"""
YOLOPv2 ONNX Perception Engine:
- Multi-Class Traffic Object Detection (Bounding Boxes, Confidence Scores, Class IDs)
- Drivable Area Segmentation (Pixel-level binary mask)
- Lane Line Segmentation (Pixel-level binary mask)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

# Ensure local libs directory is included in sys.path
_LIBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
if os.path.exists(_LIBS_DIR) and _LIBS_DIR not in sys.path:
    sys.path.insert(0, _LIBS_DIR)

import cv2
import numpy as np
import onnxruntime as ort


class DetectionResult:
    """Represents a single detected traffic object from YOLOPv2."""

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: str = "vehicle",
    ):
        self.bbox = [float(v) for v in bbox]  # [x, y, w, h]
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = class_name


class YOLOPv2Detector:
    """
    YOLOPv2 Multi-Task Autonomous Driving Perception Engine:
    Runs inference on ONNX Runtime to produce:
    1. Traffic Object Bounding Boxes & Confidence Scores
    2. Drivable Area Segmentation Mask
    3. Lane Line Segmentation Mask
    """

    def __init__(
        self,
        model_path: str = "weights/YOLOPv2.onnx",
        input_size: Tuple[int, int] = (640, 640),
        conf_thresh: float = 0.28,
        iou_thresh: float = 0.45,
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[ERROR] YOLOPv2 model file not found at '{model_path}'. "
                "Please download it to the weights/ directory."
            )

        # Initialize ONNX Runtime Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = max(1, os.cpu_count() or 4)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available_providers = ort.get_available_providers()
        valid_providers = [p for p in providers if p in available_providers]

        self.session = ort.InferenceSession(model_path, sess_options, providers=valid_providers)
        self.input_name = self.session.get_inputs()[0].name

        # Stride values for 3 detection scales (P3, P4, P5)
        self.strides = [8, 16, 32]

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    def _letterbox(
        self,
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Resizes and pads image to fit model input dimensions while preserving aspect ratio."""
        shape = img.shape[:2]  # [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2.0
        dh /= 2.0

        if shape[::-1] != new_unpad:
            resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = img.copy()

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
    ) -> List[DetectionResult]:
        """Decodes YOLOPv2 multi-scale feature maps into bounding boxes with NMS."""
        z = []
        for i in range(3):
            bs, _, ny, nx = det_maps[i].shape
            d = det_maps[i].reshape(bs, 3, 85, ny, nx).transpose(0, 1, 3, 4, 2)
            d = self._sigmoid(d)

            xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
            grid = np.stack((xv, yv), 2).reshape(1, 1, ny, nx, 2).astype(np.float32)

            d[..., 0:2] = (d[..., 0:2] * 2.0 - 0.5 + grid) * self.strides[i]
            d[..., 2:4] = (d[..., 2:4] * 2.0) ** 2 * anchor_grids[i]
            z.append(d.reshape(bs, -1, 85))

        pred = np.concatenate(z, 1)[0]  # Shape: (Total_Anchors, 85)

        boxes_cxcywh = pred[:, :4]
        obj_conf = pred[:, 4]
        class_scores = pred[:, 5:]

        # Multiply class score by objectness score
        scores = obj_conf[:, None] * class_scores
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = max_scores > self.conf_thresh
        if not np.any(mask):
            return []

        filtered_boxes = boxes_cxcywh[mask]
        filtered_scores = max_scores[mask]
        filtered_class_ids = class_ids[mask]

        dw, dh = pad
        orig_h, orig_w = orig_shape

        # Convert [cx, cy, w, h] on padded input -> [x, y, w, h] on original image
        x_min = (filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2.0 - dw) / ratio
        y_min = (filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2.0 - dh) / ratio
        box_w = filtered_boxes[:, 2] / ratio
        box_h = filtered_boxes[:, 3] / ratio

        # Boundary clamping
        x_min = np.clip(x_min, 0.0, float(orig_w - 1))
        y_min = np.clip(y_min, 0.0, float(orig_h - 1))
        box_w = np.clip(box_w, 1.0, float(orig_w) - x_min)
        box_h = np.clip(box_h, 1.0, float(orig_h) - y_min)

        # Format for OpenCV NMS: [x, y, w, h] as ints
        boxes_list = [
            [int(x_min[k]), int(y_min[k]), int(box_w[k]), int(box_h[k])]
            for k in range(len(filtered_boxes))
        ]
        scores_list = [float(s) for s in filtered_scores]

        indices = cv2.dnn.NMSBoxes(
            boxes_list,
            scores_list,
            self.conf_thresh,
            self.iou_thresh,
        )

        results = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                bx, by, bw, bh = boxes_list[idx]
                if bw >= 12 and bh >= 12:  # Plausibility threshold
                    results.append(
                        DetectionResult(
                            bbox=(bx, by, bw, bh),
                            confidence=scores_list[idx],
                            class_id=int(filtered_class_ids[idx]),
                            class_name="vehicle",
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
        """Decodes drivable area and lane line segmentation masks."""
        dw, dh = pad
        orig_h, orig_w = orig_shape
        inp_h, inp_w = self.input_size

        top_crop = int(round(dh))
        bot_crop = max(top_crop + 1, inp_h - int(round(dh)))
        left_crop = int(round(dw))
        right_crop = max(left_crop + 1, inp_w - int(round(dw)))

        # 1. Drivable Area Mask (Argmax across 2 channels)
        da_mask_raw = np.argmax(da_raw[0], axis=0).astype(np.uint8)  # (640, 640)
        da_crop = da_mask_raw[top_crop:bot_crop, left_crop:right_crop]
        da_mask = cv2.resize(da_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 2. Lane Line Mask (Threshold > 0.5)
        ll_mask_raw = (ll_raw[0][0] > 0.5).astype(np.uint8)  # (640, 640)
        ll_crop = ll_mask_raw[top_crop:bot_crop, left_crop:right_crop]
        ll_mask = cv2.resize(ll_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return da_mask, ll_mask

    def detect(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[DetectionResult], np.ndarray, np.ndarray]:
        """
        Executes YOLOPv2 inference on input BGR frame.
        Returns:
            detections: List of DetectionResult objects
            da_mask: Drivable Area binary mask (H, W) uint8
            ll_mask: Lane Line binary mask (H, W) uint8
        """
        orig_shape = frame.shape[:2]

        # 1. Preprocessing (Letterbox, RGB conversion, Normalization)
        padded_img, ratio, pad = self._letterbox(frame, new_shape=self.input_size)
        input_tensor = (
            padded_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # 2. Run ONNX Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # 3. Postprocess Object Detections
        det_maps = outputs[0]
        anchor_grids = [outputs[1], outputs[2], outputs[3]]
        detections = self._decode_detections(
            det_maps, anchor_grids, ratio, pad, orig_shape
        )

        # 4. Postprocess Segmentation Masks
        da_raw = outputs[4]  # Drivable area logits (1, 2, 640, 640)
        ll_raw = outputs[5]  # Lane line logits (1, 1, 640, 640)
        da_mask, ll_mask = self._decode_segmentation(da_raw, ll_raw, pad, orig_shape)

        return detections, da_mask, ll_mask

    @staticmethod
    def draw_segmentation_overlay(
        frame: np.ndarray,
        da_mask: Optional[np.ndarray],
        ll_mask: Optional[np.ndarray],
        da_color: Tuple[int, int, int] = (255, 180, 0),  # Sleek Electric Cyan
        ll_color: Tuple[int, int, int] = (0, 0, 255),    # Crimson Red for Lane Lines
        alpha: float = 0.35,
    ):
        """Blends YOLOPv2 drivable area and lane line masks onto the frame."""
        overlay = frame.copy()

        # Drivable area overlay
        if da_mask is not None and np.any(da_mask == 1):
            overlay[da_mask == 1] = da_color

        # Lane line overlay
        if ll_mask is not None and np.any(ll_mask == 1):
            overlay[ll_mask == 1] = ll_color

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
