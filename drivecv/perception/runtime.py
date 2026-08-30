"""
Inference runtime selection for YOLOPv2.

Decision (30 FPS target on CPU-only Intel, no discrete GPU):
- Keep YOLOPv2 ONNX as the neural engine (drivable/lane heads + detection).
- Use ByteTrack-style two-stage association in the tracker, not a nano detector.
- Prefer OpenVINOExecutionProvider when the ORT build includes it.
- Direct OpenVINO IR conversion of this YOLOPv2 export fails (SequenceMark outputs),
  so we do not swap in a converted IR graph.
- A YOLOv8n/YOLO11n detect-every-frame path remains the next step if a discrete GPU
  or a convertible nano ONNX is added later.
"""

from typing import List, Tuple
import onnxruntime as ort


def create_ort_session(
    model_path: str,
    sess_options: ort.SessionOptions,
    prefer_openvino: bool = True,
) -> Tuple[ort.InferenceSession, List[str]]:
    """Create an ORT session, trying OpenVINO, CUDA, then CPU."""
    available = set(ort.get_available_providers())
    attempts: List[List] = []
    if prefer_openvino and "OpenVINOExecutionProvider" in available:
        attempts.append([
            ("OpenVINOExecutionProvider", {"device_type": "GPU_FP32"}),
            "CPUExecutionProvider",
        ])
        attempts.append(["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    if "CUDAExecutionProvider" in available:
        attempts.append(["CUDAExecutionProvider", "CPUExecutionProvider"])
    attempts.append(["CPUExecutionProvider"])

    last_err = None
    for providers in attempts:
        try:
            session = ort.InferenceSession(model_path, sess_options, providers=providers)
            return session, list(session.get_providers())
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Failed to create ONNX Runtime session: {last_err}")
