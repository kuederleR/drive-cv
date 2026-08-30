#!/usr/bin/env python3
"""
DriveCV Model Weights Downloader
Downloads YOLOPv2 ONNX multi-task perception weights into weights/ directory.
"""

import os
import sys
import urllib.request

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "YOLOPv2.onnx")

# Primary and mirror download links for YOLOPv2 ONNX weights
DOWNLOAD_URLS = [
    "https://huggingface.co/models/YOLOPv2/resolve/main/YOLOPv2.onnx",
    "https://raw.githubusercontent.com/CAIC-AD/YOLOPv2/main/weights/YOLOPv2.onnx",
    "https://github.com/PINTO0309/PINTO_model_zoo/releases/download/v1.0.0/326_YOLOPv2.onnx",
]


def download_progress_hook(count, block_size, total_size):
    """Displays progress bar for download."""
    downloaded = count * block_size
    if total_size > 0:
        percent = min(100.0, (downloaded / total_size) * 100.0)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r[INFO] Downloading YOLOPv2.onnx: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
    else:
        mb_downloaded = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r[INFO] Downloading YOLOPv2.onnx: {mb_downloaded:.1f} MB downloaded")
    sys.stdout.flush()


def download_weights(target_path: str = WEIGHTS_PATH) -> bool:
    """Downloads YOLOPv2 ONNX model weights if not already present."""
    if os.path.exists(target_path) and os.path.getsize(target_path) > 10 * 1024 * 1024:
        print(f"[INFO] YOLOPv2 ONNX weights already exist at '{target_path}'.")
        return True

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"[INFO] YOLOPv2 ONNX weights missing at '{target_path}'. Initiating automatic download...")

    for url in DOWNLOAD_URLS:
        try:
            print(f"[INFO] Trying download source: {url}")
            urllib.request.urlretrieve(url, target_path, reporthook=download_progress_hook)
            print("\n[INFO] YOLOPv2 ONNX weights downloaded successfully!")
            return True
        except Exception as e:
            print(f"\n[WARNING] Failed to download from '{url}': {e}")
            if os.path.exists(target_path):
                os.remove(target_path)

    print("[ERROR] Could not download YOLOPv2 ONNX model weights automatically from mirror URLs.")
    print("[INFO] DriveCV will operate in Classical Computer Vision Mode (160+ FPS).")
    return False


if __name__ == "__main__":
    download_weights()
