#!/usr/bin/env python3
"""
Export trained weights to ONNX / TensorRT engine (run on NVIDIA GPU machine, e.g. Windows 4080).

Requires: TensorRT + CUDA toolkit aligned with your PyTorch CUDA build.

Usage:
  python export_tensorrt.py --weights runs/detect/neu_det_yolov8n_baseline/weights/best.pt --imgsz 640

Notes:
  - First export often produces ONNX; TensorRT .engine depends on environment.
  - See Ultralytics docs for `model.export(format='engine', ...)`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO weights to ONNX / TensorRT")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--format",
        type=str,
        default="engine",
        choices=("onnx", "engine"),
        help="onnx is easier to debug; engine needs TensorRT",
    )
    p.add_argument("--half", action="store_true", help="FP16 export when supported")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    w = Path(args.weights)
    if not w.is_file():
        raise FileNotFoundError(w)

    model = YOLO(str(w))
    model.export(format=args.format, imgsz=args.imgsz, half=args.half)


if __name__ == "__main__":
    main()
