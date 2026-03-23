#!/usr/bin/env python3
"""
Validate a trained YOLOv8n checkpoint on the same dataset YAML.

Usage:
  python val_baseline.py --weights runs/detect/neu_det_yolov8n_baseline/weights/best.pt --data data/neu_det.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate YOLOv8n weights")
    p.add_argument("--weights", type=str, required=True, help="Path to best.pt or last.pt")
    p.add_argument("--data", type=str, default="data/neu_det.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    w = Path(args.weights)
    if not w.is_file():
        raise FileNotFoundError(w)
    data = Path(args.data)
    if not data.is_file():
        raise FileNotFoundError(data)

    model = YOLO(str(w))
    metrics = model.val(
        data=str(data.resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    print(metrics)


if __name__ == "__main__":
    main()
