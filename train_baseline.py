#!/usr/bin/env python3
"""
Baseline training script: YOLOv8n on NEU-DET (or any YOLO-format dataset).

Usage (on Windows 4080 with CUDA):
  python train_baseline.py --data data/neu_det.yaml --epochs 100 --imgsz 640

Default model: yolov8n.pt (nano).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8n baseline (Ultralytics)")
    p.add_argument(
        "--data",
        type=str,
        default="data/neu_det.yaml",
        help="Path to dataset YAML (path/train/val/names/nc)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Checkpoint: yolov8n.pt (baseline nano)",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0", help="GPU id or cpu")
    p.add_argument("--workers", type=int, default=8, help="dataloader workers")
    p.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Ultralytics project directory",
    )
    p.add_argument(
        "--name",
        type=str,
        default="neu_det_yolov8n_baseline",
        help="Run name under project/",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without improvement)",
    )
    p.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting existing run name",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset yaml not found: {data_path.resolve()}\n"
            "Copy data/neu_det_template.yaml to data/neu_det.yaml and set `path:`."
        )

    model = YOLO(args.model)

    model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        patience=args.patience,
        # Single-GPU baseline; keep optimizer/aug as Ultralytics defaults unless you lock them in paper
        optimizer="auto",
        verbose=True,
    )

    # Validate best weights
    weights_dir = Path(args.project) / args.name / "weights"
    best = weights_dir / "best.pt"
    if best.is_file():
        val_model = YOLO(str(best))
        metrics = val_model.val(data=str(data_path.resolve()), imgsz=args.imgsz, batch=args.batch, device=args.device)
        print(metrics)


if __name__ == "__main__":
    main()
