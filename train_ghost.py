#!/usr/bin/env python3
"""
Ghost-YOLOv8n training script.

Architecture: YOLOv8n with backbone downsampling Conv replaced by GhostConv.
Model config:  models/yolov8n-ghost.yaml

For fair paper comparison, keep ALL training hyperparameters identical to
train_baseline.py (epochs, imgsz, batch, seed, patience, optimizer).

Usage:
  python train_ghost.py --data data/neu_det.yaml --epochs 150 --imgsz 640 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

# Path to Ghost model config (relative to this script)
_SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL_YAML = str(_SCRIPT_DIR / "models" / "yolov8n-ghost.yaml")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Ghost-YOLOv8n on NEU-DET")
    p.add_argument(
        "--data",
        type=str,
        default="data/neu_det.yaml",
        help="Dataset YAML (same file used for baseline training)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_YAML,
        help="Ghost model YAML (trains from scratch — no COCO pretrained)",
    )
    # ---- keep the following identical to train_baseline.py for fair comparison ----
    p.add_argument("--epochs",   type=int, default=150)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--device",   type=str, default="0")
    p.add_argument("--workers",  type=int, default=8)
    p.add_argument("--seed",     type=int, default=42,  help="Fix for reproducibility")
    p.add_argument("--patience", type=int, default=50,  help="Early-stopping patience")
    # -------------------------------------------------------------------------------
    p.add_argument(
        "--project",
        type=str,
        default="runs/detect",
    )
    p.add_argument(
        "--name",
        type=str,
        default="neu_det_yolov8n_ghost",
        help="Run name; results saved under project/name/",
    )
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset yaml not found: {data_path.resolve()}\n"
            "Use the same --data yaml as train_baseline.py."
        )

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Ghost model yaml not found: {model_path.resolve()}\n"
            "Expected: models/yolov8n-ghost.yaml in the project root."
        )

    # Load from YAML (trains from scratch — architecture paper standard)
    model = YOLO(str(model_path))

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
        optimizer="auto",
        verbose=True,
    )

    # Auto-validate best weights after training
    weights_dir = Path(args.project) / args.name / "weights"
    best = weights_dir / "best.pt"
    if best.is_file():
        val_model = YOLO(str(best))
        metrics = val_model.val(
            data=str(data_path.resolve()),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        print(metrics)


if __name__ == "__main__":
    main()
