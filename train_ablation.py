#!/usr/bin/env python3
"""
Ablation study: GhostConv layer count in YOLOv8n backbone.

Runs 3 experiments sequentially (Ghost-1 → Ghost-2 → Ghost-4) under the same
training protocol, so results are directly comparable to the baseline and to
each other in the paper's ablation table.

Ablation design:
  Ghost-1  models/yolov8n-ghost-1.yaml   layer 7 only
  Ghost-2  models/yolov8n-ghost-2.yaml   layers 5, 7
  Ghost-4  models/yolov8n-ghost.yaml     layers 1, 3, 5, 7  (full Ghost)

Baseline (YOLOv8n) is NOT re-run here — use the existing best.pt from
train_baseline.py and record its metrics manually in the ablation table.

Usage:
  python train_ablation.py --data "D:/path/to/data.yaml" --epochs 150 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

_SCRIPT_DIR = Path(__file__).parent

# Ordered from least to most GhostConv substitution
ABLATION_CONFIGS: list[dict] = [
    {
        "yaml": "models/yolov8n-ghost-1.yaml",
        "name": "ablation_ghost1_layer7",
        "label": "Ghost-1 (layer 7 only)",
    },
    {
        "yaml": "models/yolov8n-ghost-2.yaml",
        "name": "ablation_ghost2_layers5_7",
        "label": "Ghost-2 (layers 5, 7)",
    },
    {
        "yaml": "models/yolov8n-ghost.yaml",
        "name": "ablation_ghost4_layers1_3_5_7",
        "label": "Ghost-4 (layers 1, 3, 5, 7) — full Ghost",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GhostConv ablation experiments")
    p.add_argument(
        "--data",
        type=str,
        default="data/neu_det.yaml",
        help="Dataset YAML — must be the same file used for baseline training",
    )
    # ---- keep identical to train_baseline.py for fair comparison ----
    p.add_argument("--epochs",   type=int, default=150)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--device",   type=str, default="0")
    p.add_argument("--workers",  type=int, default=8)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--patience", type=int, default=50)
    # -----------------------------------------------------------------
    p.add_argument("--project",  type=str, default="runs/ablation")
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def train_one(cfg: dict, args: argparse.Namespace, data_path: Path) -> None:
    yaml_path = _SCRIPT_DIR / cfg["yaml"]
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"Model yaml not found: {yaml_path.resolve()}\n"
            "Make sure models/ directory contains the ablation yaml files."
        )

    print(f"\n{'=' * 60}")
    print(f"  Starting: {cfg['label']}")
    print(f"  Config  : {yaml_path}")
    print(f"  Run name: {cfg['name']}")
    print(f"{'=' * 60}\n")

    model = YOLO(str(yaml_path))
    model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=cfg["name"],
        exist_ok=args.exist_ok,
        seed=args.seed,
        patience=args.patience,
        optimizer="auto",
        verbose=True,
    )

    # Validate best weights and print summary
    best = Path(args.project) / cfg["name"] / "weights" / "best.pt"
    if best.is_file():
        val_model = YOLO(str(best))
        metrics = val_model.val(
            data=str(data_path.resolve()),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        map50     = metrics.box.map50
        map50_95  = metrics.box.map
        print(f"\n[{cfg['label']}]  mAP@0.5={map50:.4f}  mAP@0.5:0.95={map50_95:.4f}")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset yaml not found: {data_path.resolve()}\n"
            "Pass the same --data yaml used in train_baseline.py."
        )

    print("\nAblation: GhostConv layer-count study (Ghost-1 / Ghost-2 / Ghost-4)")
    print(f"Epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}  seed={args.seed}\n")

    for cfg in ABLATION_CONFIGS:
        train_one(cfg, args, data_path)

    print("\n\nAll ablation runs finished.")
    print(f"Results saved under: {Path(args.project).resolve()}/")
    print("\nFor the paper ablation table, also include:")
    print("  Baseline (YOLOv8n): mAP@0.5=0.830  Params=3.0M  FLOPs=8.1G")
    print("  Ghost-4 result already in: runs/detect/neu_det_yolov8n_ghost/")


if __name__ == "__main__":
    main()
