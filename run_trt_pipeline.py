#!/usr/bin/env python3
"""
One-shot TensorRT pipeline: export + benchmark for Baseline and Ghost-4.

Must run on Windows 4080 (CUDA + TensorRT environment).

Step 1 - Export both models to TensorRT FP16 engine:
  python run_trt_pipeline.py --mode export --imgsz 640

Step 2 - Benchmark PyTorch vs TensorRT for both models:
  python run_trt_pipeline.py --mode benchmark --imgsz 640

Step 3 - Do both in one command:
  python run_trt_pipeline.py --mode all --imgsz 640

Edit BASELINE_PT and GHOST4_PT below if your paths differ.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from ultralytics import YOLO


# ── Edit these paths to match your Windows run directories ──────────────────
BASELINE_PT = Path(
    "runs/detect/runs/detect/neu_det_yolov8n_baseline2/weights/best.pt"
)
GHOST4_PT = Path(
    "runs/detect/runs/ablation/ablation_ghost4_layers1_3_5_7/weights/best.pt"
)
# ─────────────────────────────────────────────────────────────────────────────

WARMUP_RUNS  = 50
MEASURE_RUNS = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export + Benchmark TensorRT pipeline")
    p.add_argument(
        "--mode",
        choices=("export", "benchmark", "all"),
        default="all",
        help="export: only export .engine  |  benchmark: only time  |  all: both",
    )
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--batch",  type=int, default=1)
    p.add_argument("--device", type=str, default="0")
    return p.parse_args()


def export_engine(pt_path: Path, imgsz: int) -> Path:
    """Export .pt → TensorRT FP16 .engine alongside the .pt file."""
    print(f"\n[EXPORT] {pt_path.name}  →  TensorRT FP16")
    model = YOLO(str(pt_path))
    model.export(format="engine", imgsz=imgsz, half=True)
    engine_path = pt_path.with_suffix(".engine")
    if engine_path.is_file():
        print(f"  Saved: {engine_path}")
    else:
        # Ultralytics sometimes saves engine next to .pt with same stem
        candidates = list(pt_path.parent.glob("*.engine"))
        engine_path = candidates[0] if candidates else engine_path
        print(f"  Engine: {engine_path}")
    return engine_path


def benchmark(pt_path: Path, engine_path: Path, imgsz: int, batch: int, device: str) -> dict:
    """Benchmark PyTorch FP32 vs TensorRT FP16 and return stats dict."""
    dummy = [torch.zeros(imgsz, imgsz, 3).numpy()] * batch

    def _run(model: YOLO, label: str) -> dict:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        for _ in range(WARMUP_RUNS):
            model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
        torch.cuda.synchronize()

        lats: list[float] = []
        for _ in range(MEASURE_RUNS):
            t0 = time.perf_counter()
            model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)

        avg   = sum(lats) / len(lats)
        fps   = 1000.0 / avg * batch
        peak  = torch.cuda.max_memory_allocated() / 1024 ** 2
        return {"label": label, "avg_ms": avg, "fps": fps, "peak_mb": peak}

    print(f"\n[BENCHMARK] {pt_path.stem}")

    pt_model  = YOLO(str(pt_path))
    pt_s      = _run(pt_model, "PyTorch  FP32")
    del pt_model; torch.cuda.empty_cache()

    trt_model = YOLO(str(engine_path))
    trt_s     = _run(trt_model, "TensorRT FP16")
    del trt_model; torch.cuda.empty_cache()

    speedup = pt_s["avg_ms"] / trt_s["avg_ms"]
    print(f"\n  {'':20s} {'Latency':>10} {'FPS':>8} {'PeakMem':>10}")
    print(f"  {'PyTorch FP32':20s} {pt_s['avg_ms']:>9.2f}ms {pt_s['fps']:>7.1f} {pt_s['peak_mb']:>9.1f}MB")
    print(f"  {'TensorRT FP16':20s} {trt_s['avg_ms']:>9.2f}ms {trt_s['fps']:>7.1f} {trt_s['peak_mb']:>9.1f}MB")
    print(f"  Speedup: {speedup:.2f}x  ({(speedup-1)*100:.1f}% faster)")
    return {"name": pt_path.stem, "pt": pt_s, "trt": trt_s, "speedup": speedup}


def main() -> None:
    args = parse_args()

    for p in (BASELINE_PT, GHOST4_PT):
        if not p.is_file():
            raise FileNotFoundError(
                f"Weights not found: {p.resolve()}\n"
                "Edit BASELINE_PT / GHOST4_PT at the top of this script."
            )

    pairs: list[tuple[Path, Path]] = []

    # ── Export phase ─────────────────────────────────────────────────────────
    if args.mode in ("export", "all"):
        for pt in (BASELINE_PT, GHOST4_PT):
            eng = export_engine(pt, args.imgsz)
            pairs.append((pt, eng))
    else:
        # benchmark-only: engines must already exist
        for pt in (BASELINE_PT, GHOST4_PT):
            candidates = list(pt.parent.glob("*.engine"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .engine found next to {pt}. Run with --mode export first."
                )
            pairs.append((pt, candidates[0]))

    # ── Benchmark phase ───────────────────────────────────────────────────────
    if args.mode in ("benchmark", "all"):
        all_stats: list[dict] = []
        for pt, eng in pairs:
            stats = benchmark(pt, eng, args.imgsz, args.batch, args.device)
            all_stats.append(stats)

        # Final paper-ready summary
        print("\n\n" + "=" * 68)
        print("  PAPER DEPLOYMENT TABLE  (imgsz={}, batch={}, device=RTX 4080 Laptop)".format(
            args.imgsz, args.batch))
        print("=" * 68)
        print(f"  {'Method':<28} {'Backend':<14} {'Lat(ms)':>8} {'FPS':>7} {'Mem(MB)':>9}")
        print("-" * 68)
        for s in all_stats:
            name = s["name"].replace("neu_det_yolov8n_", "").replace("ablation_", "")
            print(f"  {name:<28} {'PyTorch FP32':<14} {s['pt']['avg_ms']:>8.2f} "
                  f"{s['pt']['fps']:>7.1f} {s['pt']['peak_mb']:>9.1f}")
            print(f"  {name:<28} {'TensorRT FP16':<14} {s['trt']['avg_ms']:>8.2f} "
                  f"{s['trt']['fps']:>7.1f} {s['trt']['peak_mb']:>9.1f}")
            print(f"  {'  → TRT speedup':28s} {s['speedup']:.2f}x")
            print()
        print("=" * 68)
        print("\n→ Copy the table above into your paper's deployment section.")


if __name__ == "__main__":
    main()
