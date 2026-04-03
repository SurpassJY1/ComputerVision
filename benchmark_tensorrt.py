#!/usr/bin/env python3
"""
TensorRT inference benchmark: compare PyTorch vs TensorRT FP16 speed.

Must run on NVIDIA GPU machine (Windows 4080) with TensorRT installed.

Usage (step 1 — export engine first if not done):
  python export_tensorrt.py --weights path/to/best.pt --imgsz 640 --format engine --half

Usage (step 2 — benchmark):
  python benchmark_tensorrt.py --pt path/to/best.pt --engine path/to/best.engine --imgsz 640

Output (example):
  [PyTorch  FP32]  avg=1.23ms  fps=813   mem=xxx MB
  [TensorRT FP16]  avg=0.45ms  fps=2222  mem=xxx MB
  Speedup: 2.73x
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from ultralytics import YOLO


WARMUP_RUNS  = 50   # discard first N runs to let GPU reach steady state
MEASURE_RUNS = 200  # number of timed inference runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT inference speed")
    p.add_argument("--pt",     type=str, required=True,  help="PyTorch weights (.pt)")
    p.add_argument("--engine", type=str, required=True,  help="TensorRT engine file (.engine)")
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--device", type=str, default="0",    help="GPU device id")
    p.add_argument("--batch",  type=int, default=1,      help="Inference batch size (paper usually uses 1)")
    return p.parse_args()


def gpu_mem_mb() -> float:
    """Return current GPU allocated memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def benchmark_model(model: YOLO, imgsz: int, batch: int, device: str, label: str) -> dict:
    """
    Run warmup + timed inference and return latency/FPS stats.
    Ultralytics model.predict() with a synthetic black image.
    """
    # Build a synthetic input: list of batch dummy images
    dummy = [torch.zeros(imgsz, imgsz, 3).numpy()] * batch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(WARMUP_RUNS):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    torch.cuda.synchronize()

    # Timed runs
    latencies: list[float] = []
    for _ in range(MEASURE_RUNS):
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    avg_ms  = sum(latencies) / len(latencies)
    min_ms  = min(latencies)
    max_ms  = max(latencies)
    fps     = 1000.0 / avg_ms * batch
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    print(f"\n[{label}]")
    print(f"  Avg latency : {avg_ms:.2f} ms   (min={min_ms:.2f}  max={max_ms:.2f})")
    print(f"  Throughput  : {fps:.1f} FPS  (batch={batch})")
    print(f"  Peak GPU mem: {peak_mb:.1f} MB")

    return {"label": label, "avg_ms": avg_ms, "fps": fps, "peak_mb": peak_mb}


def main() -> None:
    args = parse_args()

    pt_path     = Path(args.pt)
    engine_path = Path(args.engine)

    if not pt_path.is_file():
        raise FileNotFoundError(f"PyTorch weights not found: {pt_path}")
    if not engine_path.is_file():
        raise FileNotFoundError(
            f"TensorRT engine not found: {engine_path}\n"
            "Export first:  python export_tensorrt.py --weights {pt_path} --imgsz {args.imgsz} --format engine --half"
        )

    print(f"\nBenchmark settings: imgsz={args.imgsz}  batch={args.batch}  "
          f"warmup={WARMUP_RUNS}  runs={MEASURE_RUNS}  device={args.device}")
    print(f"  PyTorch model  : {pt_path}")
    print(f"  TensorRT engine: {engine_path}")

    # --- PyTorch FP32 ---
    pt_model = YOLO(str(pt_path))
    pt_stats  = benchmark_model(pt_model, args.imgsz, args.batch, args.device, "PyTorch  FP32")
    del pt_model
    torch.cuda.empty_cache()

    # --- TensorRT FP16 ---
    trt_model = YOLO(str(engine_path))
    trt_stats  = benchmark_model(trt_model, args.imgsz, args.batch, args.device, "TensorRT FP16")
    del trt_model
    torch.cuda.empty_cache()

    # --- Summary ---
    speedup = pt_stats["avg_ms"] / trt_stats["avg_ms"]
    print("\n" + "=" * 52)
    print("  SUMMARY")
    print("=" * 52)
    print(f"  {'':20s} {'Latency(ms)':>12} {'FPS':>8} {'PeakMem(MB)':>12}")
    print(f"  {'PyTorch FP32':20s} {pt_stats['avg_ms']:>12.2f} {pt_stats['fps']:>8.1f} {pt_stats['peak_mb']:>12.1f}")
    print(f"  {'TensorRT FP16':20s} {trt_stats['avg_ms']:>12.2f} {trt_stats['fps']:>8.1f} {trt_stats['peak_mb']:>12.1f}")
    print(f"\n  TensorRT speedup: {speedup:.2f}x  ({(speedup-1)*100:.1f}% faster)")
    print("=" * 52)
    print("\n→ Copy these numbers into your paper's deployment table.")


if __name__ == "__main__":
    main()
