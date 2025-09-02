#!/usr/bin/env python3
"""CPU-only benchmark: transformer model inference time vs batch size (optimized, fixed)."""

from __future__ import annotations
import os, sys, time, gc
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ---------- CPU threading & env (set before importing torch) ----------
CPU_THREADS = min(os.cpu_count() or 4, 6)   # tweak if you want to sweep
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(CPU_THREADS))           # macOS Accelerate
os.environ.setdefault("ACCELERATE_MATMUL_MULTITHREADING", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Project imports path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import load_config

# ---------- Globals ----------
MODEL_DIR = Path("../models/trained_model")
MODEL_FILE = "final_model.pt2"   # torch.export artifact
BATCH_SIZES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Warmup/timing (amortize Python overhead)
WARMUP_CALLS = 20                # forwards per warmup block
CALLS_PER_MEASUREMENT = 32       # forwards per timed block
REPEATS = 5                      # number of timed blocks (for mean/std)

# Plotting
PLOT_DIR = MODEL_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Utils ----------
def _set_torch_threads(threads: int) -> None:
    torch.set_num_threads(int(threads))
    torch.set_num_interop_threads(max(1, int(threads // 2)))

def _safe_style():
    try:
        plt.style.use("science.mplstyle")
    except Exception:
        pass  # fall back to default style

def load_exported_module() -> Tuple[Any, Dict[str, Any], torch.device, object]:
    """Load exported program, config, device, and cache the callable once."""
    cfg_path = MODEL_DIR / "train_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    config = load_config(cfg_path)

    device = torch.device("cpu")
    _set_torch_threads(CPU_THREADS)

    prog_path = MODEL_DIR / MODEL_FILE
    if not prog_path.exists():
        raise FileNotFoundError(f"Model not found: {prog_path}")
    print(f"Loading exported model: {prog_path}")
    exported_prog = torch.export.load(str(prog_path))

    # IMPORTANT: materialize the callable once; DO NOT call .eval() (unsupported here)
    fn = exported_prog.module()
    return exported_prog, config, device, fn

def infer_seq_len_from_config(config: dict) -> int:
    try:
        return int(config["model_hyperparameters"]["max_sequence_length"])
    except Exception:
        raise KeyError("max_sequence_length not found in config['model_hyperparameters'].")

def create_batch(bs: int, seq_len: int, config: dict, device: torch.device) -> Dict[str, torch.Tensor]:
    data_spec = config["data_specification"]
    input_dim = int(len(data_spec["input_variables"]))
    global_dim = int(len(data_spec.get("global_variables", [])))

    batch = {
        "sequence": torch.randn(bs, seq_len, input_dim, device=device, dtype=torch.float32).contiguous(),
        "sequence_mask": torch.zeros(bs, seq_len, dtype=torch.bool, device=device),
    }
    if global_dim > 0:
        batch["global_features"] = torch.randn(bs, global_dim, device=device, dtype=torch.float32).contiguous()
    return batch

@torch.inference_mode()
def run_forwards(fn, batch: Dict[str, torch.Tensor], calls: int) -> None:
    for _ in range(calls):
        _ = fn(**batch)

@torch.inference_mode()
def benchmark_bs(fn, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    """Return (mean_ms_per_call, std_ms_per_call) using amortized timing."""
    # Warmup on this batch
    run_forwards(fn, batch, WARMUP_CALLS)

    # Timed blocks
    times_ms = []
    for _ in range(REPEATS):
        t0 = time.perf_counter_ns()
        run_forwards(fn, batch, CALLS_PER_MEASUREMENT)
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        times_ms.append(dt_ms / CALLS_PER_MEASUREMENT)  # per forward

    arr = np.asarray(times_ms, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

def plot_results(batch_sizes: List[int], mean_ms: List[float], std_ms: List[float], threads: int) -> None:
    _safe_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    per_sample_mean = [m / bs for m, bs in zip(mean_ms, batch_sizes)]
    per_sample_std = [s / bs for s, bs in zip(std_ms, batch_sizes)]
    ax1.errorbar(batch_sizes, per_sample_mean, yerr=per_sample_std, marker="o", markersize=6, capsize=4, linewidth=2)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Time per sample (ms)")
    ax1.set_title("Inference time per sample")
    ax1.grid(True, alpha=0.3)

    throughput = [bs / (m / 1000.0) for bs, m in zip(batch_sizes, mean_ms)]
    ax2.plot(batch_sizes, throughput, marker="s", markersize=6, linewidth=2)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Throughput (samples/s)")
    ax2.set_title("Inference throughput")
    ax2.grid(True, alpha=0.3)

    best = int(np.argmax(throughput))
    ax2.scatter(batch_sizes[best], throughput[best], s=160, marker="*", zorder=5)

    fig.suptitle(f"CPU Inference Benchmark (threads={threads})")
    fig.tight_layout()

    out_png = PLOT_DIR / "benchmark_cpu.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")

def main():
    print("=" * 60)
    print("CPU INFERENCE BENCHMARK (optimized)")
    print("=" * 60)
    print(f"CPU threads (intra-op): {CPU_THREADS}")
    print(f"Warmup calls: {WARMUP_CALLS} | Calls/measurement: {CALLS_PER_MEASUREMENT} | Repeats: {REPEATS}")

    # Reduce noise from GC and background allocations
    gc.disable()
    torch.set_grad_enabled(False)

    # Load once; cache callable
    exported_prog, config, device, fn = load_exported_module()
    seq_len = infer_seq_len_from_config(config)

    # Model summary
    mh = config.get("model_hyperparameters", {})
    print(f"\nModel config: d_model={mh.get('d_model')}  nhead={mh.get('nhead')}  layers={mh.get('num_encoder_layers')}")
    print(f"Sequence length (from config): {seq_len}")

    # Preallocate batches for all batch sizes (reused across warmup + repeats)
    batches: Dict[int, Dict[str, torch.Tensor]] = {
        bs: create_batch(bs, seq_len, config, device) for bs in BATCH_SIZES
    }

    # Benchmarks
    means, stds = [], []
    print("\nRunning benchmarks (amortized timing)...")
    print("-" * 60)
    print(f"{'Batch':>8} | {'Mean (ms)':>12} | {'Std (ms)':>10} | {'Throughput':>12}")
    print("-" * 60)

    for bs in BATCH_SIZES:
        mean_ms, std_ms = benchmark_bs(fn, batches[bs])
        means.append(mean_ms)
        stds.append(std_ms)
        thr = bs / (mean_ms / 1000.0)
        print(f"{bs:>8} | {mean_ms:>12.2f} | {std_ms:>10.2f} | {thr:>10.0f} s/s")

    print("-" * 60)

    if means:
        best_idx = int(np.argmax([bs / (m / 1000.0) for bs, m in zip(BATCH_SIZES, means)]))
        print(f"\nOptimal batch size: {BATCH_SIZES[best_idx]}")
        print(f"Peak throughput: {BATCH_SIZES[best_idx] / (means[best_idx] / 1000.0):.0f} samples/s")
        plot_results(BATCH_SIZES, means, stds, CPU_THREADS)

    gc.enable()
    print("\nDone.")

if __name__ == "__main__":
    main()
