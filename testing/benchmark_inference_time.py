#!/usr/bin/env python3
"""CPU-only benchmark: transformer model inference time vs batch size."""

# ======================= Global CPU threading control =======================
# Set how many CPU threads to use across libraries (Torch, MKL/OpenBLAS/NumExpr).
# Increase this to use more CPU cores. Common choice: CPU_THREADS = os.cpu_count()
import os as _os
CPU_THREADS = min(_os.cpu_count(), 4)   # e.g., set to 8 for a fixed value
# ===========================================================================

# ---- Force CPU and configure threading BEFORE importing numpy/torch ----
_os.environ["CUDA_VISIBLE_DEVICES"] = ""                # hide all CUDA devices
_os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"        # ensure CPU fallback on macOS
_os.environ["OMP_DYNAMIC"] = "FALSE"
_os.environ["MKL_DYNAMIC"] = "FALSE"
_os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
_os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
_os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
_os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid Intel MKL warnings

import sys
sys.path.append("../src")

from pathlib import Path
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import create_prediction_model
from utils import load_config

# Use science style if available
try:
    plt.style.use('science.mplstyle')
except Exception:
    pass

MODEL_DIR = Path("../models/trained_model")
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
N_WARMUP = 10          # Warmup iterations
N_BENCHMARK = 50       # Benchmark iterations
SEQ_LENGTH = 100       # Fixed sequence length for benchmarking


def load_model() -> Tuple[torch.nn.Module, Dict[str, Any], torch.device]:
    """Load trained model and config (CPU-only)."""
    # Try different config names
    config_paths = [
        MODEL_DIR / "train_config.json",
        MODEL_DIR / "best_config.json",
        MODEL_DIR / "normalize_config.json",
    ]

    config = None
    for config_path in config_paths:
        if config_path.exists():
            config = load_config(config_path)
            break

    if config is None:
        raise FileNotFoundError(f"No config file found in {MODEL_DIR}")

    device = torch.device("cpu")

    # Configure PyTorch threading
    try:
        torch.set_num_threads(CPU_THREADS)        # intra-op parallelism
        torch.set_num_interop_threads(1)          # inter-op; keep low for stability
        torch.set_flush_denormal(True)
    except Exception:
        pass

    # Load model
    model = create_prediction_model(config, device, compile_model=False)
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Handle compiled model state dict
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Optional: try torch.compile on CPU (safe fallback if unsupported)
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(
                model, backend="inductor", mode="max-autotune",
                fullgraph=False, dynamic=False
            )
        except Exception:
            pass

    return model, config, device


def create_dummy_batch(batch_size: int, config: dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """Create dummy input batch for benchmarking (CPU tensors)."""
    data_spec = config["data_specification"]
    input_dim = len(data_spec["input_variables"])
    global_dim = len(data_spec.get("global_variables", []))

    sequence = torch.randn(batch_size, SEQ_LENGTH, input_dim, device=device)
    batch: Dict[str, torch.Tensor] = {"sequence": sequence}

    if global_dim > 0:
        batch["global_features"] = torch.randn(batch_size, global_dim, device=device)

    # No padding for benchmark (all positions valid)
    batch["sequence_mask"] = torch.zeros(batch_size, SEQ_LENGTH, dtype=torch.bool, device=device)

    return batch


def benchmark_batch_size(model: torch.nn.Module, batch_size: int,
                         config: dict, device: torch.device) -> Tuple[float, float]:
    """CPU timing via perf_counter. Returns (mean_ms, std_ms)."""
    batch = create_dummy_batch(batch_size, config, device)

    # Warmup
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            _ = model(**batch)

    # Benchmark
    times_ms: List[float] = []
    with torch.inference_mode():
        for _ in range(N_BENCHMARK):
            t0 = time.perf_counter()
            _ = model(**batch)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def plot_results(batch_sizes: List[int], mean_times: List[float],
                 std_times: List[float], device_name: str):
    """Plot results with per-sample error bars and throughput."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    per_sample_mean = [m / bs for m, bs in zip(mean_times, batch_sizes)]
    per_sample_std  = [s / bs for s, bs in zip(std_times, batch_sizes)]

    # Plot 1: Inference time per sample (with error bars)
    ax1.errorbar(batch_sizes, per_sample_mean, yerr=per_sample_std,
                 marker='o', markersize=7, capsize=5, linewidth=2,
                 label='Mean ± Std per sample')
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Inference Time per Sample (ms)', fontsize=12)
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Annotate a few points
    idxs = [0, len(batch_sizes) // 2, len(batch_sizes) - 1]
    for i in idxs:
        ax1.annotate(f'{per_sample_mean[i]:.3f} ms',
                     xy=(batch_sizes[i], per_sample_mean[i]),
                     xytext=(8, 8), textcoords='offset points',
                     fontsize=9, alpha=0.8)

    # Plot 2: Throughput (samples/second)
    throughput = [bs / (mt / 1000.0) for bs, mt in zip(batch_sizes, mean_times)]
    ax2.plot(batch_sizes, throughput, marker='s', markersize=7, linewidth=2)
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Throughput (samples/second)', fontsize=12)
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)

    # Add optimal batch size annotation
    max_throughput_idx = int(np.argmax(throughput))
    ax2.annotate(f'Peak: {throughput[max_throughput_idx]:.0f} samples/s\n'
                 f'@ batch_size={batch_sizes[max_throughput_idx]}',
                 xy=(batch_sizes[max_throughput_idx], throughput[max_throughput_idx]),
                 xytext=(20, -20), textcoords='offset points',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.suptitle(f'Inference Benchmark (CPU, threads={CPU_THREADS})', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save plot
    save_path = MODEL_DIR / "plots" / "inference_benchmark_cpu.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot: {save_path}")
    plt.close(fig)


def main():
    print("=" * 60)
    print("TRANSFORMER INFERENCE BENCHMARK — CPU ONLY")
    print("=" * 60)
    print(f"CPU threads requested: {CPU_THREADS}")

    # Load model
    print("\nLoading model...")
    model, config, device = load_model()
    device_name = "CPU"

    # Model info
    n_params = sum(p.numel() for p in model.parameters())
    model_config = config.get("model_hyperparameters", {})
    print(f"\nModel Configuration:")
    print(f"  Parameters: {n_params:,}")
    print(f"  d_model: {model_config.get('d_model', 'n/a')}")
    print(f"  nhead: {model_config.get('nhead', 'n/a')}")
    print(f"  layers: {model_config.get('num_encoder_layers', model_config.get('num_layers', 'n/a'))}")
    print(f"  Device: {device_name}")

    print(f"\nBenchmark Settings:")
    print(f"  Sequence length: {SEQ_LENGTH}")
    print(f"  Warmup iterations: {N_WARMUP}")
    print(f"  Benchmark iterations: {N_BENCHMARK}")

    # Run benchmarks
    mean_times: List[float] = []
    std_times: List[float] = []

    print("\nRunning benchmarks...")
    print("-" * 60)
    print(f"{'Batch Size':>12} | {'Mean (ms)':>12} | {'Std (ms)':>10} | {'Throughput':>12}")
    print("-" * 60)

    for batch_size in BATCH_SIZES:
        try:
            mean_time, std_time = benchmark_batch_size(model, batch_size, config, device)
            mean_times.append(mean_time)
            std_times.append(std_time)
            throughput = batch_size / (mean_time / 1000.0)

            print(f"{batch_size:>12} | {mean_time:>12.2f} | {std_time:>10.2f} | "
                  f"{throughput:>12.0f} samples/s")

        except RuntimeError as e:
            print(f"{batch_size:>12} | {'ERROR':>12} | {'---':>10} | {'---':>12}")
            raise

    print("-" * 60)

    # Trim batch_sizes to match successful runs
    successful_batch_sizes = BATCH_SIZES[:len(mean_times)]

    if mean_times:
        # Calculate statistics
        print("\nSummary Statistics:")
        fastest_idx = int(np.argmin(mean_times))
        slowest_idx = int(np.argmax(mean_times))
        print(f"  Fastest: {mean_times[fastest_idx]:.2f} ms @ batch_size={successful_batch_sizes[fastest_idx]}")
        print(f"  Slowest: {mean_times[slowest_idx]:.2f} ms @ batch_size={successful_batch_sizes[slowest_idx]}")

        throughputs = [bs / (mt / 1000.0) for bs, mt in zip(successful_batch_sizes, mean_times)]
        best_throughput_idx = int(np.argmax(throughputs))
        print(f"  Best throughput: {throughputs[best_throughput_idx]:.0f} samples/s "
              f"@ batch_size={successful_batch_sizes[best_throughput_idx]}")

        # Efficiency vs single-sample
        if successful_batch_sizes and successful_batch_sizes[0] == 1:
            single_ms = mean_times[0]
            print("\nBatch Efficiency (vs single sample):")
            for bs, mt in zip(successful_batch_sizes[1:], mean_times[1:]):
                eff = (single_ms * bs) / mt
                print(f"  Batch {bs:3}: {eff:.1%} efficient")

        # Plot results
        plot_results(successful_batch_sizes, mean_times, std_times, device_name)
    else:
        print("\nNo successful benchmarks completed.")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
