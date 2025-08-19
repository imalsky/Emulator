#!/usr/bin/env python3
"""Benchmark transformer model inference time vs batch size."""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

sys.path.append("../src")

from pathlib import Path
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from model import create_prediction_model
from utils import load_config

# Use science style if available
try:
    plt.style.use('science.mplstyle')
except:
    pass

MODEL_DIR = Path("../models/trained_model")
BATCH_SIZES = [1, 2, 4, 8, 16]
N_WARMUP = 10  # Warmup iterations
N_BENCHMARK = 50  # Benchmark iterations
SEQ_LENGTH = 100  # Fixed sequence length for benchmarking


def load_model():
    """Load trained model and config."""
    # Try different config names
    config_paths = [
        MODEL_DIR / "train_config.json",
        MODEL_DIR / "best_config.json",
        MODEL_DIR / "normalize_config.json"
    ]

    config = None
    for config_path in config_paths:
        if config_path.exists():
            config = load_config(config_path)
            break

    if config is None:
        raise FileNotFoundError(f"No config file found in {MODEL_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_prediction_model(config, device, compile_model=False)
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Handle compiled model state dict
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    return model, config, device


def create_dummy_batch(batch_size: int, config: dict, device: torch.device) -> dict:
    """Create dummy input batch for benchmarking."""
    data_spec = config["data_specification"]
    input_dim = len(data_spec["input_variables"])
    global_dim = len(data_spec.get("global_variables", []))

    # Create random inputs
    sequence = torch.randn(batch_size, SEQ_LENGTH, input_dim, device=device)

    batch = {"sequence": sequence}

    if global_dim > 0:
        batch["global_features"] = torch.randn(batch_size, global_dim, device=device)

    # No padding for benchmark (all positions valid)
    batch["sequence_mask"] = torch.zeros(batch_size, SEQ_LENGTH, dtype=torch.bool, device=device)

    return batch


def benchmark_batch_size(model: torch.nn.Module, batch_size: int,
                         config: dict, device: torch.device) -> Tuple[float, float]:
    """Benchmark inference for a specific batch size.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Create dummy batch
    batch = create_dummy_batch(batch_size, config, device)

    # Warmup
    for _ in range(N_WARMUP):
        with torch.no_grad():
            _ = model(**batch)

    # Synchronize before timing (important for GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(N_BENCHMARK):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            _ = model(**batch)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)
    return np.mean(times), np.std(times)


def plot_results(batch_sizes: List[int], mean_times: List[float],
                 std_times: List[float], device_name: str):
    """Plot benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    time_per_sample = []
    for i, batch_size in enumerate(batch_sizes):
        time_per_sample.append(mean_times[i] / batch_size)
    print(time_per_sample)

    # Plot 1: Inference time vs batch size
    ax1.plot(batch_sizes, time_per_sample)
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Inference Time per sample (ms)', fontsize=12)
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add text annotations for key points
    for i in [0, len(batch_sizes) // 2, -1]:
        ax1.annotate(f'{mean_times[i]:.1f} ms',
                     xy=(batch_sizes[i], mean_times[i]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, alpha=0.7)

    # Plot 2: Throughput (samples/second)
    throughput = [bs / (mt / 1000) for bs, mt in zip(batch_sizes, mean_times)]
    ax2.plot(batch_sizes, throughput, marker='s', markersize=8,
             linewidth=2, color='green')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Throughput (samples/second)', fontsize=12)
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)

    # Add optimal batch size annotation
    max_throughput_idx = np.argmax(throughput)
    ax2.annotate(f'Peak: {throughput[max_throughput_idx]:.0f} samples/s\n'
                 f'@ batch_size={batch_sizes[max_throughput_idx]}',
                 xy=(batch_sizes[max_throughput_idx], throughput[max_throughput_idx]),
                 xytext=(20, -20), textcoords='offset points',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    # Save plot
    save_path = MODEL_DIR / "plots" / "inference_benchmark.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot: {save_path}")


def main():
    """Main benchmark function."""
    print("=" * 60)
    print("TRANSFORMER INFERENCE BENCHMARK")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, config, device = load_model()
    device_name = device.type.upper()
    if device.type == 'cuda':
        device_name += f" ({torch.cuda.get_device_name()})"

    # Model info
    n_params = sum(p.numel() for p in model.parameters())
    model_config = config["model_hyperparameters"]
    print(f"\nModel Configuration:")
    print(f"  Parameters: {n_params:,}")
    print(f"  d_model: {model_config.get('d_model', 256)}")
    print(f"  nhead: {model_config.get('nhead', 8)}")
    print(f"  layers: {model_config.get('num_encoder_layers', 6)}")
    print(f"  Device: {device_name}")
    print(f"\nBenchmark Settings:")
    print(f"  Sequence length: {SEQ_LENGTH}")
    print(f"  Warmup iterations: {N_WARMUP}")
    print(f"  Benchmark iterations: {N_BENCHMARK}")

    # Run benchmarks
    mean_times = []
    std_times = []

    print("\nRunning benchmarks...")
    print("-" * 40)
    print(f"{'Batch Size':>12} | {'Mean (ms)':>12} | {'Std (ms)':>10} | {'Throughput':>12}")
    print("-" * 40)

    for batch_size in BATCH_SIZES:
        try:
            mean_time, std_time = benchmark_batch_size(model, batch_size, config, device)
            mean_times.append(mean_time)
            std_times.append(std_time)
            throughput = batch_size / (mean_time / 1000)

            print(f"{batch_size:>12} | {mean_time:>12.2f} | {std_time:>10.2f} | "
                  f"{throughput:>12.0f} samples/s")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{batch_size:>12} | {'OOM':>12} | {'---':>10} | {'---':>12}")
                break
            else:
                raise

    print("-" * 40)

    # Trim batch_sizes to match successful runs
    successful_batch_sizes = BATCH_SIZES[:len(mean_times)]

    if mean_times:
        # Calculate statistics
        print("\nSummary Statistics:")
        print(f"  Fastest: {min(mean_times):.2f} ms @ batch_size={successful_batch_sizes[np.argmin(mean_times)]}")
        print(f"  Slowest: {max(mean_times):.2f} ms @ batch_size={successful_batch_sizes[np.argmax(mean_times)]}")

        throughputs = [bs / (mt / 1000) for bs, mt in zip(successful_batch_sizes, mean_times)]
        best_throughput_idx = np.argmax(throughputs)
        print(f"  Best throughput: {throughputs[best_throughput_idx]:.0f} samples/s "
              f"@ batch_size={successful_batch_sizes[best_throughput_idx]}")

        # Calculate efficiency
        single_batch_time = mean_times[0] if successful_batch_sizes[0] == 1 else None
        if single_batch_time:
            print("\nBatch Efficiency (vs single sample):")
            for bs, mt in zip(successful_batch_sizes[1:], mean_times[1:]):
                efficiency = (single_batch_time * bs) / mt
                print(f"  Batch {bs:3}: {efficiency:.1%} efficient")

        # Plot results
        plot_results(successful_batch_sizes, mean_times, std_times, device_name)
    else:
        print("\nNo successful benchmarks completed.")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()