#!/usr/bin/env python3
"""Benchmark model inference speed across batch sizes on CPU."""

import json
import time
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# Configuration
MODEL_NAME = "trained_model"
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
N_WARMUP = 10
N_RUNS = 50
DEVICE = torch.device("cpu")

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.extend([str(ROOT), str(ROOT / "src")])

from model import create_prediction_model
from utils import load_config


def load_model(model_dir: Path) -> Tuple[torch.nn.Module, float, str, Dict]:
    """
    Load model from checkpoint or exported format.
    
    Returns:
        Tuple of (model, size_mb, model_type, config)
    """
    # Load configuration
    config_path = next(model_dir.glob("*_config.json"), model_dir / "config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"No config found in {model_dir}")
    config = load_config(config_path)
    
    # Try loading exported model first
    exported_paths = sorted(model_dir.glob("*_exported*.pt2"))
    if exported_paths:
        # Prefer static export if available
        static = [p for p in exported_paths if "static" in p.name]
        export_path = static[0] if static else exported_paths[0]
        
        try:
            print(f"Loading exported model: {export_path.name}")
            exported_prog = torch.export.load(str(export_path))
            model = exported_prog.module()
            model.eval()
            size_mb = export_path.stat().st_size / (1024 * 1024)
            return model, size_mb, f"exported", config
        except Exception as e:
            print(f"Failed to load export ({e}), falling back to checkpoint")
    
    # Load from checkpoint
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Create model
    model = create_prediction_model(config, device=DEVICE, compile_model=False)
    
    # Load weights
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Remove compilation wrapper if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    return model, size_mb, "checkpoint", config


def get_model_dimensions(config: Dict) -> Tuple[int, int, int]:
    """
    Extract model dimensions from config.
    
    Returns:
        Tuple of (sequence_length, input_dim, global_dim)
    """
    data_spec = config.get("data_specification", {})
    model_params = config.get("model_hyperparameters", {})
    
    # Get dimensions
    input_vars = data_spec.get("input_variables", [])
    global_vars = data_spec.get("global_variables", [])
    seq_length = model_params.get("max_sequence_length", 64)
    
    return seq_length, len(input_vars), len(global_vars)


def create_batch(batch_size: int, seq_len: int, input_dim: int, global_dim: int) -> Dict[str, torch.Tensor]:
    """Create dummy batch for benchmarking."""
    # All tensors on CPU, float32
    batch = {
        "sequence": torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32, device=DEVICE)
    }
    
    if global_dim > 0:
        batch["global_features"] = torch.randn(batch_size, global_dim, dtype=torch.float32, device=DEVICE)
    
    # Add padding mask (no padding for benchmark)
    batch["sequence_mask"] = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=DEVICE)
    
    return batch


def benchmark_model(
    model: torch.nn.Module,
    seq_len: int,
    input_dim: int, 
    global_dim: int,
    batch_sizes: list = BATCH_SIZES
) -> Dict[str, list]:
    """
    Benchmark model inference across batch sizes.
    
    Returns:
        Dictionary with batch_size, latency_ms, throughput, and latency_per_sample
    """
    results = {
        "batch_size": [],
        "latency_ms": [],
        "latency_std": [],
        "throughput": [],
        "latency_per_sample": []
    }
    
    print(f"\nBenchmarking with {N_WARMUP} warmup and {N_RUNS} timed runs per batch size")
    print("-" * 60)
    
    # Force CPU and ensure no gradients
    model = model.to(DEVICE)
    model.eval()
    torch.set_grad_enabled(False)
    
    for bs in batch_sizes:
        batch = create_batch(bs, seq_len, input_dim, global_dim)
        
        # Warmup runs
        for _ in range(N_WARMUP):
            with torch.no_grad():
                _ = model(**batch)
        
        # Clear any caches
        if hasattr(torch, '_C') and hasattr(torch._C, '_jit_clear_class_registry'):
            torch._C._jit_clear_class_registry()
        
        # Timed runs
        times_ms = []
        for _ in range(N_RUNS):
            # Force CPU sync (though not needed for CPU)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(**batch)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed_ms)
        
        # Calculate statistics
        mean_ms = np.mean(times_ms)
        std_ms = np.std(times_ms)
        throughput = bs / (mean_ms / 1000)  # samples per second
        latency_per_sample = mean_ms / bs
        
        # Store results
        results["batch_size"].append(bs)
        results["latency_ms"].append(float(mean_ms))
        results["latency_std"].append(float(std_ms))
        results["throughput"].append(float(throughput))
        results["latency_per_sample"].append(float(latency_per_sample))
        
        print(f"BS={bs:3d}: {mean_ms:7.2f} ± {std_ms:5.2f} ms | "
              f"{throughput:8.1f} samples/s | "
              f"{latency_per_sample:6.2f} ms/sample")
    
    torch.set_grad_enabled(True)  # Re-enable gradients
    return results


def plot_results(results: Dict, model_info: Dict, save_dir: Path) -> None:
    """Create benchmark visualization plots."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    bs = results["batch_size"]
    
    # 1. Latency with error bars
    axes[0, 0].errorbar(bs, results["latency_ms"], yerr=results["latency_std"], 
                        marker='o', capsize=3, capthick=1)
    axes[0, 0].set_xlabel("Batch Size")
    axes[0, 0].set_ylabel("Latency (ms)")
    axes[0, 0].set_title("Inference Latency")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Throughput
    axes[0, 1].plot(bs, results["throughput"], "o-", color="green")
    axes[0, 1].set_xlabel("Batch Size")
    axes[0, 1].set_ylabel("Throughput (samples/s)")
    axes[0, 1].set_title("Inference Throughput")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Latency per sample
    axes[1, 0].plot(bs, results["latency_per_sample"], "o-", color="orange")
    axes[1, 0].set_xlabel("Batch Size")
    axes[1, 0].set_ylabel("Latency per Sample (ms)")
    axes[1, 0].set_title("Amortized Latency")
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Efficiency (samples per second per batch size)
    efficiency = [t/b for t, b in zip(results["throughput"], bs)]
    axes[1, 1].plot(bs, efficiency, "o-", color="purple")
    axes[1, 1].set_xlabel("Batch Size")
    axes[1, 1].set_ylabel("Efficiency (throughput/batch_size)")
    axes[1, 1].set_title("Batching Efficiency")
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add overall title
    model_type = model_info["type"]
    size_mb = model_info["size_mb"]
    plt.suptitle(f"CPU Inference Benchmark | {model_type} ({size_mb:.1f} MB)", fontsize=14)
    
    plt.tight_layout()
    output_path = save_dir / "inference_benchmark.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved benchmark plot: {output_path}")
    plt.close()


def save_results(results: Dict, model_info: Dict, save_dir: Path) -> None:
    """Save benchmark results to JSON."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata to results
    full_results = {
        "model": model_info,
        "benchmark_config": {
            "warmup_runs": N_WARMUP,
            "timed_runs": N_RUNS,
            "device": str(DEVICE),
        },
        "results": results,
        "summary": {
            "best_throughput": max(results["throughput"]),
            "best_throughput_batch": results["batch_size"][np.argmax(results["throughput"])],
            "min_latency": min(results["latency_ms"]),
            "min_latency_batch": results["batch_size"][np.argmin(results["latency_ms"])],
        }
    }
    
    output_path = save_dir / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"✓ Saved results: {output_path}")


def main():
    """Main benchmark pipeline."""
    print("=" * 60)
    print("CPU Inference Benchmark")
    print("=" * 60)
    
    # Setup paths
    model_dir = ROOT / "models" / MODEL_NAME
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    output_dir = model_dir / "benchmark"
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model, size_mb, model_type, config = load_model(model_dir)
    seq_len, input_dim, global_dim = get_model_dimensions(config)
    
    model_info = {
        "name": MODEL_NAME,
        "type": model_type,
        "size_mb": size_mb,
        "seq_length": seq_len,
        "input_dim": input_dim,
        "global_dim": global_dim,
    }
    
    print(f"Model type: {model_type} ({size_mb:.1f} MB)")
    print(f"Dimensions: seq={seq_len}, input={input_dim}, global={global_dim}")
    
    # Run benchmark
    results = benchmark_model(model, seq_len, input_dim, global_dim, BATCH_SIZES)
    
    # Save and plot results
    save_results(results, model_info, output_dir)
    plot_results(results, model_info, output_dir)
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()