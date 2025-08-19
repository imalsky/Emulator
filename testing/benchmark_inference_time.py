#!/usr/bin/env python3
"""Benchmark inference speed on CPU/GPU."""

import json
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# Configuration
MODEL_DIR = Path("../models/trained_model")
BATCH_SIZES = [1, 4, 16]
N_WARMUP = 5
N_RUNS = 20

def load_model(model_dir):
    """Load model from checkpoint."""
    import sys
    sys.path.append("../src")
    from model import create_prediction_model
    from utils import load_config
    
    config = load_config(model_dir / "train_config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_prediction_model(config, device, compile_model=False)
    
    # Load weights
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, config, device

def benchmark(model, config, device):
    """Run benchmark across batch sizes."""
    data_spec = config["data_specification"]
    model_params = config["model_hyperparameters"]
    
    seq_len = model_params["max_sequence_length"]
    input_dim = len(data_spec["input_variables"])
    global_dim = len(data_spec.get("global_variables", []))
    
    results = {"batch_size": [], "latency_ms": [], "throughput": []}
    
    for bs in BATCH_SIZES:
        # Create dummy batch
        inputs = {
            "sequence": torch.randn(bs, seq_len, input_dim, device=device),
            "sequence_mask": torch.zeros(bs, seq_len, dtype=torch.bool, device=device)
        }
        if global_dim > 0:
            inputs["global_features"] = torch.randn(bs, global_dim, device=device)
        
        # Warmup
        for _ in range(N_WARMUP):
            with torch.no_grad():
                _ = model(**inputs)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == "cuda" else None
        times = []
        for _ in range(N_RUNS):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs)
            torch.cuda.synchronize() if device.type == "cuda" else None
            times.append((time.perf_counter() - start) * 1000)
        
        mean_ms = sum(times) / len(times)
        throughput = bs / (mean_ms / 1000)
        
        results["batch_size"].append(bs)
        results["latency_ms"].append(mean_ms)
        results["throughput"].append(throughput)
        
        print(f"BS={bs:3d}: {mean_ms:7.2f} ms, {throughput:8.1f} samples/s")
    
    return results

def plot_results(results, device, save_dir):
    """Plot benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(results["batch_size"], results["latency_ms"], "o-")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Inference Latency ({device.type.upper()})")
    
    ax2.plot(results["batch_size"], results["throughput"], "o-", color="green")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Throughput (samples/s)")
    ax2.set_xscale("log", base=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"Inference Throughput ({device.type.upper()})")
    
    plt.tight_layout()
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "inference_benchmark.png", dpi=150)
    print(f"Saved plot: {save_dir / 'inference_benchmark.png'}")

if __name__ == "__main__":
    model, config, device = load_model(MODEL_DIR)
    print(f"Benchmarking on {device}")
    
    results = benchmark(model, config, device)
    
    # Save results
    with open(MODEL_DIR / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    plot_results(results, device, MODEL_DIR / "plots")