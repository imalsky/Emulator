# File 3: benchmark_inference_time.py

import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random

sys.path.append("../src")
from dataset import create_dataset, create_collate_fn  # noqa: E402

# Relative path to the model directory (assuming script is in 'testing/' and 'models/' is sibling)
model_dir = (
    Path(__file__).parent.parent / "models" / "trained_model_picaso_transformer_v2"
)  # Change 'trained_model_picaso_transformer' if needed
processed_dir = (
    Path(__file__).parent.parent / "data" / "processed"
)  # Assuming 'data/' is sibling


def load_model_and_data(model_dir: Path, processed_dir: Path):
    # Load config to get necessary params
    with open(model_dir / "run_config.json") as f:
        cfg = json.load(f)
    with open(processed_dir / "normalization_metadata.json") as f:
        norm = json.load(f)  # Not used here but for dataset creation
    with open(model_dir / "dataset_splits.json") as f:
        splits = json.load(f)

    device = torch.device("cpu")  # Force CPU

    # Load best JIT model
    jit_files = list(model_dir.glob("best_model_epoch_*_jit.pt"))
    if not jit_files:
        raise FileNotFoundError("No JIT model files found.")
    best_jit_path = max(
        jit_files, key=lambda p: int(p.stem.split("_epoch_")[1].split("_")[0])
    )
    model = torch.jit.load(best_jit_path, map_location=device)
    model = torch.jit.optimize_for_inference(model)
    model.eval()
    model_size_mb = best_jit_path.stat().st_size / (1024 * 1024)

    # Load test dataset
    from dataset import (
        create_dataset,
        create_collate_fn,
    )  # Assuming dataset.py is in src or path

    test_shards = sorted(processed_dir.glob("test_shard_*.npz"))
    test_indices = list(
        range(sum(np.load(s)["sequence_inputs"].shape[0] for s in test_shards))
    )
    ds = create_dataset(test_shards, cfg, test_indices)
    pad_val = float(cfg["data_specification"].get("padding_value", -9999.0))
    collate = create_collate_fn(pad_val)

    return model, ds, collate, device, model_size_mb


def benchmark(model, ds, collate, device, batches=(1, 2, 4, 8, 16, 32, 64, 128)):
    res = {"batch_size": [], "mean_ms": [], "throughput": [], "lat_per_sample_ms": []}

    for bs in batches:
        # Sample indices, collate real batch
        idx = random.sample(range(len(ds)), min(bs, len(ds)))
        samples = [ds[i] for i in idx]
        batch = collate(samples)
        if batch is None:
            continue

        inp, masks, *_ = batch
        seq = inp["sequence"].to(device)
        seq_mask = masks["sequence"].to(device)
        gbl = inp.get("global")
        if gbl is not None:
            gbl = gbl.to(device)

        # Warm-up
        with torch.inference_mode():
            for _ in range(5):
                model(seq, gbl, seq_mask)

        # Timed runs (in ms)
        times_ms = []
        for _ in range(20):
            t0 = time.perf_counter()
            with torch.inference_mode():
                model(seq, gbl, seq_mask)
            times_ms.append((time.perf_counter() - t0) * 1e3)  # ms

        mean_ms = float(np.mean(times_ms))
        n = len(samples)
        res["batch_size"].append(n)
        res["mean_ms"].append(mean_ms)
        res["throughput"].append(n / (mean_ms / 1e3))  # samples/sec
        res["lat_per_sample_ms"].append(mean_ms / n)

    return res


def plot_results(res, device, size_mb, plots_dir: Path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.plot(res["batch_size"], res["mean_ms"], "o-")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Batch latency")
    ax1.grid(alpha=0.3)

    ax2.plot(res["batch_size"], res["lat_per_sample_ms"], "o-")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Latency / sample (ms)")
    # ax2.set_yscale('log')
    ax2.set_title("Per-sample latency")
    ax2.grid(alpha=0.3)

    ax3.plot(res["batch_size"], res["throughput"], "o-")
    ax3.set_xlabel("Batch size")
    ax3.set_ylabel("Throughput (samples/s)")
    ax3.set_title("Throughput")
    ax3.grid(alpha=0.3)

    plt.suptitle(f"Device: {device} — JIT model {size_mb:.1f} MB")
    plt.tight_layout()
    plt.savefig(plots_dir / "inference_time_benchmark.png", dpi=150)


if __name__ == "__main__":
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model, ds, collate, device, size_mb = load_model_and_data(model_dir, processed_dir)
    res = benchmark(model, ds, collate, device)
    with open(plots_dir / "timing.json", "w") as f:
        json.dump(res, f, indent=2)
    plot_results(res, device, size_mb, plots_dir)
    print(f"Done. Results saved to {plots_dir}")
