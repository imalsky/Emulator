#!/usr/bin/env python3
"""Benchmark model inference speed across batch sizes (CPU)."""

import json
import time
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


# --------------------------- user-adjustables --------------------------- #
MODEL_NAME = "trained_model"      # folder under ./models/
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
N_WARMUP = 5
N_RUNS = 20
# ---------------------------------------------------------------------- #


def _try_set_log_xscale(ax, base=2):
    """Matplotlib changed the kw name over versions; be compatible."""
    try:
        ax.set_xscale("log", base=base)
    except TypeError:
        ax.set_xscale("log", basex=base)


def _load_config(model_dir: Path) -> dict:
    """Load any *config.json in the model dir, else {}."""
    # Prefer "*_config.json"; fall back to "config.json"
    cfg_candidates = sorted(list(model_dir.glob("*_config.json"))) or sorted(list(model_dir.glob("config.json")))
    if not cfg_candidates:
        return {}
    with open(cfg_candidates[0], "r") as f:
        return json.load(f)


def _infer_dims_from_config(cfg: dict) -> tuple[int, int, int]:
    """Infer (seq_len, input_dim, global_dim) with conservative fallbacks."""
    seq_len = None
    input_dim = None
    global_dim = 0

    # Try common places
    # data section
    data = cfg.get("data", {})
    if isinstance(data, dict):
        # sequence length could be named variously
        seq_len = (
            data.get("sequence_length")
            or data.get("max_sequence_length")
            or data.get("max_seq_len")
            or data.get("fixed_sequence_length")
        )
        # input/global dims can come from variable lists
        if isinstance(data.get("input_variables"), list):
            input_dim = len(data["input_variables"])
        if isinstance(data.get("global_variables"), list):
            global_dim = len(data["global_variables"])

    # model section (alternate names)
    model = cfg.get("model", {})
    if isinstance(model, dict):
        seq_len = seq_len or model.get("sequence_length")
        input_dim = input_dim or model.get("input_dim")
        global_dim = global_dim or model.get("global_dim")

    # conservative defaults if unknown
    seq_len = int(seq_len) if seq_len is not None else 64
    input_dim = int(input_dim) if input_dim is not None else 29
    global_dim = int(global_dim) if global_dim is not None else 5

    return seq_len, input_dim, global_dim


def _import_model_ctor():
    """
    Support both create_prediction_model(...) and create_model(...) and both
    project layouts: repository root and ./src/.
    """
    root = Path(__file__).parent.parent
    sys.path.extend([str(root), str(root / "src")])
    try:
        from model import create_prediction_model  # type: ignore
        return create_prediction_model
    except Exception:
        from model import create_model  # type: ignore
        return create_model


def load_model(model_dir: Path):
    """
    Load an exported model if possible; fall back to checkpoint when
    export schema version mismatches or no export exists.
    Returns: (callable model on CPU, size_mb, model_type_str, config_dict)
    """
    device = torch.device("cpu")
    cfg = _load_config(model_dir)

    # Prefer exported models, but handle schema-version mismatches gracefully.
    exported = sorted(model_dir.glob("*_exported*.pt*"))
    if exported:
        # prefer "static" exported artifact if present
        static = [p for p in exported if "static" in p.name]
        model_path = static[0] if static else exported[-1]
        try:
            from torch.export import load as export_load
            exported_prog = export_load(str(model_path))
            model = exported_prog.module()
            model.eval()
            size_mb = model_path.stat().st_size / (1024 * 1024)
            return model, size_mb, f"exported ({model_path.name})", cfg
        except RuntimeError as e:
            # Typical when PyTorch's current schema version != artifact's.
            if "Serialized version" in str(e):
                print(f"Warning: export schema mismatch for {model_path.name}; falling back to checkpoint.")
            else:
                print(f"Warning: failed to load export ({e}); falling back to checkpoint.")
        except Exception as e:
            print(f"Warning: failed to load export ({e}); falling back to checkpoint.")

    # Fallback: checkpoint + constructor
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No loadable export and checkpoint not found: {ckpt_path}.\n"
            f"Available files: {[p.name for p in model_dir.iterdir()]}"
        )

    ctor = _import_model_ctor()
    model = ctor(cfg, device=device, compile_model=False)  # ctor signature in your repo
    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt.get("state_dict", ckpt)
    # Handle compiled models that wrap keys under _orig_mod.
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    return model, size_mb, "checkpoint", cfg


def create_dummy_batch(batch_size: int, seq_len: int, input_dim: int, global_dim: int):
    """Create dummy inputs with shapes inferred from config (float32 on CPU)."""
    sequence = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32)
    inputs = {"sequence": sequence}
    if global_dim and global_dim > 0:
        inputs["global_features"] = torch.randn(batch_size, global_dim, dtype=torch.float32)
    return inputs


def benchmark_model(model, seq_len: int, input_dim: int, global_dim: int, batch_sizes=BATCH_SIZES):
    """Benchmark model across batch sizes with warmup and mean±std."""
    results = {"batch_size": [], "latency_ms": [], "throughput": []}

    # Warmup once at the largest batch to JIT/initialize paths if any
    try:
        with torch.inference_mode():
            _ = model(**create_dummy_batch(max(batch_sizes), seq_len, input_dim, global_dim))
    except TypeError:
        # Some exported modules may only accept positional args; fall back to kwargs per BS below.
        pass

    for bs in batch_sizes:
        inputs = create_dummy_batch(bs, seq_len, input_dim, global_dim)

        # Warmup
        for _ in range(N_WARMUP):
            with torch.inference_mode():
                _ = model(**inputs)

        # Timed runs
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model(**inputs)
            times.append((time.perf_counter() - t0) * 1000.0)  # ms

        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))
        thr = bs / (mean_ms / 1000.0) if mean_ms > 0 else float("inf")

        results["batch_size"].append(bs)
        results["latency_ms"].append(mean_ms)
        results["throughput"].append(thr)

        print(f"BS={bs:3d}: {mean_ms:7.2f} ± {std_ms:5.2f} ms, {thr:8.1f} samples/s")

    return results


def plot_results(results, size_mb, model_type, save_dir: Path):
    """Create benchmark visualization; robust log-x scaling."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Latency
    axes[0].plot(results["batch_size"], results["latency_ms"], "o-")
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Inference Latency")
    axes[0].grid(True, alpha=0.3)
    _try_set_log_xscale(axes[0], base=2)

    # Throughput
    axes[1].plot(results["batch_size"], results["throughput"], "o-")
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Throughput (samples/s)")
    axes[1].set_title("Inference Throughput")
    axes[1].grid(True, alpha=0.3)
    _try_set_log_xscale(axes[1], base=2)

    # Latency per sample
    lat_per_sample = [l / b if b > 0 else math.nan for l, b in zip(results["latency_ms"], results["batch_size"])]
    axes[2].plot(results["batch_size"], lat_per_sample, "o-")
    axes[2].set_xlabel("Batch Size")
    axes[2].set_ylabel("Latency per Sample (ms)")
    axes[2].set_title("Amortized Latency")
    axes[2].grid(True, alpha=0.3)
    _try_set_log_xscale(axes[2], base=2)

    plt.suptitle(f"CPU Inference Benchmark | Model: {model_type} ({size_mb:.1f} MB)")
    plt.tight_layout()
    out_path = save_dir / "inference_benchmark.png"
    plt.savefig(out_path, dpi=150)
    print(f"✓ Saved benchmark plot -> {out_path}")


if __name__ == "__main__":
    model_dir = Path(__file__).parent.parent / "models" / MODEL_NAME
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, size_mb, model_type, cfg = load_model(model_dir)
    seq_len, input_dim, global_dim = _infer_dims_from_config(cfg)
    print(f"Benchmarking {model_type} ({size_mb:.1f} MB) on CPU | "
          f"seq_len={seq_len}, input_dim={input_dim}, global_dim={global_dim}")

    results = benchmark_model(model, seq_len, input_dim, global_dim, BATCH_SIZES)

    # Save results JSON
    with open(plots_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_results(results, size_mb, model_type, plots_dir)
