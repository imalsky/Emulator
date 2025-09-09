#!/usr/bin/env python3
"""
export.py — CPU-optimized export & benchmark (pure PT2 + optional torch.compile)
No TorchScript. No ONNX. No argparse. Tunables are globals below.

Artifacts under models/<MODEL_RUN_DIR>/:
  - final_model.pt2   (torch.export program with dynamic batch)  ← baseline & usually fastest on CPU
"""

from __future__ import annotations
import os, sys, time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

# ========= GLOBALS: tune here =========
MODEL_RUN_DIR: str = "trained_model"

# Threading & benchmark knobs
CPU_THREADS: int = max(1, (os.cpu_count() or 4) // 2)
BATCH_SIZES: list[int] = [1, 4, 16]
WARMUP_ITERS: int = 5
TIMING_ITERS: int = 20

# Features
ENABLE_COMPILE: bool = True   # Try torch.compile (Inductor) on CPU

# ========= ENV (set BEFORE heavy imports) =========
# Keep CPU-only and minimize OpenMP drama (macOS/Homebrew/Conda mix)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(CPU_THREADS))            # macOS Accelerate
os.environ.setdefault("ACCELERATE_MATMUL_MULTITHREADING", "1")               # macOS Accelerate
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")                        # avoid libomp double-load aborts

# ========= PATH SETUP =========
_THIS_FILE = Path(__file__).resolve()
for _p in [
    _THIS_FILE.parent.parent / "src",
    _THIS_FILE.parent.parent,
    _THIS_FILE.parent / "src",
    _THIS_FILE.parent,
]:
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

# ========= DEPS =========
import warnings
warnings.filterwarnings("ignore", message="record_context_cpp is not support.*")

import torch
import torch.nn as nn
from torch.export import export as texport, save as tsave, Dim
from utils import load_config
from model import create_prediction_model


# ========= UTILITIES =========
def _set_cpu_threads(threads: Optional[int]) -> None:
    if not threads or threads <= 0:
        return
    torch.set_num_threads(int(threads))
    torch.set_num_interop_threads(max(1, int(threads // 2)))


class _WrappedModel(nn.Module):
    """Stable signature wrapper for torch.compile and export program parity."""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(
        self,
        sequence: torch.Tensor,
        sequence_mask: torch.Tensor,
        global_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.base(sequence, global_features, sequence_mask)


@torch.no_grad()
def _export_pt2(model: nn.Module, seq_len: int, input_dim: int, global_dim: int, out_path: Path) -> Path:
    # Sample inputs + dynamic batch shapes
    kwargs = {
        "sequence": torch.randn(2, seq_len, input_dim),
        "sequence_mask": torch.zeros(2, seq_len, dtype=torch.bool),
    }
    if global_dim > 0:
        kwargs["global_features"] = torch.randn(2, global_dim)

    batch_dim = Dim("batch", min=1, max=8192)
    dyn = {"sequence": {0: batch_dim}, "sequence_mask": {0: batch_dim}}
    if global_dim > 0:
        dyn["global_features"] = {0: batch_dim}

    prog = texport(model, args=(), kwargs=kwargs, dynamic_shapes=dyn, strict=False)
    tsave(prog, str(out_path))
    return out_path


def _build_compiled(model: nn.Module) -> Optional[nn.Module]:
    try:
        wrapped = _WrappedModel(model).eval()
        compiled = torch.compile(wrapped, backend="inductor", dynamic=True, fullgraph=False)
        return compiled
    except Exception as e:
        print(f"WARNING: torch.compile on CPU failed/unsupported: {e}")
        return None


@torch.inference_mode()
def _bench_callable(fn, inps_by_bs: Dict[int, Dict], iters: int, warmup: int) -> Dict[int, float]:
    """Return ms/forward for each batch size."""
    out: Dict[int, float] = {}
    for bs, inps in inps_by_bs.items():
        # warmup
        for _ in range(warmup):
            _ = fn(**inps)
        # timed
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fn(**inps)
        out[bs] = (time.perf_counter() - t0) / iters * 1000.0
    return out


def _make_inputs(batch_sizes, seq_len, input_dim, global_dim, numpy: bool = False) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for bs in batch_sizes:
        seq = torch.randn(bs, seq_len, input_dim, dtype=torch.float32).contiguous()
        msk = torch.zeros(bs, seq_len, dtype=torch.bool)
        d = {"sequence": seq, "sequence_mask": msk}
        if global_dim > 0:
            d["global_features"] = torch.randn(bs, global_dim, dtype=torch.float32).contiguous()
        if numpy:
            for k in list(d.keys()):
                d[k] = d[k].cpu().numpy()
        out[bs] = d
    return out


# ========= MAIN =========
def main() -> None:
    _set_cpu_threads(CPU_THREADS)
    print(f"CPU threads: intra-op={torch.get_num_threads()}, inter-op={torch.get_num_interop_threads()}")

    project_root = _THIS_FILE.parent.parent
    model_dir = project_root / "models" / MODEL_RUN_DIR
    cfg_path = model_dir / "train_config.json"
    ckpt_path = model_dir / "best_model.pt"

    # Build model (CPU)
    config = load_config(cfg_path)
    model = create_prediction_model(config, device=torch.device("cpu"), compile_model=False).eval()
    torch.set_grad_enabled(False)

    # Load weights (handle DDP prefixes)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        print("Removed '_orig_mod.' prefix from state dict keys")
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from epoch {ckpt.get('epoch', -1)} with val_loss={ckpt.get('val_loss', float('nan')):.6f}")

    # Shapes
    data_spec = config["data_specification"]
    model_params = config["model_hyperparameters"]
    seq_len = int(model_params["max_sequence_length"])
    input_dim = int(len(data_spec["input_variables"]))
    global_dim = int(len(data_spec.get("global_variables", [])))

    # Inputs for all batch sizes
    inputs_per_bs = _make_inputs(BATCH_SIZES, seq_len, input_dim, global_dim, numpy=False)

    # ---- PT2 export (baseline) ----
    pt2_path = _export_pt2(model, seq_len, input_dim, global_dim, model_dir / "final_model.pt2")
    print(f"✓ Exported model saved to: {pt2_path} ({pt2_path.stat().st_size / (1024*1024):.2f} MB)")
    prog = torch.export.load(str(pt2_path))

    # ---- torch.compile (optional) ----
    compiled_fn = None
    if ENABLE_COMPILE:
        compiled = _build_compiled(deepcopy(model))
        if compiled is not None:
            def _compiled_adapter(**kw):
                return compiled(kw["sequence"], kw["sequence_mask"], kw.get("global_features", None))
            compiled_fn = _compiled_adapter

    # ---- Benchmarks ----
    print("\nBenchmarking on CPU (ms per forward):")
    print(f"  warmup={WARMUP_ITERS} iters={TIMING_ITERS} batch_sizes={BATCH_SIZES}")

    pt2_ms = _bench_callable(prog.module(), inputs_per_bs, TIMING_ITERS, WARMUP_ITERS)
    for bs in BATCH_SIZES:
        print(f"  [pt2]   batch {bs:4d}: {pt2_ms[bs]:8.2f} ms  ({pt2_ms[bs]/max(1,bs):6.2f} ms/sample)")

    if compiled_fn is not None:
        comp_ms = _bench_callable(compiled_fn, inputs_per_bs, TIMING_ITERS, WARMUP_ITERS)
        for bs in BATCH_SIZES:
            spd = pt2_ms[bs] / comp_ms[bs] if comp_ms[bs] > 0 else float('inf')
            print(f"  [cmp]   batch {bs:4d}: {comp_ms[bs]:8.2f} ms  ({comp_ms[bs]/max(1,bs):6.2f} ms/sample)  x{spd:4.2f} vs pt2")

    print("\nArtifacts:")
    print(f"  - {model_dir / 'final_model.pt2'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
