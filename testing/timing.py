#!/usr/bin/env python3
import sys
import json
import random
import time
from pathlib import Path
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Configuration for Benchmarking ---
# Total number of distinct random profiles to load from the test set for the benchmark pool.
# These profiles will be cycled through during warmup and timed inference runs.
NUM_PROFILES_TO_LOAD_FOR_BENCHMARK_POOL = 1000
NUM_WARMUP_RUNS = 50
NUM_TIMED_RUNS = 500
# --- End Global Configuration ---

def find_project_root(marker_file: str = 'src') -> Path:
    path = Path(__file__).resolve()
    while path != path.parent:
        if (path / marker_file).is_dir(): return path
        path = path.parent
    return Path(__file__).resolve().parent # Fallback

def benchmark_cpu_inference(model: nn.Module, batched_inputs_cpu: dict, num_warmup: int, num_timed: int) -> dict:
    model.eval()
    # Ensure there's at least one tensor to get batch size from
    first_tensor_key = next((k for k, v in batched_inputs_cpu.items() if isinstance(v, torch.Tensor)), None)
    if first_tensor_key is None:
        return {"error": "No tensors found in batched_inputs_cpu"}
    actual_batch_size = batched_inputs_cpu[first_tensor_key].size(0)
    
    if actual_batch_size == 0:
        return {"error": "Loaded batch has size 0"}

    logger.info(f"Warmup ({num_warmup} runs, cycling {actual_batch_size} unique samples)...")
    with torch.no_grad():
        for i in range(num_warmup):
            inp = {k: v[i % actual_batch_size: (i % actual_batch_size) + 1] for k, v in batched_inputs_cpu.items() if isinstance(v, torch.Tensor)}
            if inp and any(t.numel() > 0 for t in inp.values()): # Check if any tensor in inp is not empty
                 _ = model(inp)

    timings_ms = []
    logger.info(f"Timed inference ({num_timed} runs, cycling {actual_batch_size} unique samples)...")
    with torch.no_grad():
        for i in range(num_timed):
            inp = {k: v[i % actual_batch_size: (i % actual_batch_size) + 1] for k, v in batched_inputs_cpu.items() if isinstance(v, torch.Tensor)}
            if not inp or not any(t.numel() > 0 for t in inp.values()): continue # Skip if no valid input
            start_time = time.perf_counter()
            _ = model(inp)
            timings_ms.append((time.perf_counter() - start_time) * 1000)
    
    if not timings_ms: return {"error": "No timings recorded"}
    timings_np = np.array(timings_ms)
    return {
        "count": len(timings_np), "mean_ms": np.mean(timings_np),
        "median_ms": np.median(timings_np), "min_ms": np.min(timings_np),
        "max_ms": np.max(timings_np), "std_ms": np.std(timings_np)
    }

if __name__ == "__main__":
    logger.info("--- Concise CPU Inference Benchmark ---")
    logger.info(f"Config: {NUM_PROFILES_TO_LOAD_FOR_BENCHMARK_POOL=}, {NUM_WARMUP_RUNS=}, {NUM_TIMED_RUNS=}")
    try:
        PROJECT_ROOT = find_project_root('src')
        SRC_DIR = PROJECT_ROOT / "src"
        if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))

        from utils import load_config
        from dataset import AtmosphericDataset, create_multi_source_collate_fn
        from model import create_prediction_model

        cfg = load_config(PROJECT_ROOT / "inputs" / "model_input_params.jsonc")
        if cfg is None: sys.exit("Config load failed.")
        device = torch.device("cpu")

        dataset = AtmosphericDataset(
            PROJECT_ROOT / "data" / "normalized_profiles", cfg["input_variables"], cfg["target_variables"],
            cfg.get("global_variables", []), cfg["sequence_types"], cfg["sequence_lengths"],
            cfg["output_seq_type"], validate_profiles=False
        )
        with (PROJECT_ROOT / "data" / "model" / "test_set_info.json").open('r') as f:
            test_filenames = json.load(f).get("test_filenames", [])
        
        all_ds_fnames = [p.name for p in dataset.valid_files]
        indices = [i for i, fname in enumerate(all_ds_fnames) if fname in test_filenames]
        if not indices: sys.exit("No test profiles found in dataset for benchmark.")
        
        # Use the global constant here
        num_to_select = min(NUM_PROFILES_TO_LOAD_FOR_BENCHMARK_POOL, len(indices))
        selected_indices = random.sample(indices, num_to_select)
        logger.info(f"Loading {len(selected_indices)} profiles for benchmark pool.")

        if not selected_indices: # Should not happen if indices is not empty, but as a safeguard
            sys.exit("No profiles were selected for loading.")

        benchmark_loader = DataLoader(
            Subset(dataset, selected_indices), batch_size=len(selected_indices), # Load all selected into one batch
            collate_fn=create_multi_source_collate_fn()
        )
        model = create_prediction_model(cfg)
        checkpoint = torch.load(PROJECT_ROOT / "data" / "model" / "best_model.pt", map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device).eval()
        logger.info("Model loaded onto CPU.")

        inputs_batch_cpu, _ = next(iter(benchmark_loader))
        
        stats = benchmark_cpu_inference(model, inputs_batch_cpu, NUM_WARMUP_RUNS, NUM_TIMED_RUNS)
        logger.info("\n--- Timing Statistics (ms) ---")
        for stat, value in stats.items():
            # Ensure value is a printable type, especially for the "error" case
            printable_value = value if isinstance(value, (int, float, str)) else f"{value:.4f}"
            logger.info(f"{stat.replace('_', ' ').capitalize():<15}: {printable_value}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)
