#!/usr/bin/env python3
"""
benchmark_inference.py - Load the model and process a sample profile,
including normalization/denorm calls, and time inference (both sequential,
parallel, and with increasing batch sizes to estimate concurrent capacity)
on a CUDA GPU or using MPS on a Mac. No plotting is performed.
"""

import sys
sys.path.append('../src')

import json
import json5
from pathlib import Path
import torch
import time
import random
from utils import create_prediction_model
from normalizer import DataNormalizer

# Hard-coded paths
DATA_PATH = "../data/normalized_profiles"
METADATA_PATH = "../data/normalized_profiles/normalization_metadata.json"
MODEL_PATH = "../data/model/best_model.pt"
CONFIG_PATH = "../inputs/model_input_params.jsonc"

def load_config(path):
    with open(path, 'r') as f:
        return json5.load(f)

def load_metadata(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_model(model_path, config, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = create_prediction_model(config).to(device)
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Loaded model weights")
    except Exception as e:
        print(f"Error loading model: {e}")
    model.eval()
    return model

def process_profile(profile, config, input_vars, coord_var, device):
    coord_vals = profile.get(coord_var, [])
    if not isinstance(coord_vals, list):
        coord_vals = [coord_vals]
    seq_len = len(coord_vals)
    inputs = {}
    seq_types = config.get("sequence_types", {"profile": []})
    seq_type = list(seq_types.keys())[0]
    seq_indices = seq_types[seq_type]
    if seq_indices:
        seq_features = []
        for idx in seq_indices:
            val = profile.get(input_vars[idx], 0)
            if isinstance(val, list):
                values = val if len(val) >= seq_len else val + [val[-1]]*(seq_len - len(val))
                values = values[:seq_len]
            else:
                values = [float(val)] * seq_len
            seq_features.append(torch.tensor(values, dtype=torch.float32, device=device))
        inputs[seq_type] = torch.stack(seq_features, dim=1).unsqueeze(0)
    global_indices = config.get("global_feature_indices", [])
    if global_indices:
        global_values = []
        for idx in global_indices:
            val = profile.get(input_vars[idx], 0)
            val_float = float(val[0]) if isinstance(val, list) and val else float(val)
            global_values.append(val_float)
        inputs["global"] = torch.tensor(global_values, dtype=torch.float32, device=device).unsqueeze(0)
    return inputs, seq_len, coord_vals

def denorm_predictions(preds, seq_len, target_vars, metadata):
    # For each sample in the batch, denormalize each target feature.
    for j in range(preds.shape[0]):
        for i, feat in enumerate(target_vars):
            _ = DataNormalizer.denormalize(preds[j, :seq_len, i].cpu().numpy(), metadata, feat)

def device_synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    # CPU operations are synchronous by default

def device_empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # MPS and CPU do not use empty_cache

def format_time(seconds):
    """Converts seconds into a human-readable format."""
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = []
    if days:
        parts.append(f"{int(days)} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
    parts.append(f"{seconds:.2f} seconds")
    return ', '.join(parts)

def test_concurrency(compiled_model, inputs, device):
    """
    Tests inference with increasing batch sizes to estimate how many inferences
    can run concurrently. For each batch size, it measures total time,
    average time per inference, throughput, and memory usage (if available).
    """
    print("\n=== Concurrency Benchmark ===")
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    results = []
    for bs in batch_sizes:
        try:
            # Prepare batched inputs by repeating along dim=0.
            current_inputs = {k: v.repeat(bs, *(1,)*(v.dim()-1)) for k, v in inputs.items()}
            device_empty_cache(device)
            device_synchronize(device)
            # Record starting memory (if CUDA)
            start_mem = torch.cuda.memory_allocated(device)/1024**2 if device.type == "cuda" else None

            start_time = time.perf_counter()
            with torch.no_grad():
                _ = compiled_model(current_inputs)
            device_synchronize(device)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            avg_time_per_inf = (total_time / bs) * 1000  # in ms
            throughput = bs / total_time

            mem_used = None
            if device.type == "cuda":
                current_mem = torch.cuda.max_memory_allocated(device)/1024**2
                mem_used = current_mem - (start_mem if start_mem is not None else 0)
                torch.cuda.reset_peak_memory_stats(device)

            results.append((bs, total_time, avg_time_per_inf, throughput, mem_used))
            print(f"Batch Size: {bs:4d} | Total Time: {total_time:.6f} sec | "
                  f"Avg Time per Inference: {avg_time_per_inf:.4f} ms | "
                  f"Throughput: {throughput:.2f} inf/sec", end="")
            if mem_used is not None:
                print(f" | Extra Memory: {mem_used:.2f} MB")
            else:
                print(" | Extra Memory: N/A")
        except Exception as e:
            print(f"Batch Size: {bs} caused an error: {e}")
            break
    return results

def main():
    # Determine device: CUDA, MPS, or CPU fallback
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(device)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) on Mac")
    else:
        print("Neither CUDA nor MPS is available. Running on CPU (this may be slow).")
        device = torch.device("cpu")
    
    config = load_config(CONFIG_PATH)
    metadata = load_metadata(METADATA_PATH)
    model = load_model(MODEL_PATH, config, device)
    
    input_vars = config["input_variables"]
    target_vars = config["target_variables"]
    coord_var = "pressure"
    
    # Pick a sample profile file (ignoring metadata file)
    all_files = [f for f in Path(DATA_PATH).glob("*.json") if f.name != "normalization_metadata.json"]
    if not all_files:
        print("No profile files found!")
        return
    with open(random.choice(all_files), 'r') as f:
        profile = json.load(f)
    
    inputs, seq_len, _ = process_profile(profile, config, input_vars, coord_var, device)
    for key, tensor in inputs.items():
        if tensor.device != device:
            inputs[key] = tensor.to(device)
    
    try:
        # Attempt to compile model with TorchScript
        device_synchronize(device)
        compiled_model = torch.jit.trace(model, inputs).to(device)
        compiled_model.eval()
        print("Successfully compiled model with TorchScript")
    except Exception as e:
        print(f"Error compiling model: {e}\nFalling back to non-compiled model")
        compiled_model = model

    # Warm-up iterations (including denorm calls)
    print("Running warm-up iterations...")
    for _ in range(10):
        with torch.no_grad():
            preds = compiled_model(inputs)
            denorm_predictions(preds, seq_len, target_vars, metadata)
        device_synchronize(device)
    
    # Measure latency of a single inference (ideal fully parallel time)
    with torch.no_grad():
        t_single_start = time.perf_counter()
        _ = compiled_model(inputs)
        device_synchronize(device)
        t_single_end = time.perf_counter()
    ideal_latency = t_single_end - t_single_start
    print("\nIdeal fully parallel execution time (one inference latency):")
    print(f"{ideal_latency:.6f} sec ({format_time(ideal_latency)})")

    # Sequential benchmark (1000 iterations)
    n_calls = 1000
    print(f"\nStarting sequential benchmark with {n_calls} iterations...")
    device_empty_cache(device)
    device_synchronize(device)
    start_time = time.perf_counter()
    for _ in range(n_calls):
        with torch.no_grad():
            preds = compiled_model(inputs)
            denorm_predictions(preds, seq_len, target_vars, metadata)
        device_synchronize(device)
    total_time = time.perf_counter() - start_time
    avg_time = total_time / n_calls
    print(f"Total time: {total_time:.6f} sec | Avg time per call: {avg_time*1000:.6f} ms | "
          f"Throughput: {n_calls/total_time:.2f} inferences/sec")

    # Parallel benchmark (10,000 inferences batched)
    print("\n" + "-"*50)
    print("Starting parallel benchmark with 10,000 inferences...")
    batch_size = 64
    n_parallel_calls = 10000
    n_batches = -(-n_parallel_calls // batch_size)  # Ceiling division
    batched_inputs = {k: v.repeat(batch_size, *(1,)*(v.dim()-1)) for k, v in inputs.items()}
    device_empty_cache(device)
    device_synchronize(device)
    if device.type == "cuda":
        start_mem = torch.cuda.memory_allocated(device) / 1024**2
    else:
        start_mem = 0
    batched_start_time = time.perf_counter()
    total_processed = 0
    for i in range(n_batches):
        current_bs = min(batch_size, n_parallel_calls - total_processed)
        current_inputs = {
            k: v.repeat(current_bs, *(1,)*(v.dim()-1)) if current_bs < batch_size else batched_inputs[k]
            for k, v in inputs.items()
        }
        with torch.no_grad():
            preds = compiled_model(current_inputs)
            denorm_predictions(preds, seq_len, target_vars, metadata)
        total_processed += current_bs
        if i % 10 == 0 or i == n_batches - 1:
            device_synchronize(device)
            print(f"Progress: {100*total_processed/n_parallel_calls:.1f}% ({total_processed}/{n_parallel_calls})", end="\r")
    device_synchronize(device)
    batched_total_time = time.perf_counter() - batched_start_time
    batched_avg_time = batched_total_time / n_parallel_calls
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak_mem = 0
    print("\n" + "-"*50)
    print(f"Parallel results: Total time: {batched_total_time:.4f} sec | "
          f"Avg time per inference: {batched_avg_time*1000:.4f} ms | "
          f"Throughput: {n_parallel_calls/batched_total_time:.2f} inferences/sec")
    if device.type == "cuda":
        print(f"Batching memory: {peak_mem - start_mem:.2f} MB | Peak GPU memory: {peak_mem:.2f} MB")
    else:
        print("Memory metrics not available for non-CUDA devices.")

    print("\nEstimated sequential performance:")
    print(f"50,000 inferences: {batched_avg_time*50000:.2f} sec")
    print(f"100,000 inferences: {batched_avg_time*100000:.2f} sec")
    print(f"1,000,000 inferences: {batched_avg_time*1000000/60:.2f} min")
    
    print("\nAssuming maximal parallelism (all inferences running concurrently):")
    print(f"Total job time would be approximately: {ideal_latency:.6f} sec ({format_time(ideal_latency)})")
    
    # Run concurrency test over increasing batch sizes
    concurrency_results = test_concurrency(compiled_model, inputs, device)

    print("\n=== Summary of Concurrency Test ===")
    print("Batch Size | Total Time (sec) | Avg Time per Inference (ms) | Throughput (inf/sec) | Extra Memory (MB)")
    for bs, tot, avg, thr, mem in concurrency_results:
        mem_str = f"{mem:.2f}" if mem is not None else "N/A"
        print(f"{bs:10d} | {tot:16.6f} | {avg:27.4f} | {thr:18.2f} | {mem_str:16}")

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"\nGPU Memory: Allocated = {allocated:.2f} MB | Reserved = {reserved:.2f} MB")

if __name__ == '__main__':
    main()
