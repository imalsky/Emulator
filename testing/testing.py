#!/usr/bin/env python3
"""
testing.py - Minimal inference speed test for 100 profiles.
Loads the first 100 profiles found and times batched inference.
Includes FP16/AMP attempt for CUDA devices. Corrected for MPS.
"""
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import contextlib # Needed for nullcontext

# --- Configuration ---
MODEL_CHECKPOINT = "../data/model/best_model.pt" # REQUIRED: Path to your model
CONFIG_PATH = "../inputs/model_input_params.jsonc"
DATA_DIR = "../data"
NUM_PROFILES_TO_TEST = 3000
BATCH_SIZE = 50     # Adjust as needed
NUM_WARMUP_BATCHES = 1 # Warmup with first few batches
# --- End Configuration ---

# Add src directory to path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path: sys.path.insert(0, str(src_path))

try:
    from utils import load_config as load_main_config
    from hardware import setup_device, get_device_type # Import get_device_type
    from model import create_prediction_model
    from dataset import AtmosphericDataset, create_multi_source_collate_fn
except ImportError as e: print(f"Import Error: {e}. Ensure 'src' is at {src_path}"); sys.exit(1)

def main():
    # --- Setup ---
    print(f"Loading config: {CONFIG_PATH}")
    cfg = load_main_config(CONFIG_PATH)
    if cfg is None: print("Failed config load"); return 1
    device = setup_device()
    is_cuda = device.type == 'cuda'
    use_amp = is_cuda # Enable AMP automatically only for CUDA
    print(f"Using device: {device.type}")
    print(f"Attempting AMP (FP16): {use_amp}")

    # --- Load Model ---
    print(f"Loading model: {MODEL_CHECKPOINT}")
    model = create_prediction_model(cfg)
    try:
        # Load the full checkpoint dictionary
        # Using weights_only=False (default) as train.py saves metadata alongside state_dict
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location="cpu")
        if 'state_dict' not in checkpoint:
             print(f"Checkpoint {MODEL_CHECKPOINT} missing 'state_dict'."); return 1
        state_dict = checkpoint['state_dict']
        # Remove potential compilation prefix
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint (Epoch: {checkpoint.get('epoch', '?')})")
    except Exception as e: print(f"Checkpoint load failed: {e}"); return 1

    model.to(device).eval()
    # NOTE: torch.compile removed for simplicity

    # --- Load Data (First N Profiles) ---
    try:
        norm_dir = Path(DATA_DIR) / "normalized_profiles"
        # Initialize dataset - validation log might still appear here
        print("Initializing dataset (validation log may appear)...")
        # Note: AtmosphericDataset performs validation during __init__
        full_dataset = AtmosphericDataset(norm_dir, cfg["input_variables"], cfg["target_variables"],
                                          cfg.get("global_variables", []), cfg["sequence_types"],
                                          cfg["sequence_lengths"], cfg["output_seq_type"], cache_size=0) # No cache

        actual_num_profiles = min(NUM_PROFILES_TO_TEST, len(full_dataset))
        if actual_num_profiles == 0: print("Dataset is empty!"); return 1
        if actual_num_profiles < NUM_PROFILES_TO_TEST:
             print(f"Warning: Found only {len(full_dataset)} profiles, testing {actual_num_profiles}.")

        profile_indices = list(range(actual_num_profiles)) # Indices for first N profiles
        subset = Subset(full_dataset, profile_indices)
        # Force 0 workers for simplicity and MPS compatibility
        # Pin memory only if CUDA for potential speedup with async transfers
        pin_memory = is_cuda
        loader = DataLoader(subset, batch_size=BATCH_SIZE, collate_fn=create_multi_source_collate_fn(), num_workers=0, pin_memory=pin_memory)
        print(f"Loaded {actual_num_profiles} profiles into {len(loader)} batches.")
    except Exception as e: print(f"Data loading failed: {e}"); return 1

    # --- Timing ---
    # FIX: Only initialize autocast for CUDA, use nullcontext otherwise
    if use_amp: # use_amp is already correctly set to True only if is_cuda
        # Ensure device_type matches the actual device for autocast when enabled
        amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)
    else:
        amp_ctx = contextlib.nullcontext() # Use a null context for non-CUDA or if AMP is disabled

    total_samples = 0
    total_time = 0.0
    actual_warmup_run = 0
    actual_timing_run = 0

    try:
        with torch.no_grad():
            data_iterator = iter(loader) # Get iterator here

            # Warmup
            print(f"Running {NUM_WARMUP_BATCHES} warmup batch(es)...")
            for i in range(NUM_WARMUP_BATCHES):
                 try:
                     inputs, _ = next(data_iterator)
                     # Use non_blocking=True if pin_memory is True (CUDA)
                     inputs = {k: v.to(device, non_blocking=pin_memory) for k, v in inputs.items()}
                     # Use the autocast context manager
                     with amp_ctx:
                         _ = model(inputs)
                     actual_warmup_run += 1
                 except StopIteration:
                     print("Warning: DataLoader exhausted during warmup.")
                     break # Stop warmup if no more data
            if device.type in ['cuda', 'mps']: torch.cuda.synchronize() if is_cuda else torch.mps.synchronize() # type: ignore
            print(f"Warmup complete ({actual_warmup_run} batches).")

            # Timing remaining batches
            print(f"Timing remaining batches...")
            t_start = time.perf_counter()
            while True: # Loop until StopIteration
                 try:
                     inputs, targets = next(data_iterator)
                     inputs = {k: v.to(device, non_blocking=pin_memory) for k, v in inputs.items()}
                     # Use the autocast context manager
                     with amp_ctx:
                         _ = model(inputs)
                     total_samples += targets.shape[0]
                     actual_timing_run += 1
                 except StopIteration:
                     break # End of data loader
            if device.type in ['cuda', 'mps']: torch.cuda.synchronize() if is_cuda else torch.mps.synchronize() # type: ignore
            total_time = time.perf_counter() - t_start
            print(f"Timing complete ({actual_timing_run} batches).")

    except Exception as e: print(f"Inference error: {e}"); return 1

    # --- Results ---
    if total_samples == 0 or total_time <= 0: print("Timing failed (0 samples or time)."); return 1
    avg_time_ms = (total_time / total_samples) * 1000
    samples_per_sec = total_samples / total_time
    print(f"\n--- Results ---")
    print(f"Total Profiles Tested: {total_samples}")
    print(f"Total Inference Time: {total_time:.4f} sec (for {actual_timing_run} batches)")
    print(f"Avg Inference Time: {avg_time_ms:.4f} ms / sample")
    print(f"Samples per Second: {samples_per_sec:.2f} samples/sec")
    print(f"Settings: Device={device.type}, BatchSize={BATCH_SIZE}, AMP={use_amp}") # Added AMP to settings
    return 0

if __name__ == "__main__":
    sys.exit(main())
