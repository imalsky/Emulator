#!/usr/bin/env python3
"""
Evaluate and visualize model predictions on test data.
Creates comparison plots of true vs predicted atmospheric fluxes.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Configuration
MODEL_NAME = "trained_model"
NUM_PROFILES = 2

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.extend([str(ROOT), str(ROOT / "src")])

from normalizer import DataNormalizer
from model import create_prediction_model


def remap_layer_keys(state_dict):
    """
    Remap checkpoint keys from old 'layers' structure to new 'blocks' structure.
    
    Old architecture: layers.0 (initial_film), layers.1 (transformer0), layers.2 (film0), ...
    New architecture: initial_film, blocks.0.transformer, blocks.0.film, ...
    """
    remapped = {}
    
    for key, value in state_dict.items():
        # Remove any _orig_mod prefix from compiled models
        key = key.replace("_orig_mod.", "")
        
        if not key.startswith("layers."):
            remapped[key] = value
            continue
            
        # Parse layer index and remaining path
        parts = key.split(".", 2)
        layer_idx = int(parts[1])
        rest = parts[2] if len(parts) > 2 else ""
        
        # Map to new structure
        if layer_idx == 0:
            remapped[f"initial_film.{rest}"] = value
        elif layer_idx % 2 == 1:  # Odd = transformer
            block_idx = layer_idx // 2
            remapped[f"blocks.{block_idx}.transformer.{rest}"] = value
        else:  # Even > 0 = film
            block_idx = (layer_idx - 2) // 2
            remapped[f"blocks.{block_idx}.film.{rest}"] = value
    
    return remapped


def load_model(model_dir):
    """Load trained model and configuration."""
    # Find config file
    config_files = list(model_dir.glob("*_config.json"))
    if not config_files:
        raise FileNotFoundError(f"No config found in {model_dir}")
    
    with open(config_files[0]) as f:
        config = json.load(f)
    
    # Load checkpoint
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"Loading model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create model and load weights
    model = create_prediction_model(config, device=torch.device("cpu"), compile_model=False)
    state_dict = remap_layer_keys(checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, config


def load_test_data(config, num_samples=2):
    """Load random samples from test/validation set."""
    # Try test directory first, fall back to validation
    data_dir = ROOT / "data/processed/test"
    if not data_dir.exists():
        data_dir = ROOT / "data/processed/val"
    
    # Load metadata
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    # Pick random shard
    shards = sorted((data_dir / "sequence_inputs").glob("shard_*.npy"))
    shard = np.random.choice(shards)
    
    # Load arrays
    sequences = np.load(data_dir / "sequence_inputs" / shard.name)
    targets = np.load(data_dir / "targets" / shard.name)
    
    # Load global features if present
    globals_array = None
    if config["data_specification"].get("global_variables"):
        globals_path = data_dir / "globals" / shard.name
        if globals_path.exists():
            globals_array = np.load(globals_path)
    
    # Select random samples
    indices = np.random.choice(len(sequences), min(num_samples, len(sequences)), replace=False)
    padding_value = config["data_specification"]["padding_value"]
    
    samples = []
    for idx in indices:
        # Create tensors
        seq_tensor = torch.from_numpy(sequences[idx]).float()
        tgt_tensor = torch.from_numpy(targets[idx]).float()
        
        # Identify padding positions
        is_padding = torch.all(torch.abs(seq_tensor - padding_value) < 1e-6, dim=-1)
        
        sample = {
            "sequence": seq_tensor,
            "target": tgt_tensor,
            "padding_mask": is_padding,  # True = padding (for model)
            "valid_mask": ~is_padding,   # True = valid (for plotting)
        }
        
        if globals_array is not None:
            sample["global_features"] = torch.from_numpy(globals_array[idx]).float()
        
        samples.append(sample)
    
    return samples


def run_inference(model, sample):
    """Run model inference on a single sample."""
    with torch.no_grad():
        # Prepare inputs with batch dimension
        inputs = {
            "sequence": sample["sequence"].unsqueeze(0),
            "sequence_mask": sample["padding_mask"].unsqueeze(0),
        }
        
        if "global_features" in sample:
            inputs["global_features"] = sample["global_features"].unsqueeze(0)
        
        # Forward pass
        output = model(**inputs)
        return output.squeeze(0)


def denormalize(tensor, variable_name, norm_metadata):
    """Denormalize a tensor using saved statistics."""
    method = norm_metadata["normalization_methods"].get(variable_name, "none")
    stats = norm_metadata["per_key_stats"].get(variable_name, {})
    
    if method != "none" and stats:
        stats["method"] = method  # Ensure method is in stats
        result = DataNormalizer.denormalize_tensor(tensor, method, stats)
        return result.cpu().numpy()
    
    return tensor.cpu().numpy()


def calculate_errors(predictions, targets):
    """Calculate percent error, capped at 10000%."""
    errors = 100 * np.abs((predictions - targets) / np.maximum(np.abs(targets), 1e-10))
    return np.minimum(errors, 1e4)


def create_plots(samples, model, config, norm_metadata, output_path):
    """Create comparison plots for model evaluation."""
    # Get variable names and indices
    input_vars = config["data_specification"]["input_variables"]
    target_vars = config["data_specification"]["target_variables"]
    
    pressure_idx = input_vars.index("pressure_bar")
    thermal_idx = target_vars.index("net_thermal_flux")
    reflected_idx = target_vars.index("net_reflected_flux")
    
    # Setup figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = plt.cm.tab10(np.arange(len(samples)))
    
    for i, sample in enumerate(samples):
        # Run inference
        predictions = run_inference(model, sample)
        
        # Get valid (non-padded) data points
        valid_indices = sample["valid_mask"].nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            continue
        
        # Extract valid data
        seq_valid = sample["sequence"][valid_indices]
        tgt_valid = sample["target"][valid_indices]
        pred_valid = predictions[valid_indices]
        
        # Denormalize all variables
        pressure = denormalize(seq_valid[:, pressure_idx], "pressure_bar", norm_metadata)
        thermal_true = denormalize(tgt_valid[:, thermal_idx], "net_thermal_flux", norm_metadata)
        thermal_pred = denormalize(pred_valid[:, thermal_idx], "net_thermal_flux", norm_metadata)
        reflected_true = denormalize(tgt_valid[:, reflected_idx], "net_reflected_flux", norm_metadata)
        reflected_pred = denormalize(pred_valid[:, reflected_idx], "net_reflected_flux", norm_metadata)
        
        # Calculate errors
        thermal_error = calculate_errors(thermal_pred, thermal_true)
        reflected_error = calculate_errors(reflected_pred, reflected_true)
        
        # Print summary
        print(f"\nProfile {i+1}:")
        print(f"  Points: {valid_indices.numel()}")
        print(f"  Mean error: Thermal={thermal_error.mean():.1f}%, Reflected={reflected_error.mean():.1f}%")
        
        # Plot results
        color = colors[i]
        
        # Thermal flux comparison
        axes[0, 0].plot(thermal_true, pressure, "o-", color=color, alpha=0.7, 
                       label=f"True {i+1}", markersize=3)
        axes[0, 0].plot(thermal_pred, pressure, "x--", color=color, alpha=0.9,
                       label=f"Pred {i+1}", markersize=3)
        
        # Reflected flux comparison
        axes[0, 1].plot(reflected_true, pressure, "o-", color=color, alpha=0.7, markersize=3)
        axes[0, 1].plot(reflected_pred, pressure, "x--", color=color, alpha=0.9, markersize=3)
        
        # Error plots
        axes[1, 0].semilogx(thermal_error, pressure, color=color, label=f"Profile {i+1}")
        axes[1, 1].semilogx(reflected_error, pressure, color=color)
    
    # Format all axes
    for ax in axes.flat:
        ax.set_yscale("log")
        ax.set_ylim(1e2, 1e-5)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylabel("Pressure (bar)")
    
    # Format flux axes
    axes[0, 0].set_xscale("symlog")
    axes[0, 1].set_xscale("symlog")
    axes[0, 0].set_xlabel("Flux (W/m²)")
    axes[0, 1].set_xlabel("Flux (W/m²)")
    
    # Format error axes
    for ax in axes[1, :]:
        ax.set_xlim(1e-2, 1e4)
        ax.set_xlabel("Percent Error (%)")
    
    # Add titles and legends
    axes[0, 0].set_title("Net Thermal Flux")
    axes[0, 1].set_title("Net Reflected Flux")
    axes[1, 0].set_title("Thermal Error")
    axes[1, 1].set_title("Reflected Error")
    
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[1, 0].legend(fontsize=8, loc="best")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved plot to: {output_path}")


def main():
    """Main evaluation pipeline."""
    # Setup paths
    model_dir = ROOT / "models" / MODEL_NAME
    plot_dir = model_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load model and metadata
    print(f"Evaluating model: {MODEL_NAME}")
    model, config = load_model(model_dir)
    
    # Load normalization metadata
    norm_path = ROOT / "data/processed/normalization_metadata.json"
    with open(norm_path) as f:
        norm_metadata = json.load(f)
    
    # Load test samples
    samples = load_test_data(config, NUM_PROFILES)
    print(f"Loaded {len(samples)} test samples")
    
    # Create evaluation plots
    output_path = plot_dir / "model_predictions.png"
    create_plots(samples, model, config, norm_metadata, output_path)


if __name__ == "__main__":
    main()