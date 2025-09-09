#!/usr/bin/env python3
"""Test model predictions with denormalization and visualization."""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

sys.path.append("../src")

import torch
import numpy as np
import matplotlib.pyplot as plt
import json

from dataset import create_dataset, create_collate_fn
from normalizer import DataNormalizer
from utils import load_config, PADDING_VALUE

plt.style.use('science.mplstyle')

# Configuration
MODEL_DIR = Path("../models/trained_model")
PROCESSED_DIR = Path("../data/processed/test")
MODEL_FILE = 'final_model.pt2'

# Sample selection - specify exact indices to plot
# Can be:
#   - A list of specific indices: [0, 2, 5, 10, 15]
#   - A range: list(range(0, 10, 2)) for even indices from 0-8
#   - A slice: list(range(5))[:3] for first 3 of first 5
SAMPLE_INDICES = [2, 4, 5, 6, 7]  # Modify this list to select specific samples


# Alternatively, use these helper functions:
def select_samples(mode='first', n=5, indices=None, total_samples=None):
    """
    Helper to select sample indices.

    Args:
        mode: 'first', 'last', 'random', 'custom', or 'evenly_spaced'
        n: number of samples for 'first', 'last', 'random', 'evenly_spaced'
        indices: list of specific indices for 'custom' mode
        total_samples: total number of available samples (for 'last', 'random', 'evenly_spaced')

    Returns:
        List of sample indices
    """
    if mode == 'custom':
        return indices if indices else []
    elif mode == 'first':
        return list(range(n))
    elif mode == 'last' and total_samples:
        return list(range(max(0, total_samples - n), total_samples))
    elif mode == 'random' and total_samples:
        import random
        return sorted(random.sample(range(total_samples), min(n, total_samples)))
    elif mode == 'evenly_spaced' and total_samples:
        step = max(1, total_samples // n)
        return list(range(0, total_samples, step))[:n]
    else:
        return list(range(n))


# Example usage (uncomment one):
# SAMPLE_INDICES = select_samples('first', n=5)
# SAMPLE_INDICES = select_samples('custom', indices=[0, 5, 10, 15, 20])
# SAMPLE_INDICES = select_samples('evenly_spaced', n=5, total_samples=100)

plt.style.use('science.mplstyle')


def load_normalization_metadata():
    """Load normalization metadata for denormalization."""
    metadata_path = Path("../data/processed/normalization_metadata.json")
    with open(metadata_path, 'r') as f:
        return json.load(f)


def denormalize_variable(data, var_name, norm_metadata):
    """Denormalize a single variable."""
    data_tensor = torch.from_numpy(data).float()
    method = norm_metadata["normalization_methods"].get(var_name, "none")
    stats = norm_metadata["per_key_stats"].get(var_name, {})

    if method != "none" and stats:
        return DataNormalizer.denormalize_tensor(data_tensor, method, stats).numpy()
    return data


def load_model_and_data(sample_indices):
    """Load exported model and test data."""
    # Find config file
    config_path = MODEL_DIR / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the exported model (.pt2 format)
    model_path = MODEL_DIR / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading exported model from: {model_path}")
    exported_model = torch.export.load(str(model_path))

    # Load test dataset with all required indices
    max_idx = max(sample_indices) if sample_indices else 0
    test_dataset = create_dataset(PROCESSED_DIR, config, list(range(max_idx + 1)))
    collate_fn = create_collate_fn(PADDING_VALUE)

    return exported_model, test_dataset, collate_fn, config, device


def run_inference(model, batch_inputs, batch_masks, device):
    """Run inference with the exported model."""
    # Move inputs to device
    inputs = {}
    inputs["sequence"] = batch_inputs["sequence"].to(device)
    inputs["sequence_mask"] = batch_masks["sequence"].to(device)

    if "global_features" in batch_inputs:
        inputs["global_features"] = batch_inputs["global_features"].to(device)

    # Run inference
    with torch.no_grad():
        output = model.module()(**inputs)

    return output


def test_predictions(sample_indices=None):
    """
    Test model on samples and visualize results.

    Args:
        sample_indices: List of specific sample indices to plot.
                       If None, uses global SAMPLE_INDICES.
    """
    if sample_indices is None:
        sample_indices = SAMPLE_INDICES

    if not sample_indices:
        raise ValueError("No sample indices specified")

    print(f"Processing samples at indices: {sample_indices}")

    model, dataset, collate_fn, config, device = load_model_and_data(sample_indices)
    norm_metadata = load_normalization_metadata()

    # Validate indices
    available_samples = len(dataset)
    valid_indices = [idx for idx in sample_indices if 0 <= idx < available_samples]

    if not valid_indices:
        raise ValueError(f"No valid indices. Dataset has {available_samples} samples, "
                         f"but requested indices were: {sample_indices}")

    if len(valid_indices) < len(sample_indices):
        invalid = [idx for idx in sample_indices if idx not in valid_indices]
        print(f"Warning: Skipping invalid indices {invalid} (dataset has {available_samples} samples)")
        sample_indices = valid_indices

    # Get variable indices
    input_vars = config["data_specification"]["input_variables"]
    target_vars = config["data_specification"]["target_variables"]

    if "pressure_bar" not in input_vars:
        raise ValueError("pressure_bar not found in input_variables")

    pressure_idx = input_vars.index("pressure_bar")
    thermal_idx = target_vars.index("net_thermal_flux") if "net_thermal_flux" in target_vars else None
    reflected_idx = target_vars.index("net_reflected_flux") if "net_reflected_flux" in target_vars else None

    if thermal_idx is None or reflected_idx is None:
        raise ValueError(f"Required flux variables not found. Available: {target_vars}")

    # Create figure
    n_samples = len(sample_indices)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Use viridis colormap for consistent, scientific visualization
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_samples))  # 0 to 0.9 to avoid very light yellows

    # Process selected samples
    for plot_idx, sample_idx in enumerate(sample_indices):
        print(f"  Processing sample {sample_idx}...")

        inputs, targets = dataset[sample_idx]
        batch_inputs, batch_masks, batch_targets, target_masks = collate_fn([(inputs, targets)])

        # Get predictions
        preds = run_inference(model, batch_inputs, batch_masks, device)

        # Extract data
        preds_np = preds.cpu().numpy()[0]
        targets_np = batch_targets.numpy()[0]
        valid_mask = ~target_masks.numpy()[0]
        valid_idx = np.where(valid_mask)[0]

        if len(valid_idx) == 0:
            print(f"    Warning: Sample {sample_idx} has no valid data points")
            continue

        # Get and denormalize pressure
        pressure_values = batch_inputs["sequence"].numpy()[0, valid_idx, pressure_idx]
        pressure_denorm = denormalize_variable(pressure_values, "pressure_bar", norm_metadata)

        # Process thermal flux
        thermal_pred = denormalize_variable(
            preds_np[valid_idx, thermal_idx], "net_thermal_flux", norm_metadata
        )
        thermal_target = denormalize_variable(
            targets_np[valid_idx, thermal_idx], "net_thermal_flux", norm_metadata
        )

        # Process reflected flux
        reflected_pred = denormalize_variable(
            preds_np[valid_idx, reflected_idx], "net_reflected_flux", norm_metadata
        )
        reflected_target = denormalize_variable(
            targets_np[valid_idx, reflected_idx], "net_reflected_flux", norm_metadata
        )

        # Calculate percent errors
        thermal_error = 100 * np.abs(thermal_pred - thermal_target) / (np.abs(thermal_target) + 1)
        reflected_error = 100 * np.abs(reflected_pred - reflected_target) / (np.abs(reflected_target) + 1)

        color = colors[plot_idx]

        # Use actual sample index in labels
        label_suffix = f"Sample {sample_idx}"

        # Plot thermal flux - only add label for first sample
        thermal_target_label = "Target" if plot_idx == 0 else ""
        thermal_pred_label = "Prediction" if plot_idx == 0 else ""

        axes[0, 0].plot(pressure_denorm, thermal_target, '-', color=color,
                        label=thermal_target_label, alpha=0.7, linewidth=3)
        axes[0, 0].plot(pressure_denorm, thermal_pred, '--', color=color,
                        label=thermal_pred_label, alpha=0.7, linewidth=3)

        # Plot thermal error - keep individual sample labels
        axes[1, 0].plot(pressure_denorm, thermal_error, color=color,
                        label=label_suffix, alpha=0.7, linewidth=3)

        # Plot reflected flux - only add label for first sample
        reflected_target_label = "Target" if plot_idx == 0 else ""
        reflected_pred_label = "Prediction" if plot_idx == 0 else ""

        axes[0, 1].plot(pressure_denorm, reflected_target, '-', color=color,
                        label=reflected_target_label, alpha=0.7, linewidth=3)
        axes[0, 1].plot(pressure_denorm, reflected_pred, '--', color=color,
                        label=reflected_pred_label, alpha=0.7, linewidth=3)

        # Plot reflected error - keep individual sample labels
        axes[1, 1].plot(pressure_denorm, reflected_error, color=color,
                        label=label_suffix, alpha=0.7, linewidth=3)

    # Format subplots
    for ax in axes.flat:
        ax.set_xscale('log')

    # Thermal flux
    axes[0, 0].set_yscale('symlog')
    axes[0, 0].set_xlabel("Pressure (bar)")
    axes[0, 0].set_ylabel("Net Thermal Flux (Ergs/cm²)")
    #axes[0, 0].set_title("Net Thermal Flux")
    axes[0, 0].set_ylim(-1e10, 1e10)
    axes[0, 0].legend(fontsize=10, loc='best')

    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel("Pressure (bar)")
    axes[1, 0].set_ylabel("Percent Error (%)")
    axes[1, 0].set_ylim(1e-3, 100)
    #axes[1, 0].set_title("Thermal Flux Error")
    #if n_samples <= 8:
    #    axes[1, 0].legend(fontsize=8, loc='best')

    # Reflected flux
    axes[0, 1].set_yscale('symlog')
    axes[0, 1].set_xlabel("Pressure (bar)")
    axes[0, 1].set_ylabel("Net Reflected Flux (Ergs/cm²)")
    #axes[0, 1].set_title("Net Reflected Flux")
    axes[0, 1].set_ylim(-1e10, 1e1)
    axes[0, 1].legend(fontsize=10, loc='best')

    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel("Pressure (bar)")
    axes[1, 1].set_ylabel("Percent Error (%)")
    #axes[1, 1].set_title("Reflected Flux Error")
    axes[1, 1].set_ylim(1e-3, 100)
    #if n_samples <= 8:
    #    axes[1, 1].legend(fontsize=8, loc='best')

    plt.tight_layout()

    # Save plot with informative filename
    indices_str = "_".join(map(str, sample_indices[:5]))  # First 5 indices in filename
    if len(sample_indices) > 5:
        indices_str += f"_and_{len(sample_indices) - 5}_more"

    save_path = MODEL_DIR / "plots" / f"flux_predictions_samples_{indices_str}.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {save_path}")

    # Calculate and print metrics
    print_metrics(model, dataset, collate_fn, config, norm_metadata, device, sample_indices)


def print_metrics(model, dataset, collate_fn, config, norm_metadata, device, sample_indices):
    """Calculate and print denormalized metrics for specified samples."""
    target_vars = config["data_specification"]["target_variables"]

    print("\n" + "=" * 60)
    print(f"DENORMALIZED METRICS (Physical Units)")
    print(f"Samples analyzed: {sample_indices}")
    print("=" * 60)

    for flux_name in ["net_thermal_flux", "net_reflected_flux"]:
        if flux_name not in target_vars:
            continue

        flux_idx = target_vars.index(flux_name)
        all_preds = []
        all_targets = []

        # Collect predictions for specified samples only
        for sample_idx in sample_indices:
            if sample_idx >= len(dataset):
                continue

            inputs, targets = dataset[sample_idx]
            batch_inputs, batch_masks, batch_targets, target_masks = collate_fn([(inputs, targets)])

            preds = run_inference(model, batch_inputs, batch_masks, device)

            preds_np = preds.cpu().numpy()[0]
            targets_np = batch_targets.numpy()[0]
            valid_mask = ~target_masks.numpy()[0]

            if np.any(valid_mask):
                flux_pred = denormalize_variable(
                    preds_np[valid_mask, flux_idx], flux_name, norm_metadata
                )
                flux_target = denormalize_variable(
                    targets_np[valid_mask, flux_idx], flux_name, norm_metadata
                )
                all_preds.append(flux_pred)
                all_targets.append(flux_target)

        if all_preds:
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)

            rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
            mae = np.mean(np.abs(all_preds - all_targets))

            print(f"\n{flux_name}:")
            print(f"  RMSE: {rmse:.3e} Ergs/cm²")
            print(f"  MAE:  {mae:.3e} Ergs/cm²")
            print(f"  Target range: [{np.min(all_targets):.3e}, {np.max(all_targets):.3e}]")
            print(f"  Pred range:   [{np.min(all_preds):.3e}, {np.max(all_preds):.3e}]")
            print(f"  Number of points: {len(all_preds)}")


if __name__ == "__main__":
    # You can override the indices here or use the global SAMPLE_INDICES
    # Examples:
    # test_predictions([0, 10, 20, 30, 40])  # Specific indices
    # test_predictions(select_samples('random', n=10, total_samples=100))  # Random selection
    test_predictions()  # Uses SAMPLE_INDICES defined at top