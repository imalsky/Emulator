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
N_SAMPLES = 5
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


def load_model_and_data():
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

    # Load test dataset
    test_dataset = create_dataset(PROCESSED_DIR, config, list(range(N_SAMPLES)))
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


def test_predictions():
    """Test model on samples and visualize results."""
    model, dataset, collate_fn, config, device = load_model_and_data()
    norm_metadata = load_normalization_metadata()

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
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Process samples
    for i in range(min(N_SAMPLES, len(dataset))):
        inputs, targets = dataset[i]
        batch_inputs, batch_masks, batch_targets, target_masks = collate_fn([(inputs, targets)])

        # Get predictions
        preds = run_inference(model, batch_inputs, batch_masks, device)

        # Extract data
        preds_np = preds.cpu().numpy()[0]
        targets_np = batch_targets.numpy()[0]
        valid_mask = ~target_masks.numpy()[0]
        valid_idx = np.where(valid_mask)[0]

        if len(valid_idx) == 0:
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

        color = colors[i % len(colors)]

        # Plot thermal flux
        axes[0, 0].plot(pressure_denorm, thermal_target, '-', color=color,
                        label=f"Target {i + 1}", alpha=0.7, linewidth=2)
        axes[0, 0].plot(pressure_denorm, thermal_pred, '--', color=color,
                        label=f"Pred {i + 1}", alpha=0.7, linewidth=2)

        # Plot thermal error
        axes[1, 0].plot(pressure_denorm, thermal_error, color=color,
                        label=f"Sample {i + 1}", alpha=0.7, linewidth=2)

        # Plot reflected flux
        axes[0, 1].plot(pressure_denorm, reflected_target, '-', color=color,
                        label=f"Target {i + 1}", alpha=0.7, linewidth=2)
        axes[0, 1].plot(pressure_denorm, reflected_pred, '--', color=color,
                        label=f"Pred {i + 1}", alpha=0.7, linewidth=2)

        # Plot reflected error
        axes[1, 1].plot(pressure_denorm, reflected_error, color=color,
                        label=f"Sample {i + 1}", alpha=0.7, linewidth=2)

    # Format subplots
    for ax in axes.flat:
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')

    # Thermal flux
    axes[0, 0].set_yscale('symlog')
    axes[0, 0].set_xlabel("Pressure (bar)")
    axes[0, 0].set_ylabel("Net Thermal Flux (Ergs/cm²)")
    axes[0, 0].set_title("Net Thermal Flux")
    axes[0, 0].legend(fontsize=8)

    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel("Pressure (bar)")
    axes[1, 0].set_ylabel("Percent Error (%)")
    axes[1, 0].set_title("Thermal Flux Error")
    axes[1, 0].legend(fontsize=8)

    # Reflected flux
    axes[0, 1].set_yscale('symlog')
    axes[0, 1].set_xlabel("Pressure (bar)")
    axes[0, 1].set_ylabel("Net Reflected Flux (Ergs/cm²)")
    axes[0, 1].set_title("Net Reflected Flux")
    axes[0, 1].legend(fontsize=8)

    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel("Pressure (bar)")
    axes[1, 1].set_ylabel("Percent Error (%)")
    axes[1, 1].set_title("Reflected Flux Error")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()

    # Save plot
    save_path = MODEL_DIR / "plots" / "flux_predictions.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {save_path}")

    # Calculate and print metrics
    print_metrics(model, dataset, collate_fn, config, norm_metadata, device)


def print_metrics(model, dataset, collate_fn, config, norm_metadata, device):
    """Calculate and print denormalized metrics."""
    target_vars = config["data_specification"]["target_variables"]

    print("\n" + "=" * 60)
    print("DENORMALIZED METRICS (Physical Units)")
    print("=" * 60)

    for flux_name in ["net_thermal_flux", "net_reflected_flux"]:
        if flux_name not in target_vars:
            continue

        flux_idx = target_vars.index(flux_name)
        all_preds = []
        all_targets = []

        # Collect predictions
        for i in range(min(N_SAMPLES, len(dataset))):
            inputs, targets = dataset[i]
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


if __name__ == "__main__":
    test_predictions()