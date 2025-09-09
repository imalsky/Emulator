#!/usr/bin/env python3
"""Plot average error vs pressure for test set."""
import sys

sys.path.append("../src")

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm import tqdm
from dataset import create_dataset, create_collate_fn
from normalizer import DataNormalizer
from utils import load_config, PADDING_VALUE

# Configuration
MODEL_DIR = Path("../models/trained_model")
DATA_DIR = Path("../data/processed/test")
TEST_FRACTION = 0.01  # Fraction of test set to analyze
RANDOM_SEED = 42

plt.style.use('science.mplstyle')


def main():
    # Load model and config
    config = load_config(MODEL_DIR / "train_config.json")
    with open("../data/processed/normalization_metadata.json") as f:
        norm_metadata = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.export.load(str(MODEL_DIR / "final_model.pt2"))

    # Get test samples
    with open(DATA_DIR / "metadata.json") as f:
        total_samples = json.load(f)['total_samples']
    n_samples = max(1, int(total_samples * TEST_FRACTION))
    np.random.seed(RANDOM_SEED)
    indices = np.sort(np.random.choice(total_samples, n_samples, replace=False))

    print(f"Processing {n_samples}/{total_samples} samples ({TEST_FRACTION * 100:.1f}%)")

    # Create dataset
    dataset = create_dataset(DATA_DIR, config, indices.tolist())
    collate_fn = create_collate_fn(PADDING_VALUE)

    # Get variable indices
    input_vars = config["data_specification"]["input_variables"]
    target_vars = config["data_specification"]["target_variables"]
    pressure_idx = input_vars.index("pressure_bar")
    flux_indices = {name: target_vars.index(name) for name in
                    ["net_thermal_flux", "net_reflected_flux"] if name in target_vars}

    # Collect pressure-error pairs
    pressure_errors = {name: {'pressures': [], 'errors': []} for name in flux_indices}

    for i in tqdm(range(n_samples)):
        inputs, targets = dataset[i]
        batch = collate_fn([(inputs, targets)])
        batch_inputs, batch_masks, batch_targets, target_masks = batch

        # Run inference
        model_inputs = {
            "sequence": batch_inputs["sequence"].to(device),
            "sequence_mask": batch_masks["sequence"].to(device)
        }
        if "global_features" in batch_inputs:
            model_inputs["global_features"] = batch_inputs["global_features"].to(device)

        with torch.no_grad():
            predictions = model.module()(**model_inputs)

        # Extract valid points
        valid_mask = ~target_masks.numpy()[0]
        if not np.any(valid_mask):
            continue

        # Get pressures
        pressures = batch_inputs["sequence"].numpy()[0, valid_mask, pressure_idx]
        pressures = DataNormalizer.denormalize_tensor(
            torch.from_numpy(pressures).float(),
            norm_metadata["normalization_methods"].get("pressure_bar", "none"),
            norm_metadata["per_key_stats"].get("pressure_bar", {})
        ).numpy()

        # Calculate errors for each flux
        for flux_name, flux_idx in flux_indices.items():
            preds = predictions.cpu().numpy()[0, valid_mask, flux_idx]
            targs = batch_targets.numpy()[0, valid_mask, flux_idx]

            # Denormalize
            preds = DataNormalizer.denormalize_tensor(
                torch.from_numpy(preds).float(),
                norm_metadata["normalization_methods"].get(flux_name, "none"),
                norm_metadata["per_key_stats"].get(flux_name, {})
            ).numpy()
            targs = DataNormalizer.denormalize_tensor(
                torch.from_numpy(targs).float(),
                norm_metadata["normalization_methods"].get(flux_name, "none"),
                norm_metadata["per_key_stats"].get(flux_name, {})
            ).numpy()

            # Calculate percent errors
            abs_errors = np.abs(preds - targs)
            percent_errors = 100 * abs_errors / np.maximum(np.abs(targs), 1.0)

            pressure_errors[flux_name]['pressures'].extend(pressures)
            pressure_errors[flux_name]['errors'].extend(percent_errors)

    # Create plot with shared y-axis
    fig, axes = plt.subplots(1, len(flux_indices), figsize=(14, 7), sharey=True)
    if len(flux_indices) == 1:
        axes = [axes]

    flux_names_list = list(flux_indices.keys())

    for idx, (ax, flux_name) in enumerate(zip(axes, flux_names_list)):
        pressures = np.array(pressure_errors[flux_name]['pressures'])
        percent_errors = np.array(pressure_errors[flux_name]['errors'])

        # Bin pressures logarithmically
        pressure_bins = np.logspace(np.log10(pressures.min()), np.log10(pressures.max()), 50)
        binned_errors = []
        bin_centers = []

        for i in range(len(pressure_bins) - 1):
            mask = (pressures >= pressure_bins[i]) & (pressures < pressure_bins[i + 1])
            if np.any(mask):
                binned_errors.append(np.mean(percent_errors[mask]))  # Use mean for average
                bin_centers.append(np.sqrt(pressure_bins[i] * pressure_bins[i + 1]))

        # Plot percent error
        ax.plot(binned_errors, bin_centers, color='black', linewidth=3)

        # Add vertical line for overall mean error
        overall_mean_error = np.mean(percent_errors)
        ax.axvline(x=overall_mean_error, color='red', linestyle='--', linewidth=2, alpha=0.7)

        # Set x-axis label for each subplot
        if flux_name == "net_thermal_flux":
            ax.set_xlabel('Thermal Channel, Mean Percent Error (%)', fontsize=16)
        elif flux_name == "net_reflected_flux":
            ax.set_xlabel('Reflected Channel, Mean Percent Error (%)', fontsize=16)
        else:
            ax.set_xlabel(f'{flux_name.replace("_", " ").title()}\nMean Percent Error (%)', fontsize=12)

        # Only label y-axis on the leftmost plot
        if idx == 0:
            ax.set_ylabel('Pressure (bar)', fontsize=16)

        ax.set_ylim(1e2, 1e-5)
        ax.set_yscale('log')
        ax.set_xlim(0.002, 99)  # Set x-axis limits from 0.001% to 100%
        ax.set_xscale('log')  # Use log scale for x-axis

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)
    save_path = MODEL_DIR / "plots/pressure_error_profile.png"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

    # Print some statistics
    print("\nError Statistics:")
    for flux_name in flux_indices:
        errors = np.array(pressure_errors[flux_name]['errors'])
        print(f"\n{flux_name.replace('_', ' ').title()}:")
        print(f"  Mean percent error: {np.mean(errors):.2f}%")
        print(f"  Median percent error: {np.median(errors):.2f}%")
        print(f"  90th percentile: {np.percentile(errors, 90):.2f}%")
        print(f"  99th percentile: {np.percentile(errors, 99):.2f}%")


if __name__ == "__main__":
    main()