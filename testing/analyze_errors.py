#!/usr/bin/env python3
"""Analyze and visualize error distributions for model predictions."""

import sys

sys.path.append("../src")

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

from dataset import create_dataset, create_collate_fn
from normalizer import DataNormalizer
from utils import load_config, PADDING_VALUE

# Configuration
MODEL_DIR = Path("../models/trained_model")
PROCESSED_DIR = Path("../data/processed/test")
N_SAMPLES = 100
MODEL_FILE = 'final_model.pt2'


def load_model_and_data():
    """Load exported model, config, and test dataset."""
    config_path = MODEL_DIR / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load exported model
    model_path = MODEL_DIR / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model: {model_path}")
    exported_model = torch.export.load(str(model_path))

    # Load normalization metadata
    metadata_path = Path("../data/processed/normalization_metadata.json")
    with open(metadata_path, 'r') as f:
        norm_metadata = json.load(f)

    # Create dataset
    test_dataset = create_dataset(PROCESSED_DIR, config, list(range(min(N_SAMPLES, 1000))))
    collate_fn = create_collate_fn(PADDING_VALUE)

    return exported_model, test_dataset, collate_fn, config, norm_metadata, device


def collect_errors(model, dataset, collate_fn, config, norm_metadata, device):
    """Collect prediction errors."""
    target_vars = config["data_specification"]["target_variables"]
    all_errors = {var: [] for var in target_vars}

    for i in range(len(dataset)):
        inputs, targets = dataset[i]
        batch_inputs, batch_masks, batch_targets, target_masks = collate_fn([(inputs, targets)])

        # Run inference
        model_inputs = {
            "sequence": batch_inputs["sequence"].to(device),
            "sequence_mask": batch_masks["sequence"].to(device)
        }
        if "global_features" in batch_inputs:
            model_inputs["global_features"] = batch_inputs["global_features"].to(device)

        with torch.no_grad():
            preds = model.module()(**model_inputs)

        # Get valid positions
        preds_np = preds.cpu().numpy()[0]
        targets_np = batch_targets.numpy()[0]
        valid_mask = ~target_masks.numpy()[0]

        if not np.any(valid_mask):
            continue

        # Calculate percent errors for each variable
        for var_idx, var_name in enumerate(target_vars):
            var_preds = preds_np[valid_mask, var_idx]
            var_targets = targets_np[valid_mask, var_idx]

            # Denormalize
            method = norm_metadata["normalization_methods"].get(var_name, "none")
            stats = norm_metadata["per_key_stats"].get(var_name, {})

            if method != "none" and stats:
                var_preds_denorm = DataNormalizer.denormalize_tensor(
                    torch.from_numpy(var_preds).float(), method, stats
                ).numpy()
                var_targets_denorm = DataNormalizer.denormalize_tensor(
                    torch.from_numpy(var_targets).float(), method, stats
                ).numpy()
            else:
                var_preds_denorm = var_preds
                var_targets_denorm = var_targets

            # Calculate percent error
            percent_error = 100 * np.abs(var_preds_denorm - var_targets_denorm) / (np.abs(var_targets_denorm) + 1e-10)
            all_errors[var_name].extend(percent_error.tolist())

    # Convert to arrays
    for var in all_errors:
        all_errors[var] = np.array(all_errors[var])

    return all_errors


def plot_error_analysis(errors):
    """Create percent error and percentile plots with log scale."""
    # Select key variables
    key_vars = ['net_thermal_flux', 'net_reflected_flux']
    if 'net_thermal_flux' not in errors:
        key_vars = list(errors.keys())[:2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for col, var_name in enumerate(key_vars[:2]):
        if var_name not in errors or len(errors[var_name]) == 0:
            continue

        percent_errors = errors[var_name]
        percent_errors = percent_errors[percent_errors > 0]  # Remove zeros for log scale

        # Top plot: Percent error histogram with log y-axis
        ax = axes[0, col]

        counts, bins, _ = ax.hist(percent_errors, bins=50, alpha=0.7,
                                  color='steelblue', edgecolor='black')
        ax.set_xlabel('Percent Error (%)')
        ax.set_ylabel('Count')
        ax.set_yscale('log')
        ax.set_title(f'{var_name}')
        ax.grid(True, alpha=0.3, which='both')

        # Add percentile lines
        percentiles = [50, 75, 90, 95]
        colors = ['green', 'orange', 'darkorange', 'red']
        for p, c in zip(percentiles, colors):
            val = np.percentile(percent_errors, p)
            ax.axvline(val, color=c, linestyle='--', alpha=0.7, label=f'{p}th: {val:.1f}%')

        ax.legend(loc='upper right')

        # Bottom plot: Percentile curve with log y-axis
        ax = axes[1, col]

        percentile_range = np.arange(0, 100, 1)
        percentile_values = np.percentile(percent_errors, percentile_range)

        ax.plot(percentile_range, percentile_values, 'b-', linewidth=2)
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Percent Error (%)')
        ax.set_yscale('log')
        ax.set_title(f'{var_name} Percentiles')
        ax.grid(True, alpha=0.3, which='both')

        # Mark key percentiles
        key_percentiles = [50, 75, 90, 95, 99]
        for p in key_percentiles:
            val = np.percentile(percent_errors, p)
            ax.plot(p, val, 'ro', markersize=6)

    plt.suptitle('Model Error Analysis', fontsize=14)
    plt.tight_layout()

    # Save
    save_path = MODEL_DIR / "plots" / "error_analysis.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def main():
    """Main analysis."""
    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    print("\nLoading model and data...")
    model, dataset, collate_fn, config, norm_metadata, device = load_model_and_data()

    print(f"Analyzing {len(dataset)} samples...")
    errors = collect_errors(model, dataset, collate_fn, config, norm_metadata, device)

    # Print summary
    print("\n" + "=" * 60)
    print("PERCENT ERROR SUMMARY")
    print("=" * 60)

    for var_name in list(errors.keys())[:4]:
        if len(errors[var_name]) > 0:
            err = errors[var_name]
            print(f"\n{var_name}:")
            print(f"  Mean:   {np.mean(err):.1f}%")
            print(f"  Median: {np.median(err):.1f}%")
            print(f"  75th:   {np.percentile(err, 75):.1f}%")
            print(f"  90th:   {np.percentile(err, 90):.1f}%")
            print(f"  95th:   {np.percentile(err, 95):.1f}%")
            print(f"  99th:   {np.percentile(err, 99):.1f}%")

    print("\nGenerating plots...")
    plot_error_analysis(errors)

    print("\nComplete!")


if __name__ == "__main__":
    main()