#!/usr/bin/env python3
"""Analyze and visualize error distributions for model predictions."""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

sys.path.append("../src")

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
from typing import Dict, Tuple, List

from dataset import create_dataset, create_collate_fn
from model import create_prediction_model
from normalizer import DataNormalizer
from utils import load_config, PADDING_VALUE

try:
    plt.style.use('science.mplstyle')
except:
    pass

MODEL_DIR = Path("../models/trained_model")
PROCESSED_DIR = Path("../data/processed/test")
N_SAMPLES = 100  # Use more samples for better statistics


def load_model_and_data():
    """Load model, config, and test dataset."""
    config_paths = [
        MODEL_DIR / "train_config.json",
        MODEL_DIR / "best_config.json",
        MODEL_DIR / "normalize_config.json"
    ]

    config = None
    for config_path in config_paths:
        if config_path.exists():
            config = load_config(config_path)
            break

    if config is None:
        raise FileNotFoundError(f"No config file found in {MODEL_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_prediction_model(config, device, compile_model=False)
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Load normalization metadata
    metadata_path = Path("../data/processed/normalization_metadata.json")
    with open(metadata_path, 'r') as f:
        norm_metadata = json.load(f)

    # Create dataset
    test_dataset = create_dataset(PROCESSED_DIR, config, list(range(min(N_SAMPLES, 1000))))
    collate_fn = create_collate_fn(PADDING_VALUE)

    return model, test_dataset, collate_fn, config, norm_metadata, device


def collect_errors(model, dataset, collate_fn, config, norm_metadata, device) -> Dict:
    """Collect prediction errors for all variables."""
    target_vars = config["data_specification"]["target_variables"]

    # Initialize error collectors
    errors = {var: {'absolute': [], 'relative': [], 'signed': []} for var in target_vars}

    # Process each sample
    for i in range(len(dataset)):
        inputs, targets = dataset[i]
        batch_inputs, batch_masks, batch_targets, target_masks = collate_fn([(inputs, targets)])

        # Move to device
        for k in batch_inputs:
            batch_inputs[k] = batch_inputs[k].to(device)
        batch_masks["sequence"] = batch_masks["sequence"].to(device)

        # Predict
        with torch.no_grad():
            preds = model(
                sequence=batch_inputs["sequence"],
                global_features=batch_inputs.get("global_features"),
                sequence_mask=batch_masks["sequence"]
            )

        # Get valid positions
        preds_np = preds.cpu().numpy()[0]
        targets_np = batch_targets.numpy()[0]
        valid_mask = ~target_masks.numpy()[0]

        if not np.any(valid_mask):
            continue

        # Calculate errors for each variable
        for var_idx, var_name in enumerate(target_vars):
            # Get predictions and targets for this variable
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

            # Calculate errors
            abs_error = np.abs(var_preds_denorm - var_targets_denorm)
            signed_error = var_preds_denorm - var_targets_denorm

            # Relative error with small epsilon to avoid division by zero
            rel_error = abs_error / (np.abs(var_targets_denorm) + 1e-10)

            errors[var_name]['absolute'].extend(abs_error.tolist())
            errors[var_name]['relative'].extend(rel_error.tolist())
            errors[var_name]['signed'].extend(signed_error.tolist())

    # Convert to arrays
    for var in errors:
        for error_type in errors[var]:
            errors[var][error_type] = np.array(errors[var][error_type])

    return errors


def plot_error_distributions(errors: Dict):
    """Create comprehensive error distribution plots."""
    # Select key variables to analyze
    key_vars = ['net_thermal_flux', 'net_reflected_flux'] if 'net_thermal_flux' in errors else list(errors.keys())[:2]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    for col, var_name in enumerate(key_vars):
        if var_name not in errors:
            continue

        var_errors = errors[var_name]

        # 1. Histogram of absolute errors with fitted distribution
        ax = axes[0, col]
        abs_errors = var_errors['absolute']

        # Remove outliers for better visualization (keep 99%)
        q99 = np.percentile(abs_errors, 99)
        abs_errors_plot = abs_errors[abs_errors <= q99]

        n, bins, patches = ax.hist(abs_errors_plot, bins=50, density=True,
                                   alpha=0.7, color='blue', edgecolor='black')

        # Fit and plot exponential distribution
        if len(abs_errors_plot) > 0:
            loc, scale = stats.expon.fit(abs_errors_plot)
            x = np.linspace(0, abs_errors_plot.max(), 100)
            ax.plot(x, stats.expon.pdf(x, loc, scale), 'r-', lw=2,
                    label=f'Exponential fit\nλ={1 / scale:.2e}')

        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Density')
        ax.set_title(f'{var_name}\nAbsolute Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text
        ax.text(0.95, 0.95, f'Mean: {np.mean(abs_errors):.2e}\n'
                            f'Median: {np.median(abs_errors):.2e}\n'
                            f'95%: {np.percentile(abs_errors, 95):.2e}',
                transform=ax.transAxes, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. Q-Q plot for normality check of signed errors
        ax = axes[1, col]
        signed_errors = var_errors['signed']

        # Standardize errors
        if len(signed_errors) > 0 and np.std(signed_errors) > 0:
            standardized = (signed_errors - np.mean(signed_errors)) / np.std(signed_errors)
            stats.probplot(standardized, dist="norm", plot=ax)
            ax.set_title(f'{var_name}\nQ-Q Plot (Normality Check)')
            ax.grid(True, alpha=0.3)

            # Add Shapiro-Wilk test result
            if len(standardized) < 5000:  # Shapiro-Wilk has sample size limit
                statistic, p_value = stats.shapiro(standardized[:5000])
                ax.text(0.05, 0.95, f'Shapiro-Wilk p={p_value:.3f}\n'
                                    f'{"Normal" if p_value > 0.05 else "Non-normal"}',
                        transform=ax.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        # 3. Box plot comparing error quantiles
        ax = axes[2, col]

        # Prepare data for box plot
        rel_errors = var_errors['relative'] * 100  # Convert to percentage

        # Create box plot with outliers
        bp = ax.boxplot([abs_errors, np.abs(signed_errors)],
                        labels=['Absolute', '|Signed|'],
                        showfliers=True, patch_artist=True)

        # Color the boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('Error Magnitude')
        ax.set_title(f'{var_name}\nError Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Add a second y-axis for relative errors
        ax2 = ax.twinx()
        bp2 = ax2.boxplot([rel_errors], positions=[3], widths=0.6,
                          labels=['Relative (%)'], patch_artist=True)
        bp2['boxes'][0].set_facecolor('lightyellow')
        ax2.set_ylabel('Relative Error (%)')
        ax2.tick_params(axis='y', labelcolor='orange')

    plt.suptitle('Error Distribution Analysis', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save plot
    save_path = MODEL_DIR / "plots" / "error_distributions.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_error_heatmap(errors: Dict, config: Dict):
    """Create heatmap showing error patterns across variables."""
    target_vars = config["data_specification"]["target_variables"]

    # Create error summary matrix
    metrics = ['Mean Abs', 'Median Abs', 'Std', '95th %ile', 'Max']
    error_matrix = np.zeros((len(target_vars), len(metrics)))

    for i, var in enumerate(target_vars):
        if var in errors:
            abs_err = errors[var]['absolute']
            if len(abs_err) > 0:
                error_matrix[i, 0] = np.mean(abs_err)
                error_matrix[i, 1] = np.median(abs_err)
                error_matrix[i, 2] = np.std(abs_err)
                error_matrix[i, 3] = np.percentile(abs_err, 95)
                error_matrix[i, 4] = np.max(abs_err)

    # Normalize each column to [0, 1] for better visualization
    for j in range(len(metrics)):
        col_max = error_matrix[:, j].max()
        if col_max > 0:
            error_matrix[:, j] /= col_max

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(error_matrix, cmap='YlOrRd', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(target_vars)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(target_vars)

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Error Magnitude', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(target_vars)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{error_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black" if error_matrix[i, j] < 0.5 else "white")

    ax.set_title('Error Metrics Heatmap (Normalized per Column)', fontsize=14, pad=20)
    plt.tight_layout()

    # Save
    save_path = MODEL_DIR / "plots" / "error_heatmap.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


def main():
    """Main analysis function."""
    print("=" * 60)
    print("ERROR DISTRIBUTION ANALYSIS")
    print("=" * 60)

    print("\nLoading model and data...")
    model, dataset, collate_fn, config, norm_metadata, device = load_model_and_data()

    print(f"Analyzing {len(dataset)} test samples...")
    errors = collect_errors(model, dataset, collate_fn, config, norm_metadata, device)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("ERROR SUMMARY STATISTICS")
    print("=" * 60)

    for var_name in list(errors.keys())[:5]:  # Show first 5 variables
        abs_err = errors[var_name]['absolute']
        rel_err = errors[var_name]['relative'] * 100

        print(f"\n{var_name}:")
        print(f"  Absolute Error:")
        print(f"    Mean:   {np.mean(abs_err):.3e}")
        print(f"    Median: {np.median(abs_err):.3e}")
        print(f"    95%:    {np.percentile(abs_err, 95):.3e}")
        print(f"  Relative Error:")
        print(f"    Mean:   {np.mean(rel_err):.1f}%")
        print(f"    Median: {np.median(rel_err):.1f}%")

    # Create visualizations
    print("\nGenerating error distribution plots...")
    plot_error_distributions(errors)

    print("Generating error heatmap...")
    plot_error_heatmap(errors, config)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()