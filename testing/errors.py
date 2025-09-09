#!/usr/bin/env python3
"""Calculate test set error statistics."""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

sys.path.append("../src")

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from dataset import create_dataset, create_collate_fn
from normalizer import DataNormalizer
from utils import load_config, PADDING_VALUE

# Configuration
MODEL_DIR = Path("../models/trained_model")
DATA_DIR = Path("../data/processed/test")
TEST_FRACTION = 0.01
RANDOM_SEED = 42


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

    # Get flux indices
    target_vars = config["data_specification"]["target_variables"]
    flux_vars = {name: target_vars.index(name) for name in
                 ["net_thermal_flux", "net_reflected_flux"] if name in target_vars}

    # Collect predictions and targets
    results = {name: {'preds': [], 'targets': []} for name in flux_vars}

    for i in tqdm(range(n_samples)):
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
            predictions = model.module()(**model_inputs)

        # Extract valid points
        valid_mask = ~target_masks.numpy()[0]
        if not np.any(valid_mask):
            continue

        # Process each flux
        for flux_name, flux_idx in flux_vars.items():
            preds = predictions.cpu().numpy()[0, valid_mask, flux_idx]
            targs = batch_targets.numpy()[0, valid_mask, flux_idx]

            # Denormalize
            method = norm_metadata["normalization_methods"].get(flux_name, "none")
            stats = norm_metadata["per_key_stats"].get(flux_name, {})
            if method != "none" and stats:
                preds = DataNormalizer.denormalize_tensor(
                    torch.from_numpy(preds).float(), method, stats).numpy()
                targs = DataNormalizer.denormalize_tensor(
                    torch.from_numpy(targs).float(), method, stats).numpy()

            results[flux_name]['preds'].append(preds)
            results[flux_name]['targets'].append(targs)

    # Calculate and print statistics
    print("\n" + "=" * 60)
    print("TEST SET ERROR STATISTICS")
    print("=" * 60)

    for flux_name in flux_vars:
        if len(results[flux_name]['preds']) == 0:
            continue

        preds = np.concatenate(results[flux_name]['preds'])
        targets = np.concatenate(results[flux_name]['targets'])

        # Calculate errors
        abs_errors = np.abs(preds - targets)
        percent_errors = 100 * abs_errors / np.maximum(np.abs(targets), 1.0)

        # Print statistics
        print(f"\n{flux_name.replace('_', ' ').title()}:")
        print(f"  MAE:                  {np.mean(abs_errors):.3e} Ergs/cm²")
        print(f"  RMSE:                 {np.sqrt(np.mean((preds - targets) ** 2)):.3e} Ergs/cm²")
        print(f"  Mean % Error:         {np.mean(percent_errors):.2f}%")
        print(f"  Median % Error:       {np.median(percent_errors):.2f}%")
        print(
            f"  R-squared:            {1 - np.sum((targets - preds) ** 2) / np.sum((targets - np.mean(targets)) ** 2):.6f}")


if __name__ == "__main__":
    main()