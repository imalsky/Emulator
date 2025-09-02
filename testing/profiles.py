#!/usr/bin/env python3
"""Plot pressure-temperature and pressure-flux profiles directly from raw HDF5 data."""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

sys.path.append("../src")

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import load_config

# Configuration - specify exact profile indices to plot
PROFILE_INDICES = [0, 2, 3, 4, 6, 7, 8, 10, 12, 17, 18, 20]  # Change these to your desired indices
MODEL_DIR = Path("../models/trained_model")
RAW_DATA_DIR = Path("../data/raw")

plt.style.use('science.mplstyle')


def main():
    # Load config to get file names and splits
    config = load_config(MODEL_DIR / "train_config.json")

    # Load splits to map indices to files
    with open(MODEL_DIR / "dataset_splits.json", 'r') as f:
        splits = json.load(f)

    # Handle compressed format if present
    if "file_stems" in splits:
        file_stems = splits["file_stems"]
        test_indices = [(file_stems[s], i) for s, i in splits["test"]]
    else:
        test_indices = splits["test"]

    # Get HDF5 filenames from config
    h5_files = config["data_paths_config"]["hdf5_dataset_filename"]
    if isinstance(h5_files, str):
        h5_files = [h5_files]

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.inferno(np.linspace(0, 0.95, len(PROFILE_INDICES)))

    # Plot each requested profile
    for plot_idx, profile_idx in enumerate(PROFILE_INDICES):
        if profile_idx >= len(test_indices):
            print(f"Warning: Profile index {profile_idx} exceeds test set size")
            continue

        file_stem, h5_idx = test_indices[profile_idx]
        h5_filename = f"{file_stem}.h5"
        h5_path = RAW_DATA_DIR / h5_filename

        if not h5_path.exists():
            print(f"Warning: File {h5_path} not found")
            continue

        # Read directly from HDF5
        with h5py.File(h5_path, 'r') as hf:
            pressure = hf['pressure_bar'][h5_idx]
            temperature = hf['temperature_k'][h5_idx]
            thermal_flux = hf['net_thermal_flux'][h5_idx]
            reflected_flux = hf['net_reflected_flux'][h5_idx]

            # Remove any invalid/padding values
            valid = (pressure > 0) & np.isfinite(pressure) & \
                   np.isfinite(temperature) & np.isfinite(thermal_flux) & \
                   np.isfinite(reflected_flux)

            # Panel 1: Temperature vs Pressure
            axes[0].plot(temperature[valid],
                        pressure[valid],
                        color=colors[plot_idx],
                        label=f'Profile {profile_idx}',
                        linewidth=2)

            # Panel 2: Thermal Flux vs Pressure
            axes[1].plot(thermal_flux[valid],
                        pressure[valid],
                        color=colors[plot_idx],
                        linewidth=2)

            # Panel 3: Reflected Flux vs Pressure
            axes[2].plot(reflected_flux[valid],
                        pressure[valid],
                        color=colors[plot_idx],
                        linewidth=2)

    # Configure Panel 1 (Temperature)
    axes[0].set_yscale('log')
    axes[0].set_xlim(0, 4000)
    axes[0].set_ylim(1e2, 1e-5)
    axes[0].set_xlabel('Temperature (K)', fontsize=14)
    axes[0].set_ylabel('Pressure (bar)', fontsize=14)

    # Configure Panel 2 (Thermal Flux)
    axes[1].set_yscale('log')
    axes[1].set_xscale('symlog', base=10)
    axes[1].set_ylim(1e2, 1e-5)
    axes[1].set_xlabel('Layer Net Thermal Flux (ergs/cm²)', fontsize=14)
    axes[1].set_ylabel('Pressure (bar)', fontsize=14)
    axes[1].set_xlim(-1e11, 1e11)
    axes[1].set_xticks([-1e10, -1e8, -1e6, -1e4, -1e2, 0, 1e2, 1e4, 1e6, 1e8, 1e10])

    # Configure Panel 3 (Reflected Flux)
    axes[2].set_yscale('log')
    axes[2].set_xscale('symlog')
    axes[2].set_ylim(1e2, 1e-5)
    axes[2].set_xlim(-1e10, 0.9)
    axes[2].set_xlabel('Layer Net Reflected Flux (ergs/cm²)', fontsize=14)
    axes[2].set_ylabel('Pressure (bar)', fontsize=14)

    # Add legend to first panel (could also add to all or create a separate legend)
    # axes[0].legend(fontsize=10, loc='best')

    plt.tight_layout()
    save_path = MODEL_DIR / "plots" / "atmospheric_profiles_3panel.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()