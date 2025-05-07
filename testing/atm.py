#!/usr/bin/env python3
"""
plot_atm.py - Plots a random selection of atmospheric profiles, using the
              first key in the JSON for the y-axis and the second for the x-axis.
              Minimal version.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import random
from pathlib import Path
from matplotlib.cm import viridis
from matplotlib.colors import to_rgba

def main():
    """Loads profile data and generates a plot."""
    try:
        SCRIPT_DIR = Path(__file__).resolve().parent
        # Assume script is in 'testing' or similar, relative to project root
        PROJECT_ROOT = SCRIPT_DIR.parent
        PROFILES_DIR = PROJECT_ROOT / "data" / "profiles"
        PLOTS_DIR = PROJECT_ROOT / "plots"
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        STYLE_FILE = SCRIPT_DIR / "science.mplstyle"

        # Apply style
        if STYLE_FILE.exists():
            plt.style.use(str(STYLE_FILE))
            print(f"Using style file: {STYLE_FILE}")
        else:
            print(f"Warning: Style file not found: {STYLE_FILE}")

        # --- Configuration ---
        base_filename = 'prof'
        num_profiles_to_plot = 20

        # --- Load Profile Files ---
        profile_files = glob.glob(os.path.join(PROFILES_DIR, f"{base_filename}_*.json"))
        if not profile_files:
            print(f"Error: No profile files found in {PROFILES_DIR}")
            sys.exit(1)

        selected_files = random.sample(profile_files, min(num_profiles_to_plot, len(profile_files)))
        print(f"Plotting {len(selected_files)} profiles.")

        # --- Plotting ---
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        colors = [to_rgba(viridis(i / max(1, len(selected_files) - 1))) for i in range(len(selected_files))]

        y_axis_key = None
        x_axis_key = None
        plot_success_count = 0

        for profile_file, color in zip(selected_files, colors):
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)

                keys = list(data.keys())
                if len(keys) < 2: continue # Skip if not enough keys

                current_y_key, current_x_key = keys[0], keys[1]
                if y_axis_key is None: y_axis_key = current_y_key
                if x_axis_key is None: x_axis_key = current_x_key

                y_data = data.get(current_y_key)
                x_data = data.get(current_x_key)

                if isinstance(y_data, list) and isinstance(x_data, list) and len(y_data) == len(x_data) and len(y_data) > 0:
                    ax.plot(np.array(x_data), np.array(y_data), color=color, alpha=0.7)
                    plot_success_count += 1
                else:
                    print(f"Warning: Skipping {Path(profile_file).name} due to data issues.")

            except Exception as e:
                print(f"Warning: Skipping {Path(profile_file).name} due to error: {e}")

        if plot_success_count == 0:
             print("Error: No profiles were successfully plotted.")
             sys.exit(1)

        # --- Final Plot Configuration ---
        x_label = x_axis_key.replace("_", " ").title() if x_axis_key else "X-Axis (Key 2)"
        y_label = y_axis_key.replace("_", " ").title() if y_axis_key else "Y-Axis (Key 1)"

        #ax.set_ylim(1e2, 1e-5) # Keep typical pressure limits
        #ax.set_xlim(0, 4000)   # Keep typical temperature limits
        ax.set_xlabel(f'{x_label}')
        ax.set_ylabel(f'{y_label}')
        ax.set_title(f'Random Sample of {plot_success_count} Profiles ({y_label} vs {x_label})')

        plt.tight_layout()

        # --- Save Figure ---
        plot_filename = PLOTS_DIR / f"profiles_plot.png"
        plt.savefig(plot_filename, dpi=250)
        print(f"Plot saved to: {plot_filename}")
        plt.close(fig)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
