#!/usr/bin/env python3
"""
plot_corner.py - Generates a corner plot for global input variables from
                 test profiles specified in test_set_info.json, showing
                 histograms with quantiles on the diagonal and scatter points
                 colored by Mean Absolute Error (MAE) on the off-diagonal plots.
"""

import sys
import json
import random
import time
import os
from pathlib import Path
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
# Import the function 'corner' from the module 'corner'
from corner import corner as corner_plot_func

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates Mean Absolute Error per sample."""
    if y_true.ndim == 1: return np.abs(y_true - y_pred)
    return np.mean(np.abs(y_true - y_pred), axis=1)

def main():
    """Loads data, model, runs inference, and generates corner plot."""
    try:
        SCRIPT_DIR = Path(__file__).resolve().parent
        PROJECT_ROOT = SCRIPT_DIR.parent # Assume script is in 'testing'
        SRC_DIR = PROJECT_ROOT / "src"
        PLOTS_DIR = PROJECT_ROOT / "plots"
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))

        # Define paths
        CONFIG_PATH = PROJECT_ROOT / "inputs" / "model_input_params.jsonc"
        MODEL_CHECKPOINT = PROJECT_ROOT / "data" / "model" / "best_model.pt"
        TEST_SET_INFO_PATH = PROJECT_ROOT / "data" / "model" / "test_set_info.json"
        DATA_DIR = PROJECT_ROOT / "data" / "normalized_profiles"
        NORMALIZATION_META_PATH = DATA_DIR / "normalization_metadata.json"
        STYLE_FILE = SCRIPT_DIR / "science.mplstyle"

        # Apply style
        if STYLE_FILE.exists(): plt.style.use(str(STYLE_FILE))
        else: logger.warning(f"Style file not found: {STYLE_FILE}")

        # --- Internal imports ---
        from utils import load_config
        from dataset import AtmosphericDataset, create_multi_source_collate_fn
        from model import create_prediction_model
        from normalizer import DataNormalizer

        # --- Config & Setup ---
        logger.info(f"Loading config: {CONFIG_PATH}")
        cfg = load_config(CONFIG_PATH)
        if not cfg: sys.exit("Failed to load config.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        num_corner_samples = cfg.get("plotting_corner_num_samples", 1000) # Use more samples for scatter
        target_var_for_mae = cfg["target_variables"][0]
        global_input_vars = cfg.get("global_variables", [])
        if not global_input_vars: sys.exit("No global variables defined.")

        # --- Load Data & Test Files ---
        logger.info(f"Loading dataset: {DATA_DIR}")
        dataset = AtmosphericDataset(
            DATA_DIR, cfg["input_variables"], cfg["target_variables"],
            global_input_vars, cfg["sequence_types"], cfg["sequence_lengths"],
            cfg["output_seq_type"], validate_profiles=False
        )
        logger.info(f"Dataset loaded ({len(dataset)} profiles).")

        logger.info(f"Loading test info: {TEST_SET_INFO_PATH}")
        with TEST_SET_INFO_PATH.open('r') as f: test_info = json.load(f)
        test_filenames = test_info.get("test_filenames")
        if not test_filenames: sys.exit("No 'test_filenames' in test_set_info.json.")

        all_ds_filenames = [p.name for p in dataset.valid_files]
        indices_for_subset = [i for i, fname in enumerate(all_ds_filenames) if fname in test_filenames]
        if not indices_for_subset: sys.exit("None of the specified test profiles found.")

        actual_num_samples = min(num_corner_samples, len(indices_for_subset))
        selected_indices = random.sample(indices_for_subset, actual_num_samples)
        logger.info(f"Selected {actual_num_samples} test profiles for plotting.")

        # --- DataLoader & Model ---
        collate_fn = create_multi_source_collate_fn()
        plot_loader = DataLoader(Subset(dataset, selected_indices), batch_size=actual_num_samples, collate_fn=collate_fn)

        logger.info(f"Loading model: {MODEL_CHECKPOINT}")
        model = create_prediction_model(cfg)
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        logger.info(f"Loading metadata: {NORMALIZATION_META_PATH}")
        with NORMALIZATION_META_PATH.open('r') as f: norm_meta = json.load(f)

        # --- Inference & Data Prep ---
        logger.info("Running inference...")
        inputs_batch, y_true_batch = next(iter(plot_loader))
        with torch.no_grad():
            predictions_batch = model({k: v.to(device) for k, v in inputs_batch.items()}).cpu()

        if "global" not in inputs_batch: sys.exit("No 'global' key in input batch.")
        global_tensors_norm = inputs_batch["global"].cpu()
        corner_data = np.stack([
            DataNormalizer.denormalize(global_tensors_norm[:, i], norm_meta, var_name).numpy()
            for i, var_name in enumerate(global_input_vars)
        ], axis=1)

        target_idx = cfg["target_variables"].index(target_var_for_mae)
        y_true_denorm = DataNormalizer.denormalize(y_true_batch[:, :, target_idx], norm_meta, target_var_for_mae).numpy()
        y_pred_denorm = DataNormalizer.denormalize(predictions_batch[:, :, target_idx], norm_meta, target_var_for_mae).numpy()
        mae_values = calculate_mae(y_true_denorm, y_pred_denorm)

        # --- Generate Corner Plot ---
        logger.info("Generating corner plot with MAE scatter...")
        # Plot histograms on diagonal, show titles and quantiles
        # Off-diagonal plots will be empty initially, we'll add scatter later
        fig = corner_plot_func(
            corner_data,
            labels=[var.replace("_", " ").title() for var in global_input_vars],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            quantiles=[0.16, 0.5, 0.84],
            plot_datapoints=False,  # Disable default scatter
            plot_density=False,    # Disable density/contours
            plot_contours=False,
            fill_contours=False,
            smooth=1.0,
            ret_fig=True
        )

        # Overlay scatter points colored by MAE (using Reds)
        num_global_vars = corner_data.shape[1]
        if num_global_vars >= 2:
            # Fix DeprecationWarning: Use plt.colormaps instead of plt.cm.get_cmap
            cmap_scatter = plt.colormaps["Reds"]
            norm_min = np.percentile(mae_values, 5)
            norm_max = np.percentile(mae_values, 95)

            for i in range(num_global_vars):
                for j in range(i):
                    ax = fig.axes[i * num_global_vars + j]
                    if ax.get_visible():
                        scatter_artist = ax.scatter(
                            corner_data[:, j], corner_data[:, i],
                            c=mae_values, cmap=cmap_scatter, s=10, alpha=1.0,
                            vmin=norm_min, vmax=norm_max, edgecolors='face',
                        )

            # Add colorbar
            cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
            fig.colorbar(scatter_artist, cax=cbar_ax, label=f"MAE ({target_var_for_mae.replace('_', ' ').title()})")
            fig.subplots_adjust(top=0.92, right=0.85)
        else:
            fig.subplots_adjust(top=0.92, right=0.95)

        fig.suptitle(f"Corner Plot (Test Set Sample, N={actual_num_samples})", fontsize=16)

        # Save the plot
        plot_filename = PLOTS_DIR / "corner_plot_globals_mae.png"
        plt.savefig(plot_filename, dpi=250)
        logger.info(f"Plot saved to: {plot_filename}")
        plt.close(fig)

    except FileNotFoundError as e:
        logger.error(f"File/Dir not found: {e}")
    except KeyError as e:
        logger.error(f"Config key error: {e}.")
    except ImportError as e:
        logger.error(f"Import error: {e}.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
