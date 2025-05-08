#!/usr/bin/env python3
"""
spectra_plotting.py - Plots a selection of test profiles comparing true and
                      predicted spectra, along with the absolute error, based
                      on filenames specified in test_set_info.json.
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
from torch.cuda.amp import autocast # Added for AMP

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_project_root(marker_file: str = 'src') -> Path:
    """Finds the project root directory by looking for a marker."""
    current_path = Path(__file__).resolve().parent
    while current_path != current_path.parent:
        if (current_path / marker_file).is_dir():
            return current_path
        current_path = current_path.parent
    logger.warning(f"Could not find '{marker_file}' dir. Assuming script is in project root.")
    return Path(__file__).resolve().parent

def main():
    """
    Loads data, model, runs inference on selected test profiles, and plots spectra.
    """
    try:
        SCRIPT_DIR = Path(__file__).resolve().parent
        try:
            PROJECT_ROOT = find_project_root('src')
        except FileNotFoundError:
            PROJECT_ROOT = SCRIPT_DIR.parent
            logger.warning(f"Could not find 'src' dir. Assuming project root is: {PROJECT_ROOT}")

        SRC_DIR = PROJECT_ROOT / "src"
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        STYLE_FILE = SCRIPT_DIR / "science.mplstyle"
        if STYLE_FILE.exists():
            plt.style.use(STYLE_FILE)
            logger.info(f"Using style file: {STYLE_FILE}")
        else:
            logger.warning(f"Style file not found: {STYLE_FILE}. Using default matplotlib styles.")

        # Define paths
        CONFIG_PATH = PROJECT_ROOT / "inputs" / "model_input_params.jsonc"
        MODEL_CHECKPOINT = PROJECT_ROOT / "data" / "model" / "best_model.pt"
        TEST_SET_INFO_PATH = PROJECT_ROOT / "data" / "model" / "test_set_info.json"
        DATA_DIR = PROJECT_ROOT / "data" / "normalized_profiles"
        NORMALIZATION_META_PATH = DATA_DIR / "normalization_metadata.json"

        # Internal imports
        from utils import load_config # type: ignore
        from dataset import AtmosphericDataset, create_multi_source_collate_fn # type: ignore
        from model import create_prediction_model # type: ignore
        from normalizer import DataNormalizer # type: ignore

        # --- Configuration & Setup ---
        logger.info(f"Loading configuration from: {CONFIG_PATH}")
        cfg = load_config(CONFIG_PATH)
        if cfg is None:
            logger.error("Failed to load configuration.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running on device: {device}")

        num_samples_to_plot = 10

        # --- Load Dataset ---
        logger.info(f"Loading dataset from: {DATA_DIR}")
        dataset = AtmosphericDataset(
            data_folder=DATA_DIR,
            input_variables=cfg["input_variables"],
            target_variables=cfg["target_variables"],
            global_variables=cfg.get("global_variables", []),
            sequence_types=cfg["sequence_types"],
            sequence_lengths=cfg["sequence_lengths"],
            output_seq_type=cfg["output_seq_type"],
            validate_profiles=False,
        )
        logger.info(f"Dataset loaded with {len(dataset)} profiles.")

        # --- Load Test Filenames and Select Samples ---
        if not TEST_SET_INFO_PATH.exists():
            logger.error(f"Test set info file not found: {TEST_SET_INFO_PATH}")
            return

        logger.info(f"Loading test set information from: {TEST_SET_INFO_PATH}")
        with TEST_SET_INFO_PATH.open('r') as f:
            test_info = json.load(f)
        test_filenames_from_json = test_info.get("test_filenames")

        if not test_filenames_from_json:
            logger.error("No 'test_filenames' found in test_set_info.json.")
            return

        all_ds_filenames_str = [p.name for p in dataset.valid_files]

        # Find indices in the current dataset corresponding to the test filenames
        indices_for_subset = []
        valid_filenames_found = []
        for fname in test_filenames_from_json:
            try:
                idx = all_ds_filenames_str.index(fname)
                indices_for_subset.append(idx)
                valid_filenames_found.append(fname)
            except ValueError:
                logger.warning(f"Test filename '{fname}' from list not found in current dataset directory.")

        if not indices_for_subset:
            logger.error("None of the specified test profiles were found in the dataset directory.")
            return

        # Select a random subset of the *found* test profiles if needed
        if len(indices_for_subset) > num_samples_to_plot:
            selected_indices = random.sample(indices_for_subset, num_samples_to_plot)
            logger.info(f"Randomly selected {num_samples_to_plot} profiles from the available test set for plotting.")
        else:
            selected_indices = indices_for_subset
            logger.info(f"Plotting all {len(selected_indices)} available test profiles found.")

        if not selected_indices:
             logger.info("No profiles to plot.")
             return

        logger.info(f"Selected profiles by index: {selected_indices}")
        logger.info(f"Corresponding filenames: {[all_ds_filenames_str[i] for i in selected_indices]}")

        # --- Create DataLoader for Selected Samples ---
        collate_fn = create_multi_source_collate_fn()
        plot_subset = Subset(dataset, selected_indices)
        plot_loader = DataLoader(
            plot_subset,
            batch_size=len(selected_indices), # Load all in one batch
            collate_fn=collate_fn
        )

        # --- Load Model ---
        logger.info(f"Loading model from: {MODEL_CHECKPOINT}")
        model = create_prediction_model(cfg)
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")

        # --- Load Normalization Metadata ---
        if not NORMALIZATION_META_PATH.exists():
            logger.error(f"Normalization metadata file not found: {NORMALIZATION_META_PATH}")
            return
        logger.info(f"Loading normalization metadata from: {NORMALIZATION_META_PATH}")
        with NORMALIZATION_META_PATH.open('r') as f:
            norm_meta = json.load(f)

        # --- Perform Inference ---
        inputs_batch, y_true_batch = next(iter(plot_loader))
        inputs_batch_device = {k: v.to(device) for k, v in inputs_batch.items()}

        logger.info("Running inference...")
        t_start = time.perf_counter()
        with torch.no_grad():
            if device.type == 'cuda':
                with autocast(): # AMP enabled for CUDA
                    predictions_batch = model(inputs_batch_device).cpu()
            else: # CPU execution
                predictions_batch = model(inputs_batch_device).cpu() # .cpu() is fine here as model is on cpu

        infer_time_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"Inference for {len(selected_indices)} profiles took {infer_time_ms:.2f} ms.")

        # --- Denormalize Data ---
        # Determine x-axis and y-axis variables from config (minimal check)
        x_axis_var_name = cfg["input_variables"][0]
        y_axis_var_name = cfg["target_variables"][0]
        input_sequence_key = next((k for k, v_list in cfg["sequence_types"].items() if x_axis_var_name in v_list), None)

        if not input_sequence_key:
            logger.error("Could not determine input sequence key for x-axis.")
            return

        x_axis_feature_index = cfg["sequence_types"][input_sequence_key].index(x_axis_var_name)
        y_axis_feature_index = cfg["target_variables"].index(y_axis_var_name) # Assuming target is single-feature sequence

        x_denorm = DataNormalizer.denormalize(
            inputs_batch[input_sequence_key][:, :, x_axis_feature_index], norm_meta, x_axis_var_name
        ).numpy()
        y_true_denorm = DataNormalizer.denormalize(
            y_true_batch[:, :, y_axis_feature_index], norm_meta, y_axis_var_name
        ).numpy()
        y_pred_denorm = DataNormalizer.denormalize(
            predictions_batch[:, :, y_axis_feature_index], norm_meta, y_axis_var_name
        ).numpy()

        # --- Plotting ---
        logger.info("Generating plots...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for i in range(len(selected_indices)):
            axes[0].plot(x_denorm[i], y_true_denorm[i], color='black', linewidth=2, alpha=0.7, label="True" if i == 0 else None)
            axes[0].plot(x_denorm[i], y_pred_denorm[i], color='red', linestyle='--', linewidth=2, alpha=0.7, label="Predicted" if i == 0 else None)

            error = y_pred_denorm[i] - y_true_denorm[i]
            axes[1].plot(x_denorm[i], error, color='black', linewidth=1, alpha=0.7, label="Error" if i == 0 else None)

        axes[0].set_xlabel(f"{x_axis_var_name.replace('_', ' ').title()} (denormalized)")
        axes[0].set_ylabel(f"{y_axis_var_name.replace('_', ' ').title()} (denormalized)")
        axes[0].legend()

        axes[1].set_xlabel(f"{x_axis_var_name.replace('_', ' ').title()} (denormalized)")
        axes[1].set_ylabel("Absolute Error (Predicted - True)")
        if len(selected_indices) > 0: axes[1].legend()

        fig.suptitle("Model Predictions vs True Values (Test Set Sample)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plot_filename = PROJECT_ROOT / "plots" / "simple.png"
        plt.savefig(plot_filename, dpi=250)
        logger.info(f"Plot saved to: {plot_filename}")
        plt.close(fig)

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
    except KeyError as e:
        logger.error(f"Configuration key error: {e}. Check config file.")
    except ImportError as e:
        logger.error(f"Import error: {e}.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()