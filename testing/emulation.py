#!/usr/bin/env python3
"""
plotting_simple.py - Minimal script to load the model and profiles specified
                     in test_set_info.json, plot model predictions against
                     true values for two features in a 4-panel plot.
"""

import sys
import json
import json5
from pathlib import Path
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import logging

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

def process_profile(profile, config, coord_var):
    """
    Builds the input dictionary from a profile using variable names from config.
    Returns the inputs dict, sequence length, and coord values. Minimal checks.
    """
    coord_vals = profile.get(coord_var, [])
    if not isinstance(coord_vals, list): coord_vals = [coord_vals]
    seq_len = len(coord_vals)
    if seq_len == 0: return {}, 0, []

    inputs = {}
    for seq_type, var_list in config.get("sequence_types", {}).items():
        if var_list:
            seq_features = []
            for var_name in var_list:
                val = profile.get(var_name, 0)
                if isinstance(val, list):
                    if len(val) < seq_len: val = val + [val[-1]] * (seq_len - len(val))
                    else: val = val[:seq_len]
                else:
                    val = [float(val)] * seq_len
                seq_features.append(torch.tensor(val, dtype=torch.float32))
            inputs[seq_type] = torch.stack(seq_features, dim=1).unsqueeze(0)
            break

    if config.get("global_variables"):
        global_values = [float(profile.get(var_name, [0])[0]) if isinstance(profile.get(var_name, [0]), list) else float(profile.get(var_name, 0)) for var_name in config["global_variables"]]
        inputs["global"] = torch.tensor(global_values, dtype=torch.float32).unsqueeze(0)
    return inputs, seq_len, coord_vals

def main():
    """
    Loads data, model, runs inference on specified test profiles, and generates plot.
    """
    try:
        SCRIPT_DIR = Path(__file__).resolve().parent
        PROJECT_ROOT = find_project_root('data')

        # Define paths
        DATA_PATH = PROJECT_ROOT / "data" / "normalized_profiles"
        METADATA_PATH = DATA_PATH / "normalization_metadata.json"
        MODEL_PATH = PROJECT_ROOT / "data" / "model" / "best_model.pt"
        CONFIG_PATH = PROJECT_ROOT / "inputs" / "model_input_params.jsonc"
        TEST_SET_INFO_PATH = PROJECT_ROOT / "data" / "model" / "test_set_info.json"
        PLOTS_DIR = SCRIPT_DIR / ".." / "plots"
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        STYLE_FILE = SCRIPT_DIR / "science.mplstyle"
        if STYLE_FILE.exists():
            plt.style.use(STYLE_FILE)
            logger.info(f"Using style file: {STYLE_FILE}")
        else:
            logger.warning(f"Style file not found: {STYLE_FILE}. Using default matplotlib styles.")

        # --- Loading configuration, metadata, and model ---
        logger.info(f"Loading config: {CONFIG_PATH}")
        with open(CONFIG_PATH, 'r') as f: config = json5.load(f)
        logger.info(f"Loading metadata: {METADATA_PATH}")
        with open(METADATA_PATH, 'r') as f: metadata = json.load(f)

        logger.info(f"Loading model: {MODEL_PATH}")
        model = create_prediction_model(config)
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        logger.info("Model loaded.")

        target_vars = config["target_variables"]
        feature1 = target_vars[0]
        feature2 = None
        if len(target_vars) > 1:
            feature2 = target_vars[1]
            logger.info(f"Using feature1: {feature1}, feature2: {feature2}")
        else:
            logger.warning("Only one target variable found. Plotting only for feature1.")

        coord_var = "pressure"
        num_profiles_to_plot = config.get("plotting_simple_num_samples", 3)

        # --- Select profiles based on test_set_info.json ---
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

        # Find which test files actually exist in the data directory
        all_files_in_dir = {f.name: f for f in Path(DATA_PATH).glob("*.json") if f.name != "normalization_metadata.json"}
        available_test_files = [all_files_in_dir[fname] for fname in test_filenames_from_json if fname in all_files_in_dir]

        if not available_test_files:
            logger.error("None of the test filenames from list were found in the data directory.")
            return

        # Select a random subset if needed
        if len(available_test_files) > num_profiles_to_plot:
            selected_files = random.sample(available_test_files, num_profiles_to_plot)
            logger.info(f"Randomly selected {num_profiles_to_plot} profiles from the available test set for plotting.")
        else:
            selected_files = available_test_files
            logger.info(f"Plotting all {len(selected_files)} available test profiles found.")


        if not selected_files:
             logger.info("No profiles to plot.")
             return

        # --- Plotting Setup ---
        num_rows = 2 if feature2 else 1
        fig, axs = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows), sharey=True)
        if num_rows == 1: axs = np.array([axs])

        profile_times = []
        errors_feature1 = []
        errors_feature2 = [] if feature2 else None

        legend_labels = {
            feature1: {"true": f"True {feature1}", "pred": f"Predicted {feature1}"}
        }
        if feature2:
            legend_labels[feature2] = {"true": f"True {feature2}", "pred": f"Predicted {feature2}"}

        lw = 2

        # --- Process and Plot ---
        logger.info(f"Processing {len(selected_files)} selected test profiles...")
        for file_index, file_path in enumerate(selected_files):
            try:
                start_time = time.time()
                with open(file_path, 'r') as f: profile = json.load(f)

                inputs, seq_len, coord_vals = process_profile(profile, config, coord_var)
                if not inputs: continue

                with torch.no_grad():
                    predictions = model(inputs)
                    if predictions.dim() == 2: predictions = predictions.unsqueeze(1)

                profile_times.append(time.time() - start_time)

                features_to_plot = [feature1]
                if feature2: features_to_plot.append(feature2)

                for j, feature in enumerate(features_to_plot):
                    true_vals = profile.get(feature, [0] * seq_len)
                    if not isinstance(true_vals, list): true_vals = [true_vals] * seq_len
                    target_idx = target_vars.index(feature)
                    pred_vals = predictions[0, :seq_len, target_idx].cpu().numpy()

                    # Import here to ensure it's available after path modification
                    from normalizer import DataNormalizer
                    true_denorm = np.array(DataNormalizer.denormalize(true_vals, metadata, feature)).flatten()
                    pred_denorm = np.array(DataNormalizer.denormalize(pred_vals, metadata, feature)).flatten()
                    coord_denorm = np.array(DataNormalizer.denormalize(coord_vals, metadata, coord_var)).flatten()

                    min_len = min(len(true_denorm), len(pred_denorm), len(coord_denorm))
                    true_denorm, pred_denorm, coord_denorm = true_denorm[:min_len], pred_denorm[:min_len], coord_denorm[:min_len]

                    epsilon = 1e-10
                    error = 100 * (pred_denorm - true_denorm) / np.maximum(np.abs(true_denorm), epsilon)

                    if j == 0: errors_feature1.extend(error.tolist())
                    elif errors_feature2 is not None: errors_feature2.extend(error.tolist())

                    ax_pred = axs[j, 0]
                    ax_err = axs[j, 1]

                    label_true = legend_labels[feature]["true"] if file_index == 0 else None
                    label_pred = legend_labels[feature]["pred"] if file_index == 0 else None

                    ax_pred.plot(true_denorm, coord_denorm, color='black', linestyle='solid', linewidth=lw, label=label_true)
                    ax_pred.plot(pred_denorm, coord_denorm, color='firebrick', linestyle='dashed', linewidth=lw, label=label_pred)
                    ax_err.plot(error, coord_denorm, color='firebrick', linestyle='solid', linewidth=1, alpha=0.7)

            except Exception as e:
                logger.warning(f"Error processing {file_path.name}: {e}")

        # --- Final Plot Configuration ---
        if profile_times:
            logger.info(f"Average processing time per profile: {1000 * np.mean(profile_times):.4e} ms")

        def print_error_metrics(errors, feat):
            if errors:
                mean_abs_err = np.mean(np.abs(errors))
                median_abs_err = np.median(np.abs(errors))
                logger.info(f"Feature {feat}: Mean Abs Err = {mean_abs_err:.2f}%, Median Abs Err = {median_abs_err:.2f}%")

        print_error_metrics(errors_feature1, feature1)
        if feature2: print_error_metrics(errors_feature2, feature2)

        axs[0, 0].set_xlabel(f'{feature1.replace("_", " ").title()}')
        axs[0, 0].set_ylabel("Pressure (bar)")
        axs[0, 0].set_xscale("symlog", linthresh=1e-10)
        axs[0, 0].set_yscale("log")
        axs[0, 0].set_ylim(1e2, 1e-5)
        axs[0, 0].legend(loc="best", fontsize="small")

        axs[0, 1].set_xlabel("Percent Error")
        axs[0, 1].set_xscale("symlog", linthresh=1)
        axs[0, 1].set_yscale("log")
        axs[0, 1].set_ylim(1e2, 1e-5)
        axs[0, 1].set_xlim(-100, 100)
        axs[0, 1].axvline(x=0, color="black", linestyle="dotted")

        if feature2:
            axs[1, 0].set_xlabel(f'{feature2.replace("_", " ").title()}')
            axs[1, 0].set_ylabel("Pressure (bar)")
            axs[1, 0].set_xscale("symlog", linthresh=1e-10)
            axs[1, 0].set_yscale("log")
            axs[1, 0].set_ylim(1e2, 1e-5)
            axs[1, 0].legend(loc="best", fontsize="small")

            axs[1, 1].set_xlabel("Percent Error")
            axs[1, 1].set_xscale("symlog", linthresh=1)
            axs[1, 1].set_yscale("log")
            axs[1, 1].set_ylim(1e2, 1e-5)
            axs[1, 1].set_xlim(-100, 100)
            axs[1, 1].axvline(x=0, color="black", linestyle="dotted")

        fig.suptitle("Model Predictions vs True Values (Test Set Sample)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the plot
        plot_filename = PLOTS_DIR / "simple_model_comparison.png"
        plt.savefig(plot_filename, dpi=250)
        logger.info(f"Plot saved to: {plot_filename}")
        plt.close(fig)

    except FileNotFoundError as e:
        logger.error(f"Required file or directory not found: {e}.")
    except KeyError as e:
        logger.error(f"Configuration key error: {e}.")
    except ImportError as e:
        logger.error(f"Import error: {e}.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
