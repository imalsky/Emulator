#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import optuna # For type hinting Optuna.Trial
import torch

# Local application imports
from dataset import AtmosphericDataset, PadCollate, create_multi_source_collate_fn
from hardware import setup_device
from hyperparams import run_hyperparameter_search
from normalizer import DataNormalizer
from train import ModelTrainer
from utils import (
    ensure_dirs,
    load_config,
    save_json,
    seed_everything,
    setup_logging,
)

# Suppress specific PyTorch UserWarnings for cleaner output.
# These warnings are often informational and don't indicate critical issues.
warnings.filterwarnings(
    "ignore",
    message=(
        "enable_nested_tensor is True, but self.use_nested_tensor is False "
        "because encoder_layer.norm_first was True"
    ),
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="torch.nn.functional.glu is deprecated.",
    category=UserWarning
)


logger = logging.getLogger(__name__)

# Default paths and constants for the pipeline.
DEFAULT_CONFIG_PATH = Path("inputs/model_input_params.jsonc")
DEFAULT_DATA_DIR_PATH = Path("data")
DEFAULT_OPTUNA_TRIALS = 100


def _normalize_data(config: Dict[str, Any], data_root_dir: Path) -> bool:
    """
    Performs data normalization based on the provided configuration.

    This function initializes a DataNormalizer, calculates global statistics
    from the raw profiles, and then processes these profiles to create
    normalized versions.

    Args:
        config: The main configuration dictionary, containing normalization
                settings and variable definitions.
        data_root_dir: The root directory for data, expected to contain
                       a 'profiles' subdirectory with raw data.

    Returns:
        True if normalization was successful.
        The function relies on DataNormalizer to call sys.exit(1)
        on critical, unrecoverable errors.
    """
    raw_profiles_dir = data_root_dir / "profiles"
    normalized_profiles_dir = data_root_dir / "normalized_profiles"
    ensure_dirs(raw_profiles_dir, normalized_profiles_dir)

    norm_config_section = config.get("normalization", {})
    normalizer_instance = DataNormalizer(
        input_folder=raw_profiles_dir,
        output_folder=normalized_profiles_dir,
        config_data=config,
        epsilon=norm_config_section.get("epsilon"),
    )
    global_stats = normalizer_instance.calculate_global_stats(
        key_methods=norm_config_section.get("key_methods"),
        default_method=norm_config_section.get("default_method", "standard"),
        clip_outliers_before_scaling=norm_config_section.get(
            "clip_outliers_before_scaling", False
        ),
        symlog_percentile=norm_config_section.get("symlog_percentile", 0.5),
        symlog_thresholds=norm_config_section.get("symlog_thresholds"),
    )
    # DataNormalizer.calculate_global_stats will sys.exit on critical errors.
    # If it returns None without exiting, it's an unexpected state.
    if global_stats is None:
        logger.error(
            "Normalization statistics calculation returned None unexpectedly. "
            "This indicates an issue within DataNormalizer if sys.exit was not called."
        )
        return False
    normalizer_instance.process_profiles(global_stats)
    return True


def _initialize_dataset_and_collate(
    config: Dict[str, Any], data_dir: Path
) -> Optional[Tuple[AtmosphericDataset, PadCollate]]:
    """
    Loads and prepares the dataset and the custom collate function.

    Args:
        config: The main configuration dictionary.
        data_dir: Root directory where normalized data ('normalized_profiles')
                  is expected to be located.

    Returns:
        A tuple containing (AtmosphericDataset instance, PadCollate instance),
        or None if the setup fails (e.g., normalized data not found).
        AtmosphericDataset itself will sys.exit on critical initialization errors.
    """
    normalized_data_path = data_dir / "normalized_profiles"
    if not normalized_data_path.exists():
        logger.error(
            "Normalized data directory not found at %s. "
            "Please run the pipeline with the --normalize flag first.",
            normalized_data_path
        )
        return None
    try:
        padding_val = float(config.get("padding_value", 0.0))
        dataset_instance = AtmosphericDataset(
            data_folder=normalized_data_path,
            input_variables=config["input_variables"],
            target_variables=config["target_variables"],
            global_variables=config.get("global_variables", []),
            sequence_types=config["sequence_types"],
            output_seq_type=config["output_seq_type"],
            padding_value=padding_val,
            cache_size=config.get("dataset_cache_size", 1024),
            validate_profiles=config.get("validate_profiles", True),
        )
        collate_function = create_multi_source_collate_fn(
            padding_value=padding_val
        )
        return dataset_instance, collate_function
    except Exception as exc:
        # Catch any other unexpected errors during dataset or collate setup.
        # AtmosphericDataset is expected to sys.exit on its own critical errors.
        logger.error(
            "Dataset and collate function setup failed with an unexpected error: %s",
            exc, exc_info=True
        )
        return None


def _execute_model_training(
    optuna_trial_obj: Optional[optuna.Trial],
    train_config: Dict[str, Any],
    compute_device: torch.device,
    dataset_obj: AtmosphericDataset,
    collate_fn_obj: Callable,
    model_save_dir: Path,
) -> float:
    """
    Initializes and runs the ModelTrainer for a single training process.

    Args:
        optuna_trial_obj: An Optuna Trial object if part of a tuning run.
        train_config: The configuration dictionary for this training run.
        compute_device: The PyTorch device for training.
        dataset_obj: The pre-loaded AtmosphericDataset instance.
        collate_fn_obj: The collate function for DataLoaders.
        model_save_dir: Directory where model artifacts are saved.

    Returns:
        The best validation loss achieved during this training run.
    Raises:
        RuntimeError: If ModelTrainer.train() does not return a float.
    """
    trainer = ModelTrainer(
        config=train_config,
        device=compute_device,
        save_dir=model_save_dir,
        dataset=dataset_obj,
        collate_fn=collate_fn_obj,
        optuna_trial=optuna_trial_obj,
    )
    best_metric_from_train = trainer.train()
    if not isinstance(best_metric_from_train, float):
        error_msg = (
            f"ModelTrainer.train() returned type {type(best_metric_from_train)} "
            "instead of float. This indicates an issue in the training loop."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    return best_metric_from_train


def _initiate_hyperparameter_tuning(
    base_run_config: Dict[str, Any], data_directory: Path
) -> bool:
    """
    Manages the hyperparameter tuning process using Optuna.

    Args:
        base_run_config: The base configuration dictionary.
        data_directory: The root data directory.

    Returns:
        True if tuning completes successfully, False otherwise.
    """
    num_optuna_trials = base_run_config.get(
        "optuna_num_trials", DEFAULT_OPTUNA_TRIALS
    )
    logger.info(
        "Starting hyperparameter search with %d trials.", num_optuna_trials
    )
    tuning_output_dir = data_directory / "tuning_results"
    ensure_dirs(tuning_output_dir)

    def optuna_objective_wrapper(
        current_trial: optuna.Trial,
        current_trial_config: Dict[str, Any],
        device_for_trial: torch.device,
        dataset_for_trial: AtmosphericDataset,
        collate_for_trial: Callable,
        trial_artifacts_root: Path,
    ) -> float:
        return _execute_model_training(
            current_trial, current_trial_config, device_for_trial,
            dataset_for_trial, collate_for_trial, trial_artifacts_root
        )

    best_configuration_found = run_hyperparameter_search(
        base_config=base_run_config,
        data_dir=str(data_directory),
        output_dir=str(tuning_output_dir),
        setup_dataset_func=_initialize_dataset_and_collate,
        train_model_func=optuna_objective_wrapper,
        setup_device_func=setup_device,
        save_config_func=save_json,
        num_trials=num_optuna_trials,
    )
    return best_configuration_found is not None


def _parse_command_line_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    Returns:
        An argparse.Namespace object with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Atmospheric transformer modeling pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="Path to the main configuration file (JSON or JSONC/JSON5)."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR_PATH,
        help="Path to the root data directory."
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Run the data normalisation step using profiles in data-dir/profiles."
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train a model using the fixed configuration specified in the config file."
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter search to find an optimal model configuration."
    )
    parsed_args = parser.parse_args()

    if not (parsed_args.normalize or parsed_args.train or parsed_args.tune):
        parser.error(
            "No action specified. Please choose at least one: "
            "--normalize, --train, or --tune."
        )
    return parsed_args


def main() -> int:
    """
    Main entry point for the script.

    Orchestrates actions based on command-line arguments and configuration.

    Returns:
        0 on success, 1 on failure.
    """
    cli_args = _parse_command_line_args()
    setup_logging(level=logging.INFO)

    logger.info("Using configuration file: %s", cli_args.config.resolve())
    logger.info("Using data directory: %s", cli_args.data_dir.resolve())

    main_config = load_config(cli_args.config)
    if main_config is None:
        logger.critical(
            "Main configuration could not be loaded or validated from %s. Terminating.",
            cli_args.config
            )
        return 1

    seed_everything(main_config.get("random_seed", 42))
    ensure_dirs(
        cli_args.data_dir / "profiles",
        cli_args.data_dir / "normalized_profiles",
        cli_args.data_dir / "model",
        cli_args.data_dir / "tuning_results",
    )

    if cli_args.normalize:
        logger.info("Starting data normalization process...")
        if not _normalize_data(main_config, cli_args.data_dir):
            logger.error("Data normalization step reported failure. Terminating.")
            return 1
        logger.info("Data normalization finished successfully.")

    if cli_args.train:
        logger.info("Starting model training with fixed configuration...")
        selected_device = setup_device()
        dataset_setup_result = _initialize_dataset_and_collate(
            main_config, cli_args.data_dir
        )
        if dataset_setup_result is None:
            logger.error("Failed to set up dataset for training. Terminating.")
            return 1
        main_dataset, main_collate_fn = dataset_setup_result

        try:
            fixed_train_model_dir = cli_args.data_dir / "model"
            ensure_dirs(fixed_train_model_dir)
            _execute_model_training(
                optuna_trial_obj=None,
                train_config=main_config,
                compute_device=selected_device,
                dataset_obj=main_dataset,
                collate_fn_obj=main_collate_fn,
                model_save_dir=fixed_train_model_dir
            )
            logger.info(
                "Fixed configuration model training finished. Artifacts saved in %s",
                fixed_train_model_dir,
            )
        except Exception as e:
            logger.error(
                "Fixed configuration training failed unexpectedly: %s",
                e, exc_info=True
            )
            return 1

    if cli_args.tune:
        logger.info("Starting hyperparameter tuning process...")
        if not _initiate_hyperparameter_tuning(main_config, cli_args.data_dir):
            logger.error(
                "Hyperparameter tuning process reported failure or was interrupted."
            )
            return 1
        logger.info("Hyperparameter tuning finished successfully.")

    logger.info("All requested pipeline actions completed.")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except SystemExit as e:
        # Catch sys.exit to ensure the process terminates with the correct code
        sys.exit(e.code)
    except Exception as e:
        # Fallback for any other unhandled exceptions from main()
        logger.critical(
            "An unhandled error occurred in main execution: %s", e, exc_info=True
        )
        sys.exit(1) # General error code for unhandled exceptions
