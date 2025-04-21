#!/usr/bin/env python3
"""
main.py — Command-line entry point for pipeline.

Handles command-line arguments, configuration loading/validation, and
orchestrates the normalization, training, and hyperparameter tuning processes
via helper functions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from optuna.exceptions import TrialPruned

# Import necessary components from other modules
from hardware import setup_device
from normalizer import DataNormalizer
from dataset import AtmosphericDataset, create_multi_source_collate_fn
from train import ModelTrainer
from hyperparams import run_hyperparameter_search
from utils import (
    ensure_dirs,
    load_config,  # Handles internal validation
    setup_logging,
    save_json,  # For saving tuning results
    seed_everything,  # For reproducibility
)

logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions (Orchestration Logic)
# =============================================================================


def _normalize(cfg: Dict[str, Any], data_root: Path) -> bool:
    """
    Runs the data normalization process using DataNormalizer.

    Args:
        cfg: The main configuration dictionary.
        data_root: The base data directory path.

    Returns:
        True if normalization completes successfully, False otherwise.
    """
    raw_dir = data_root / "profiles"
    out_dir = data_root / "normalized_profiles"
    ensure_dirs(raw_dir, out_dir)

    try:
        logger.info(
            "Starting normalization: profiles from %s to %s",
            raw_dir.name,
            out_dir.name,
        )
        norm_cfg = cfg.get("normalization", {})
        normalizer = DataNormalizer(raw_dir, out_dir)

        stats = normalizer.calculate_global_stats(
            key_methods=norm_cfg.get("key_methods"),
            default_method=norm_cfg.get("default_method", "standard"),
            clip_outliers_before_scaling=norm_cfg.get(
                "clip_outliers_before_scaling", False
            ),
            symlog_percentile=norm_cfg.get("symlog_percentile", 0.5),
            symlog_thresholds=norm_cfg.get("symlog_thresholds"),
        )

        if stats is None:
            raise RuntimeError("Normalization statistics calculation failed.")

        normalizer.process_profiles(stats)
        logger.info("Normalization finished successfully.")
        return True

    except Exception as e:
        logger.exception("Normalization failed: %s", e)
        return False


def _setup_dataset(
    config: Dict[str, Any], data_dir: Path
) -> Optional[Tuple[AtmosphericDataset, Callable]]:
    """
    Initializes the AtmosphericDataset using parameters from the configuration.

    Performs strict length validation based on `sequence_lengths` in the config.

    Args:
        config: The main configuration dictionary.
        data_dir: The base data directory path.

    Returns:
        A tuple (dataset, collate_fn) on success, or None on failure. Returns
        None if the dataset setup fails.
    """
    logger.info("Initializing dataset (strict lengths from config)...")
    norm_dir = data_dir / "normalized_profiles"
    if not norm_dir.exists():
        logger.error("Normalized data directory not found: %s", norm_dir)
        logger.error("Please run with --normalize first.")
        return None

    try:
        sequence_lengths = config["sequence_lengths"]
        output_seq_type = config["output_seq_type"]

        ds = AtmosphericDataset(
            data_folder=str(norm_dir),
            input_variables=config["input_variables"],
            target_variables=config["target_variables"],
            global_variables=config.get("global_variables", []),
            sequence_types=config["sequence_types"],
            sequence_lengths=sequence_lengths,
            output_seq_type=output_seq_type,
            cache_size=config.get("dataset_cache_size", 1024),
        )

        logger.info("Dataset loaded: %d profiles.", len(ds))
        collate_fn = create_multi_source_collate_fn()
        return ds, collate_fn

    except (KeyError, ValueError, FileNotFoundError) as e:
        logger.error("Dataset setup failed: %s", e)
        return None
    except Exception as e:
        logger.exception(
            "Dataset setup failed with an unexpected error: %s", e
        )
        return None


def train_model(
    config: Dict[str, Any],
    device: torch.device,
    dataset: AtmosphericDataset,
    collate_fn: Callable,
    save_dir: Path,
) -> float:
    """
    Initializes and runs the ModelTrainer for a single training run.

    Args:
        config: The configuration dictionary for this run.
        device: The torch device to train on.
        dataset: The initialized dataset instance.
        collate_fn: The collate function for the DataLoader.
        save_dir: The directory to save model artifacts (checkpoints, logs).

    Returns:
        The best validation loss achieved during training (float). Returns
        float('inf') on failure.

    Raises:
        RuntimeError: If training encounters an unrecoverable error.
    """
    logger.info("Starting model training run...")
    ensure_dirs(save_dir)

    try:
        trainer = ModelTrainer(
            config=config,
            device=device,
            save_dir=save_dir,
            dataset=dataset,
            collate_fn=collate_fn,
        )
        best_loss = trainer.train()

        if not isinstance(best_loss, float):
            logger.error("Trainer did not return a float validation loss!")
            return float("inf")

        logger.info("Training run complete (Best Val Loss = %.4e)", best_loss)
        return best_loss

    except Exception as e:
        logger.exception("Model training failed: %s", e)
        raise RuntimeError(f"Training failed: {e}") from e


def _run_tuning(cfg: Dict[str, Any], data_dir: Path, num_trials: int) -> bool:
    """
    Runs hyperparameter tuning using Optuna.

    Args:
        cfg: The base configuration dictionary.
        data_dir: The base data directory path.
        num_trials: The number of trials to run.

    Returns:
        True if tuning completes and finds a best config, False otherwise.
    """
    logger.info("Starting hyper-parameter search (%d trials)...", num_trials)
    seed_everything(cfg.get("random_seed", 42))

    tuning_output_dir = data_dir / "tuning_results"
    ensure_dirs(tuning_output_dir)

    try:
        def _train_objective_wrapper(
            trial_cfg: Dict[str, Any],
            device: torch.device,
            ds: AtmosphericDataset,
            collate: Callable,
            data_root: Path,
        ) -> float:
            trial_num = trial_cfg.get("optuna_trial_number", "N/A")
            trial_save_dir = data_root / "tuning_trials" / f"trial_{trial_num}"
            try:
                best_loss = train_model(
                    config=trial_cfg,
                    device=device,
                    dataset=ds,
                    collate_fn=collate,
                    save_dir=trial_save_dir,
                )
                return best_loss
            except Exception as trial_exc:
                logger.error(
                    f"Trial {trial_num} training function failed: {trial_exc}"
                )
                raise TrialPruned(f"Training failed: {trial_exc}") from trial_exc

        best_config_found = run_hyperparameter_search(
            base_config=cfg,
            data_dir=str(data_dir),
            output_dir=str(tuning_output_dir),
            setup_dataset_func=_setup_dataset,
            train_model_func=_train_objective_wrapper,
            setup_device_func=setup_device,
            save_config_func=save_json,
            num_trials=num_trials,
        )

        if best_config_found:
            logger.info(
                "Tuning finished. Best configuration saved in %s.",
                tuning_output_dir,
            )
            return True
        else:
            logger.error(
                "Tuning finished but no best configuration was determined."
            )
            return False

    except Exception as e:
        logger.exception(
            "Hyperparameter tuning failed unexpectedly: %s", e
        )
        return False


# =============================================================================
# Command-Line Interface Setup
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """Defines and parses command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Atmospheric profile prediction pipeline"
    )

    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--normalize", action="store_true", help="Run data normalization only."
    )
    action_group.add_argument(
        "--train",
        action="store_true",
        help="Train the model using the specified configuration.",
    )
    action_group.add_argument(
        "--tune",
        action="store_true",
        help="Run hyper-parameter search using Optuna.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("inputs/model_input_params.jsonc"),
        help="Path to the config file (JSON or JSONC format).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base directory for data and model artifacts.",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Number of trials for tuning (used with --tune).",
    )

    return parser.parse_args()


# =============================================================================
# Main Execution Logic
# =============================================================================


def main() -> int:
    """
    Parses command-line arguments and executes the requested pipeline steps.

    Returns:
        0 on success, 1 on failure.
    """
    args = _parse_args()
    setup_logging()

    logger.info("Loading configuration from: %s", args.config)
    config = load_config(args.config)
    if config is None:
        logger.error("Failed to load or validate configuration. Aborting.")
        return 1
    logger.info("Configuration loaded and validated successfully.")

    ensure_dirs(
        args.data_dir / "profiles",
        args.data_dir / "normalized_profiles",
        args.data_dir / "model",
        args.data_dir / "tuning_results",
    )

    seed = config.get("random_seed", 42)
    seed_everything(seed)

    exit_code = 0
    action_performed = False

    if args.normalize:
        action_performed = True
        if not _normalize(config, args.data_dir):
            exit_code = 1
            logger.error("Normalization step failed.")

    if args.train and exit_code == 0:
        action_performed = True
        logger.info("--- Starting Training ---")
        device = setup_device()
        dataset_info = _setup_dataset(config, args.data_dir)
        if dataset_info is None:
            exit_code = 1
            logger.error("Training step failed due to dataset setup error.")
        else:
            dataset, collate_fn = dataset_info
            try:
                train_model(
                    config=config,
                    device=device,
                    dataset=dataset,
                    collate_fn=collate_fn,
                    save_dir=(args.data_dir / "model"),
                )
            except Exception:
                exit_code = 1
                logger.error("Training step failed.")

    if args.tune and exit_code == 0:
        action_performed = True
        logger.info("--- Starting Hyperparameter Tuning ---")
        if not _run_tuning(config, args.data_dir, args.trials):
            exit_code = 1
            logger.error("Tuning step failed.")

    if not action_performed:
        logger.error("No action (--normalize, --train, --tune) was performed.")
        exit_code = 1

    if exit_code == 0:
        logger.info("Pipeline finished successfully.")
    else:
        logger.error("Pipeline finished with errors.")

    return exit_code


if __name__ == "__main__":
    try:
        exit_status = main()
        sys.exit(exit_status)
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logger.exception("An unexpected error occurred at the top level: %s", e)
        sys.exit(1)