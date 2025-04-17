#!/usr/bin/env python3
"""
main.py — Streamlined entry-point for the atmospheric profile prediction pipeline.

Handles command-line arguments, configuration loading/validation, and orchestrates
the normalization, training, and hyperparameter tuning processes.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional # Added Optional

import torch
import numpy as np # Needed for seeding

# Import necessary components from other modules
from hardware import setup_device
from normalizer import DataNormalizer
from dataset import AtmosphericDataset, create_multi_source_collate_fn
from train import ModelTrainer
from hyperparams import run_hyperparameter_search
from utils import (
    ensure_dirs,
    load_config,
    setup_logging,
    validate_config, # Use the strict validation
    save_json,       # For saving tuning results
    seed_everything # For reproducibility
)


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper Functions (Orchestration Logic)
# -----------------------------------------------------------------------------

def _normalize(cfg: dict, data_root: Path) -> bool:
    """Run data normalization; returns True on success."""
    raw_dir = data_root / "profiles"
    out_dir = data_root / "normalized_profiles"

    try:
        logger.info("Normalising %s → %s", raw_dir, out_dir)
        # Pass normalization sub-config or empty dict
        norm_cfg = cfg.get("normalization", {})
        normalizer = DataNormalizer(raw_dir, out_dir)
        stats = normalizer.calculate_global_stats(
            key_methods=norm_cfg.get("key_methods"),
            default_method=norm_cfg.get("default_method", "standard"), # Match default in config if exists
            clip_outliers_before_scaling=norm_cfg.get("clip_outliers_before_scaling", False),
            symlog_percentile=norm_cfg.get("symlog_percentile", 0.5),
            symlog_thresholds=norm_cfg.get("symlog_thresholds")
        )
        if not stats:
             raise RuntimeError("Normalization statistics calculation failed.")
        normalizer.process_profiles(stats)
        logger.info("Normalisation finished successfully.")
        return True
    except Exception as e:
        logger.exception("Normalisation failed: %s", e)
        return False


def _setup_dataset(config: dict, data_dir: Path) -> Optional[Tuple[AtmosphericDataset, Callable]]:
    """
    Initialize the dataset using parameters from config.
    Returns (dataset, collate_fn) tuple or (None, None) on failure.
    """
    try:
        logger.info("Initializing dataset (strict lengths from config)...")
        norm_dir = data_dir / "normalized_profiles"
        if not norm_dir.exists():
             logger.error(f"Normalized data directory not found: {norm_dir}")
             logger.error("Please run with --normalize first.")
             return None, None

        # --- Crucial validation moved to utils.validate_config ---
        # Ensure required keys for dataset exist (checked by validate_config)
        sequence_lengths = config["sequence_lengths"]
        output_seq_type = config["output_seq_type"]
        # ---

        ds = AtmosphericDataset(
            data_folder=str(norm_dir),
            input_variables=config["input_variables"],
            target_variables=config["target_variables"],
            global_variables=config.get("global_variables", []),
            sequence_types=config["sequence_types"],
            sequence_lengths=sequence_lengths, # Pass dict from config
            output_seq_type=output_seq_type, # Pass key from config
            cache_size=config.get("dataset_cache_size", 1024) # Optional cache size
        )
        logger.info("Dataset loaded: %d profiles.", len(ds))
        collate_fn = create_multi_source_collate_fn()
        return ds, collate_fn
    except KeyError as e:
         logger.error(f"Dataset setup failed: Missing required configuration key: {e}")
         return None, None
    except Exception as e:
        logger.exception("Dataset setup failed: %s", e)
        return None, None


def _train(cfg: dict, data_dir: Path) -> bool:
    """Run model training; returns True on success."""
    logger.info("Starting model training...")
    device = setup_device()
    seed_everything(cfg.get("random_seed", 42)) # Seed before dataset/model init

    dataset, collate_fn = _setup_dataset(cfg, data_dir)
    if dataset is None:
        return False # Error logged in _setup_dataset

    try:
        trainer = ModelTrainer(
            config=cfg,
            device=device,
            save_dir=data_dir / "model",
            dataset=dataset,
            collate_fn=collate_fn,
        )
        # train() method now returns best validation loss
        best_loss = trainer.train()
        logger.info("Training complete (Best Val Loss = %.4e)", best_loss)
        # Save final metrics including best loss
        save_json({"best_val_loss": best_loss}, data_dir / "model" / "metrics.json")
        return True
    except Exception as e:
         logger.exception("Model training failed: %s", e)
         return False


def _tune(cfg: dict, data_dir: Path, num_trials: int) -> bool:
    """Run hyperparameter tuning; returns True on success."""
    logger.info("Starting hyper-parameter search (%d trials)...", num_trials)
    seed_everything(cfg.get("random_seed", 42)) # Seed for reproducibility in Optuna

    try:
        # This is the function Optuna will call for each trial's training run
        # It needs to return the metric to minimize (best validation loss)
        def _train_for_tuning(
             trial_cfg: dict, device: torch.device, ds, collate, data_root: Path
        ) -> float:
            # Use a temporary directory for trial artifacts to avoid conflicts
            trial_save_dir = data_root / f"tuning_trial_{trial_cfg.get('optuna_trial_number', 0)}"
            trainer = ModelTrainer(
                config=trial_cfg,
                device=device,
                save_dir=trial_save_dir, # Save trial models separately
                dataset=ds,
                collate_fn=collate,
            )
            best_loss = trainer.train() # Returns best validation loss
            # Clean up trial directory? Optional.
            # import shutil
            # shutil.rmtree(trial_save_dir, ignore_errors=True)
            return best_loss

        # Run the search
        best_config = run_hyperparameter_search(
            base_config=cfg,
            data_dir=str(data_dir),
            output_dir=str(data_dir / "tuning_results"),
            # Pass _setup_dataset directly
            setup_dataset_func=_setup_dataset,
            # Pass the wrapper function that returns the float metric
            train_model_func=_train_for_tuning,
            setup_device_func=setup_device,
            # Pass the save_json utility
            save_config_func=save_json,
            num_trials=num_trials,
        )
        if best_config:
            logger.info("Tuning finished. Best configuration saved.")
            return True
        else:
            logger.error("Tuning finished but no best configuration was determined (all trials may have failed).")
            return False
    except Exception as e:
         logger.exception("Hyperparameter tuning failed: %s", e)
         return False

# -----------------------------------------------------------------------------
# Command-Line Interface Setup
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Atmospheric profile prediction pipeline")
    # Actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--normalize", action="store_true", help="Run data normalization only")
    action_group.add_argument("--train", action="store_true", help="Train the model using config")
    action_group.add_argument("--tune", action="store_true", help="Run hyper-parameter search (Optuna)")

    # Common arguments
    parser.add_argument("--config", type=Path, default=Path("inputs/model_input_params.jsonc"),
                        help="Path to configuration file (JSON/JSONC)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Base directory for data (profiles, normalized_profiles, model)")

    # Tuning specific arguments
    parser.add_argument("--trials", type=int, default=25,
                        help="Number of trials for hyper-parameter tuning (used with --tune)")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main Execution Logic
# -----------------------------------------------------------------------------

def main() -> int:
    """Parses arguments and runs the requested pipeline steps."""
    args = _parse_args()
    setup_logging() # Setup logging first

    # --- Load and Validate Configuration ---
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    if config is None:
        # Error logged in load_config
        return 1 # Exit code 1 for failure
    logger.info("Validating configuration...")
    if not validate_config(config):
        # Error logged in validate_config
        return 1 # Exit code 1 for failure
    logger.info("Configuration loaded and validated successfully.")
    # --- End Config Handling ---

    # Ensure base directories exist
    ensure_dirs(
        args.data_dir / "profiles",
        args.data_dir / "normalized_profiles",
        args.data_dir / "model",
        args.data_dir / "tuning_results" # Also ensure tuning results dir
    )

    exit_code = 0 # Default to success

    # --- Execute Actions ---
    if args.normalize:
        if not _normalize(config, args.data_dir):
            exit_code = 1 # Normalization failed
    elif args.train:
        if not _train(config, args.data_dir):
            exit_code = 1 # Training failed
    elif args.tune:
        if not _tune(config, args.data_dir, args.trials):
             exit_code = 1 # Tuning failed
    # --- End Actions ---

    if exit_code == 0:
        logger.info("Pipeline finished successfully.")
    else:
        logger.error("Pipeline finished with errors.")

    return exit_code


if __name__ == "__main__":
    # Set matmul precision (useful for TF32 on Ampere GPUs)
    # torch.set_float32_matmul_precision("medium") # Or 'high'
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user (Ctrl+C).")
        sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
         # Catch any unexpected errors at the top level
         logger.exception("An unexpected error occurred: %s", e)
         sys.exit(1) # General error exit code