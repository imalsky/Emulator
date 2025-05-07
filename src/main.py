#!/usr/bin/env python3
"""
main.py – command‑line entry point for the atmospheric‑flux pipeline.

This script handles command-line arguments to orchestrate data normalization,
model training with a fixed configuration, or hyperparameter tuning using Optuna.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import optuna
import torch

from dataset import AtmosphericDataset, create_multi_source_collate_fn
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

warnings.filterwarnings(
    "ignore",
    message=(
        "enable_nested_tensor is True, but self.use_nested_tensor is False "
        "because encoder_layer.norm_first was True"
    ),
)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("inputs/model_input_params.jsonc")
DATA_DIR_PATH = Path("data")
DEFAULT_NUM_TRIALS = 100


def _normalize(cfg: Dict[str, Any], data_root: Path) -> bool:
    """
    Performs data normalization based on the provided configuration.

    Calculates global statistics and applies normalization to profile data.

    Args:
        cfg: The main configuration dictionary.
        data_root: The root directory for data (containing 'profiles').

    Returns:
        True if normalization was successful, False otherwise.
    """
    raw_dir = data_root / "profiles"
    out_dir = data_root / "normalized_profiles"
    ensure_dirs(raw_dir, out_dir)
    norm_cfg = cfg.get("normalization", {})
    normalizer = DataNormalizer(
        input_folder=raw_dir,
        output_folder=out_dir,
        config_data=cfg,
        epsilon=norm_cfg.get("epsilon"),
    )
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
        logger.error("Normalization statistics calculation failed.")
        return False
    normalizer.process_profiles(stats)
    return True


def _setup_dataset(
    cfg: Dict[str, Any], data_dir: Path
) -> Optional[Tuple[AtmosphericDataset, Callable]]:
    """
    Loads and prepares the dataset for training or evaluation.

    Args:
        cfg: The main configuration dictionary.
        data_dir: The root directory where normalized data is located.

    Returns:
        A tuple containing the AtmosphericDataset instance and the collate
        function, or None if dataset setup fails.
    """
    norm_dir = data_dir / "normalized_profiles"
    if not norm_dir.exists():
        logger.error(
            "Normalised data not found at %s; run --normalize first", norm_dir
        )
        return None
    try:
        ds = AtmosphericDataset(
            data_folder=norm_dir,
            input_variables=cfg["input_variables"],
            target_variables=cfg["target_variables"],
            global_variables=cfg.get("global_variables", []),
            sequence_types=cfg["sequence_types"],
            sequence_lengths=cfg["sequence_lengths"],
            output_seq_type=cfg["output_seq_type"],
            cache_size=cfg.get("dataset_cache_size", 1024),
            validate_profiles=cfg.get("validate_profiles", True),
        )
        collate = create_multi_source_collate_fn()
        return ds, collate
    except Exception as exc:
        logger.error("Dataset setup failed: %s", exc, exc_info=True)
        return None


def train_model(
    trial: Optional[optuna.Trial],
    config: Dict[str, Any],
    device: torch.device,
    dataset: AtmosphericDataset,
    collate_fn: Callable,
    save_dir: Path,
) -> float:
    """
    Initializes and runs the ModelTrainer for a single training process.

    This function serves as a wrapper around the ModelTrainer class,
    facilitating both standard training runs and individual trials
    within a hyperparameter tuning session.

    Args:
        trial: An Optuna Trial object if part of a tuning run, else None.
        config: The configuration dictionary for this training run.
        device: The PyTorch device (CPU or CUDA) to use for training.
        dataset: The pre-loaded AtmosphericDataset instance.
        collate_fn: The collate function for the DataLoader.
        save_dir: The directory where model artifacts and logs will be saved.

    Returns:
        The best validation loss achieved during training.
    """
    trainer = ModelTrainer(
        config=config,
        device=device,
        save_dir=save_dir,
        dataset=dataset,
        collate_fn=collate_fn,
        optuna_trial=trial,
    )
    best = trainer.train()
    if not isinstance(best, float):
        raise RuntimeError("ModelTrainer did not return a float validation loss.")
    return best


def _run_tuning(cfg: Dict[str, Any], data_dir: Path) -> bool:
    """
    Manages the hyperparameter tuning process using Optuna.

    It sets up the tuning study and calls the `run_hyperparameter_search`
    function.

    Args:
        cfg: The base configuration dictionary.
        data_dir: The root data directory.

    Returns:
        True if the tuning process completes and finds a best configuration,
        False otherwise.
    """
    num_trials = cfg.get("optuna_num_trials", DEFAULT_NUM_TRIALS)
    logger.info("Starting hyper‑parameter search: %d trials", num_trials)
    tuning_results_dir = data_dir / "tuning_results"

    def _train_wrapper(
        trial: optuna.Trial,
        trial_cfg: Dict[str, Any],
        device: torch.device,
        ds: AtmosphericDataset,
        collate: Callable,
        root: Path,
    ) -> float:
        """
        A wrapper for `train_model` to match the signature expected by
        `run_hyperparameter_search`.
        """
        return train_model(trial, trial_cfg, device, ds, collate, root)

    best_cfg = run_hyperparameter_search(
        base_config=cfg,
        data_dir=str(data_dir),
        output_dir=str(tuning_results_dir),
        setup_dataset_func=_setup_dataset,
        train_model_func=_train_wrapper,
        setup_device_func=setup_device,
        save_config_func=save_json,
        num_trials=num_trials,
    )
    return best_cfg is not None


def _parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    p = argparse.ArgumentParser(
        description="Atmospheric‑flux transformer pipeline"
    )
    p.add_argument(
        "--normalize", action="store_true", help="Run data normalisation"
    )
    p.add_argument(
        "--train", action="store_true", help="Train a model using fixed config"
    )
    p.add_argument(
        "--tune", action="store_true", help="Run Optuna hyper‑parameter search"
    )
    args = p.parse_args()
    if not (args.normalize or args.train or args.tune):
        p.error("Specify at least one action: --normalize, --train, --tune")
    return args


def main() -> int:
    """
    The main entry point for the script.

    Orchestrates the selected actions (normalize, train, tune) based on
    command-line arguments.

    Returns:
        0 on successful completion, 1 on failure.
    """
    args = _parse_args()
    setup_logging()
    logger.info("Using configuration file: %s", CONFIG_PATH)
    logger.info("Using data directory: %s", DATA_DIR_PATH)

    cfg = load_config(CONFIG_PATH)
    if cfg is None:
        logger.error("Configuration load/validation failed. Exiting.")
        return 1
    seed_everything(cfg.get("random_seed", 42))
    ensure_dirs(
        DATA_DIR_PATH / "profiles",
        DATA_DIR_PATH / "normalized_profiles",
        DATA_DIR_PATH / "model",
        DATA_DIR_PATH / "tuning_results",
    )

    if args.normalize:
        logger.info("Starting data normalization...")
        if not _normalize(cfg, DATA_DIR_PATH):
            logger.error("Data normalization failed. Exiting.")
            return 1
        logger.info("Data normalization finished.")

    if args.train:
        logger.info("Starting model training with fixed configuration...")
        device = setup_device()
        ds_info = _setup_dataset(cfg, DATA_DIR_PATH)
        if ds_info is None:
            return 1
        ds, coll = ds_info
        try:
            model_save_dir = DATA_DIR_PATH / "model"
            train_model(None, cfg, device, ds, coll, model_save_dir)
            logger.info(
                "Model training finished. Artifacts saved in %s",
                model_save_dir,
            )
        except Exception as exc:
            logger.error(
                "Training failed unexpectedly: %s", exc, exc_info=True
            )
            return 1

    if args.tune:
        logger.info("Starting hyper-parameter tuning...")
        if not _run_tuning(cfg, DATA_DIR_PATH):
            logger.error("Hyper-parameter tuning failed.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
