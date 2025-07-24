#!/usr/bin/env python3
"""
hyperparam_search.py - Optuna hyperparameter optimization logic.

This module orchestrates hyperparameter optimization using Optuna. It defines
the objective function to be minimized and manages the creation and execution
of the Optuna study.
"""
from __future__ import annotations

import logging
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import optuna
import torch

# Assuming these are your project's modules
from dataset import create_collate_fn
from train import ModelTrainer
from utils import ensure_dirs, save_json, setup_logging

logger = logging.getLogger(__name__)


def run_optuna(
    config: Dict[str, Any],
    args: Namespace,
    device: torch.device,
    processed_dir: Path,
    splits: Dict[str, List[Tuple[str, int]]],
    padding_val: float,
    model_save_dir: Path,
) -> None:
    """
    Sets up and runs an Optuna hyperparameter search.
    """
    ensure_dirs(model_save_dir)
    search_space = config.get("hyperparameter_search", {})
    if not search_space:
        logger.error("'hyperparameter_search' section not found in config. Aborting.")
        return

    def objective(trial: optuna.Trial) -> float:
        trial_config = deepcopy(config)

        d_model = trial.suggest_categorical("d_model", search_space["d_model"])
        trial_config["model_hyperparameters"]["d_model"] = d_model

        possible_nheads = [
            div for div in search_space["nhead_divisors"] if d_model % div == 0
        ]
        if not possible_nheads:
            logger.warning(
                f"For trial {trial.number}, no valid nhead_divisor for d_model={d_model}. Defaulting to 1."
            )
            possible_nheads = [1]
        nhead = trial.suggest_categorical("nhead", possible_nheads)
        trial_config["model_hyperparameters"]["nhead"] = nhead

        layers_range = search_space["num_encoder_layers"]
        num_encoder_layers = trial.suggest_int(
            "num_encoder_layers", low=layers_range[0], high=layers_range[1]
        )
        trial_config["model_hyperparameters"]["num_encoder_layers"] = num_encoder_layers

        dim_feedforward = trial.suggest_categorical(
            "dim_feedforward", search_space["dim_feedforward"]
        )
        trial_config["model_hyperparameters"]["dim_feedforward"] = dim_feedforward

        dropout = trial.suggest_float("dropout", **search_space["dropout"])
        trial_config["model_hyperparameters"]["dropout"] = dropout

        learning_rate = trial.suggest_float(
            "learning_rate", **search_space["learning_rate"]
        )
        trial_config["training_hyperparameters"]["learning_rate"] = learning_rate

        batch_size = trial.suggest_categorical("batch_size", search_space["batch_size"])
        trial_config["training_hyperparameters"]["batch_size"] = batch_size

        weight_decay = trial.suggest_float(
            "weight_decay", **search_space["weight_decay"]
        )
        trial_config["training_hyperparameters"]["weight_decay"] = weight_decay

        trial_save_dir = model_save_dir / f"trial_{trial.number:04d}"
        ensure_dirs(trial_save_dir)
        save_json(trial_config, trial_save_dir / "trial_config.json")

        trial_log_file = trial_save_dir / "trial_log.log"
        setup_logging(log_file=trial_log_file, force=True)
        logger.info(f"Starting Trial {trial.number} with params: {trial.params}")

        collate_fn = create_collate_fn(padding_val)

        try:
            trainer = ModelTrainer(
                config=trial_config,
                device=device,
                save_dir=trial_save_dir,
                processed_dir=processed_dir,
                splits=splits,
                collate_fn=collate_fn,
                optuna_trial=trial,
            )
            best_val_loss = trainer.train()
            return best_val_loss
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.error(
                f"Trial {trial.number} failed with exception: {e}", exc_info=True
            )
            return float("inf")

    study_name = args.optuna_study_name or "atmospheric_transformer_study"
    storage_path = f"sqlite:///{model_save_dir / 'optuna_study.db'}"
    sampler_seed = config.get("miscellaneous_settings", {}).get("random_seed", 42)
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=3)

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    logger.info(
        f"Starting/resuming Optuna study '{study_name}' with {args.num_trials} trials."
    )
    logger.info(
        f"Sampler: {sampler.__class__.__name__}, Pruner: {pruner.__class__.__name__}"
    )
    logger.info(f"Results will be saved in: {storage_path}")

    study.optimize(objective, n_trials=args.num_trials, gc_after_trial=True)

    logger.info("\n--- Optuna Hyperparameter Search Complete ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    if study.best_trial:
        logger.info(f"Best trial: Trial #{study.best_trial.number}")
        logger.info(f"  Best Validation Loss: {study.best_value:.6f}")
        logger.info("  Best Hyperparameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"    {key}: {value}")

        best_params_path = model_save_dir / "best_hyperparameters.json"
        results_summary = {
            "best_value": study.best_value,
            "best_params": study.best_trial.params,
            "best_trial_number": study.best_trial.number,
        }
        save_json(results_summary, best_params_path)
        logger.info(f"Best hyperparameters saved to {best_params_path}")
    else:
        logger.warning("No successful trials completed in the study.")


__all__ = ["run_optuna"]
