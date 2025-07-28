#!/usr/bin/env python3
"""
hyperparam_search.py - Optuna hyperparameter optimization with aggressive pruning.

This module orchestrates hyperparameter optimization using Optuna with efficient
data reuse and aggressive early stopping for bad trials.
"""
from __future__ import annotations

import logging
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner
from optuna.samplers import TPESampler
import torch

from dataset import create_collate_fn
from train import ModelTrainer
from utils import ensure_dirs, save_json, setup_logging

logger = logging.getLogger(__name__)


class AggressivePruner(optuna.pruners.BasePruner):
    """
    Custom aggressive pruner that combines multiple pruning strategies.
    Prunes if ANY of the following conditions are met:
    1. Current loss is worse than median of previous trials at same step
    2. Current loss is in the worst 25th percentile
    3. Loss hasn't improved for patience steps
    """
    
    def __init__(
        self,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 3,
        patience: int = 3,
        min_improvement: float = 0.001,
        percentile: float = 75.0,  # Prune bottom 25%
    ):
        self._median_pruner = MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
        )
        self._percentile_pruner = PercentilePruner(
            percentile=percentile,
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
        )
        self.patience = patience
        self.min_improvement = min_improvement
        self._trial_best_values: Dict[int, Tuple[float, int]] = {}
    
    def prune(self, study: optuna.Study, trial: optuna.FrozenTrial) -> bool:
        # Use median pruner
        if self._median_pruner.prune(study, trial):
            logger.info(f"Trial {trial.number} pruned by median pruner")
            return True
        
        # Use percentile pruner
        if self._percentile_pruner.prune(study, trial):
            logger.info(f"Trial {trial.number} pruned by percentile pruner")
            return True
        
        # Check patience-based pruning
        step = trial.last_step
        if step is None:
            return False
        
        current_value = trial.intermediate_values[step]
        trial_id = trial.number
        
        if trial_id not in self._trial_best_values:
            self._trial_best_values[trial_id] = (current_value, step)
        else:
            best_value, best_step = self._trial_best_values[trial_id]
            if current_value < best_value - self.min_improvement:
                self._trial_best_values[trial_id] = (current_value, step)
            elif step - best_step >= self.patience:
                logger.info(
                    f"Trial {trial.number} pruned by patience "
                    f"(no improvement for {self.patience} steps)"
                )
                return True
        
        return False


def create_reduced_config(
    config: Dict[str, Any], 
    fraction: float = 0.1,
    max_epochs: int = 30,
) -> Dict[str, Any]:
    """
    Create a reduced configuration for faster hyperparameter search.
    
    Args:
        config: Original configuration
        fraction: Fraction of data to use
        max_epochs: Maximum epochs for trials
    
    Returns:
        Modified configuration
    """
    reduced_config = deepcopy(config)
    
    # Reduce dataset size
    reduced_config["training_hyperparameters"]["dataset_fraction_to_use"] = fraction
    
    # Reduce epochs
    current_epochs = reduced_config["training_hyperparameters"].get("epochs", 100)
    reduced_config["training_hyperparameters"]["epochs"] = min(max_epochs, current_epochs)
    
    # More aggressive early stopping for trials
    reduced_config["training_hyperparameters"]["early_stopping_patience"] = 5
    reduced_config["training_hyperparameters"]["min_delta"] = 1e-4
    
    # Disable model export during trials to save time
    reduced_config["miscellaneous_settings"]["torch_export"] = False
    
    return reduced_config


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
    Sets up and runs an Optuna hyperparameter search with aggressive pruning
    and efficient data reuse.
    """
    ensure_dirs(model_save_dir)
    search_space = config.get("hyperparameter_search", {})
    if not search_space:
        logger.error("'hyperparameter_search' section not found in config. Aborting.")
        return

    # Create reduced config for faster trials
    trial_config_base = create_reduced_config(config, fraction=0.2, max_epochs=50)
    logger.info(
        f"Using reduced config for trials: "
        f"{trial_config_base['training_hyperparameters']['dataset_fraction_to_use']:.0%} of data, "
        f"max {trial_config_base['training_hyperparameters']['epochs']} epochs"
    )

    def objective(trial: optuna.Trial) -> float:
        """Objective function for a single trial."""
        trial_config = deepcopy(trial_config_base)

        # Model architecture parameters
        d_model = trial.suggest_categorical("d_model", search_space["d_model"])
        trial_config["model_hyperparameters"]["d_model"] = d_model

        possible_nheads = [
            div for div in search_space["nhead_divisors"] if d_model % div == 0
        ]
        if not possible_nheads:
            logger.warning(
                f"For trial {trial.number}, no valid nhead_divisor for d_model={d_model}. "
                f"Defaulting to 1."
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

        # Regularization
        dropout = trial.suggest_float("dropout", **search_space["dropout"])
        trial_config["model_hyperparameters"]["dropout"] = dropout

        # Training parameters
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

        # Create trial directory
        trial_save_dir = model_save_dir / f"trial_{trial.number:04d}"
        ensure_dirs(trial_save_dir)
        save_json(trial_config, trial_save_dir / "trial_config.json")

        # Setup trial logging
        trial_log_file = trial_save_dir / "trial_log.log"
        setup_logging(log_file=trial_log_file, force=True)
        logger.info(f"Starting Trial {trial.number} with params: {trial.params}")

        collate_fn = create_collate_fn(padding_val)

        try:
            # Data is already preprocessed and will be reused
            logger.info("Reusing preprocessed data from disk/cache")
            
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
            
            # Save trial results
            trial_results = {
                "trial_number": trial.number,
                "best_val_loss": best_val_loss,
                "params": trial.params,
                "state": trial.state.name,
            }
            save_json(trial_results, trial_save_dir / "trial_results.json")
            
            return best_val_loss
            
        except optuna.exceptions.TrialPruned:
            logger.info(f"Trial {trial.number} was pruned.")
            raise
        except Exception as e:
            logger.error(
                f"Trial {trial.number} failed with exception: {e}", exc_info=True
            )
            return float("inf")

    # Create study with aggressive pruning
    study_name = args.optuna_study_name or "atmospheric_transformer_study"
    storage_path = f"sqlite:///{model_save_dir / 'optuna_study.db'}"
    
    # Use TPE sampler with more startup trials for better exploration
    sampler_seed = config.get("miscellaneous_settings", {}).get("random_seed", 42)
    sampler = TPESampler(
        seed=sampler_seed,
        n_startup_trials=10,  # More exploration initially
        n_ei_candidates=24,   # More candidates for acquisition function
    )
    
    # Use our custom aggressive pruner
    pruner = AggressivePruner(
        n_startup_trials=5,
        n_warmup_steps=3,
        patience=3,
        min_improvement=0.001,
        percentile=75.0,  # Prune bottom 25%
    )

    # Create or load study
    if args.resume and Path(storage_path.replace("sqlite:///", "")).exists():
        logger.info(f"Resuming existing study '{study_name}'")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_path,
            sampler=sampler,
            pruner=pruner,
        )
    else:
        logger.info(f"Creating new study '{study_name}'")
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_path,
            load_if_exists=False,
            sampler=sampler,
            pruner=pruner,
        )

    logger.info(
        f"Starting Optuna study '{study_name}' with {args.num_trials} trials."
    )
    logger.info(f"Sampler: {sampler.__class__.__name__}")
    logger.info(f"Pruner: {pruner.__class__.__name__} (aggressive settings)")
    logger.info(f"Results will be saved in: {model_save_dir}")

    # Run optimization
    study.optimize(
        objective,
        n_trials=args.num_trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    # Log results
    logger.info("\n--- Optuna Hyperparameter Search Complete ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    if study.best_trial:
        logger.info(f"Best trial: Trial #{study.best_trial.number}")
        logger.info(f"  Best Validation Loss: {study.best_value:.6f}")
        logger.info("  Best Hyperparameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"    {key}: {value}")

        # Save best parameters
        best_params_path = model_save_dir / "best_hyperparameters.json"
        results_summary = {
            "best_value": study.best_value,
            "best_params": study.best_trial.params,
            "best_trial_number": study.best_trial.number,
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        }
        save_json(results_summary, best_params_path)
        logger.info(f"Best hyperparameters saved to {best_params_path}")
        
        # Create final config with best params for full training
        final_config = deepcopy(config)
        model_hp = final_config["model_hyperparameters"]
        train_hp = final_config["training_hyperparameters"]
        
        # Apply best parameters
        best_params = study.best_trial.params
        model_hp["d_model"] = best_params["d_model"]
        model_hp["nhead"] = best_params["nhead"]
        model_hp["num_encoder_layers"] = best_params["num_encoder_layers"]
        model_hp["dim_feedforward"] = best_params["dim_feedforward"]
        model_hp["dropout"] = best_params["dropout"]
        train_hp["learning_rate"] = best_params["learning_rate"]
        train_hp["batch_size"] = best_params["batch_size"]
        train_hp["weight_decay"] = best_params["weight_decay"]
        
        save_json(final_config, model_save_dir / "best_config.json")
        logger.info("Created best_config.json for full training run")
        
    else:
        logger.warning("No successful trials completed in the study.")
    
    # Print pruning statistics
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if pruned_trials:
        logger.info(f"\nPruning statistics:")
        logger.info(f"  Total pruned: {len(pruned_trials)}/{len(study.trials)}")
        logger.info(f"  Pruning rate: {len(pruned_trials)/len(study.trials):.1%}")


__all__ = ["run_optuna", "AggressivePruner", "create_reduced_config"]