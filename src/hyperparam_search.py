#!/usr/bin/env python3
"""
hyperparam_search.py - Optuna hyperparameter optimization with aggressive pruning.

This module orchestrates hyperparameter optimization using Optuna with:
- Efficient data reuse (preprocessed data loaded once)
- Aggressive early stopping for bad trials
- Reduced configurations for faster exploration
"""
from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

from dataset import create_collate_fn
from train import ModelTrainer
from utils import ensure_dirs, save_json, setup_logging

logger = logging.getLogger(__name__)


class AggressivePruner(optuna.pruners.BasePruner):
    """
    Custom aggressive pruner combining multiple strategies.
        
    Prunes if ANY of the following conditions are met:
    1. Current loss is worse than median of previous trials at same step
    2. Loss hasn't improved for patience steps
    """
    
    def __init__(
        self,
        n_startup_trials: int = 3,
        n_warmup_steps: int = 2,
        patience: int = 2,
        min_improvement: float = 0.001,
    ):
        """
        Initialize aggressive pruner.
        
        Args:
            n_startup_trials: Number of trials before pruning starts
            n_warmup_steps: Number of steps before pruning in each trial
            patience: Steps without improvement before pruning
            min_improvement: Minimum improvement to reset patience
        """
        self._median_pruner = MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
        )
        self.patience = patience
        self.min_improvement = min_improvement
        # FIXED: Removed unused _trial_best dictionary
    
    def prune(self, study: optuna.Study, trial: optuna.FrozenTrial) -> bool:
        """
        Determine if trial should be pruned.
        
        Args:
            study: Optuna study
            trial: Current trial
            
        Returns:
            True if trial should be pruned
        """
        # Check median pruner first (fast)
        if self._median_pruner.prune(study, trial):
            return True
        
        # Check patience-based pruning
        step = trial.last_step
        if step is None or step < self.patience:
            return False
        
        values = list(trial.intermediate_values.values())
        if len(values) >= self.patience:
            # Check if no improvement in last patience steps
            recent = values[-self.patience:]
            if min(recent) >= recent[0] - self.min_improvement:
                return True
        
        return False


def create_reduced_config(
    config: Dict[str, Any],
    fraction: float = 0.1,  # Reduced from 0.2
    max_epochs: int = 20,   # Reduced from 30
) -> Dict[str, Any]:
    """
    Create a reduced configuration for faster hyperparameter search.
    
    Args:
        config: Original configuration
        fraction: Fraction of data to use
        max_epochs: Maximum epochs for trials
        
    Returns:
        Modified configuration for faster trials
    """
    # Only modify what we need - avoid full deepcopy
    reduced_config = {
        "data_specification": config["data_specification"],
        "data_paths_config": config["data_paths_config"],
        "normalization": config["normalization"],
        "output_paths_config": config["output_paths_config"],
        "model_hyperparameters": dict(config["model_hyperparameters"]),
        "training_hyperparameters": dict(config["training_hyperparameters"]),
        "miscellaneous_settings": dict(config["miscellaneous_settings"]),
    }
    
    # Reduce dataset size and epochs
    reduced_config["training_hyperparameters"]["dataset_fraction_to_use"] = fraction
    reduced_config["training_hyperparameters"]["epochs"] = max_epochs
    reduced_config["training_hyperparameters"]["early_stopping_patience"] = 3  # More aggressive
    reduced_config["training_hyperparameters"]["min_delta"] = 5e-4  # Less sensitive
    
    # Disable expensive features during trials
    reduced_config["miscellaneous_settings"]["torch_export"] = False
    reduced_config["miscellaneous_settings"]["torch_compile"] = False
    reduced_config["miscellaneous_settings"]["detect_anomaly"] = False
    
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
    Run Optuna hyperparameter search with aggressive pruning.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        device: Compute device
        processed_dir: Directory with preprocessed data
        splits: Data splits
        padding_val: Padding value
        model_save_dir: Directory for saving results
    """
    ensure_dirs(model_save_dir)
    
    # Get search space from config
    search_space = config.get("hyperparameter_search", {})
    if not search_space:
        logger.error("'hyperparameter_search' section not found in config.")
        return
    
    # Pre-compute valid d_model/nhead combinations (faster than computing in trial)
    valid_combinations = {}
    for d_model in search_space["d_model"]:
        valid_combinations[d_model] = [
            n for n in search_space["nhead_divisors"] if d_model % n == 0
        ]
    
    # Create base config once
    trial_config_base = create_reduced_config(config, fraction=0.1, max_epochs=20)
    
    logger.info(
        f"Using reduced config: {trial_config_base['training_hyperparameters']['dataset_fraction_to_use']:.0%} of data, "
        f"max {trial_config_base['training_hyperparameters']['epochs']} epochs"
    )
    
    # Create collate function once
    collate_fn = create_collate_fn(padding_val)
    
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for a single trial.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Best validation loss achieved
        """
        # Lightweight config update (no deepcopy)
        trial_config = {
            "data_specification": trial_config_base["data_specification"],
            "data_paths_config": trial_config_base["data_paths_config"],
            "normalization": trial_config_base["normalization"],
            "output_paths_config": trial_config_base["output_paths_config"],
            "miscellaneous_settings": trial_config_base["miscellaneous_settings"],
            "model_hyperparameters": {},
            "training_hyperparameters": {},
        }
        
        # Model architecture
        d_model = trial.suggest_categorical("d_model", search_space["d_model"])
        nhead = trial.suggest_categorical("nhead", valid_combinations[d_model])
        
        layers_range = search_space["num_encoder_layers"]
        num_encoder_layers = trial.suggest_int(
            "num_encoder_layers", layers_range[0], layers_range[1]
        )
        
        dim_feedforward = trial.suggest_categorical(
            "dim_feedforward", search_space["dim_feedforward"]
        )
        
        dropout = trial.suggest_float("dropout", **search_space["dropout"])
        
        # Training parameters
        learning_rate = trial.suggest_float("learning_rate", **search_space["learning_rate"])
        batch_size = trial.suggest_categorical("batch_size", search_space["batch_size"])
        weight_decay = trial.suggest_float("weight_decay", **search_space["weight_decay"])
        
        # Update config
        trial_config["model_hyperparameters"] = {
            **trial_config_base["model_hyperparameters"],
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        }
        
        trial_config["training_hyperparameters"] = {
            **trial_config_base["training_hyperparameters"],
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
        }
        
        # Create trial directory
        trial_save_dir = model_save_dir / f"trial_{trial.number:04d}"
        ensure_dirs(trial_save_dir)
        
        # Minimal logging - no per-trial log files
        logger.info(f"Trial {trial.number}: d={d_model}, n={nhead}, L={num_encoder_layers}, lr={learning_rate:.1e}")
        
        try:
            # Create trainer
            trainer = ModelTrainer(
                config=trial_config,
                device=device,
                save_dir=trial_save_dir,
                processed_dir=processed_dir,
                splits=splits,
                collate_fn=collate_fn,
                optuna_trial=trial,
            )
            
            # Train and get best validation loss
            best_val_loss = trainer.train()
            
            # Save only essential results
            save_json({
                "trial": trial.number,
                "loss": best_val_loss,
                "params": trial.params,
            }, trial_save_dir / "result.json")
            
            return best_val_loss
            
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float("inf")
    
    # Create study with aggressive pruning
    study_name = args.optuna_study_name or "atmospheric_transformer_study"
    storage_path = f"sqlite:///{model_save_dir / 'optuna_study.db'}"
    
    # Simpler sampler with fewer startup trials
    sampler = TPESampler(
        seed=config.get("miscellaneous_settings", {}).get("random_seed", 42),
        n_startup_trials=5,  # Reduced from 10
        n_ei_candidates=20,  # Reduced from 24
    )
    
    # More aggressive pruner
    pruner = AggressivePruner(
        n_startup_trials=3,
        n_warmup_steps=2,
        patience=2,
        min_improvement=0.001,
    )
    
    # Create or load study
    if args.resume and Path(storage_path.replace("sqlite:///", "")).exists():
        logger.info(f"Resuming study '{study_name}'")
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
    
    logger.info(f"Starting Optuna with {args.num_trials} trials")
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=args.num_trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )
    
    # Log results
    logger.info(f"\nCompleted {len(study.trials)} trials")
    
    if study.best_trial:
        logger.info(f"Best trial: #{study.best_trial.number}, Loss: {study.best_value:.6f}")
        logger.info("Best params:")
        for key, value in study.best_trial.params.items():
            logger.info(f"  {key}: {value}")
        
        # Save best parameters and create final config
        results_summary = {
            "best_value": study.best_value,
            "best_params": study.best_trial.params,
            "best_trial_number": study.best_trial.number,
            "n_trials": len(study.trials),
            "n_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
        }
        save_json(results_summary, model_save_dir / "best_hyperparameters.json")
        
        # Create final config with best params
        final_config = dict(config)
        final_config["model_hyperparameters"].update({
            "d_model": study.best_trial.params["d_model"],
            "nhead": study.best_trial.params["nhead"],
            "num_encoder_layers": study.best_trial.params["num_encoder_layers"],
            "dim_feedforward": study.best_trial.params["dim_feedforward"],
            "dropout": study.best_trial.params["dropout"],
        })
        final_config["training_hyperparameters"].update({
            "learning_rate": study.best_trial.params["learning_rate"],
            "batch_size": study.best_trial.params["batch_size"],
            "weight_decay": study.best_trial.params["weight_decay"],
        })
        
        save_json(final_config, model_save_dir / "best_config.json")
        logger.info(f"\nTrain with best params: python main.py train --config {model_save_dir / 'best_config.json'}")
    else:
        logger.warning("No successful trials completed")


__all__ = ["run_optuna", "AggressivePruner", "create_reduced_config"]