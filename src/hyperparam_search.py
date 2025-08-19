#!/usr/bin/env python3
"""
hyperparam_search.py - Flexible Optuna hyperparameter optimization.

Features:
- Supports optional parameters (comment out in config to use defaults)
- Aggressive pruning for efficiency
- Automatic handling of d_model/nhead compatibility
"""
from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

from dataset import create_collate_fn
from train import ModelTrainer
from utils import ensure_dirs, save_json

logger = logging.getLogger(__name__)


class AggressivePruner(optuna.pruners.BasePruner):
    """Custom aggressive pruner combining multiple strategies."""
    
    def __init__(
        self,
        n_startup_trials: int = 3,
        n_warmup_steps: int = 2,
        patience: int = 2,
        min_improvement: float = 0.001,
    ):
        self._median_pruner = MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
        )
        self.patience = patience
        self.min_improvement = min_improvement
    
    def prune(self, study: optuna.Study, trial: optuna.FrozenTrial) -> bool:
        if self._median_pruner.prune(study, trial):
            return True
        
        step = trial.last_step
        if step is None or step < self.patience:
            return False
        
        values = list(trial.intermediate_values.values())
        if len(values) >= self.patience:
            recent = values[-self.patience:]
            if min(recent) >= recent[0] - self.min_improvement:
                return True
        
        return False


def create_reduced_config(
    config: Dict[str, Any],
    fraction: float = 0.1,
    max_epochs: int = 20,
) -> Dict[str, Any]:
    """Create a reduced configuration for faster trials."""
    reduced_config = {
        "data_specification": config["data_specification"],
        "data_paths_config": config["data_paths_config"],
        "normalization": config["normalization"],
        "output_paths_config": config["output_paths_config"],
        "model_hyperparameters": dict(config["model_hyperparameters"]),
        "training_hyperparameters": dict(config["training_hyperparameters"]),
        "miscellaneous_settings": dict(config["miscellaneous_settings"]),
    }
    
    reduced_config["training_hyperparameters"]["dataset_fraction_to_use"] = fraction
    reduced_config["training_hyperparameters"]["epochs"] = max_epochs
    reduced_config["training_hyperparameters"]["early_stopping_patience"] = 3
    reduced_config["training_hyperparameters"]["min_delta"] = 5e-4
    
    reduced_config["miscellaneous_settings"]["torch_export"] = False
    reduced_config["miscellaneous_settings"]["torch_compile"] = False
    reduced_config["miscellaneous_settings"]["detect_anomaly"] = False
    
    return reduced_config


def suggest_parameter(
    trial: optuna.Trial,
    name: str,
    search_config: Optional[Union[List, Dict]],
    default_value: Any,
) -> Any:
    """
    Suggest a parameter value or use default if not in search space.
    
    Args:
        trial: Optuna trial
        name: Parameter name
        search_config: Search configuration (None, list, or dict with low/high)
        default_value: Default value to use if not searching
        
    Returns:
        Suggested or default value
    """
    if search_config is None:
        # Parameter not in search space, use default
        logger.debug(f"Using default value for {name}: {default_value}")
        return default_value
    
    if isinstance(search_config, list):
        # Categorical parameter
        return trial.suggest_categorical(name, search_config)
    
    elif isinstance(search_config, dict):
        # Continuous parameter
        if "low" in search_config and "high" in search_config:
            if search_config.get("log", False):
                return trial.suggest_float(
                    name,
                    search_config["low"],
                    search_config["high"],
                    log=True
                )
            elif "step" in search_config:
                return trial.suggest_float(
                    name,
                    search_config["low"],
                    search_config["high"],
                    step=search_config["step"]
                )
            else:
                return trial.suggest_float(
                    name,
                    search_config["low"],
                    search_config["high"]
                )
        else:
            logger.warning(f"Invalid search config for {name}, using default")
            return default_value
    
    else:
        logger.warning(f"Unknown search config type for {name}, using default")
        return default_value


def run_optuna(
    config: Dict[str, Any],
    args: Namespace,
    device: torch.device,
    processed_dir: Path,
    splits: Dict[str, List[Tuple[str, int]]],
    padding_val: float,
    model_save_dir: Path,
) -> None:
    """Run Optuna hyperparameter search with flexible parameter selection."""
    ensure_dirs(model_save_dir)
    
    # Get search space from config
    search_space = config.get("hyperparameter_search", {})
    if not search_space:
        logger.error(
            "'hyperparameter_search' section not found in config. "
            "Add it to define which parameters to optimize."
        )
        return
    
    # Get default values from config
    default_model = config.get("model_hyperparameters", {})
    default_train = config.get("training_hyperparameters", {})
    
    # Pre-compute all possible nhead values if searching both d_model and nhead
    all_possible_nheads = set()
    if "d_model" in search_space and "nhead_divisors" in search_space:
        # For each possible d_model, find valid nhead values
        for d_model in search_space["d_model"]:
            for n in search_space["nhead_divisors"]:
                if d_model % n == 0:
                    all_possible_nheads.add(n)
        
        # Convert to sorted list for consistency
        all_possible_nheads = sorted(list(all_possible_nheads))
        
        if not all_possible_nheads:
            # If no valid combinations, use common divisors
            all_possible_nheads = [4, 8, 16]
        
        logger.info(f"Using fixed nhead search space: {all_possible_nheads}")
    
    # Create base config for trials
    trial_config_base = create_reduced_config(config, fraction=0.1, max_epochs=20)
    
    # Log which parameters will be searched
    searchable_params = [k for k, v in search_space.items() if v is not None]
    logger.info(f"Parameters to optimize: {searchable_params}")
    
    # Create collate function once
    collate_fn = create_collate_fn(padding_val)
    
    def objective(trial: optuna.Trial) -> float:
        """Objective function for a single trial."""
        trial_config = {
            "data_specification": trial_config_base["data_specification"],
            "data_paths_config": trial_config_base["data_paths_config"],
            "normalization": trial_config_base["normalization"],
            "output_paths_config": trial_config_base["output_paths_config"],
            "miscellaneous_settings": trial_config_base["miscellaneous_settings"],
            "model_hyperparameters": {},
            "training_hyperparameters": {},
        }
        
        # Model architecture parameters
        d_model = suggest_parameter(
            trial, "d_model",
            search_space.get("d_model"),
            default_model.get("d_model", 256)
        )
        
        # Handle nhead - must be divisor of d_model
        if "nhead_divisors" in search_space and search_space["nhead_divisors"]:
            # Suggest from all possible nheads, then validate
            nhead = trial.suggest_categorical("nhead", all_possible_nheads)
            
            # If this combination is invalid, return a bad score
            if d_model % nhead != 0:
                logger.info(
                    f"Trial {trial.number}: Invalid combination d_model={d_model}, "
                    f"nhead={nhead} (not divisible). Skipping."
                )
                return float("inf")
        else:
            # Not searching nhead, use default
            nhead = default_model.get("nhead", 8)
            # Ensure compatibility
            if d_model % nhead != 0:
                # Find closest valid nhead
                valid_nheads = [n for n in [4, 8, 16, 32] if d_model % n == 0]
                if valid_nheads:
                    nhead = valid_nheads[0]
                else:
                    nhead = 1  # Fallback
        
        # Handle num_encoder_layers (can be a range [min, max])
        if "num_encoder_layers" in search_space:
            layers_config = search_space["num_encoder_layers"]
            if isinstance(layers_config, list) and len(layers_config) == 2:
                num_encoder_layers = trial.suggest_int(
                    "num_encoder_layers", layers_config[0], layers_config[1]
                )
            else:
                num_encoder_layers = suggest_parameter(
                    trial, "num_encoder_layers",
                    layers_config,
                    default_model.get("num_encoder_layers", 6)
                )
        else:
            num_encoder_layers = default_model.get("num_encoder_layers", 6)
        
        dim_feedforward = suggest_parameter(
            trial, "dim_feedforward",
            search_space.get("dim_feedforward"),
            default_model.get("dim_feedforward", 1024)
        )
        
        dropout = suggest_parameter(
            trial, "dropout",
            search_space.get("dropout"),
            default_model.get("dropout", 0.1)
        )
        
        # Handle attention_dropout
        attention_dropout = suggest_parameter(
            trial, "attention_dropout",
            search_space.get("attention_dropout"),
            default_model.get("attention_dropout", dropout)  # Default to dropout value
        )
        
        # Handle film_clamp
        film_clamp = suggest_parameter(
            trial, "film_clamp",
            search_space.get("film_clamp"),
            2.0  # Default value
        )
        
        # Training parameters
        learning_rate = suggest_parameter(
            trial, "learning_rate",
            search_space.get("learning_rate"),
            default_train.get("learning_rate", 1e-4)
        )
        
        batch_size = suggest_parameter(
            trial, "batch_size",
            search_space.get("batch_size"),
            default_train.get("batch_size", 256)
        )
        
        weight_decay = suggest_parameter(
            trial, "weight_decay",
            search_space.get("weight_decay"),
            default_train.get("weight_decay", 1e-5)
        )
        
        # Handle gradient_accumulation_steps
        gradient_accumulation_steps = suggest_parameter(
            trial, "gradient_accumulation_steps",
            search_space.get("gradient_accumulation_steps"),
            default_train.get("gradient_accumulation_steps", 1)
        )
        
        # Update config with suggested values
        trial_config["model_hyperparameters"] = {
            **trial_config_base["model_hyperparameters"],
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "film_clamp": film_clamp,
        }
        
        trial_config["training_hyperparameters"] = {
            **trial_config_base["training_hyperparameters"],
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }
        
        # Create trial directory
        trial_save_dir = model_save_dir / f"trial_{trial.number:04d}"
        ensure_dirs(trial_save_dir)
        
        # Log trial parameters
        logger.info(
            f"Trial {trial.number}: d={d_model}, n={nhead}, L={num_encoder_layers}, "
            f"ff={dim_feedforward}, lr={learning_rate:.1e}, bs={batch_size}, "
            f"grad_acc={gradient_accumulation_steps}"
        )
        
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
            
            # Save trial results
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
    
    # Create study
    study_name = args.optuna_study_name or "atmospheric_transformer_study"
    storage_path = f"sqlite:///{model_save_dir / 'optuna_study.db'}"
    
    sampler = TPESampler(
        seed=config.get("miscellaneous_settings", {}).get("random_seed", 42),
        n_startup_trials=5,
        n_ei_candidates=20,
    )
    
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
        
        # Save results
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
        
        # Only update parameters that were actually searched
        for param in ["d_model", "nhead", "num_encoder_layers", "dim_feedforward", 
                      "dropout", "attention_dropout", "film_clamp"]:
            if param in study.best_trial.params:
                final_config["model_hyperparameters"][param] = study.best_trial.params[param]
        
        for param in ["learning_rate", "batch_size", "weight_decay", "gradient_accumulation_steps"]:
            if param in study.best_trial.params:
                final_config["training_hyperparameters"][param] = study.best_trial.params[param]
        
        save_json(final_config, model_save_dir / "best_config.json")
        logger.info(f"\nTrain with best params: python main.py train --config {model_save_dir / 'best_config.json'}")
    else:
        logger.warning("No successful trials completed")


__all__ = ["run_optuna", "AggressivePruner", "create_reduced_config"]