#!/usr/bin/env python3
"""
hyperparam_search.py - Flexible Optuna hyperparameter optimization.

Features:
- Supports optional parameters (comment out in config to use defaults)
- Reasonable pruning for efficiency
- Automatic handling of d_model/nhead compatibility
- Preserves config values unless explicitly in search space
- Trial logging to text file
"""
from __future__ import annotations

import logging
from argparse import Namespace
from datetime import datetime
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


def log_trial_to_file(log_path: Path, trial_number: int, params: Dict[str, Any],
                      result: float, status: str = "COMPLETE") -> None:
    """
    Log trial results to a text file for easy tracking.

    Args:
        log_path: Path to the log file
        trial_number: Trial number
        params: Trial parameters
        result: Trial result (validation loss)
        status: Trial status (COMPLETE, PRUNED, FAILED)
    """
    # Create header if file doesn't exist
    if not log_path.exists():
        with open(log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HYPERPARAMETER OPTIMIZATION TRIAL LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    # Append trial results
    with open(log_path, 'a') as f:
        f.write(f"\nTrial {trial_number:04d} - {status}\n")
        f.write(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
        f.write(f"Loss: {result:.6e}\n")
        f.write("Parameters:\n")

        # Skip combo_idx in output, show actual d_model and nhead
        for key, value in sorted(params.items()):
            if key == "combo_idx":
                continue
            if isinstance(value, float):
                f.write(f"  {key:25s}: {value:.6f}\n")
            else:
                f.write(f"  {key:25s}: {value}\n")
        f.write("-" * 40 + "\n")


class AggressivePruner(optuna.pruners.BasePruner):
    """Custom pruner that protects startup trials and has reasonable patience."""

    def __init__(
            self,
            n_startup_trials: int = 10,
            n_warmup_steps: int = 10,
            patience: int = 5,
            min_improvement: float = 0.0001,  #
    ):


        self._median_pruner = MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
        )
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.patience = patience
        self.min_improvement = min_improvement

    def prune(self, study: optuna.Study, trial: optuna.FrozenTrial) -> bool:
        # Don't prune the first n_startup_trials to establish baseline
        completed_trials = len([t for t in study.trials
                                if t.state == optuna.trial.TrialState.COMPLETE])
        if completed_trials < self.n_startup_trials:
            return False

        # Don't prune during warmup steps
        step = trial.last_step
        if step is None or step < self.n_warmup_steps:
            return False

        if step % 3 != 0:
            return False

        # Use median pruner
        if self._median_pruner.prune(study, trial):
            return True

        # Check for stagnation
        values = list(trial.intermediate_values.values())
        if len(values) >= self.patience:
            recent = values[-self.patience:]
            best_recent = min(recent)
            # Only prune if no improvement
            if best_recent >= recent[0] - self.min_improvement:
                return True

        return False


def create_reduced_config(
        config: Dict[str, Any],
        fraction: float = 1.0,
        override_epochs: bool = False,
        max_epochs: int = 50,
) -> Dict[str, Any]:
    """
    Create a configuration for trials with specific optimizations.

    Args:
        config: Original configuration
        fraction: Fraction of dataset to use
        override_epochs: If True, override epochs to max_epochs (for faster search)
        max_epochs: Maximum epochs if override_epochs is True

    Returns:
        Modified configuration
    """
    reduced_config = {
        "data_specification": config["data_specification"],
        "data_paths_config": config["data_paths_config"],
        "normalization": config["normalization"],
        "output_paths_config": config["output_paths_config"],
        "model_hyperparameters": dict(config["model_hyperparameters"]),
        "training_hyperparameters": dict(config["training_hyperparameters"]),
        "miscellaneous_settings": dict(config["miscellaneous_settings"]),
    }

    # Use subset of data for faster trials
    reduced_config["training_hyperparameters"]["dataset_fraction_to_use"] = fraction

    # Keep original epochs from config unless explicitly overriding
    if override_epochs:
        reduced_config["training_hyperparameters"]["epochs"] = max_epochs
        logger.info(f"Overriding epochs from {config['training_hyperparameters']['epochs']} to {max_epochs} for search")
    else:
        # Keep original epochs but adjust early stopping for faster pruning
        original_epochs = config["training_hyperparameters"]["epochs"]
        logger.info(f"Keeping original epochs setting: {original_epochs}")
        reduced_config["training_hyperparameters"]["early_stopping_patience"] = min(10, original_epochs // 3)
        reduced_config["training_hyperparameters"]["min_delta"] = 1e-4

    # Log what we're disabling for tuning
    logger.info("=== Settings disabled for hyperparameter tuning ===")

    # Disable expensive features during search
    if reduced_config["miscellaneous_settings"].get("torch_export", False):
        logger.info("  - torch_export: True -> False (model export disabled)")
        reduced_config["miscellaneous_settings"]["torch_export"] = False

    if reduced_config["miscellaneous_settings"].get("torch_compile", False):
        logger.info("  - torch_compile: True -> False (compilation disabled)")
        reduced_config["miscellaneous_settings"]["torch_compile"] = False

    if reduced_config["miscellaneous_settings"].get("detect_anomaly", False):
        logger.info("  - detect_anomaly: True -> False (anomaly detection disabled)")
        reduced_config["miscellaneous_settings"]["detect_anomaly"] = False

    if fraction < 1.0:
        logger.info(
            f"  - dataset_fraction_to_use: {config['training_hyperparameters'].get('dataset_fraction_to_use', 1.0)} -> {fraction}")

    logger.info("=== End of tuning modifications ===\n")

    return reduced_config


def suggest_parameter(
        trial: optuna.Trial,
        name: str,
        search_config: Optional[Union[List, Dict]],
        default_value: Any,
) -> Any:
    """
    Suggest a parameter value or use default if not in search space.

    IMPORTANT: Only suggests values if parameter is in search space,
    otherwise returns the default (config) value unchanged.

    Args:
        trial: Optuna trial
        name: Parameter name
        search_config: Search configuration (None, list, or dict with low/high)
        default_value: Default value to use if not searching

    Returns:
        Suggested or default value
    """
    if search_config is None:
        # Parameter not in search space, use config default unchanged
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
            else:
                return trial.suggest_float(
                    name,
                    search_config["low"],
                    search_config["high"]
                )
        else:
            logger.warning(f"Invalid search config for {name}, using config default: {default_value}")
            return default_value

    else:
        logger.warning(f"Unknown search config type for {name}, using config default: {default_value}")
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

    # Check if directory already exists
    if model_save_dir.exists() and any(model_save_dir.iterdir()):
        if not args.resume:
            logger.error(f"Hyperparameter search directory already exists: {model_save_dir}")
            logger.error("To run a new search, either:")
            logger.error("  1. Delete the existing directory")
            logger.error("  2. Use --resume flag to continue the existing study")
            logger.error("  3. Change the output directory in config")
            return

    ensure_dirs(model_save_dir)

    # Initialize trial log file
    trial_log_path = model_save_dir / "trial_log.txt"

    # Get search space from config
    search_space = config.get("hyperparameter_search", {})
    if not search_space:
        logger.error(
            "'hyperparameter_search' section not found in config. "
            "Add it to define which parameters to optimize."
        )
        return

    # Analyze config for potential issues
    logger.info("=== Configuration Analysis ===")

    # Check scheduler compatibility with epochs
    scheduler_type = config.get("training_hyperparameters", {}).get("scheduler_type", "reduce_on_plateau")
    epochs = config.get("training_hyperparameters", {}).get("epochs", 100)
    warmup_epochs = config.get("training_hyperparameters", {}).get("warmup_epochs", 0)

    if scheduler_type == "cosine" and warmup_epochs >= epochs:
        logger.error(f"ERROR: warmup_epochs ({warmup_epochs}) >= epochs ({epochs}). This will cause training to fail!")
        return

    # Check for very high film_clamp
    film_clamp = config.get("model_hyperparameters", {}).get("film_clamp", 2.0)
    if film_clamp > 100:
        logger.warning(f"WARNING: film_clamp is very high ({film_clamp}). This essentially disables FiLM modulation.")
        logger.warning("         Consider adding 'film_clamp': {'low': 1.0, 'high': 5.0} to hyperparameter_search")

    # Check for disabled early stopping
    early_stopping = config.get("training_hyperparameters", {}).get("early_stopping_patience", 20)
    if early_stopping > 1000:
        logger.info(f"NOTE: Early stopping is effectively disabled (patience={early_stopping})")

    logger.info("=== End Configuration Analysis ===\n")

    # Get default values from config
    default_model = config.get("model_hyperparameters", {})
    default_train = config.get("training_hyperparameters", {})

    # Pre-compute valid d_model/nhead combinations
    valid_combinations = []
    if "d_model" in search_space and "nhead_divisors" in search_space:
        for d_model in search_space["d_model"]:
            for nhead in search_space["nhead_divisors"]:
                if d_model % nhead == 0:
                    valid_combinations.append((d_model, nhead))

        if not valid_combinations:
            logger.error("No valid d_model/nhead combinations found!")
            logger.error("Ensure at least one d_model is divisible by at least one nhead_divisor")
            return

        logger.info(f"Found {len(valid_combinations)} valid d_model/nhead combinations")

    # Create base config for trials (keeping original epochs)
    trial_config_base = create_reduced_config(
        config,
        fraction=1.0,
        override_epochs=False  # Don't override epochs - use config value
    )

    # Log which parameters will be searched
    searchable_params = [k for k, v in search_space.items() if v is not None and k != "nhead_divisors"]
    logger.info(f"Parameters to optimize: {searchable_params}")

    # Log parameters that will use config defaults
    all_params = set([
        "d_model", "nhead", "num_encoder_layers", "dim_feedforward",
        "dropout", "attention_dropout", "film_clamp",
        "learning_rate", "batch_size", "weight_decay", "gradient_accumulation_steps"
    ])
    using_defaults = all_params - set(searchable_params) - {"nhead"}  # nhead is special case
    if using_defaults:
        logger.info(f"Parameters using config defaults: {list(using_defaults)}")

    logger.info("")  # Blank line for readability

    # Create collate function once
    collate_fn = create_collate_fn(padding_val)

    def objective(trial: optuna.Trial) -> float:
        """Objective function for a single trial."""
        # Start with base config
        trial_config = {
            "data_specification": trial_config_base["data_specification"],
            "data_paths_config": trial_config_base["data_paths_config"],
            "normalization": trial_config_base["normalization"],
            "output_paths_config": trial_config_base["output_paths_config"],
            "miscellaneous_settings": trial_config_base["miscellaneous_settings"],
            "model_hyperparameters": dict(trial_config_base["model_hyperparameters"]),
            "training_hyperparameters": dict(trial_config_base["training_hyperparameters"]),
        }

        # Handle d_model and nhead together if both are being searched
        if valid_combinations:
            # Choose a valid combination
            combo_idx = trial.suggest_int("combo_idx", 0, len(valid_combinations) - 1)
            d_model, nhead = valid_combinations[combo_idx]
        else:
            # Use individual parameters or defaults
            d_model = suggest_parameter(
                trial, "d_model",
                search_space.get("d_model"),
                default_model.get("d_model", 256)
            )

            # Handle nhead - must be divisor of d_model
            if "nhead_divisors" in search_space:
                # Get valid nheads for this d_model
                valid_nheads = [n for n in search_space["nhead_divisors"] if d_model % n == 0]
                if valid_nheads:
                    nhead = trial.suggest_categorical("nhead", valid_nheads)
                else:
                    # No valid nhead in search space, use default
                    nhead = default_model.get("nhead", 8)
                    if d_model % nhead != 0:
                        # Find any valid nhead
                        for n in [4, 8, 16, 32, 64, 128, 256]:
                            if d_model % n == 0:
                                nhead = n
                                break
                        else:
                            nhead = 1  # Last resort
            else:
                nhead = default_model.get("nhead", 8)

        # Model parameters - only change if in search space
        num_encoder_layers = suggest_parameter(
            trial, "num_encoder_layers",
            search_space.get("num_encoder_layers"),
            default_model.get("num_encoder_layers", 6)
        )

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

        attention_dropout = suggest_parameter(
            trial, "attention_dropout",
            search_space.get("attention_dropout"),
            default_model.get("attention_dropout", dropout)
        )

        film_clamp = suggest_parameter(
            trial, "film_clamp",
            search_space.get("film_clamp"),
            default_model.get("film_clamp", 2.0)
        )

        # Training parameters - only change if in search space
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

        gradient_accumulation_steps = suggest_parameter(
            trial, "gradient_accumulation_steps",
            search_space.get("gradient_accumulation_steps"),
            default_train.get("gradient_accumulation_steps", 1)
        )

        # Update config with suggested values
        trial_config["model_hyperparameters"].update({
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "film_clamp": film_clamp,
        })

        trial_config["training_hyperparameters"].update({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        })

        # Create trial directory
        trial_save_dir = model_save_dir / f"trial_{trial.number:04d}"
        ensure_dirs(trial_save_dir)

        # Log trial parameters
        logger.info(
            f"Trial {trial.number}: d={d_model}, n={nhead}, L={num_encoder_layers}, "
            f"ff={dim_feedforward}, lr={learning_rate:.1e}, bs={batch_size}"
        )

        # Prepare params for logging (with actual d_model and nhead)
        log_params = dict(trial.params)
        if "combo_idx" in log_params:
            log_params["d_model"] = d_model
            log_params["nhead"] = nhead

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

            # Log to text file
            log_trial_to_file(trial_log_path, trial.number, log_params,
                              best_val_loss, "COMPLETE")

            return best_val_loss

        except optuna.exceptions.TrialPruned:
            # Log pruned trial
            log_trial_to_file(trial_log_path, trial.number, log_params,
                              float("inf"), "PRUNED")
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Log failed trial
            log_trial_to_file(trial_log_path, trial.number, log_params,
                              float("inf"), f"FAILED: {str(e)[:50]}")
            return float("inf")

    # Create study
    study_name = args.optuna_study_name or "atmospheric_transformer_study"
    storage_path = f"sqlite:///{model_save_dir / 'optuna_study.db'}"

    sampler = TPESampler(
        seed=config.get("miscellaneous_settings", {}).get("random_seed", 42),
        n_startup_trials=10,
        n_ei_candidates=20,
    )

    # Adjust pruner based on actual epochs being used
    actual_epochs = trial_config_base["training_hyperparameters"]["epochs"]
    warmup_epochs = trial_config_base["training_hyperparameters"].get("warmup_epochs", 0)
    warmup_steps = max(warmup_epochs + 5, 12)  # was: min(8, actual_epochs // 4)

    pruner = AggressivePruner(
        n_startup_trials=10,  # keep
        n_warmup_steps=warmup_steps,
        patience=10,  # was 5
        min_improvement=5e-5,  # was 1e-5; avoid death by noise
    )

    logger.info(f"Pruner settings: warmup_steps={warmup_steps} (based on {actual_epochs} epochs)")

    # Create or load study
    if args.resume:
        # Try to load existing study
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_path,
                sampler=sampler,
                pruner=pruner,
            )
            logger.info(f"Resumed existing study '{study_name}' with {len(study.trials)} trials")
        except KeyError:
            logger.error(f"No existing study found with name '{study_name}'. Cannot resume.")
            return
    else:
        # Create new study
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
            if key != "combo_idx":  # Don't show internal combo_idx
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

        # Map combo_idx back to d_model and nhead if it was used
        best_params = dict(study.best_trial.params)
        if "combo_idx" in best_params and valid_combinations:
            combo_idx = best_params["combo_idx"]
            d_model, nhead = valid_combinations[combo_idx]
            best_params["d_model"] = d_model
            best_params["nhead"] = nhead
            del best_params["combo_idx"]

        # Update config with best params (only for parameters that were searched)
        for param in ["d_model", "nhead", "num_encoder_layers", "dim_feedforward",
                      "dropout", "attention_dropout", "film_clamp"]:
            if param in best_params:
                final_config["model_hyperparameters"][param] = best_params[param]

        for param in ["learning_rate", "batch_size", "weight_decay", "gradient_accumulation_steps"]:
            if param in best_params:
                final_config["training_hyperparameters"][param] = best_params[param]

        save_json(final_config, model_save_dir / "best_config.json")
        logger.info(f"\nTrain with best params: python main.py --train --config {model_save_dir / 'best_config.json'}")

        # Add summary to trial log
        with open(trial_log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("OPTIMIZATION COMPLETE\n")
            f.write(f"Total trials: {len(study.trials)}\n")
            f.write(
                f"Successful trials: {len(study.trials) - sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}\n")
            f.write(f"Best trial: #{study.best_trial.number}\n")
            f.write(f"Best loss: {study.best_value:.6e}\n")
            f.write("=" * 80 + "\n")
    else:
        logger.warning("No successful trials completed")


__all__ = ["run_optuna", "AggressivePruner", "create_reduced_config", "log_trial_to_file"]