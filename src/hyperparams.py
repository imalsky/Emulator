#!/usr/bin/env python3
"""
hyperparams.py – Hyperparameter tuning using Optuna for atmospheric-transformer models.

Defines:
- objective(): samples hyperparameters, trains model, returns validation loss
- run_hyperparameter_search(): orchestrates an Optuna study with progress and best-config saving
"""
from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import optuna
from optuna import Trial, Study

# Assuming utils.py contains save_json
# Import necessary functions if they are in utils or elsewhere
# from utils import save_json # Example import

logger = logging.getLogger(__name__)


def objective(
    trial: Trial,
    base_config: Dict[str, Any],
    data_dir: Path,
    setup_dataset_func: Callable[[Dict[str, Any], Path], tuple],
    train_model_func: Callable[[Dict[str, Any], Any, Any, Any, Path], float],
    setup_device_func: Callable[[], Any],
) -> float:
    """
    Optuna objective: sample hyperparameters, train, and return best validation loss.
    """
    # Copy base config to avoid mutation during trial
    config = copy.deepcopy(base_config)

    # --- Sample hyperparameters relevant to the simpler model ---
    config.update({
        "d_model": trial.suggest_categorical("d_model", [256, 512]),
        "nhead": trial.suggest_categorical("nhead", [4, 8, 16]), # Ensure d_model is divisible
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 8), # Adjusted range for simpler model
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [1024, 2048]), # Adjusted options
        "dropout": trial.suggest_float("dropout", 0.0, 0.2), # Slightly wider range common
        # "mlp_layers" and "mlp_hidden_dim" might be relevant if your GlobalEncoder or Head uses them
        # Add sampling here if they are configurable and you want to tune them, e.g.:
        # "global_encoder_layers": trial.suggest_int("global_encoder_layers", 1, 3),
        # "head_mlp_layers": trial.suggest_int("head_mlp_layers", 1, 3),
    })

    # Ensure d_model is divisible by nhead after sampling
    if config["d_model"] % config["nhead"] != 0:
        # Prune trial if constraints are violated - Optuna handles this
        raise optuna.exceptions.TrialPruned(
            f"d_model ({config['d_model']}) is not divisible by nhead ({config['nhead']}). Pruning trial."
        )

    # Ensure PE is 'sine' as per config/validation
    config["positional_encoding"] = "sine"
    # Remove sampling for features not present in the simple model
    # config["layerdrop"] = ...
    # config["token_dropout"] = ...
    # --- End Sampling ---


    # Setup device and dataset
    device = setup_device_func()
    try:
        # setup_dataset_func should return (dataset, collate_fn) or similar
        dataset, collate_fn = setup_dataset_func(config, data_dir)
        if dataset is None:
            logger.warning(f"Dataset setup failed for trial {trial.number}. Pruning.")
            raise optuna.exceptions.TrialPruned()
    except Exception as e:
        logger.warning(f"Exception during dataset setup for trial {trial.number}: {e}. Pruning.")
        raise optuna.exceptions.TrialPruned()


    # Train the model using the provided function
    # train_model_func is expected to return the best validation loss (float)
    try:
        # Pass only necessary arguments expected by train_model function signature
        # Note: Assumes train_model_func matches signature (config, device, dataset, collate_fn, data_dir) -> float
        #       Adjust if the signature is different.
        best_val = train_model_func(config, device, dataset, collate_fn, data_dir)
        if not isinstance(best_val, float):
             logger.error(f"train_model_func did not return float, got {type(best_val)}. Pruning.")
             raise optuna.exceptions.TrialPruned()
        return best_val
    except Exception as e:
         logger.error(f"Training failed for trial {trial.number}: {e}", exc_info=True)
         raise optuna.exceptions.TrialPruned() # Prune if training crashes


def run_hyperparameter_search(
    base_config: Dict[str, Any],
    data_dir: str,
    output_dir: str,
    setup_dataset_func: Callable[[Dict[str, Any], Path], tuple],
    train_model_func: Callable[[Dict[str, Any], Any, Any, Any, Path], float],
    setup_device_func: Callable[[], Any],
    save_config_func: Callable[[Dict[str, Any], str], bool], # Assumes save_json signature
    num_trials: int = 10,
) -> Optional[Dict[str, Any]]: # Return Optional[Dict] to indicate possible failure
    """
    Run an Optuna study to minimize validation loss, saving best configs along the way.
    Returns the best configuration dict found, or None if the study fails.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    # Callback to save the current best configuration after each trial
    def _save_current_best(study: Study, trial: Trial) -> None:
        # Check if the current trial is the best one so far or if it's the first trial
        if trial.state == optuna.trial.TrialState.COMPLETE and (study.best_trial is None or trial.number == study.best_trial.number):
            best_cfg = copy.deepcopy(base_config)
            best_cfg.update(study.best_params) # Use study.best_params directly
            curr_path = out_path / "best_config_current.json"
            if save_config_func(best_cfg, str(curr_path)): # Check if save was successful
                 logger.info(
                     f"Trial {trial.number} finished. Current best value={study.best_value:.4e}. "
                     f"Saved current best config to {curr_path}"
                 )
            else:
                 logger.warning(f"Failed to save current best config at trial {trial.number}")
        elif trial.state != optuna.trial.TrialState.COMPLETE:
             logger.info(f"Trial {trial.number} finished with state: {trial.state}")


    # Create study and optimize
    study = optuna.create_study(direction="minimize")
    try:
        study.optimize(
            lambda trial: objective( # Pass trial object correctly
                trial,
                base_config,
                data_path,
                setup_dataset_func,
                train_model_func,
                setup_device_func,
            ),
            n_trials=num_trials,
            callbacks=[_save_current_best],
            catch=(Exception,), # Catch exceptions to allow pruning
        )
    except Exception as e:
         logger.error(f"Optuna study optimization failed: {e}", exc_info=True)
         return None # Indicate failure


    # Check if any trials completed successfully
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
         logger.error("No trials completed successfully in the Optuna study.")
         return None


    # Finalize and save the best configuration found
    best_cfg = copy.deepcopy(base_config)
    best_cfg.update(study.best_params) # study.best_params holds the best hyperparameters

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = out_path / f"best_config_{timestamp}.json"
    if save_config_func(best_cfg, str(best_path)):
        logger.info(f"Best configuration overall (value={study.best_value:.6e}) saved to {best_path}")
    else:
         logger.error(f"Failed to save final best configuration to {best_path}")
         # Still return the config, but log the error


    # Optionally save full study object using joblib
    try:
        import joblib
        study_path = out_path / f"study_{timestamp}.pkl"
        joblib.dump(study, study_path)
        logger.info(f"Full Optuna study object saved to {study_path}")
    except ImportError:
        logger.info("joblib not installed; skipping study object serialization.")
    except Exception as e:
        logger.warning(f"Failed to save Optuna study object: {e}")


    return best_cfg


__all__ = ["run_hyperparameter_search"]