#!/usr/bin/env python3
"""
hyperparams.py – Optuna-based hyperparameter tuning for the atmospheric‑flux transformer.

This module defines the hyperparameter search space, the Optuna objective function,
and manages the overall optimization process, including dataset caching,
checkpointing, and saving results.
"""
from __future__ import annotations

import copy
import logging
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import optuna
import torch
from optuna import Study, Trial
from optuna.exceptions import TrialPruned

logger = logging.getLogger(__name__)

# Attempt to import joblib for optional study saving
_HAS_JOBLIB = False
try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    # Joblib is optional; study saving will be skipped if not installed.
    logger.debug("joblib not found, study object saving will be skipped.")


# --------------------------------------------------------------------------- #
# Dataset Cache (Shared Across Trials)                                        #
# --------------------------------------------------------------------------- #

# Cache to store loaded dataset objects, keyed by data path,
# to avoid redundant loading in consecutive trials with the same data.
_DATASET_CACHE: dict[str, Tuple[Any, Callable]] = {}


def _load_or_cache_dataset(
    cfg: Dict[str, Any],
    data_root: Path,
    loader_fn: Callable[[Dict[str, Any], Path], Optional[Tuple[Any, Callable]]],
) -> Optional[Tuple[Any, Callable]]:
    """
    Loads dataset using loader_fn or retrieves it from cache if data_root matches.

    Assumes dataset content at data_root does not change between trials.
    """
    key = str(data_root.resolve())
    if key in _DATASET_CACHE:
        logger.debug("Reusing cached dataset for key: %s", key)
        return _DATASET_CACHE[key]

    logger.debug("Loading dataset for key: %s", key)
    dataset_info = loader_fn(cfg, data_root)
    if dataset_info:
        _DATASET_CACHE[key] = dataset_info
    return dataset_info

# --------------------------------------------------------------------------- #
# Hyperparameter Space Definition                                             #
# --------------------------------------------------------------------------- #

def _suggest_hyperparams(trial: Trial, cfg: Dict[str, Any]) -> None:
    """
    Suggests hyperparameters using Optuna's trial object and updates cfg in-place.
    """
    cfg.update(
        {
            "d_model": trial.suggest_categorical("d_model", [256, 512]),
            "nhead": trial.suggest_categorical("nhead", [4, 8, 16]),
            "num_encoder_layers": trial.suggest_int("num_layers", 2, 8),
            "dim_feedforward": trial.suggest_categorical("dim_ff", [1024, 2048]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.2, step=0.05),
            "positional_encoding": "sine", # Assuming sine is standard here
        }
    )

# --------------------------------------------------------------------------- #
# Optuna Objective Function                                                   #
# --------------------------------------------------------------------------- #

def _objective(
    trial: Trial,
    base_config: Dict[str, Any],
    data_dir: Path,
    setup_dataset_fn: Callable[[Dict[str, Any], Path], Optional[Tuple[Any, Callable]]],
    train_fn: Callable[[Dict[str, Any], Any, Any, Any, Path], float],
    device_fn: Callable[[], Any],
    ckpt_root: Path, # Directory to save trial-specific checkpoints
) -> float:
    """
    Objective function for an Optuna trial.

    Sets up config, loads data, trains model, returns validation metric.
    """
    # Create a trial-specific config copy
    cfg = copy.deepcopy(base_config)
    cfg["optuna_trial"] = trial.number
    _suggest_hyperparams(trial, cfg)

    # Prune invalid parameter combinations early
    if cfg["d_model"] % cfg["nhead"] != 0:
        raise TrialPruned("d_model must be divisible by nhead")

    device = device_fn()

    # Load dataset (potentially from cache)
    dataset_info = _load_or_cache_dataset(cfg, data_dir, setup_dataset_fn)
    if dataset_info is None:
        logger.error("Trial %d: Dataset loading failed.", trial.number)
        raise TrialPruned("dataset load failed")
    dataset, collate = dataset_info

    # Run training for this trial
    # Checkpoints will be saved under the trial-specific ckpt_root
    best_val = train_fn(cfg, device, dataset, collate, ckpt_root)

    # Check if training returned a valid metric
    if not (isinstance(best_val, float) and torch.isfinite(torch.tensor(best_val))):
        logger.warning("Trial %d: Received non-finite validation loss: %s", trial.number, best_val)
        raise TrialPruned("non-finite val loss")

    return best_val

# --------------------------------------------------------------------------- #
# Public API: Main Search Function                                            #
# --------------------------------------------------------------------------- #

def run_hyperparameter_search(
    base_config: Dict[str, Any],
    data_dir: str,
    output_dir: str, # Root directory for all tuning results
    setup_dataset_func: Callable[[Dict[str, Any], Path], Optional[Tuple[Any, Callable]]],
    train_model_func: Callable[[Dict[str, Any], Any, Any, Any, Path], float],
    setup_device_func: Callable[[], Any],
    save_config_func: Callable[[Dict[str, Any], Union[str, Path]], bool],
    *,
    num_trials: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Executes the Optuna hyperparameter search.

    Args:
        base_config: Base configuration dictionary.
        data_dir: Path to the data directory.
        output_dir: Path to save tuning results (configs, study object).
        setup_dataset_func: Function to load/cache the dataset.
        train_model_func: Function to train the model for one trial.
        setup_device_func: Function to get the compute device.
        save_config_func: Function to save JSON configurations.
        num_trials: Number of Optuna trials to run.

    Returns:
        The best hyperparameter configuration found, or None if search fails.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Configure Optuna sampler and pruner
    sampler = optuna.samplers.TPESampler(
        seed=base_config.get("random_seed", 42), multivariate=True, group=True
    )
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5) # Prune after 5 epochs/steps

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    # Callback to save the best config found so far after each trial
    def _checkpoint_best(study_: Study, trial_: Trial) -> None:
        # Ensure best_trial is not None before accessing its attributes
        if study_.best_trial is None or study_.best_trial.number != trial_.number:
            return
        logger.debug("Trial %d is new best, saving current best config.", trial_.number)
        best_cfg = copy.deepcopy(base_config)
        best_cfg.update(study_.best_params)
        # Overwrite the 'current best' file
        save_config_func(best_cfg, out_path / "best_current.json")

    # Signal handler for graceful interruption (e.g., Ctrl+C)
    def _signal_handler(signum, frame):  # type: ignore[no-untyped-def]
        logger.warning("Interrupt signal received. Attempting to save study state...")
        if _HAS_JOBLIB:
            try:
                # Save the Optuna study object if joblib is available
                joblib.dump(study, out_path / "study_interrupt.pkl")
                logger.info("Saved Optuna study state to study_interrupt.pkl")
            except Exception as exc:
                logger.error("Failed to save study state during interrupt: %s", exc)
        else:
            logger.warning("Cannot save study state: joblib is not installed.")
        # Re-raise KeyboardInterrupt to ensure script termination
        raise KeyboardInterrupt

    # Register the signal handler
    signal.signal(signal.SIGINT, _signal_handler)

    logger.info("Starting Optuna hyperparameter search for %d trials...", num_trials)
    try:
        # Run the optimization loop
        study.optimize(
            # Lambda function wraps the objective to pass necessary arguments
            lambda t: _objective(
                t,
                base_config,
                data_path,
                setup_dataset_func,
                train_model_func,
                setup_device_func,
                out_path / f"trial_{t.number}", # Pass unique path for trial artifacts
            ),
            n_trials=num_trials,
            callbacks=[_checkpoint_best], # Save best config periodically
            gc_after_trial=True, # Help manage memory
            show_progress_bar=os.getenv("DISABLE_TQDM", "0") == "0", # Optional progress bar
            catch=(Exception,), # Catch trial failures to allow search continuation
        )
    except KeyboardInterrupt:
        logger.info("Search interrupted by user.")
    except Exception as exc:
        logger.error("Hyperparameter search failed unexpectedly: %s", exc, exc_info=True)
        return None
    finally:
        # Restore default signal handler after search completion or interruption
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    # --- Post-search processing ---

    # Check if any trials completed successfully
    if not study.trials or study.best_trial is None or study.best_trial.state != optuna.trial.TrialState.COMPLETE:
        logger.error("No successful trials were completed during the search.")
        return None

    # Prepare the final best configuration
    final_cfg = copy.deepcopy(base_config)
    final_cfg.update(study.best_params)
    logger.info("Best trial completed: #%d with value: %.4e", study.best_trial.number, study.best_trial.value)

    # Save the final best configuration with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_path = out_path / f"best_config_{timestamp}.json"
    if save_config_func(final_cfg, cfg_path):
        logger.info("Saved final best configuration to %s", cfg_path.name)

    # Optionally save the final study object using joblib
    if _HAS_JOBLIB:
        try:
            study_path = out_path / f"study_{timestamp}.pkl"
            joblib.dump(study, study_path)
            logger.info("Saved final Optuna study object to %s", study_path.name)
        except Exception as exc: # pragma: no cover
            # Catch potential issues during pickling/saving
            logger.warning("Could not save final study pickle: %s", exc)
    else:
         logger.warning("Cannot save final study object: joblib is not installed.")

    return final_cfg


__all__ = ["run_hyperparameter_search"]