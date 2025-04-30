#!/usr/bin/env python3
"""
hyperparams.py – Optuna-based hyperparameter tuning for the atmospheric-flux transformer.

This module defines the hyperparameter search space, the Optuna objective function,
and manages the overall optimization process, including checkpointing the best
configuration found so far and saving the final best configuration.
"""
from __future__ import annotations

import copy
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import optuna
import torch
from optuna import Study, Trial
from optuna.exceptions import TrialPruned

logger = logging.getLogger(__name__)

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
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 8),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", [1024, 2048, 4096]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
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
    ckpt_root: Path,
) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object.
        base_config: Base configuration dictionary.
        data_dir: Path to the data directory.
        setup_dataset_fn: Function to set up the dataset and collate function.
        train_fn: Function to train the model and return best validation score.
        device_fn: Function to get the computational device.
        ckpt_root: Root directory for trial-specific checkpoints.

    Returns:
        Best validation score achieved during training for this trial.

    Raises:
        TrialPruned: If the trial should be pruned early.
    """
    cfg = copy.deepcopy(base_config)
    cfg["optuna_trial"] = trial.number
    _suggest_hyperparams(trial, cfg)

    if cfg["d_model"] % cfg["nhead"] != 0:
        logger.warning(
            "Trial %d pruned: d_model (%d) not divisible by nhead (%d).",
            trial.number, cfg["d_model"], cfg["nhead"]
        )
        raise TrialPruned("d_model must be divisible by nhead")

    device = device_fn()
    dataset_info = setup_dataset_fn(cfg, data_dir)
    if dataset_info is None:
        logger.warning("Trial %d pruned: dataset loading failed.", trial.number)
        raise TrialPruned("dataset load failed")
    dataset, collate = dataset_info

    try:
        best_val = train_fn(cfg, device, dataset, collate, ckpt_root)
    except TrialPruned as e:
        logger.info("Trial %d pruned during training: %s", trial.number, e)
        raise
    except Exception as e:
        logger.error("Trial %d training failed: %s", trial.number, e, exc_info=True)
        raise TrialPruned(f"Training exception: {e}") from e

    if not (isinstance(best_val, float) and torch.isfinite(torch.tensor(best_val))):
        logger.warning(
            "Trial %d pruned: non-finite validation loss (%.4e).",
            trial.number, best_val
        )
        raise TrialPruned("non-finite validation loss")

    logger.info("Trial %d finished with validation loss: %.4e", trial.number, best_val)
    return best_val


# --------------------------------------------------------------------------- #
# Public API: Main Search Function                                            #
# --------------------------------------------------------------------------- #

def run_hyperparameter_search(
    base_config: Dict[str, Any],
    data_dir: str,
    output_dir: str,
    setup_dataset_func: Callable[[Dict[str, Any], Path], Optional[Tuple[Any, Callable]]],
    train_model_func: Callable[[Dict[str, Any], Any, Any, Any, Path], float],
    setup_device_func: Callable[[], Any],
    save_config_func: Callable[[Dict[str, Any], Union[str, Path]], bool],
    *,
    num_trials: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Runs the hyperparameter search using Optuna.

    Args:
        base_config: The starting configuration dictionary.
        data_dir: Path to the directory containing the dataset.
        output_dir: Path to the directory where results and checkpoints are saved.
        setup_dataset_func: Callable that returns the dataset and collate function.
        train_model_func: Callable that trains the model and returns the best score.
        setup_device_func: Callable that returns the compute device (e.g., torch.device).
        save_config_func: Callable to save the configuration dictionary to a file.
        num_trials: The number of Optuna trials to run.

    Returns:
        The best configuration dictionary found, or None if the search fails.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(
        seed=base_config.get("random_seed", 42),
        multivariate=True,
        group=True,
        warn_independent_sampling=False
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(5, num_trials // 10),
        n_warmup_steps=3,
        interval_steps=1
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def _checkpoint_best_config(study_: Study, trial_: Trial) -> None:
        """Saves the config of the best trial so far."""
        if study_.best_trial is None or study_.best_trial.number != trial_.number:
            return

        logger.info(
            "Trial %d is new best (loss=%.4e), saving config.",
            trial_.number, trial_.value
        )
        best_cfg = copy.deepcopy(base_config)
        best_cfg.update(trial_.params)
        try:
            save_config_func(best_cfg, out_path / "best_current.json")
        except Exception as e:
            logger.error(
                "Failed to save best current config for trial %d: %s",
                trial_.number, e, exc_info=True
            )

    logger.info("Starting Optuna hyperparameter search for %d trials...", num_trials)
    start_time = datetime.now()

    try:
        study.optimize(
            lambda t: _objective(
                t,
                base_config,
                data_path,
                setup_dataset_func,
                train_model_func,
                setup_device_func,
                out_path / f"trial_{t.number}",
            ),
            n_trials=num_trials,
            callbacks=[_checkpoint_best_config],
            gc_after_trial=True,
            show_progress_bar=os.getenv("DISABLE_TQDM", "0") == "0",
            catch=(TrialPruned,),
        )
    except KeyboardInterrupt:
        logger.warning("Search interrupted by user.")
    except Exception as exc:
        logger.error("Search failed unexpectedly: %s", exc, exc_info=True)
        return None
    finally:
        duration = datetime.now() - start_time
        logger.info("Optuna search duration: %s", duration)

    if not study.trials:
        logger.error("No trials were run.")
        return None

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        logger.error("No trials completed successfully.")
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        fail_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        logger.info(
            "Trial summary: %d pruned, %d failed.",
            len(pruned_trials), len(fail_trials)
        )
        return None

    best_trial = study.best_trial
    logger.info(
        "Best trial: #%d, Value: %.4e, Params: %s",
        best_trial.number, best_trial.value, best_trial.params
    )

    best_cfg = copy.deepcopy(base_config)
    best_cfg.update(best_trial.params)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = out_path / f"best_config_{timestamp}.json"

    try:
        if save_config_func(best_cfg, final_path):
            logger.info(
                "Hyperparameter search complete. Best config saved to %s",
                final_path.name
            )
        else:
            logger.error("Failed to save final best configuration.")
            return None
    except Exception as e:
        logger.error(
            "Exception during final config save: %s", e, exc_info=True
        )
        return None

    return best_cfg


__all__ = ["run_hyperparameter_search"]