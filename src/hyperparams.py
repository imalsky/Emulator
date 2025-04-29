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
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", [1024, 2048]),
        }
    )
    if "positional_encoding" not in cfg:
         cfg["positional_encoding"] = "sine"


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
        raise TrialPruned("dataset load failed")
    dataset, collate = dataset_info

    try:
        best_val = train_fn(cfg, device, dataset, collate, ckpt_root)
    except Exception as e:
        logger.error("Trial %d training failed: %s", trial.number, e, exc_info=True)
        raise TrialPruned(f"Training exception: {e}") from e

    if not (isinstance(best_val, float) and torch.isfinite(torch.tensor(best_val))):
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
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=base_config.get("random_seed", 42), multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def _checkpoint_best_config(study_: Study, trial_: Trial) -> None:
        # Only save if this trial just became the best
        if study_.best_trial is None or study_.best_trial.number != trial_.number:
            return
        logger.info("Trial %d is new best (loss=%.4e), saving config.", trial_.number, trial_.value)
        best_cfg = copy.deepcopy(base_config)
        # trial_.params now has correct keys
        best_cfg.update(trial_.params)
        save_config_func(best_cfg, out_path / "best_current.json")

    logger.info("Starting Optuna hyperparameter search for %d trials...", num_trials)
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
            catch=(TrialPruned, Exception),
        )
    except KeyboardInterrupt:
        logger.warning("Search interrupted by user.")
    except Exception as exc:
        logger.error("Search failed: %s", exc, exc_info=True)
        return None

    if study.best_trial is None:
        logger.error("No successful trials completed.")
        return None

    # Finalize and save
    best_cfg = copy.deepcopy(base_config)
    best_cfg.update(study.best_params)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = out_path / f"best_config_{timestamp}.json"
    save_config_func(best_cfg, final_path)
    logger.info("Hyperparameter search complete. Best config → %s", final_path.name)
    return best_cfg


__all__ = ["run_hyperparameter_search"]
