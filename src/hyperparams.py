#!/usr/bin/env python3
"""
hyperparams.py – Optuna tuner for the atmospheric-flux transformer.

Updates
-------
* FIX: sample feed-forward size is written to ``dim_feedforward`` (was ``dim_ff``)
  so create_prediction_model receives the correct key.
* Each trial’s checkpoints are stored under  <output_dir>/trial_<n>/  so partial
  searches survive interruptions.
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

# --------------------------------------------------------------------------- #
# dataset cache (shared across trials)                                         #
# --------------------------------------------------------------------------- #

_DATASET_CACHE: dict[str, Tuple[Any, Callable]] = {}


def _load_or_cache_dataset(
    cfg: Dict[str, Any],
    data_root: Path,
    loader_fn: Callable[[Dict[str, Any], Path], Optional[Tuple[Any, Callable]]],
) -> Optional[Tuple[Any, Callable]]:
    """Reuse the same Dataset object when data_layout is unchanged."""
    key = str(data_root.resolve())
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    dataset_info = loader_fn(cfg, data_root)
    if dataset_info:
        _DATASET_CACHE[key] = dataset_info
    return dataset_info


# --------------------------------------------------------------------------- #
# hyper-parameter space                                                        #
# --------------------------------------------------------------------------- #

def _suggest_hyperparams(trial: Trial, cfg: Dict[str, Any]) -> None:
    """Add Optuna-sampled parameters into *cfg* (in-place)."""
    cfg.update(
        {
            "d_model": trial.suggest_categorical("d_model", [256, 512]),
            "nhead": trial.suggest_categorical("nhead", [4, 8, 16]),
            "num_encoder_layers": trial.suggest_int("num_layers", 2, 8),
            # Key name fixed here ↓
            "dim_feedforward": trial.suggest_categorical("dim_ff", [1024, 2048]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.2, step=0.05),
            "positional_encoding": "sine",
        }
    )


# --------------------------------------------------------------------------- #
# Optuna objective                                                             #
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

    # basic validity check
    if cfg["d_model"] % cfg["nhead"] != 0:
        raise TrialPruned("d_model not divisible by nhead")

    device = device_fn()

    dataset_info = _load_or_cache_dataset(cfg, data_dir, setup_dataset_fn)
    if dataset_info is None:
        raise TrialPruned("dataset load failed")
    dataset, collate = dataset_info

    best_val = train_fn(cfg, device, dataset, collate, ckpt_root)
    if not (isinstance(best_val, float) and torch.isfinite(torch.tensor(best_val))):
        raise TrialPruned("non-finite val loss")

    return best_val


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #

def run_hyperparameter_search(
    base_config: Dict[str, Any],
    data_dir: str,
    output_dir: str,
    setup_dataset_func: Callable[[Dict[str, Any], Path], Optional[Tuple[Any, Callable]]],
    train_model_func: Callable[[Dict[str, Any], Any, Any, Any, Path], float],
    setup_device_func: Callable[[], Any],
    save_config_func: Callable[[Dict[str, Any], Union[str, Path]], bool],
    num_trials: int = 10,
) -> Optional[Dict[str, Any]]:

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(
        seed=base_config.get("random_seed", 42), multivariate=True, group=True
    )
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    # checkpoint best-so-far configuration
    def _checkpoint_best(study_: Study, trial_: Trial) -> None:
        if study_.best_trial.number != trial_.number:
            return
        best_cfg = copy.deepcopy(base_config)
        best_cfg.update(study_.best_params)
        save_config_func(best_cfg, out_path / "best_current.json")

    # graceful Ctrl-C
    def _signal_handler(signum, frame):  # type: ignore
        logger.warning("Interrupted – saving study and exiting …")
        import joblib

        joblib.dump(study, out_path / "study_interrupt.pkl")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _signal_handler)

    logger.info("Optuna search (%d trials) starting …", num_trials)
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
            callbacks=[_checkpoint_best],
            gc_after_trial=True,
            show_progress_bar=os.getenv("DISABLE_TQDM", "0") == "0",
            catch=(Exception,),
        )
    except KeyboardInterrupt:
        logger.info("Search interrupted by user.")
    except Exception as exc:
        logger.error("Search failed: %s", exc, exc_info=True)
        return None

    if study.best_trial.state != optuna.trial.TrialState.COMPLETE:
        logger.error("No successful trials.")
        return None

    final_cfg = copy.deepcopy(base_config)
    final_cfg.update(study.best_params)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_path = out_path / f"best_config_{timestamp}.json"
    if save_config_func(final_cfg, cfg_path):
        logger.info("Best configuration written to %s", cfg_path.name)

    try:
        import joblib
        joblib.dump(study, out_path / f"study_{timestamp}.pkl")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not save study pickle: %s", exc)

    return final_cfg


__all__ = ["run_hyperparameter_search"]
