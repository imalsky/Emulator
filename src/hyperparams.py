#!/usr/bin/env python3
"""
hyperparams.py – Optuna-based hyperparameter tuning.

This module defines the hyperparameter search space, the Optuna objective
function, and manages the overall optimization process.
"""
from __future__ import annotations

import copy
import json as std_json
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


def _suggest_hyperparams(trial: Trial, cfg: Dict[str, Any]) -> None:
    """
    Suggests hyperparameters using Optuna's trial object.

    This function updates the provided configuration dictionary `cfg` in-place
    with hyperparameter values suggested by the Optuna `trial`.

    Args:
        trial: The Optuna trial object for the current optimization iteration.
        cfg: The configuration dictionary to be updated with suggested values.
    """
    cfg.update(
        {
            "d_model": trial.suggest_categorical("d_model", [256, 512]),
            "nhead": trial.suggest_categorical("nhead", [4, 8, 16]),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 8),
            "dim_feedforward": trial.suggest_categorical(
                "dim_feedforward", [1024, 2048, 4096]
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
        }
    )


def _objective(
    trial: Trial,
    base_config: Dict[str, Any],
    data_dir: Path,
    setup_dataset_fn: Callable[
        [Dict[str, Any], Path], Optional[Tuple[Any, Callable]]
    ],
    train_fn: Callable[
        [Trial, Dict[str, Any], Any, Any, Any, Path], float
    ],
    device_fn: Callable[[], Any],
    save_config_fn: Callable[[Dict[str, Any], Union[str, Path]], bool],
    ckpt_root: Path,
) -> float:
    """
    Defines the objective function for Optuna hyperparameter optimization.

    This function takes an Optuna trial, suggests hyperparameters, sets up
    the dataset and device, runs the training process, and returns the
    metric to be optimized (e.g., validation loss).

    Args:
        trial: The Optuna trial object.
        base_config: The base configuration dictionary.
        data_dir: Path to the data directory.
        setup_dataset_fn: Function to set up the dataset and collate function.
        train_fn: Function to train the model and return the optimization metric.
        device_fn: Function to get the computational device.
        save_config_fn: Function to save the configuration dictionary.
        ckpt_root: Root directory for trial-specific checkpoints and configurations.

    Returns:
        The value of the metric to be minimized (e.g., best validation loss).

    Raises:
        TrialPruned: If the trial is pruned by Optuna or due to an error.
    """
    cfg = copy.deepcopy(base_config)
    cfg["optuna_trial_number"] = trial.number
    _suggest_hyperparams(trial, cfg)

    if cfg["d_model"] % cfg["nhead"] != 0:
        logger.warning(
            "Trial %d pruned: d_model (%d) not divisible by nhead (%d).",
            trial.number,
            cfg["d_model"],
            cfg["nhead"],
        )
        raise TrialPruned("d_model must be divisible by nhead")

    ckpt_root.mkdir(parents=True, exist_ok=True)
    trial_config_path = ckpt_root / "trial_config.json"
    try:
        save_config_fn(cfg, trial_config_path)
        logger.info(
            f"Trial {trial.number}: Saved trial-specific config to "
            f"{trial_config_path}"
        )
        trial.set_user_attr("full_config_dict_path", str(trial_config_path))
    except Exception as e:
        logger.error(
            f"Trial {trial.number}: Error saving/setting trial config: {e}",
            exc_info=True,
        )

    device = device_fn()
    dataset_info = setup_dataset_fn(cfg, data_dir)
    if dataset_info is None:
        logger.warning("Trial %d pruned: dataset loading failed.", trial.number)
        raise TrialPruned("dataset load failed")
    dataset, collate = dataset_info

    try:
        best_val = train_fn(trial, cfg, device, dataset, collate, ckpt_root)
    except TrialPruned:
        raise
    except Exception as e:
        logger.error(
            "Trial %d training failed: %s", trial.number, e, exc_info=True
        )
        raise TrialPruned(f"Training exception: {e}") from e

    if not (
        isinstance(best_val, float) and torch.isfinite(torch.tensor(best_val))
    ):
        logger.warning(
            "Trial %d pruned: non-finite validation loss (%.4e).",
            trial.number,
            best_val,
        )
        raise TrialPruned("non-finite validation loss")

    logger.info(
        "Trial %d finished with validation loss: %.4e", trial.number, best_val
    )
    return best_val


def run_hyperparameter_search(
    base_config: Dict[str, Any],
    data_dir: str,
    output_dir: str,
    setup_dataset_func: Callable[
        [Dict[str, Any], Path], Optional[Tuple[Any, Callable]]
    ],
    train_model_func: Callable[
        [Trial, Dict[str, Any], Any, Any, Any, Path], float
    ],
    setup_device_func: Callable[[], Any],
    save_config_func: Callable[[Dict[str, Any], Union[str, Path]], bool],
    *,
    num_trials: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Executes the Optuna hyperparameter search process.

    This function initializes an Optuna study, defines a sampler and pruner,
    and runs the optimization loop for a specified number of trials.
    It saves the best configuration found.

    Args:
        base_config: The starting configuration dictionary.
        data_dir: Path to the directory containing the dataset.
        output_dir: Path to the directory where results and checkpoints are saved.
        setup_dataset_func: Callable that returns the dataset and collate function.
        train_model_func: Callable that trains the model and returns the best score.
        setup_device_func: Callable that returns the compute device.
        save_config_func: Callable to save the configuration dictionary to a file.
        num_trials: The number of Optuna trials to run.

    Returns:
        The best configuration dictionary found, or None if the search fails
        or no trials complete.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        save_config_func(
            base_config, out_path / "base_config_for_tuning_run.json"
        )
    except Exception as e:
        logger.error(
            f"Failed to save base configuration for tuning run: {e}",
            exc_info=True,
        )

    n_startup_trials = base_config.get(
        "optuna_pruner_startup_trials", max(1, num_trials // 20)
    )
    n_warmup_steps = base_config.get("optuna_pruner_warmup_steps", 1)
    interval_steps = base_config.get("optuna_pruner_interval_steps", 1)

    sampler = optuna.samplers.TPESampler(
        seed=base_config.get("random_seed", 42),
        multivariate=True,
        group=True,
        warn_independent_sampling=False,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
        interval_steps=interval_steps,
    )
    study = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=pruner
    )

    def _checkpoint_best_config(study_: Study, trial_: Trial) -> None:
        """
        Callback to save a summary of the best trial found so far.
        """
        if study_.best_trial is None or study_.best_trial.number != trial_.number:
            return
        logger.info(
            "Trial %d is new best (loss=%.4e). Optuna params: %s",
            trial_.number,
            trial_.value,
            trial_.params,
        )
        summary = {
            "best_trial_number": trial_.number,
            "best_value": trial_.value,
            "optuna_suggested_params": trial_.params,
            "trial_config_location": trial_.user_attrs.get(
                "full_config_dict_path", "N/A"
            ),
        }
        try:
            save_config_func(
                summary, out_path / "best_trial_summary_current.json"
            )
        except Exception as e:
            logger.error(
                f"Failed to save best current trial summary for trial "
                f"{trial_.number}: {e}",
                exc_info=True,
            )

    logger.info(
        f"Starting Optuna search ({num_trials} trials) with pruner: "
        f"MedianPruner(startup={n_startup_trials}, warmup={n_warmup_steps}, "
        f"interval={interval_steps})"
    )
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
                save_config_func,
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
        return None
    except Exception as exc:
        logger.error(
            "Optuna search failed unexpectedly: %s", exc, exc_info=True
        )
        return None
    finally:
        logger.info("Optuna search duration: %s", datetime.now() - start_time)

    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        logger.error("No trials completed successfully.")
        return None

    best_trial = study.best_trial
    logger.info(
        "Best trial overall: #%d, Value: %.4e, Optuna Params: %s",
        best_trial.number,
        best_trial.value,
        best_trial.params,
    )

    best_cfg_overall = None
    best_trial_cfg_path = (
        out_path / f"trial_{best_trial.number}" / "trial_config.json"
    )
    if best_trial_cfg_path.exists():
        try:
            with open(best_trial_cfg_path, "r") as f:
                best_cfg_overall = std_json.load(f)
        except Exception as e:
            logger.error(f"Failed to load best trial's config: {e}")

    if best_cfg_overall is None:
        best_cfg_overall = copy.deepcopy(base_config)
        best_cfg_overall.update(best_trial.params)
        logger.warning("Reconstructed best_cfg_overall from base and Optuna params.")

    best_cfg_overall["optuna_final_best_trial_number"] = best_trial.number
    best_cfg_overall["optuna_final_best_value"] = best_trial.value

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = out_path / f"best_config_overall_{timestamp}.json"
    try:
        save_config_func(best_cfg_overall, final_path)
        logger.info(f"Best overall config saved to {final_path.name}")
        save_config_func(best_cfg_overall, out_path / "best_config.json")
    except Exception as e:
        logger.error(
            f"Exception during final config save: {e}", exc_info=True
        )
        return None

    return best_cfg_overall


__all__ = ["run_hyperparameter_search"]
