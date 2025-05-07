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
import torch # Keep for torch.isfinite
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
            # Example for other tunable parameters if they are in your base_config:
            # "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            # "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            # "optimizer": trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd"]),
        }
    )
    # If learning_rate, batch_size etc. are part of base_config and meant to be tuned,
    # ensure they are also updated here. For example:
    # if "learning_rate" in cfg: # Check if it exists in base_config to be overridden
    #     cfg["learning_rate"] = trial.suggest_float("learning_rate_tuned", 1e-5, 1e-3, log=True)


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
    save_config_fn: Callable[[Dict[str, Any], Union[str, Path]], bool], # Explicitly pass this
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
        save_config_fn: Function to save the configuration dictionary.
        ckpt_root: Root directory for trial-specific checkpoints.

    Returns:
        Best validation score achieved during training for this trial.

    Raises:
        TrialPruned: If the trial should be pruned early.
    """
    cfg = copy.deepcopy(base_config)
    cfg["optuna_trial_number"] = trial.number # Add trial number for reference
    _suggest_hyperparams(trial, cfg)

    if cfg["d_model"] % cfg["nhead"] != 0:
        logger.warning(
            "Trial %d pruned: d_model (%d) not divisible by nhead (%d).",
            trial.number, cfg["d_model"], cfg["nhead"]
        )
        raise TrialPruned("d_model must be divisible by nhead")

    # Ensure the trial's checkpoint directory exists before saving config
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Save the current trial's full configuration
    trial_config_path = ckpt_root / "trial_config.json" # Changed from .json5 to .json for consistency
    try:
        if save_config_fn(cfg, trial_config_path):
            logger.info(f"Trial {trial.number}: Saved trial-specific config to {trial_config_path}")
        else:
            logger.warning(f"Trial {trial.number}: Failed to save trial-specific config to {trial_config_path}")
    except Exception as e:
        logger.error(f"Trial {trial.number}: Error saving trial-specific config: {e}", exc_info=True)
    
    # Store the full config in trial user_attrs for potential access in _checkpoint_best_config
    try:
        trial.set_user_attr("full_config_dict", cfg)
    except Exception as e: # Optuna might have limitations on attribute size
        logger.warning(f"Trial {trial.number}: Could not set full_config_dict as user_attr: {e}")


    device = device_fn()
    dataset_info = setup_dataset_fn(cfg, data_dir)
    if dataset_info is None:
        logger.warning("Trial %d pruned: dataset loading failed.", trial.number)
        raise TrialPruned("dataset load failed")
    dataset, collate = dataset_info

    try:
        best_val = train_fn(cfg, device, dataset, collate, ckpt_root)
    except TrialPruned as e: # Let Optuna handle pruning gracefully
        logger.info("Trial %d pruned during training: %s", trial.number, e)
        raise
    except Exception as e:
        logger.error("Trial %d training failed: %s", trial.number, e, exc_info=True)
        # Consider returning a high loss value instead of pruning for unexpected errors,
        # or re-raise if it's critical. For now, pruning.
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
    (Args description remains the same as your provided version)
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save the base configuration used for this tuning session for reference
    try:
        save_config_func(base_config, out_path / "base_config_for_tuning_run.json")
        logger.info(f"Saved base configuration for this tuning run to {out_path / 'base_config_for_tuning_run.json'}")
    except Exception as e:
        logger.error(f"Failed to save base configuration for tuning run: {e}", exc_info=True)


    sampler = optuna.samplers.TPESampler(
        seed=base_config.get("random_seed", 42),
        multivariate=True,
        group=True,
        warn_independent_sampling=False
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(5, num_trials // 10),
        n_warmup_steps=base_config.get("optuna_pruner_warmup_steps", 3),
        interval_steps=base_config.get("optuna_pruner_interval_steps", 1)
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def _checkpoint_best_config(study_: Study, trial_: Trial) -> None:
        """Saves information about the best trial so far."""
        if study_.best_trial is None or study_.best_trial.number != trial_.number:
            return

        logger.info(
            "Trial %d is new best (loss=%.4e). Optuna params: %s",
            trial_.number, trial_.value, trial_.params
        )
        # The full configuration for this best trial is already saved in its trial_X directory.
        # Here, we save a summary of the best Optuna parameters.
        best_optuna_summary = {
            "best_trial_number": trial_.number,
            "best_value": trial_.value,
            "optuna_suggested_params": trial_.params,
            "trial_config_location": str(out_path / f"trial_{trial_.number}" / "trial_config.json")
        }
        try:
            save_config_func(best_optuna_summary, out_path / "best_trial_summary_current.json")
        except Exception as e:
            logger.error(
                f"Failed to save best current trial summary for trial {trial_.number}: {e}",
                exc_info=True
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
                save_config_func,  # Pass the save_config_func here
                out_path / f"trial_{t.number}", # This is ckpt_root
            ),
            n_trials=num_trials,
            callbacks=[_checkpoint_best_config],
            gc_after_trial=True,
            show_progress_bar=os.getenv("DISABLE_TQDM", "0") == "0",
            catch=(TrialPruned,), # Ensure TrialPruned exceptions are caught and handled by Optuna
        )
    except KeyboardInterrupt:
        logger.warning("Search interrupted by user.")
    except Exception as exc: # Catch other unexpected errors during study.optimize
        logger.error("Optuna search failed unexpectedly: %s", exc, exc_info=True)
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
        "Best trial overall: #%d, Value: %.4e, Optuna Params: %s",
        best_trial.number, best_trial.value, best_trial.params
    )

    # Construct the best_cfg using the full configuration stored in the best trial's directory
    best_trial_config_path = out_path / f"trial_{best_trial.number}" / "trial_config.json"
    best_cfg_overall = None
    if best_trial_config_path.exists():
        try:
            with open(best_trial_config_path, 'r') as f:
                # Assuming trial_config.json is standard JSON. If it could be jsonc,
                # utils.load_config (which handles jsonc) should be used.
                # For simplicity, using standard json.load here as save_config_func uses json.dump.
                import json as std_json 
                best_cfg_overall = std_json.load(f)
            # Ensure Optuna specific values are also in the final best_cfg if not already captured
            best_cfg_overall["optuna_final_best_trial_number"] = best_trial.number
            best_cfg_overall["optuna_final_best_value"] = best_trial.value
            best_cfg_overall["optuna_suggested_params_for_best_trial"] = best_trial.params

        except Exception as e:
            logger.error(f"Failed to load the best trial's config from {best_trial_config_path}: {e}. Falling back to reconstructing from base_config and params.")
            best_cfg_overall = copy.deepcopy(base_config)
            best_cfg_overall.update(best_trial.params)
            best_cfg_overall["optuna_final_best_trial_number"] = best_trial.number
            best_cfg_overall["optuna_final_best_value"] = best_trial.value
    else:
        logger.warning(f"Best trial's config file not found at {best_trial_config_path}. Reconstructing from base_config and params.")
        best_cfg_overall = copy.deepcopy(base_config)
        best_cfg_overall.update(best_trial.params)
        best_cfg_overall["optuna_final_best_trial_number"] = best_trial.number
        best_cfg_overall["optuna_final_best_value"] = best_trial.value


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = out_path / f"best_config_overall_{timestamp}.json"

    try:
        if save_config_func(best_cfg_overall, final_path):
            logger.info(
                "Hyperparameter search complete. Best overall config (from trial %d) saved to %s",
                best_trial.number, final_path.name
            )
            # Also save to a non-timestamped file for easy access to the latest best
            fixed_best_path = out_path / "best_config.json"
            if save_config_func(best_cfg_overall, fixed_best_path):
                 logger.info("Also saved best overall config to %s", fixed_best_path.name)
        else:
            logger.error("Failed to save final best overall configuration.")
            return None
    except Exception as e:
        logger.error(
            "Exception during final config save: %s", e, exc_info=True
        )
        return None

    return best_cfg_overall


__all__ = ["run_hyperparameter_search"]