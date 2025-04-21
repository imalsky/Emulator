#!/usr/bin/env python3
"""
hyperparams.py – Hyperparameter tuning using Optuna.

Defines the Optuna objective function and the main study runner.
This version is configured to tune hyperparameters relevant to the simpler
transformer model architecture (e.g., d_model, nhead, dropout) and expects
the training function passed to it to return the best validation loss.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import optuna
import torch  # Added import
from optuna import Study, Trial
from optuna.exceptions import TrialPruned

# Assuming utils.py contains save_json and is importable
# from utils import save_json

logger = logging.getLogger(__name__)


# =============================================================================
# Optuna Objective Function
# =============================================================================


def objective(
    trial: Trial,
    base_config: Dict[str, Any],
    data_dir: Path,
    setup_dataset_func: Callable[
        [Dict[str, Any], Path], Optional[Tuple[Any, Callable]]
    ],
    train_model_func: Callable[[Dict[str, Any], Any, Any, Any, Path], float],
    setup_device_func: Callable[[], Any],
) -> float:
    """
    Optuna objective function called for each trial.

    Steps:
    1. Creates a trial-specific configuration by sampling hyperparameters.
    2. Sets up the device (CPU/GPU).
    3. Sets up the dataset using the provided function. Prunes if setup fails.
    4. Trains the model using the provided training function. Prunes if training fails.
    5. Returns the best validation loss obtained during training.

    Args:
        trial: The Optuna Trial object for this run.
        base_config: The base configuration dictionary, which will be updated
                     with sampled hyperparameters.
        data_dir: The root directory containing the data.
        setup_dataset_func: A function to set up the dataset. Expected to return
                            a tuple (dataset, collate_fn) or None on failure.
        train_model_func: The function that performs model training. Expected to
                          accept (config, device, dataset, collate_fn, data_dir)
                          and return the best validation loss (float).
        setup_device_func: A function to set up the torch device.

    Returns:
        The best validation loss (float) achieved by the model in this trial.

    Raises:
        TrialPruned: If dataset setup or model training fails, Optuna is notified
                     to prune the trial.
    """
    config = copy.deepcopy(base_config)
    config["optuna_trial_number"] = trial.number

    config["d_model"] = trial.suggest_categorical("d_model", [256, 512])
    config["nhead"] = trial.suggest_categorical("nhead", [4, 8, 16])
    config["num_encoder_layers"] = trial.suggest_int("num_encoder_layers", 2, 8)
    config["dim_feedforward"] = trial.suggest_categorical(
        "dim_feedforward", [1024, 2048]
    )
    config["dropout"] = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)
    # config["mlp_layers"] = trial.suggest_categorical("mlp_layers", [2, 4])
    # config["mlp_hidden_dim_factor"] = trial.suggest_categorical("mlp_hidden_dim_factor", [1, 2])
    config["positional_encoding"] = "sine"

    if config["d_model"] % config["nhead"] != 0:
        msg = (
            f"d_model ({config['d_model']}) is not divisible by "
            f"nhead ({config['nhead']}). Pruning trial."
        )
        logger.debug(msg)
        raise TrialPruned(msg)

    try:
        device = setup_device_func()
        logger.info(f"[Trial {trial.number}] Setting up dataset...")
        dataset_info = setup_dataset_func(config, data_dir)

        if dataset_info is None or dataset_info[0] is None:
            logger.warning(f"[Trial {trial.number}] Dataset setup failed. Pruning.")
            raise TrialPruned("Dataset setup returned None.")
        dataset, collate_fn = dataset_info

    except Exception as e:
        logger.warning(
            f"[Trial {trial.number}] Exception during dataset/device setup: {e}. Pruning."
        )
        # logger.debug("Setup exception details:", exc_info=True)
        raise TrialPruned("Exception during setup.")

    logger.info(
        f"[Trial {trial.number}] Starting training with config: {trial.params}"
    )
    try:
        best_val_loss = train_model_func(
            config, device, dataset, collate_fn, data_dir
        )

        if not isinstance(best_val_loss, float):
            logger.error(
                f"[Trial {trial.number}] train_model_func did not return float, got {type(best_val_loss)}. Pruning."
            )
            raise TrialPruned("Invalid return type from train_model_func.")

        if not torch.isfinite(torch.tensor(best_val_loss)):
            logger.warning(
                f"[Trial {trial.number}] Training returned non-finite validation loss ({best_val_loss}). Pruning."
            )
            raise TrialPruned("Non-finite validation loss.")

        logger.info(
            f"[Trial {trial.number}] Training finished. Best validation loss: {best_val_loss:.6e}"
        )
        return best_val_loss

    except TrialPruned:
        raise
    except Exception as e:
        logger.error(
            f"[Trial {trial.number}] Training failed unexpectedly: {e}",
            exc_info=True,
        )
        raise TrialPruned("Exception during training.")


# =============================================================================
# Study Orchestration
# =============================================================================


def run_hyperparameter_search(
    base_config: Dict[str, Any],
    data_dir: str,
    output_dir: str,
    setup_dataset_func: Callable[
        [Dict[str, Any], Path], Optional[Tuple[Any, Callable]]
    ],
    train_model_func: Callable[[Dict[str, Any], Any, Any, Any, Path], float],
    setup_device_func: Callable[[], Any],
    save_config_func: Callable[[Dict[str, Any], Union[str, Path]], bool],
    num_trials: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Orchestrates an Optuna hyperparameter search study.

    Minimizes the validation loss returned by the objective function. Saves the
    best configuration found so far after each trial and the final best
    configuration at the end. Optionally serializes the full Optuna study object.

    Args:
        base_config: The base configuration dictionary.
        data_dir: Path to the root data directory.
        output_dir: Path to the directory where tuning results (configs, study object)
                    will be saved.
        setup_dataset_func: Function to set up the dataset for a trial.
        train_model_func: Function to train the model for a trial, returning validation loss.
        setup_device_func: Function to set up the compute device.
        save_config_func: Function to save a configuration dictionary to a JSON file.
                          Expected signature: `save_config_func(config, filepath) -> bool`.
        num_trials: The maximum number of trials to run.

    Returns:
        The best hyperparameter configuration dictionary found, or None if the
        study fails or no trials complete successfully.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _save_current_best(study: Study, trial: Trial) -> None:
        """Callback executed after each Optuna trial."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(f"Trial {trial.number} finished. Value: {trial.value:.6e}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(f"Trial {trial.number} was pruned.")
        else:
            logger.info(
                f"Trial {trial.number} finished with state: {trial.state}"
            )

        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and study.best_trial is not None
            and trial.number == study.best_trial.number
        ):
            current_best_cfg = copy.deepcopy(base_config)
            current_best_cfg.update(study.best_params)
            current_best_path = out_path / "best_config_current.json"

            if save_config_func(current_best_cfg, str(current_best_path)):
                logger.info(
                    f"  -> Current best value={study.best_value:.6e}. Saved current best config to {current_best_path.name}"
                )
            else:
                logger.warning(
                    f"  -> Failed to save current best config at trial {trial.number}"
                )

    study = optuna.create_study(
        direction="minimize",
        # sampler=optuna.samplers.TPESampler(seed=base_config.get("random_seed", 42)),
        # pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    logger.info(f"Starting Optuna study with {num_trials} trials...")
    try:
        study.optimize(
            lambda trial_obj: objective(
                trial_obj,
                base_config,
                data_path,
                setup_dataset_func,
                train_model_func,
                setup_device_func,
            ),
            n_trials=num_trials,
            callbacks=[_save_current_best],
            catch=(Exception,),
        )
    except Exception as e:
        logger.error(f"Optuna study optimization failed: {e}", exc_info=True)
        return None

    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        logger.error("No Optuna trials completed successfully.")
        return None

    logger.info(
        f"Optuna study finished. Best trial: {study.best_trial.number}, Value: {study.best_value:.6e}"
    )

    final_best_cfg = copy.deepcopy(base_config)
    final_best_cfg.update(study.best_params)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_best_path = out_path / f"best_config_{timestamp}.json"
    if save_config_func(final_best_cfg, str(final_best_path)):
        logger.info(f"Final best configuration saved to {final_best_path.name}")
    else:
        logger.error(
            f"Failed to save final best configuration to {final_best_path.name}"
        )

    try:
        import joblib

        study_path = out_path / f"study_{timestamp}.pkl"
        joblib.dump(study, study_path)
        logger.info(f"Full Optuna study object saved to {study_path.name}")
    except ImportError:
        logger.debug(
            "joblib not installed; skipping Optuna study object serialization."
        )
    except Exception as e:
        logger.warning(f"Failed to save Optuna study object: {e}")

    return final_best_cfg


# =============================================================================
# Module Exports
# =============================================================================

__all__ = ["run_hyperparameter_search"]
