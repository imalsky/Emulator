#!/usr/bin/env python3
"""
hyperparams.py - Hyperparameter tuning for atmospheric profile prediction

This version uses Optuna for efficient hyperparameter sampling with pruning.
It defines an objective function that samples hyperparameters, sets up the device and dataset,
and trains the model while reporting intermediate validation metrics for early stopping.

Intended to be used with main.py, which calls:
    from hyperparams import run_hyperparameter_search
"""

import copy
import json
import logging
import inspect
from datetime import datetime
from pathlib import Path

import optuna
import optuna.exceptions

logger = logging.getLogger(__name__)

def objective(trial, base_config, data_dir, train_model_func, setup_dataset_func, setup_device_func):
    """
    Objective function for Optuna hyperparameter tuning with pruning.

    Parameters
    ----------
    trial : optuna.trial.Trial
        An Optuna trial object.
    base_config : dict
        Base configuration dictionary.
    data_dir : str
        Base data directory.
    train_model_func : callable
        Function to train the model. It should ideally accept a "trial" keyword argument
        to report intermediate metrics; if not, a wrapper will be used.
    setup_dataset_func : callable
        Function to set up the dataset.
    setup_device_func : callable
        Function to set up the device.

    Returns
    -------
    float
        The best validation metric achieved during training.
    """
    config = copy.deepcopy(base_config)

    # Sample hyperparameters using Optuna's suggestion methods.
    config["d_model"] = trial.suggest_categorical("d_model", [256, 512])
    config["nhead"] = trial.suggest_categorical("nhead", [4, 8, 16])
    config["num_encoder_layers"] = trial.suggest_categorical("num_encoder_layers", [4, 8, 16])
    config["dim_feedforward"] = trial.suggest_categorical("dim_feedforward", [1024, 2048])
    config["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    config["layer_scale"] = trial.suggest_categorical("layer_scale", [0, 0.05, 0.1])
    config["mlp_layers"] = trial.suggest_categorical("mlp_layers", [2, 4, 8])
    config["mlp_hidden_dim"] = trial.suggest_categorical("mlp_hidden_dim", [32, 64, 128])
    #config["activation"] = trial.suggest_categorical("activation", ["gelu", "relu"])
    config["positional_encoding"] = trial.suggest_categorical("positional_encoding", ["rotary", "sine", "learned"])
    #config["batch_size"] = trial.suggest_categorical("batch_size", [32])
    #config["learning_rate"] = trial.suggest_categorica("learning_rate", [1e-4])
    #config["weight_decay"] = trial.suggest_categorical("weight_decay", [0, 1e-6, 1e-4])
    #config["optimizer"] = trial.suggest_categorical("optimizer", ["adamw"])
    #config["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [10])

    device = setup_device_func()
    dataset = setup_dataset_func(config, data_dir)
    if dataset is None:
        raise ValueError("Dataset initialization failed")

    sig = inspect.signature(train_model_func)
    if "trial" in sig.parameters:
        best_metric = train_model_func(config, device, dataset, data_dir, trial=trial)
    else:
        best_metric = train_model_for_tuning(config, device, dataset, data_dir, trial)

    return best_metric

def train_model_for_tuning(config, device, dataset, data_dir, trial):
    """
    Wrapper training function to support per-epoch reporting for pruning.
    
    This function creates a ModelTrainer instance and runs a training loop
    that reports intermediate validation metrics to Optuna via `trial.report`.
    It checks if the trainer has a `train_one_epoch` method, and if not, falls back
    to using the `_train_epoch` method.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    device : torch.device or similar
        Device to train on.
    dataset : torch.utils.data.Dataset
        Dataset to train on.
    data_dir : str
        Base data directory.
    trial : optuna.trial.Trial
        An Optuna trial object.
        
    Returns
    -------
    float
        The best validation metric achieved.
    """
    from train import ModelTrainer  # Import here to avoid circular imports.
    model_dir = Path(data_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = ModelTrainer(
        config=config,
        device=device,
        save_path=model_dir,
        dataset=dataset,
        collate_fn=config.get("collate_fn")
    )
    
    # Choose the per-epoch training method.
    if hasattr(trainer, "train_one_epoch"):
        epoch_fn = trainer.train_one_epoch
    elif hasattr(trainer, "_train_epoch"):
        epoch_fn = trainer._train_epoch
    else:
        raise AttributeError("No per-epoch training method found on ModelTrainer")
    
    epochs = config.get("epochs", 100)
    patience = config.get("early_stopping_patience", 15)
    best_metric = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        sig = inspect.signature(epoch_fn)
        if len(sig.parameters) == 0:
            current_metric = epoch_fn()
        else:
            current_metric = epoch_fn(epoch)
        
        trial.report(current_metric, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch
        
        if epoch - best_epoch >= patience:
            break

    return best_metric

def run_hyperparameter_search(base_config, param_grid_config, data_dir, output_dir,
                              setup_dataset_func, train_model_func, setup_device_func,
                              ensure_dirs_func, save_config_func, num_trials=10):
    """
    Run hyperparameter search using Optuna with early stopping (pruning).
    The current best configuration is saved after each trial via a callback.
    
    Parameters
    ----------
    base_config : dict
        Base configuration dictionary.
    param_grid_config : dict
        Parameter grid configuration dictionary (unused in this version).
    data_dir : str
        Base data directory.
    output_dir : str
        Directory to store tuning results.
    setup_dataset_func : callable
        Function to set up the dataset.
    train_model_func : callable
        Function to train the model (ideally supports a 'trial' parameter).
    setup_device_func : callable
        Function to set up the device.
    ensure_dirs_func : callable
        Function to ensure necessary directories exist.
    save_config_func : callable
        Function to save a configuration (should accept (config, file_path)).
    num_trials : int, optional
        Number of trials to run (default is 10).
    
    Returns
    -------
    dict
        Best configuration based on the validation metric.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # We'll store the total number of trials in a local variable.
    # This is because Study objects in recent Optuna versions don't have a public n_trials attribute.
    max_trials = num_trials

    def progress_callback(study, trial):
        """Logs progress after each trial completes."""
        completed_trials = len(study.trials)
        logger.info(f"Trial {trial.number} finished with value: {trial.value:.3e}")
        logger.info(
            f"Completed {completed_trials} / {max_trials} trials. "
            f"Best value so far: {study.best_value:.3e}"
        )

    def save_best_callback(study, trial):
        """Saves current best configuration after each trial completes."""
        current_best = copy.deepcopy(base_config)
        for key, value in study.best_trial.params.items():
            current_best[key] = value
        current_best_path = output_path / "best_config_current.json"
        save_config_func(current_best, str(current_best_path))
        logger.info(f"Updated current best configuration at trial {trial.number} saved to {current_best_path}")

    # Create the study and optimize with callbacks for progress and saving.
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: objective(t, base_config, data_dir, train_model_func,
                            setup_dataset_func, setup_device_func),
        n_trials=num_trials,
        callbacks=[save_best_callback, progress_callback]
    )

    # Build the final best configuration after all trials finish.
    best_config = copy.deepcopy(base_config)
    for key, value in study.best_trial.params.items():
        best_config[key] = value

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_config_path = output_path / f"best_config_{timestamp}.json"
    save_config_func(best_config, str(best_config_path))

    # Save the study object if joblib is installed (for full trial history).
    study_path = output_path / f"study_{timestamp}.pkl"
    try:
        import joblib
        joblib.dump(study, study_path)
    except ImportError:
        logger.warning("joblib is not installed; study object will not be saved.")

    logger.info(f"Best configuration saved to {best_config_path}")
    logger.info(f"Best validation metric: {study.best_value:.6e}")

    return best_config
