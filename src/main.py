#!/usr/bin/env python3
"""
main.py - Entry point for emulator

Provides command-line interface to normalize data, train models, and make predictions
using an encoder-only transformer with separate encoders for different
data types (global features and sequence data).
"""

import sys
import argparse
import logging
from pathlib import Path
import torch
import numpy as np

from utils import (setup_logging, load_config, ensure_dirs, save_config)
from hardware import setup_device
from normalizer import DataNormalizer
from dataset import AtmosphericDataset, MultiSourceCollate
from train import ModelTrainer
from hyperparams import run_hyperparameter_search

logger = logging.getLogger(__name__)

def normalize_data(config=None, data_dir="data"):
    """
    Normalize raw profile data using global statistics.
    """
    try:
        logger.info("Starting data normalization...")
        raw_dir = Path(data_dir) / "profiles"
        norm_dir = Path(data_dir) / "normalized_profiles"
        
        normalizer = DataNormalizer(raw_dir, norm_dir)
        norm_config = config.get("normalization", {}) if config is not None else {}
        key_methods = norm_config.get("key_methods", {})
        default_method = norm_config.get("default_method", "iqr")
        clip_outliers = norm_config.get("clip_outliers_before_scaling", False)
        
        # Extract symlog-specific parameters
        symlog_percentile = norm_config.get("symlog_percentile", 0.5)
        symlog_thresholds = norm_config.get("symlog_thresholds", {})
        
        logger.info(f"Calculating global normalization statistics using {default_method} as default method...")
        stats = normalizer.calculate_global_stats(
            key_methods=key_methods,
            default_method=default_method,
            clip_outliers_before_scaling=clip_outliers,
            symlog_percentile=symlog_percentile,
            symlog_thresholds=symlog_thresholds
        )
        logger.info("Applying normalization to profiles...")
        normalizer.process_profiles(stats)
        
        logger.info("Data normalization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Data normalization failed: {e}", exc_info=True)
        return False


def setup_dataset(config, data_dir="data"):
    """
    Initialize dataset with separation of data types.
    """
    try:
        logger.info("Initializing dataset with data type separation...")
        
        if "input_variables" not in config or "target_variables" not in config:
            raise ValueError("Missing required input_variables or target_variables in config")
        
        # Ensure required fields exist in the config
        if "global_variables" not in config:
            logger.warning("No global_variables specified in config, using empty list")
            config["global_variables"] = []
            
        if "sequence_types" not in config:
            logger.warning("No sequence_types defined in config, will use defaults in dataset constructor")
            config["sequence_types"] = {}
        
        full_dataset = AtmosphericDataset(
            data_folder=Path(data_dir) / "normalized_profiles",
            input_variables=config["input_variables"],
            target_variables=config["target_variables"],
            global_variables=config["global_variables"],
            sequence_types=config["sequence_types"]
            # Removed allow_variable_length parameter
        )
        
        if hasattr(full_dataset, "sequence_lengths"):
            config["sequence_lengths"] = dict(full_dataset.sequence_lengths)
            for seq_type, length in config["sequence_lengths"].items():
                if length > 10000:
                    raise ValueError(f"Sequence length for {seq_type} is {length}, exceeding maximum allowed 10000")
            logger.info(f"Detected sequence lengths: {config['sequence_lengths']}")
        
        # Use the MultiSourceCollate class directly instead of function
        config["collate_fn"] = MultiSourceCollate()
        
        if config.get("frac_of_data", 1.0) < 1.0:
            dataset_size = len(full_dataset)
            subset_size = int(dataset_size * config["frac_of_data"])
            indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42))[:subset_size]
            dataset = torch.utils.data.Subset(full_dataset, indices.tolist())
            logger.info(f"Using {subset_size}/{dataset_size} samples ({config['frac_of_data']:.1%})")
        else:
            dataset = full_dataset
        
        return dataset
    except Exception as e:
        logger.error(f"Dataset initialization failed: {e}", exc_info=True)
        return None


def train_model(config, device, dataset, data_dir="data"):
    """
    Train a prediction model with the encoder-only architecture.
    """
    try:
        if dataset is None:
            logger.error("Cannot train model: dataset is None")
            return False
        
        model_dir = Path(data_dir) / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if config.get("use_amp", False):
            logger.info("Enabling automatic mixed precision")
        
        trainer = ModelTrainer(
            config=config,
            device=device,
            save_path=model_dir,
            dataset=dataset,
            collate_fn=config.get("collate_fn")
        )
        
        epochs = config.get("epochs", 100)
        patience = config.get("early_stopping_patience", 15)
        final_model_path = trainer.train(num_epochs=epochs, early_stopping_patience=patience)
        
        if final_model_path:
            test_metrics = trainer.test()
            logger.info(f"Test metrics: {test_metrics:.3e}")
            
            best_val_metric = trainer.best_val_loss if hasattr(trainer, "best_val_loss") else None
            if best_val_metric is not None:
                metric_path = model_dir / "best_val_metric.txt"
                with open(metric_path, 'w') as f:
                    f.write(f"{best_val_metric:.10e}")
            
            return True
        else:
            logger.warning("Training did not produce a final model")
            return False
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        return False


def run_hyperparameter_tuning(base_config, data_dir, output_dir, num_trials=100):
    """
    Run hyperparameter tuning with a limited number of trials.
    """
    try:
        logger.info(f"Starting hyperparameter search with {num_trials} trials")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import random
        random.seed(42)
        
        # Pass an empty dict for the param grid (unused in the new version)
        best_config = run_hyperparameter_search(
            base_config=base_config,
            param_grid_config={},
            data_dir=data_dir,
            output_dir=output_dir,
            setup_dataset_func=setup_dataset,
            train_model_func=train_model,
            setup_device_func=setup_device,
            ensure_dirs_func=ensure_dirs,
            save_config_func=save_config,
            num_trials=num_trials
        )
        
        if best_config:
            logger.info("Hyperparameter tuning complete. Best configuration saved.")
            logger.info(f"To use the best configuration for training, run:")
            logger.info(f"python main.py --train --config {output_dir}/best_config.json")
        
        return best_config is not None
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}", exc_info=True)
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Atmospheric profile prediction pipeline")
    parser.add_argument("--normalize", action="store_true", help="Normalize data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials for hyperparameter tuning")
    parser.add_argument("--config", type=str, default="inputs/model_input_params.jsonc", help="Configuration file path")
    parser.add_argument("--data-dir", type=str, default="data", help="Base data directory")
    return parser.parse_args()


def main():
    """Main entry point for the atmospheric prediction pipeline."""
    args = parse_arguments()
    setup_logging()
    
    if not (args.normalize or args.train or args.tune):
        logger.error("No action specified. Use --normalize, --train, or --tune")
        return False
    
    data_dir = args.data_dir
    ensure_dirs(
        Path(data_dir) / "profiles", 
        Path(data_dir) / "normalized_profiles", 
        Path(data_dir) / "model"
    )
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False
    
    config = load_config(str(config_path))
    if not config:
        logger.error("Failed to load config file")
        return False
    
    overall_success = True
    
    if args.normalize:
        normalize_success = normalize_data(config, data_dir)
        overall_success = overall_success and normalize_success
    
    if (args.train or args.tune) and overall_success:
        device = setup_device()
        random_seed = config.get("random_seed", 42)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(random_seed)
        
        if args.train:
            dataset = setup_dataset(config, data_dir)
            if dataset is None:
                logger.error("Failed to setup dataset")
                return False
            train_success = train_model(config, device, dataset, data_dir)
            overall_success = overall_success and train_success
        
        if args.tune:
            output_dir = "tuning_results"
            tune_success = run_hyperparameter_tuning(config, data_dir, output_dir, args.trials)
            overall_success = overall_success and tune_success
    
    return overall_success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        sys.exit(1)