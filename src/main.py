#!/usr/bin/env python3
"""
main_spectral.py - Entry point for spectral prediction pipeline
Handles ultra-large spectral sequences (up to 40,000 points)
"""

import sys
import argparse
import logging
from pathlib import Path
import os
import torch
import numpy as np

from utils import setup_logging, load_config, setup_device, ensure_dirs, save_json

logger = logging.getLogger("spectral_prediction")


def normalize_spectral_data(config=None):
    """
    Normalize input profiles and spectral data.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary that may contain coordinate_variable info
    """
    try:
        from normalizer import SpectralDataNormalizer
        
        raw_dir = "data/profiles"
        norm_dir = "data/normalized_profiles"
        Path(norm_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up variable methods - adjust based on your data characteristics
        variable_methods = {
            "pressure": "log",           # Log scaling works well for pressure values
            "temperature": "standard",   # Z-score for temperature
            "transit_depth": "standard",  # Z-score for target values
            "wavenumber": "log"
        }
        
        # Add coordinate variables dynamically from config if available
        #if config and "coordinate_variable" in config:
        #    for coord_var in config["coordinate_variable"]:
        #        logger.info(f"Setting normalization method for coordinate variable '{coord_var}' to 'log'")
        #        variable_methods[coord_var] = "log"
        
        normalizer = SpectralDataNormalizer(raw_dir, norm_dir)
        stats = normalizer.calculate_global_stats(
            variable_methods=variable_methods,
            default_method="standard"
        )
        
        normalizer.process_data(stats)
        return True
    except Exception as e:
        logger.error(f"Data normalization failed: {e}")
        return False


def setup_spectral_dataset(config):
    """Initialize dataset with improved auto-detection for ultra-large spectral sequences."""
    try:
        from dataset import SpectralProfileDataset
        
        logger.info("Initializing spectral dataset with auto variable detection...")
        
        # Check for required configuration
        required_keys = ["input_variables", "target_variables"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration: {key}")
        
        # Get coordinate variables from config
        coordinate_variable = config.get("coordinate_variable", [])
        if coordinate_variable:
            logger.info(f"Using coordinate variables: {coordinate_variable}")
        
        # Auto-detect input sequence length from pressure range or first profile variable
        input_seq_length = config.get("input_seq_length")
        if not input_seq_length and "pressure_range" in config:
            input_seq_length = config["pressure_range"].get("points", 100)
            logger.info(f"Using pressure_range.points ({input_seq_length}) for input_seq_length")
        
        # Let the dataset auto-detect output sequence length if not specified
        output_seq_length = config.get("output_seq_length", None)
        
        full_dataset = SpectralProfileDataset(
            data_folder="data/normalized_profiles",
            input_seq_length=input_seq_length,
            output_seq_length=output_seq_length,
            input_variables=config["input_variables"],
            target_variables=config["target_variables"],
            coordinate_variable=coordinate_variable
        )
        
        # Update config with detected sequence lengths and variable types
        if hasattr(full_dataset, "output_seq_length") and full_dataset.output_seq_length:
            detected_length = full_dataset.output_seq_length
            logger.info(f"Auto-detected output_seq_length: {detected_length}")
            config["output_seq_length"] = detected_length
            
            # Adjust chunk size based on sequence length
            if detected_length > 100000:
                logger.warning(f"Extremely large sequence length detected ({detected_length} points)")
                config["chunk_size"] = min(config.get("chunk_size", 100), 50)  # Ultra-conservative
            elif detected_length > 40000:
                logger.warning(f"Ultra-large sequence length detected ({detected_length} points)")
                config["chunk_size"] = min(config.get("chunk_size", 100), 100)  # Very conservative
            elif detected_length > 10000:
                logger.warning(f"Very large sequence length detected ({detected_length} points)")
                config["chunk_size"] = min(config.get("chunk_size", 250), 250)  # Conservative
            
            # Auto-enable mixed precision for large sequences
            if detected_length > 20000 and not config.get("use_mixed_precision") and not config.get("use_amp"):
                logger.info("Auto-enabling mixed precision for large sequence")
                config["use_mixed_precision"] = True
                config["use_amp"] = True
        
        # Optionally use only a fraction of the data for quick experiments
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
        logger.error(f"Dataset initialization failed: {e}")
        return None


def train_spectral_model(config, device, dataset):
    """Train a spectral prediction model with optimizations for large sequences."""
    try:
        if not dataset:
            return False
        
        model_dir = "data/spectral_model"
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Adjust batch size automatically for ultra-large sequences
        if config.get("output_seq_length", 0) > 20000 and config.get("batch_size", 32) > 4:
            original_batch = config.get("batch_size", 32)
            config["batch_size"] = min(4, original_batch)
            logger.info(f"Reduced batch size from {original_batch} to {config['batch_size']} "
                       f"due to large sequence length ({config['output_seq_length']})")
            
        # Enable mixed precision by default for ultra-large sequences
        if config.get("output_seq_length", 0) > 20000:
            config["use_mixed_precision"] = config.get("use_mixed_precision", True)
            config["use_amp"] = config.get("use_amp", True)
            
        from train import ModelTrainer
        
        trainer = ModelTrainer(
            config=config,
            device=device,
            save_path=model_dir,
            dataset=dataset
        )
        
        epochs = config.get("epochs", 100)
        patience = config.get("early_stopping_patience", 15)
        
        final_model_path = trainer.train(
            num_epochs=epochs,
            early_stopping_patience=patience
        )
        
        if final_model_path:
            test_metrics = trainer.test()
            logger.info(f"Test metrics: {test_metrics}")
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False


def parse_arguments():
    parser = argparse.ArgumentParser(description="Spectral prediction pipeline")
    parser.add_argument("--normalize-data", action="store_true", help="Normalize data")
    parser.add_argument("--train-model", action="store_true", help="Train model")
    parser.add_argument("--config", type=str, default="inputs/model_input_params.jsonc", 
                        help="Configuration file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    ensure_dirs("data/profiles", "data/normalized_profiles", "data/spectral_model")
    
    overall_success = True
    
    # Load config early so we can use it for normalization
    config = None
    if args.train_model or args.normalize_data:
        config = load_config(args.config)
        if not config and (args.train_model):
            return False
    
    if args.normalize_data:
        success = normalize_spectral_data(config)
        overall_success = overall_success and success
    
    if overall_success and (args.train_model):
        if not config:
            # Should never happen due to earlier check, but just in case
            config = load_config(args.config)
            if not config:
                return False
        
        device = setup_device()
        
        # Set random seed
        random_seed = config.get("random_seed", 42)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(random_seed)
        
        dataset = setup_spectral_dataset(config)
        if dataset is None:
            return False
        
        if args.train_model:
            success = train_spectral_model(config, device, dataset)
            overall_success = overall_success and success
    
    return overall_success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        sys.exit(1)