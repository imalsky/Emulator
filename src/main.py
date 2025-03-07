#!/usr/bin/env python3
"""
main.py - Entry point 
"""

import sys
import argparse
import logging
from pathlib import Path
import torch
import numpy as np

from utils import setup_logging, load_config, setup_device, ensure_dirs, save_json

from normalizer import DataNormalizer
from dataset import Dataset
from train import ModelTrainer

logger = logging.getLogger("Prediction")

def normalize_data(config=None):
    """
    Normalize raw profile data using global statistics.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary which may contain normalization settings
    
    Returns
    -------
    bool
        True if normalization succeeds, False otherwise
    """
    try:
        logger.info("Starting data normalization...")
        
        # Setup directories with fixed paths
        raw_dir = "data/profiles"
        norm_dir = "data/normalized_profiles"
        norm_dir_path = Path(norm_dir)
            
        norm_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize normalizer
        normalizer = DataNormalizer(raw_dir, norm_dir)
        
        # Get normalization settings from config if available
        key_methods = {}
        default_method = "iqr"
        clip_outliers = False
        
        if config is not None:
            # Extract normalization settings from config
            norm_config = config.get("normalization", {})
            key_methods = norm_config.get("key_methods", {})
            default_method = norm_config.get("default_method", default_method)
            clip_outliers = norm_config.get("clip_outliers_before_scaling", clip_outliers)
        
        # Calculate global statistics
        logger.info("Calculating global normalization statistics...")
        stats = normalizer.calculate_global_stats(
            key_methods=key_methods,
            default_method=default_method,
            clip_outliers_before_scaling=clip_outliers,
        )
        
        # Process and normalize profiles
        logger.info("Applying normalization to profiles...")
        normalizer.process_profiles(stats)
        
        logger.info("Data normalization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Data normalization failed: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Detailed error:", exc_info=True)
        return False


def setup_dataset(config):
    """Initialize dataset with improved auto-detection for variable sequence lengths."""
    try:        
        logger.info("Initializing dataset with auto variable detection...")
        
        # Check for required configuration
        required_keys = ["input_variables", "target_variables"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration: {key}")
        
        # Get sequence lengths from config if provided
        input_seq_length = config.get("input_seq_length")
        output_seq_length = config.get("output_seq_length")
        
        # Initialize dataset with auto-detection for sequence lengths
        full_dataset = Dataset(
            data_folder="data/normalized_profiles",
            input_seq_length=input_seq_length,
            output_seq_length=output_seq_length,
            input_variables=config["input_variables"],
            target_variables=config["target_variables"]
        )
        
        # Update config with detected sequence lengths and variable types
        if hasattr(full_dataset, "output_seq_length") and full_dataset.output_seq_length:
            detected_length = full_dataset.output_seq_length
            logger.info(f"Auto-detected output_seq_length: {detected_length}")
            config["output_seq_length"] = detected_length
            
            # Auto-enable mixed precision for large sequences
            if detected_length > 20000 and not config.get("use_mixed_precision") and not config.get("use_amp"):
                logger.info("Auto-enabling mixed precision for large sequence")
                config["use_mixed_precision"] = True
                config["use_amp"] = True
        
        if hasattr(full_dataset, "input_seq_length") and full_dataset.input_seq_length:
            config["input_seq_length"] = full_dataset.input_seq_length
        
        # Update config with detected variable types
        if hasattr(full_dataset, "global_variables"):
            global_indices = [i for i, var in enumerate(config["input_variables"]) 
                            if var in full_dataset.global_variables]
            if global_indices:
                config["global_feature_indices"] = global_indices
                logger.info(f"Setting global_feature_indices: {global_indices}")
                
        if hasattr(full_dataset, "sequence_variables"):
            seq_indices = [i for i, var in enumerate(config["input_variables"]) 
                        if var in full_dataset.sequence_variables]
            if seq_indices:
                config["seq_feature_indices"] = seq_indices
                logger.info(f"Setting seq_feature_indices: {seq_indices}")
            
        # Ensure all input variables are accounted for
        all_indices = set(config.get("global_feature_indices", [])) | set(config.get("seq_feature_indices", []))
        if len(all_indices) != len(config["input_variables"]):
            logger.warning("Some input variables couldn't be classified as global or sequential")
            # Default unclassified variables to sequential
            unclassified = [i for i in range(len(config["input_variables"])) if i not in all_indices]
            if "seq_feature_indices" not in config:
                config["seq_feature_indices"] = []
            config["seq_feature_indices"].extend(unclassified)
            logger.info(f"Added unclassified variables to seq_feature_indices: {unclassified}")
        
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


def train_model(config, device, dataset):
    """Train a prediction model with optimizations for large sequences."""
    try:
        if dataset is None:
            logger.error("Cannot train model: dataset is None")
            return False
        
        model_dir = "data/model"
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
            logger.info(f"Test metrics: {test_metrics:.3e}")
            return True
        else:
            logger.warning("Training did not produce a final model")
            return False
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False


def parse_arguments():
    # Get directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent  # Go up one level to project root
    default_config = project_root / "inputs" / "model_input_params.jsonc"
    
    parser = argparse.ArgumentParser(description="Prediction pipeline")
    parser.add_argument("--normalize-data", action="store_true", help="Normalize data")
    parser.add_argument("--train-model", action="store_true", help="Train model")
    parser.add_argument("--config", type=str, default=str(default_config), 
                        help="Configuration file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    ensure_dirs("data/profiles", "data/normalized_profiles", "data/model")
    
    overall_success = True
    
    # Load config early so we can use it for normalization
    config = None
    if args.train_model or args.normalize_data:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return False
            
        config = load_config(str(config_path))
        if not config and args.train_model:
            logger.error("Failed to load config file and training was requested")
            return False
    
    if args.normalize_data:
        success = normalize_data(config)
        overall_success = overall_success and success
    
    if overall_success and args.train_model:
        if not config:
            # Should never happen due to earlier check, but just in case
            logger.error("Config is required for training")
            return False
        
        device = setup_device()
        
        # Set random seed
        random_seed = config.get("random_seed", 42)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(random_seed)
        
        dataset = setup_dataset(config)
        if dataset is None:
            logger.error("Failed to setup dataset")
            return False
        
        success = train_model(config, device, dataset)
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