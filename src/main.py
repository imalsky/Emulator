#!/usr/bin/env python3
"""
main.py - Entry point with separate normalize, train, and tune commands.

Commands:
- normalize: Preprocess and normalize the data
- train: Train a model with current configuration
- tune: Run hyperparameter optimization with Optuna
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.profiler import profile, ProfilerActivity, schedule

from dataset import create_collate_fn
from hardware import setup_device
from preprocess import preprocess_data
from train import ModelTrainer
from utils import (
    DEFAULT_SEED,
    PADDING_VALUE,
    ensure_dirs,
    get_config_str,
    load_config,
    load_or_generate_splits,
    save_json,
    seed_everything,
    setup_logging,
)
from hyperparam_search import run_optuna

# Prevent MKL library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Default paths
DEFAULT_CONFIG_PATH = Path("config/config.jsonc")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_MODELS_DIR = Path("models")

logger = logging.getLogger(__name__)


def _get_raw_hdf5_paths(config: Dict[str, Any], raw_dir: Path) -> List[Path]:
    """
    Get list of raw HDF5 file paths from configuration.
    
    Args:
        config: Configuration dictionary
        raw_dir: Directory containing raw HDF5 files
        
    Returns:
        List of HDF5 file paths
        
    Raises:
        ValueError: If no files specified
        FileNotFoundError: If files are missing
    """
    h5_filenames = config.get("data_paths_config", {}).get("hdf5_dataset_filename", [])
    
    if not isinstance(h5_filenames, list) or not h5_filenames:
        raise ValueError("No raw HDF5 files specified in config.")
    
    raw_paths = [raw_dir / fname for fname in h5_filenames]
    missing = [p for p in raw_paths if not p.is_file()]
    
    if missing:
        raise FileNotFoundError(f"Missing raw HDF5 files: {missing}")
    
    return raw_paths


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with subcommands."""
    parser = argparse.ArgumentParser(
        description="Atmospheric profile transformer pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Common arguments for all commands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file.",
    )
    parent_parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root data directory (contains raw/ and processed/).",
    )
    parent_parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory for saved models.",
    )
    
    # Normalize command
    normalize_parser = subparsers.add_parser(
        'normalize',
        parents=[parent_parser],
        help='Preprocess and normalize the data'
    )
    normalize_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-normalization even if cached data exists.",
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        parents=[parent_parser],
        help='Train a model with current config'
    )
    train_parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler for performance analysis.",
    )
    train_parser.add_argument(
        "--profile-wait",
        type=int,
        default=1,
        help="Number of steps to wait before profiling.",
    )
    train_parser.add_argument(
        "--profile-warmup",
        type=int,
        default=1,
        help="Number of warmup steps for profiler.",
    )
    train_parser.add_argument(
        "--profile-active",
        type=int,
        default=3,
        help="Number of steps to actively profile.",
    )
    train_parser.add_argument(
        "--profile-epochs",
        type=int,
        default=1,
        help="Number of epochs to profile (only used with --profile).",
    )
    
    # Tune command (hyperparameter search)
    tune_parser = subparsers.add_parser(
        'tune',
        parents=[parent_parser],
        help='Run hyperparameter optimization with Optuna'
    )
    tune_parser.add_argument(
        "--num-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter optimization.",
    )
    tune_parser.add_argument(
        "--optuna-study-name",
        type=str,
        default=None,
        help="Optuna study name.",
    )
    tune_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing Optuna study.",
    )
    
    args = parser.parse_args()
    
    # Check that a command was specified
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    return args


def run_normalize(
    config: Dict[str, Any],
    raw_hdf5_paths: List[Path],
    processed_dir: Path,
    model_save_dir: Path,
    force: bool = False,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Run data normalization step.
    
    Args:
        config: Configuration dictionary
        raw_hdf5_paths: List of raw HDF5 files
        processed_dir: Directory for processed data
        model_save_dir: Directory for saving splits info
        force: If True, force reprocessing
        
    Returns:
        Dictionary of data splits
    """
    logger.info("=== Running Data Normalization ===")
    
    # Load or generate splits
    splits, splits_path = load_or_generate_splits(
        config, processed_dir.parent, raw_hdf5_paths, model_save_dir
    )
    
    # Save splits info
    save_json(
        {"splits_file": str(splits_path.resolve())},
        model_save_dir / "splits_info.json",
    )
    
    # Force reprocessing if requested
    if force:
        logger.info("Force flag set - removing existing processed data")
        import shutil
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
    
    # Run preprocessing
    success = preprocess_data(
        config=config,
        raw_hdf5_paths=raw_hdf5_paths,
        splits=splits,
        processed_dir=processed_dir,
    )
    
    if not success:
        raise RuntimeError("Preprocessing failed due to invalid data.")
    
    logger.info("=== Data Normalization Complete ===")
    return splits


def run_training_with_profiler(
    config: Dict[str, Any],
    device: torch.device,
    model_save_dir: Path,
    processed_dir: Path,
    splits: Dict[str, List[Tuple[str, int]]],
    padding_val: float,
    profile_config: Dict[str, int],
) -> None:
    """
    Run training with PyTorch profiler enabled.
    
    Args:
        config: Configuration dictionary
        device: Compute device
        model_save_dir: Directory for saving models
        processed_dir: Directory with processed data
        splits: Data splits
        padding_val: Padding value
        profile_config: Profiler configuration
    """
    # Temporarily reduce epochs for profiling
    original_epochs = config["training_hyperparameters"]["epochs"]
    config["training_hyperparameters"]["epochs"] = profile_config["epochs"]
    
    try:
        logger.info("Starting training with PyTorch profiler...")
        logger.info(f"Profiling {profile_config['epochs']} epoch(s)")
        logger.info(
            f"Profile schedule: wait={profile_config['wait']}, "
            f"warmup={profile_config['warmup']}, active={profile_config['active']}"
        )
        
        # Create profiler schedule
        prof_schedule = schedule(
            wait=profile_config["wait"],
            warmup=profile_config["warmup"],
            active=profile_config["active"],
            repeat=1
        )
        
        # Profile trace directory
        trace_dir = model_save_dir / "profiler_traces"
        ensure_dirs(trace_dir)
        
        # Run training with profiler
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True
        ) as prof:
            
            collate_fn = create_collate_fn(padding_val)
            
            # Pass profiler to trainer
            trainer = ModelTrainer(
                config=config,
                device=device,
                save_dir=model_save_dir,
                processed_dir=processed_dir,
                splits=splits,
                collate_fn=collate_fn,
                profiler=prof,
            )
            
            trainer.train()
        
        # Export profiler results - only executed if no exceptions
        chrome_trace_path = trace_dir / "chrome_trace.json"
        prof.export_chrome_trace(str(chrome_trace_path))
        logger.info(f"Chrome trace saved to: {chrome_trace_path}")
        
        # Print profiler summary
        logger.info("\n=== Profiler Summary ===")
        logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        # Export detailed stats
        stats_path = trace_dir / "profiler_stats.txt"
        with open(stats_path, "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total"))
        logger.info(f"Detailed stats saved to: {stats_path}")
        
        logger.info("\nProfiling complete! View results with:")
        logger.info(f"  tensorboard --logdir={trace_dir}")
        logger.info(f"  chrome://tracing (load {chrome_trace_path})")
        
    finally:
        # Restore original epochs - guaranteed to execute
        config["training_hyperparameters"]["epochs"] = original_epochs


def run_train(
    config: Dict[str, Any],
    device: torch.device,
    model_save_dir: Path,
    processed_dir: Path,
    raw_hdf5_paths: List[Path],
    args: argparse.Namespace,
) -> None:
    """
    Run the training pipeline.
    
    Args:
        config: Configuration dictionary
        device: Compute device
        model_save_dir: Directory for saving models
        processed_dir: Directory with processed data
        raw_hdf5_paths: List of raw HDF5 files
        args: Command line arguments
    """
    logger.info("=== Running Model Training ===")
    
    # Ensure data is preprocessed
    splits = run_normalize(
        config, raw_hdf5_paths, processed_dir, model_save_dir, force=False
    )
    
    # Get padding value
    padding_val = float(
        config.get("data_specification", {}).get("padding_value", PADDING_VALUE)
    )
    
    # Run with profiler if requested
    if args.profile:
        profile_config = {
            "wait": args.profile_wait,
            "warmup": args.profile_warmup,
            "active": args.profile_active,
            "epochs": args.profile_epochs,
        }
        run_training_with_profiler(
            config, device, model_save_dir, processed_dir,
            splits, padding_val, profile_config
        )
    else:
        # Normal training
        collate_fn = create_collate_fn(padding_val)
        trainer = ModelTrainer(
            config=config,
            device=device,
            save_dir=model_save_dir,
            processed_dir=processed_dir,
            splits=splits,
            collate_fn=collate_fn,
        )
        trainer.train()
        trainer.test()
    
    logger.info("=== Model Training Complete ===")


def run_tune(
    config: Dict[str, Any],
    device: torch.device,
    model_save_dir: Path,
    processed_dir: Path,
    raw_hdf5_paths: List[Path],
    args: argparse.Namespace,
) -> None:
    """
    Run hyperparameter tuning.
    
    Args:
        config: Configuration dictionary
        device: Compute device
        model_save_dir: Directory for saving models
        processed_dir: Directory with processed data
        raw_hdf5_paths: List of raw HDF5 files
        args: Command line arguments
    """
    logger.info("=== Running Hyperparameter Tuning ===")
    
    # Ensure data is preprocessed
    splits = run_normalize(
        config, raw_hdf5_paths, processed_dir, model_save_dir, force=False
    )
    
    # Get padding value
    padding_val = float(
        config.get("data_specification", {}).get("padding_value", PADDING_VALUE)
    )
    
    # Create subdirectory for hyperparameter search
    hyperparam_dir = model_save_dir / "hyperparam_search"
    ensure_dirs(hyperparam_dir)
    
    # Run Optuna optimization
    run_optuna(
        config, args, device, processed_dir, splits, padding_val, hyperparam_dir
    )
    
    logger.info("=== Hyperparameter Tuning Complete ===")


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = _parse_arguments()
    
    # Setup directories
    ensure_dirs(args.data_dir, args.models_dir)
    setup_logging()
    
    try:
        logger.info(f"Running command: {args.command}")
        logger.info(f"Using config: {args.config.resolve()}")
        
        # Load configuration
        config = load_config(args.config)
        
        # Set random seed for reproducibility
        seed = config.get("miscellaneous_settings", {}).get("random_seed", DEFAULT_SEED)
        seed_everything(seed)
        
        # Setup compute device
        device = setup_device()
        
        # Apply hardware-specific settings
        if device.type == "cuda":
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.set_float32_matmul_precision("high")
                logger.info("Set float32 matmul precision to 'high' for newer GPUs")
        
        # Setup paths
        raw_dir = args.data_dir / "raw"
        processed_dir = args.data_dir / "processed"
        raw_hdf5_paths = _get_raw_hdf5_paths(config, raw_dir)
        
        # Create model save directory
        model_folder = get_config_str(
            config, "output_paths_config", "fixed_model_foldername", "model training"
        )
        model_save_dir = args.models_dir / model_folder
        ensure_dirs(model_save_dir)
        
        # Setup logging to file
        log_file = model_save_dir / f"{args.command}_run.log"
        setup_logging(log_file=log_file, force=True)
        
        # Save configuration
        save_json(config, model_save_dir / f"{args.command}_config.json")
        
        # Execute the appropriate command
        if args.command == 'normalize':
            run_normalize(
                config, raw_hdf5_paths, processed_dir, model_save_dir,
                force=args.force
            )
        
        elif args.command == 'train':
            run_train(
                config, device, model_save_dir, processed_dir,
                raw_hdf5_paths, args
            )
        
        elif args.command == 'tune':
            run_tune(
                config, device, model_save_dir, processed_dir,
                raw_hdf5_paths, args
            )
        
        else:
            raise ValueError(f"Unknown command: {args.command}")
        
        logger.info(f"{args.command.capitalize()} completed successfully.")
        return 0
        
    except RuntimeError as e:
        logger.error(f"Pipeline error: {e}", exc_info=False)
        return 1
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C).")
        return 130
        
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())