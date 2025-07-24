#!/usr/bin/env python3
"""
main.py - Entry point with profiling support for performance analysis.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

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




DEFAULT_CONFIG_PATH = Path("config/config.jsonc")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_MODELS_DIR = Path("models")

logger = logging.getLogger(__name__)


def _get_raw_hdf5_paths(config: Dict[str, Any], raw_dir: Path) -> List[Path]:
    h5_filenames = config.get("data_paths_config", {}).get("hdf5_dataset_filename", [])
    if not isinstance(h5_filenames, list) or not h5_filenames:
        raise ValueError("No raw HDF5 files specified in config.")

    raw_paths = [raw_dir / fname for fname in h5_filenames]
    missing = [p for p in raw_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw HDF5 files: {missing}. Exiting.")

    return raw_paths


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atmospheric profile transformer training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root data directory (contains raw/ and processed/).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory for saved models.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=0,
        help="Number of Optuna trials for hyperparameter optimization (0 to disable).",
    )
    parser.add_argument(
        "--optuna-study-name", type=str, default=None, help="Optuna study name."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler for performance analysis.",
    )
    parser.add_argument(
        "--profile-wait",
        type=int,
        default=1,
        help="Number of steps to wait before profiling.",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=1,
        help="Number of warmup steps for profiler.",
    )
    parser.add_argument(
        "--profile-active",
        type=int,
        default=3,
        help="Number of steps to actively profile.",
    )
    parser.add_argument(
        "--profile-epochs",
        type=int,
        default=1,
        help="Number of epochs to profile (only used with --profile).",
    )
    return parser.parse_args()


def run_training_with_profiler(
    config: Dict[str, Any],
    device: torch.device,
    model_save_dir: Path,
    processed_dir: Path,
    splits: Dict[str, List[Tuple[str, int]]],
    padding_val: float,
    profile_config: Dict[str, int],
) -> None:
    """Run training with PyTorch profiler enabled."""
    
    # Temporarily reduce epochs for profiling
    original_epochs = config["training_hyperparameters"]["epochs"]
    config["training_hyperparameters"]["epochs"] = profile_config["epochs"]
    
    logger.info("Starting training with PyTorch profiler...")
    logger.info(f"Profiling {profile_config['epochs']} epoch(s)")
    logger.info(f"Profile schedule: wait={profile_config['wait']}, "
               f"warmup={profile_config['warmup']}, active={profile_config['active']}")
    
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
        trainer = ModelTrainer(
            config=config,
            device=device,
            save_dir=model_save_dir,
            processed_dir=processed_dir,
            splits=splits,
            collate_fn=collate_fn,
        )
        
        # Override train method to include profiler step
        original_run_epoch = trainer._run_epoch
        
        def profiled_run_epoch(loader, is_train):
            result = original_run_epoch(loader, is_train)
            prof.step()  # Advance profiler
            return result
        
        trainer._run_epoch = profiled_run_epoch
        
        # Run training
        trainer.train()
    
    # Export Chrome trace
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
    
    # Restore original epochs
    config["training_hyperparameters"]["epochs"] = original_epochs
    
    logger.info("\nProfiling complete! View results with:")
    logger.info(f"  tensorboard --logdir={trace_dir}")
    logger.info(f"  chrome://tracing (load {chrome_trace_path})")


def main() -> int:
    args = _parse_arguments()

    ensure_dirs(args.data_dir, args.models_dir)
    setup_logging()

    try:
        logger.info("Pipeline started")
        logger.info(f"Using config: {args.config.resolve()}")
        config = load_config(args.config)

        seed = config.get("miscellaneous_settings", {}).get("random_seed", DEFAULT_SEED)
        seed_everything(seed)

        device = setup_device()
        if device.type == "cuda":
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.set_float32_matmul_precision("high")
                logger.info("Set float32 matmul precision to 'high' for A100/newer GPU")

        raw_dir = args.data_dir / "raw"
        processed_dir = args.data_dir / "processed"
        raw_hdf5_paths = _get_raw_hdf5_paths(config, raw_dir)

        model_folder = get_config_str(
            config, "output_paths_config", "fixed_model_foldername", "model training"
        )
        model_save_dir = args.models_dir / model_folder
        ensure_dirs(model_save_dir)

        log_file = model_save_dir / "run.log"
        setup_logging(log_file=log_file, force=True)

        save_json(config, model_save_dir / "run_config.json")

        splits, splits_path = load_or_generate_splits(
            config, args.data_dir, raw_hdf5_paths, model_save_dir
        )

        save_json(
            {"splits_file": str(splits_path.resolve())},
            model_save_dir / "splits_info.json",
        )

        padding_val = float(
            config.get("data_specification", {}).get("padding_value", PADDING_VALUE)
        )

        if not preprocess_data(
            config=config,
            raw_hdf5_paths=raw_hdf5_paths,
            splits=splits,
            processed_dir=processed_dir,
        ):
            raise RuntimeError("Preprocessing failed due to invalid data. Exiting.")

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
        elif args.num_trials > 0:
            run_optuna(
                config, args, device, processed_dir, splits, padding_val, model_save_dir
            )
        else:
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

        logger.info("Pipeline completed successfully.")
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