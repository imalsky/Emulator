#!/usr/bin/env python3
"""
main.py – command‑line entry point for the atmospheric‑flux pipeline.

Updates (2025‑04‑28)
--------------------
* Reads optional `validate_profiles` boolean from the top‑level config; defaults
  to *True* for backwards compatibility.
* Passes `validate_profiles` through to `AtmosphericDataset`, allowing users to
  skip the expensive one‑off profile scan when desired.
* Minor docstring/typing tidy‑ups; functional flow unchanged.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from hardware import setup_device
from normalizer import DataNormalizer
from dataset import AtmosphericDataset, create_multi_source_collate_fn
from train import ModelTrainer
from hyperparams import run_hyperparameter_search
from utils import (
    ensure_dirs,
    load_config,
    setup_logging,
    save_json,
    seed_everything,
)

# suppress noisy nested‑tensor warning on pre‑norm encoder layers
import warnings  # noqa: E402
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _normalize(cfg: Dict[str, Any], data_root: Path) -> bool:
    """Run normalisation once; return *True* on success."""
    raw_dir = data_root / "profiles"
    out_dir = data_root / "normalized_profiles"
    ensure_dirs(raw_dir, out_dir)

    norm_cfg = cfg.get("normalization", {})
    normalizer = DataNormalizer(raw_dir, out_dir)
    stats = normalizer.calculate_global_stats(
        key_methods=norm_cfg.get("key_methods"),
        default_method=norm_cfg.get("default_method", "standard"),
        clip_outliers_before_scaling=norm_cfg.get("clip_outliers_before_scaling", False),
        symlog_percentile=norm_cfg.get("symlog_percentile", 0.5),
        symlog_thresholds=norm_cfg.get("symlog_thresholds"),
    )
    if stats is None:
        logger.error("normalizer.calculate_global_stats returned None")
        return False
    normalizer.process_profiles(stats)
    logger.info("Normalisation completed")
    return True


# dataset cache so train/tune can share the same object in one run
_DATASET_CACHE: dict[str, Tuple[AtmosphericDataset, Callable]] = {}

def _setup_dataset(cfg: Dict[str, Any], data_dir: Path) -> Optional[Tuple[AtmosphericDataset, Callable]]:
    key = str(data_dir.resolve())
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    norm_dir = data_dir / "normalized_profiles"
    if not norm_dir.exists():
        logger.error("Normalised data not found at %s; run --normalize first", norm_dir)
        return None

    try:
        ds = AtmosphericDataset(
            data_folder=norm_dir,
            input_variables=cfg["input_variables"],
            target_variables=cfg["target_variables"],
            global_variables=cfg.get("global_variables", []),
            sequence_types=cfg["sequence_types"],
            sequence_lengths=cfg["sequence_lengths"],
            output_seq_type=cfg["output_seq_type"],
            cache_size=cfg.get("dataset_cache_size", 1024),
            validate_profiles=cfg.get("validate_profiles", True),
        )
        collate = create_multi_source_collate_fn()
        _DATASET_CACHE[key] = (ds, collate)
        return ds, collate
    except Exception as exc:
        logger.error("Dataset setup failed: %s", exc, exc_info=True)
        return None


def train_model(
    config: Dict[str, Any],
    device: torch.device,
    dataset: AtmosphericDataset,
    collate_fn: Callable,
    save_dir: Path,
) -> float:
    """Wrapper around ModelTrainer so Optuna can call it transparently."""
    trainer = ModelTrainer(
        config=config,
        device=device,
        save_dir=save_dir,
        dataset=dataset,
        collate_fn=collate_fn,
    )
    best = trainer.train()
    if not isinstance(best, float):
        raise RuntimeError("ModelTrainer did not return float")
    return best

# --------------------------------------------------------------------------- #
# Hyper‑parameter tuning                                                      #
# --------------------------------------------------------------------------- #

def _run_tuning(cfg: Dict[str, Any], data_dir: Path, n_trials: int) -> bool:
    logger.info("Hyper‑parameter search: %d trials", n_trials)

    def _train_wrapper(
        trial_cfg: Dict[str, Any],
        device: torch.device,
        ds: AtmosphericDataset,
        collate: Callable,
        root: Path,
    ) -> float:
        return train_model(trial_cfg, device, ds, collate, root)

    best_cfg = run_hyperparameter_search(
        base_config=cfg,
        data_dir=str(data_dir),
        output_dir=str(data_dir / "tuning_results"),
        setup_dataset_func=_setup_dataset,
        train_model_func=_train_wrapper,
        setup_device_func=setup_device,
        save_config_func=save_json,
        num_trials=n_trials,
    )
    return best_cfg is not None

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Atmospheric‑flux transformer pipeline")
    p.add_argument("--normalize", action="store_true", help="Run data normalisation")
    p.add_argument("--train", action="store_true", help="Train a model")
    p.add_argument("--tune", action="store_true", help="Optuna hyper‑parameter search")
    p.add_argument("--config", type=Path, default=Path("inputs/model_input_params.jsonc"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--trials", type=int, default=25, help="Optuna trials")
    args = p.parse_args()

    if not (args.normalize or args.train or args.tune):
        p.error("Specify at least one of --normalize, --train, --tune")
    return args

# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    args = _parse_args()
    setup_logging()

    cfg = load_config(args.config)
    if cfg is None:
        logger.error("Configuration load/validation failed")
        return 1

    seed_everything(cfg.get("random_seed", 42))
    ensure_dirs(
        args.data_dir / "profiles",
        args.data_dir / "normalized_profiles",
        args.data_dir / "model",
        args.data_dir / "tuning_results",
    )

    # ----------------------- normalise -----------------------
    if args.normalize and not _normalize(cfg, args.data_dir):
        return 1

    # ----------------------- train ---------------------------
    if args.train:
        device = setup_device()
        ds_info = _setup_dataset(cfg, args.data_dir)
        if ds_info is None:
            return 1
        ds, coll = ds_info
        try:
            train_model(cfg, device, ds, coll, args.data_dir / "model")
        except Exception as exc:
            logger.error("Training failed: %s", exc, exc_info=True)
            return 1

    # ----------------------- tune ----------------------------
    if args.tune and not _run_tuning(cfg, args.data_dir, args.trials):
        return 1

    logger.info("Pipeline finished successfully")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
