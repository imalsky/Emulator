#!/usr/bin/env python3
"""
utils.py – Core utilities for the atmospheric prediction pipeline.

Features:
- Strict config validation (no silent adjustments)
- JSON/JSONC loading with optional json5
- Directory creation
- JSON save/load handling numpy/torch types
- Reproducible seeding
"""
from __future__ import annotations

import logging
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

# Try json5 for JSONC support
try:
    import json5  # type: ignore
    _HAS_JSON5 = True
except ImportError:
    _HAS_JSON5 = False

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None) -> None:
    """
    Configure root logger to console (and file, if provided).
    """
    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers to avoid duplicates
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(fmt)
        root.addHandler(fh)


def load_config(config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load a JSON or JSONC (with comments) configuration file.
    Returns None on failure.
    """
    path = Path(config_path)
    if not path.is_file():
        logger.error("Config file not found: %s", config_path)
        return None
    text = path.read_text()
    try:
        if _HAS_JSON5:
            cfg = json5.loads(text)
        else:
            # strip // comments
            text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
            # strip /* */ comments
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
            cfg = json.loads(text)
        logger.info("Configuration loaded from %s", config_path)
        return cfg
    except Exception as e:
        logger.error("Failed to parse config '%s': %s", config_path, e)
        return None


def validate_config(cfg: Dict[str, Any]) -> bool:
    """
    Validate essential keys in model configuration. Raises ValueError on failure.

    Checks:
      - non-empty input_variables list
      - non-empty target_variables list
      - explicit output_seq_type present in config and sequence_types
      - d_model divisible by nhead (error if not)
    """
    try:
        # required lists
        iv = cfg.get('input_variables')
        tv = cfg.get('target_variables')
        if not isinstance(iv, list) or not iv:
            raise ValueError("'input_variables' must be a non-empty list")
        if not isinstance(tv, list) or not tv:
            raise ValueError("'target_variables' must be a non-empty list")

        # sequence_types mapping must include output_seq_type
        seq_types = cfg.get('sequence_types')
        if not isinstance(seq_types, dict) or not seq_types:
            raise ValueError("'sequence_types' must be a non-empty dict")
        out_seq = cfg.get('output_seq_type')
        if not isinstance(out_seq, str) or out_seq not in seq_types or not seq_types[out_seq]:
            raise ValueError("'output_seq_type' must be explicitly set to a non-empty sequence type key")

        # d_model and nhead divisibility
        dm = cfg.get('d_model', 256)
        nh = cfg.get('nhead', 8)
        if not isinstance(dm, int) or not isinstance(nh, int) or dm % nh != 0:
            raise ValueError(f"d_model ({dm}) must be an integer divisible by nhead ({nh})")

        return True
    except ValueError as e:
        logger.error("Configuration validation error: %s", e)
        return False


def ensure_dirs(*dirs: Union[str, Path]) -> None:
    """
    Create directories if they do not exist.
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> bool:
    """
    Save a Python dict to JSON, handling numpy arrays and torch tensors.
    Returns True on success, False otherwise.
    """
    path = Path(filepath)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        def _encoder(obj: Any):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)): return obj.item()
            if isinstance(obj, (np.bool_)): return bool(obj)
            if torch.is_tensor(obj): return obj.cpu().detach().numpy().tolist()
            if isinstance(obj, set): return list(obj)
            raise TypeError(f"{type(obj)} not JSON serializable")
        with path.open('w') as f:
            json.dump(data, f, indent=2, default=_encoder)
        # logger.info("Saved JSON to %s", filepath) # Optional: can be verbose
        return True
    except Exception as e:
        logger.error("Failed to save JSON '%s': %s", filepath, e)
        return False


def seed_everything(seed: int = 42) -> None:
    """
    Seed built‑ins, numpy, and torch for reproducibility.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info("Random seed set to %d", seed)


__all__ = [
    "setup_logging",
    "load_config",
    "validate_config",
    "ensure_dirs",
    "save_json",
    "seed_everything",
]