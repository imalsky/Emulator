#!/usr/bin/env python3
"""
utils.py – Core utilities for the atmospheric prediction pipeline.

Features:
- JSON/JSONC loading with optional json5 support.
- Configuration validation.
- Directory creation helper.
- JSON serialization supporting numpy/torch types.
- Reproducible seeding utility.
"""
from __future__ import annotations

import logging
import os
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

# Attempt to import json5 for enhanced JSONC parsing
try:
    import json5
    _HAS_JSON5 = True
except ImportError:
    _HAS_JSON5 = False

logger = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None
) -> None:
    """
    Configure the root logger for console and optional file output.

    Removes existing handlers to prevent duplicate logging.

    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a file for log output.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to prevent duplicates if called multiple times
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close() # Close file handlers before removing

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File Handler (Optional)
    if log_file:
        try:
            fh = logging.FileHandler(str(log_file))
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception as e:
            root.error(f"Failed to set up file logging to {log_file}: {e}")


def load_config(config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load a JSON or JSONC (with comments/trailing commas) configuration file.

    Uses json5 if available for robust JSONC parsing, otherwise falls back
    to manual comment stripping before using standard json.

    Args:
        config_path: Path to the configuration file.

    Returns:
        A dictionary representing the configuration, or None on failure.
    """
    path = Path(config_path)
    if not path.is_file():
        logger.error("Config file not found: %s", config_path)
        return None

    try:
        text = path.read_text(encoding="utf-8")
        if _HAS_JSON5:
            cfg = json5.loads(text)
        else:
            # Fallback: remove C-style comments manually
            # Remove // comments
            text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
            # Remove /* ... */ comments
            text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
            # Remove trailing commas before closing brackets/braces
            text = re.sub(r",\s*([}\]])", r"\1", text)
            cfg = json.loads(text)

        logger.info("Configuration loaded from %s", config_path)

        # Perform validation after successful loading
        if not validate_config(cfg):
            logger.error("Configuration validation failed for %s", config_path)
            return None # Return None if validation fails

        return cfg
    except FileNotFoundError: # Should be caught by is_file() check, but safeguard
        logger.error("Config file not found during read attempt: %s", config_path)
        return None
    except Exception as e:
        logger.error("Failed to parse config '%s': %s", config_path, e)
        return None


def validate_config(cfg: Dict[str, Any]) -> bool:
    """
    Validate essential keys and constraints in the model configuration.

    Logs specific errors found during validation.

    Args:
        cfg: The configuration dictionary to validate.

    Returns:
        True if the configuration is valid, False otherwise.
    """
    errors = []

    # Check required top-level keys and types
    required_lists = ["input_variables", "target_variables"]
    for key in required_lists:
        val = cfg.get(key)
        if not isinstance(val, list) or not val:
            errors.append(f"'{key}' must be a non-empty list.")

    seq_types = cfg.get("sequence_types")
    if not isinstance(seq_types, dict) or not seq_types:
        errors.append("'sequence_types' must be a non-empty dictionary.")
    else:
        # Check sequence_types values are lists
        if not all(isinstance(v, list) for v in seq_types.values()):
             errors.append("'sequence_types' values must all be lists.")

        # Check output_seq_type validity
        out_seq = cfg.get("output_seq_type")
        if not isinstance(out_seq, str):
            errors.append("'output_seq_type' must be a string.")
        elif out_seq not in seq_types:
            errors.append(
                f"'output_seq_type' ('{out_seq}') not found as a key in 'sequence_types'."
            )
        elif not seq_types.get(out_seq): # Check if the list for the key is empty
             errors.append(f"'output_seq_type' ('{out_seq}') corresponds to an empty variable list in 'sequence_types'.")

        # Check sequence_lengths existence if sequence_types is valid
        seq_lengths = cfg.get("sequence_lengths")
        if not isinstance(seq_lengths, dict):
             errors.append("'sequence_lengths' must be a dictionary.")
        else:
             # Check all keys in sequence_types have a length defined
             missing_len_keys = set(seq_types.keys()) - set(seq_lengths.keys())
             if missing_len_keys:
                  errors.append(f"Missing keys in 'sequence_lengths' for sequence types: {missing_len_keys}")
             # Check all lengths are positive integers
             for k, length in seq_lengths.items():
                  if not isinstance(length, int) or length <= 0:
                       errors.append(f"'sequence_lengths' value for '{k}' must be a positive integer, got: {length}")


    # Check model dimension constraints
    d_model = cfg.get("d_model")
    nhead = cfg.get("nhead")
    if not isinstance(d_model, int) or d_model <= 0:
        errors.append(f"'d_model' must be a positive integer, got: {d_model}")
    if not isinstance(nhead, int) or nhead <= 0:
        errors.append(f"'nhead' must be a positive integer, got: {nhead}")
    if isinstance(d_model, int) and isinstance(nhead, int) and d_model > 0 and nhead > 0:
        if d_model % nhead != 0:
            errors.append(f"'d_model' ({d_model}) must be divisible by 'nhead' ({nhead}).")

    # Check other essential model parameters exist and have correct basic types
    essential_params = {
        "num_encoder_layers": int,
        "dim_feedforward": int,
        "dropout": float,
        "max_sequence_length": int,
        # Optional params with defaults checked elsewhere: norm_first, global_variables
    }
    for key, expected_type in essential_params.items():
        val = cfg.get(key)
        if val is None:
             errors.append(f"Missing required parameter: '{key}'.")
        elif not isinstance(val, expected_type):
             errors.append(f"Parameter '{key}' must be type {expected_type.__name__}, got {type(val).__name__}.")
        elif expected_type in (int, float) and val < 0:
             # Basic sanity check for non-negative values where applicable
             if key not in ["dropout"]: # Allow dropout=0
                  if val <= 0 and key in ["num_encoder_layers", "dim_feedforward", "max_sequence_length"]:
                       errors.append(f"Parameter '{key}' must be positive, got {val}.")


    if errors:
        for error in errors:
            logger.error("Configuration validation error: %s", error)
        return False
    else:
        # logger.debug("Configuration validated successfully.") # Optional: debug log
        return True


def ensure_dirs(*dirs: Union[str, Path]) -> None:
    """
    Create directories if they do not exist.

    Args:
        *dirs: One or more directory paths (string or Path objects).
    """
    for d in dirs:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {d}: {e}")
            # Depending on severity, you might want to raise the error
            # raise


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> bool:
    """
    Save a Python dictionary to a JSON file with indentation.

    Handles common non-serializable types like numpy arrays, numpy scalars,
    torch tensors, and sets by converting them to lists or standard Python types.

    Args:
        data: The dictionary to save.
        filepath: The path to the output JSON file.

    Returns:
        True if saving was successful, False otherwise.
    """
    path = Path(filepath)
    try:
        # Ensure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Custom encoder function for json.dump
        def _encoder(obj: Any):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item() # Convert numpy scalars to Python types
            if isinstance(obj, (np.bool_)):
                return bool(obj) # Convert numpy bool to Python bool
            if torch.is_tensor(obj):
                # Detach tensor from graph, move to CPU, convert to numpy, then list
                return obj.cpu().detach().numpy().tolist()
            if isinstance(obj, set):
                return list(obj) # Convert sets to lists
            # Let the default encoder raise the TypeError for other types
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )

        # Write the JSON file
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_encoder)

        # logger.debug("Saved JSON to %s", filepath) # Optional debug log
        return True

    except TypeError as e:
        logger.error("Failed to serialize data for JSON '%s': %s", filepath, e)
        return False
    except OSError as e:
        logger.error("Failed to write JSON file '%s': %s", filepath, e)
        return False
    except Exception as e:
        logger.error("An unexpected error occurred saving JSON '%s': %s", filepath, e)
        return False


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for Python built-ins, numpy, and torch for reproducibility.

    Also sets the PYTHONHASHSEED environment variable.

    Args:
        seed: The integer value to use for seeding.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Optional: Set deterministic algorithms if needed, but can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Global random seed set to %d", seed)


__all__ = [
    "setup_logging",
    "load_config",
    "validate_config", # Keep even if not directly used in main now
    "ensure_dirs",
    "save_json",
    "seed_everything",
]
