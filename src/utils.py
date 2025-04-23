#!/usr/bin/env python3
"""
utils.py – Core utilities for the atmospheric prediction pipeline.

Features:
- JSON/JSONC loading with optional json5 support.
- Configuration validation (updated for dynamic sequence length).
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
    root_logger = logging.getLogger()
    # Set level on the root logger
    root_logger.setLevel(level)

    # Remove existing handlers to prevent duplicates if called multiple times
    if root_logger.hasHandlers():
        for h in list(root_logger.handlers):
            # Flush and close handlers before removing
            if hasattr(h, 'flush'): h.flush()
            if hasattr(h, 'close'): h.close()
            root_logger.removeHandler(h)

    # Define format
    # Consider adding %(name)s to see logger source if using multiple loggers
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S" # Optional: customize date format
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    # Set level for the handler itself (optional, defaults to root logger level)
    # console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        try:
            log_file_path = Path(log_file)
            # Ensure parent directory exists
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file_path), mode='a') # Append mode
            file_handler.setFormatter(log_format)
            # Set level for the file handler (optional)
            # file_handler.setLevel(level)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging configured. Level: {logging.getLevelName(level)}. Outputting to console and file: {log_file_path}")
        except Exception as e:
            # Log error using the already configured console handler
            root_logger.error(f"Failed to set up file logging to {log_file}: {e}", exc_info=False) # Don't log traceback for config error
            root_logger.info(f"Logging configured. Level: {logging.getLevelName(level)}. Outputting to console only.")
    else:
         root_logger.info(f"Logging configured. Level: {logging.getLevelName(level)}. Outputting to console only.")


def load_config(config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load a JSON or JSONC (with comments/trailing commas) configuration file.

    Uses json5 if available for robust JSONC parsing, otherwise falls back
    to manual comment stripping before using standard json. Validates the
    configuration after loading.

    Args:
        config_path: Path to the configuration file.

    Returns:
        A dictionary representing the configuration, or None on failure
        (file not found, parsing error, or validation error).
    """
    path = Path(config_path)
    if not path.is_file():
        logger.error("Config file not found: %s", config_path)
        return None

    try:
        text = path.read_text(encoding="utf-8-sig") # Handle potential BOM
        if _HAS_JSON5:
            logger.debug("Using json5 library for parsing config.")
            cfg = json5.loads(text)
        else:
            logger.debug("json5 library not found. Using regex fallback for JSONC parsing.")
            # Fallback: remove C-style comments manually
            # Remove // comments
            text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
            # Remove /* ... */ comments (non-greedy)
            text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
            # Remove trailing commas before closing brackets/braces
            # This regex is simplified and might fail on complex cases (e.g., commas in strings)
            text = re.sub(r",(\s*[]}])", r"\1", text)
            try:
                cfg = json.loads(text)
            except json.JSONDecodeError as json_err:
                 logger.error("Failed to parse config '%s' using standard json after comment removal: %s", config_path, json_err)
                 logger.error("Consider installing 'json5' (pip install json5) for more robust JSONC support.")
                 return None


        logger.info("Configuration loaded from %s", config_path)

        # Perform validation after successful loading
        if not validate_config(cfg):
            logger.error("Configuration validation failed for %s. See previous errors for details.", config_path)
            return None # Return None if validation fails

        return cfg
    except FileNotFoundError: # Should be caught by is_file() check, but safeguard
        logger.error("Config file not found during read attempt: %s", config_path)
        return None
    except Exception as e:
        # Catch other potential errors (e.g., permissions)
        logger.error("Failed to load or parse config '%s': %s", config_path, e, exc_info=True)
        return None


def validate_config(cfg: Dict[str, Any]) -> bool:
    """
    Validate essential keys and constraints in the model configuration.

    Logs specific errors found during validation. Updated to remove
    `max_sequence_length` check, but keeps `sequence_lengths` check
    as it's needed for target validation.

    Args:
        cfg: The configuration dictionary to validate.

    Returns:
        True if the configuration is valid, False otherwise.
    """
    errors = []
    is_valid = True # Assume valid initially

    # Check required top-level keys and types
    required_lists = ["input_variables", "target_variables"]
    for key in required_lists:
        val = cfg.get(key)
        if not isinstance(val, list) or not val:
            errors.append(f"Config Error: '{key}' must be a non-empty list.")
            is_valid = False

    seq_types = cfg.get("sequence_types")
    if not isinstance(seq_types, dict) or not seq_types:
        errors.append("Config Error: 'sequence_types' must be a non-empty dictionary.")
        is_valid = False
    else:
        # Check sequence_types values are lists of strings
        if not all(isinstance(v, list) and all(isinstance(s, str) for s in v) for v in seq_types.values()):
             errors.append("Config Error: 'sequence_types' values must all be non-empty lists of variable name strings.")
             is_valid = False
        # Check if any sequence type has an empty list of variables
        if any(not v for v in seq_types.values()):
             errors.append("Config Error: All lists of variables in 'sequence_types' must be non-empty.")
             is_valid = False


        # Check output_seq_type validity
        out_seq = cfg.get("output_seq_type")
        if not isinstance(out_seq, str):
            errors.append("Config Error: 'output_seq_type' must be a string.")
            is_valid = False
        elif out_seq not in seq_types:
            errors.append(
                f"Config Error: 'output_seq_type' ('{out_seq}') not found as a key in 'sequence_types'."
            )
            is_valid = False
        elif not seq_types.get(out_seq): # Check if the list for the key is empty
             errors.append(f"Config Error: 'output_seq_type' ('{out_seq}') corresponds to an empty variable list in 'sequence_types'.")
             is_valid = False

        # Check sequence_lengths existence and validity (still needed for target length)
        seq_lengths = cfg.get("sequence_lengths")
        if not isinstance(seq_lengths, dict):
             errors.append("Config Error: 'sequence_lengths' must be a dictionary.")
             is_valid = False
        else:
             # Check that the output_seq_type has a length defined
             if out_seq and out_seq in seq_types and out_seq not in seq_lengths:
                  errors.append(f"Config Error: 'output_seq_type' ('{out_seq}') requires a corresponding length in 'sequence_lengths' for target validation.")
                  is_valid = False

             # Check all defined lengths are positive integers
             for k, length in seq_lengths.items():
                  if not isinstance(length, int) or length <= 0:
                       errors.append(f"Config Error: 'sequence_lengths' value for '{k}' must be a positive integer, got: {length}")
                       is_valid = False


    # Check model dimension constraints
    d_model = cfg.get("d_model")
    nhead = cfg.get("nhead")
    if not isinstance(d_model, int) or d_model <= 0:
        errors.append(f"Config Error: 'd_model' must be a positive integer, got: {d_model}")
        is_valid = False
    if not isinstance(nhead, int) or nhead <= 0:
        errors.append(f"Config Error: 'nhead' must be a positive integer, got: {nhead}")
        is_valid = False
    # Only check divisibility if both are valid positive integers
    if is_valid and isinstance(d_model, int) and isinstance(nhead, int) and d_model > 0 and nhead > 0:
        if d_model % nhead != 0:
            errors.append(f"Config Error: 'd_model' ({d_model}) must be divisible by 'nhead' ({nhead}).")
            is_valid = False

    # Check other essential model parameters exist and have correct basic types
    essential_params = {
        "num_encoder_layers": int,
        "dim_feedforward": int,
        "dropout": float,
        # "max_sequence_length": int, # REMOVED
        # Optional params with defaults checked elsewhere: norm_first, global_variables
    }
    for key, expected_type in essential_params.items():
        val = cfg.get(key)
        if val is None:
             # Allow dropout to be missing (defaults often handled in model/trainer)
             if key != "dropout":
                  errors.append(f"Config Error: Missing required parameter: '{key}'.")
                  is_valid = False
        elif not isinstance(val, expected_type):
             errors.append(f"Config Error: Parameter '{key}' must be type {expected_type.__name__}, got {type(val).__name__}.")
             is_valid = False
        elif expected_type == int and val <= 0:
             # Check positivity for integer parameters
             if key in ["num_encoder_layers", "dim_feedforward"]: # Removed max_sequence_length
                  errors.append(f"Config Error: Parameter '{key}' must be positive, got {val}.")
                  is_valid = False
        elif expected_type == float and not (0.0 <= val < 1.0):
             # Check dropout range
             if key == "dropout":
                  errors.append(f"Config Error: Parameter '{key}' (dropout) must be >= 0.0 and < 1.0, got {val}.")
                  is_valid = False


    # Log all errors found
    if not is_valid:
        for error in errors:
            logger.error(error) # Log as error
    # else:
    #     logger.debug("Configuration validated successfully.") # Optional: debug log

    return is_valid


def ensure_dirs(*dirs: Union[str, Path]) -> None:
    """
    Create directories if they do not exist.

    Args:
        *dirs: One or more directory paths (string or Path objects).

    Raises:
        OSError: If directory creation fails for reasons other than existence.
    """
    for d in dirs:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {d}: {e}")
            raise # Re-raise the error as this is likely critical


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
            # Handle numpy scalar types explicitly
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                # Check for NaN/Inf before converting
                if np.isnan(obj): return None # Represent NaN as null
                if np.isinf(obj): return str(obj) # Represent Inf as "Infinity" or "-Infinity"
                return float(obj)
            elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
                return {'real': obj.real, 'imag': obj.imag} # Represent complex as dict
            elif isinstance(obj, (np.bool_)):
                return bool(obj) # Convert numpy bool to Python bool
            elif isinstance(obj, (np.void)):
                 return None # Represent void type as null or handle appropriately
            # Handle torch tensors
            elif torch.is_tensor(obj):
                # Detach tensor from graph, move to CPU, convert to numpy, then list
                np_array = obj.cpu().detach().numpy()
                # Recursively handle potential non-serializable types within the array if needed
                # For simplicity, assume basic numeric types after conversion
                return np_array.tolist()
            elif isinstance(obj, set):
                return sorted(list(obj)) # Convert sets to sorted lists for deterministic output
            elif isinstance(obj, Path):
                 return str(obj) # Convert Path objects to strings
            # Let the default encoder raise the TypeError for other types
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )

        # Write the JSON file
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_encoder, ensure_ascii=False) # ensure_ascii=False for broader char support

        logger.debug("Saved JSON to %s", filepath) # Optional debug log
        return True

    except TypeError as e:
        logger.error("Failed to serialize data for JSON '%s': %s", filepath, e)
        return False
    except OSError as e:
        logger.error("Failed to write JSON file '%s': %s", filepath, e)
        return False
    except Exception as e:
        logger.error("An unexpected error occurred saving JSON '%s': %s", filepath, e, exc_info=True)
        return False


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for Python built-ins, numpy, and torch for reproducibility.

    Also sets the PYTHONHASHSEED environment variable. Attempts to configure
    deterministic algorithms in PyTorch, warning if they might impact performance.

    Args:
        seed: The integer value to use for seeding.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
        # Using deterministic algorithms can negatively impact performance.
        # Set these only if strict reproducibility is required over speed.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # logger.warning("PyTorch deterministic algorithms enabled for CUDA. This may impact performance.")

    logger.info("Global random seed set to %d", seed)


__all__ = [
    "setup_logging",
    "load_config",
    "validate_config",
    "ensure_dirs",
    "save_json",
    "seed_everything",
]
