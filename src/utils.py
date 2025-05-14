#!/usr/bin/env python3
"""
utils.py – core helpers for the atmospheric-profile pipeline.

This module provides utility functions for logging setup, configuration loading
and validation, directory creation, JSON serialization, and random seed setting.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys # For sys.exit on critical config validation errors
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

# Attempt to use json5 for more flexible JSON parsing (comments, trailing commas)
# Fall back to standard json if json5 is not available.
try:
    import json5 as _json_backend
    _HAS_JSON5 = True
    logger_for_json = logging.getLogger(__name__) # Define logger if json5 is used
    logger_for_json.info("Using json5 for configuration file parsing.")
except ImportError:
    _json_backend = json
    _HAS_JSON5 = False
    logger_for_json = logging.getLogger(__name__) # Define logger if json5 is not used
    logger_for_json.info(
        "json5 not found. Using standard json library for configuration. "
        "Ensure config files are strict JSON if not using json5."
    )


logger = logging.getLogger(__name__)

# Regexes for stripping comments and trailing commas if not using json5
_JSONC_COM_RE = re.compile(r"//.*?$|/\*.*?\*/", re.DOTALL | re.MULTILINE)
_JSONC_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def setup_logging(
    level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None
) -> None:
    """
    Initialise the root logger with console (and optional file) output.

    Configures a specific format for log messages and handles potential
    issues with file logging.

    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a file where logs should also be written.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to prevent duplicate logging if re-called
    while root_logger.handlers:
        handler = root_logger.handlers.pop()
        handler.close()

    log_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                file_path, mode="a", encoding="utf-8"
            )
            file_handler.setFormatter(log_formatter)
            root_logger.addHandler(file_handler)
            logger.info("Logging to console and file: %s", file_path)
        except Exception as exc:
            logger.error(
                "File logging setup failed for %s: %s. "
                "Falling back to console logging only.", file_path, exc
            )


def _strip_jsonc_features(text: str) -> str:
    """
    Strips C-style comments and trailing commas from a JSON string.
    This is used as a fallback if json5 is not available.

    Args:
        text: The JSON string, potentially with comments/trailing commas.

    Returns:
        A cleaned JSON string.
    """
    text_no_comments = _JSONC_COM_RE.sub("", text)
    text_no_trailing_commas = _JSONC_TRAILING_COMMA_RE.sub(r"\1", text_no_comments)
    return text_no_trailing_commas


def load_config(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load JSON or JSONC/JSON5 configuration file and validate its basic structure.

    Args:
        path: Path to the configuration file.

    Returns:
        A dictionary with the loaded configuration, or None if loading
        or validation fails. Critical validation errors will call sys.exit.
    """
    config_path = Path(path)
    if not config_path.is_file():
        logger.critical("Configuration file not found: %s. Exiting.", config_path)
        sys.exit(1) # Critical error: config file missing

    try:
        raw_config_text = config_path.read_text(encoding="utf-8-sig") # Handle BOM
        if _HAS_JSON5:
            config_dict = _json_backend.loads(raw_config_text)
        else:
            cleaned_text = _strip_jsonc_features(raw_config_text)
            config_dict = json.loads(cleaned_text)
    except Exception as exc:
        logger.critical(
            "Failed to parse configuration file %s: %s. Exiting.", config_path, exc
        )
        sys.exit(1) # Critical error: config parsing failed

    logger.info("Successfully loaded configuration from %s.", config_path)

    # Perform basic structural validation on the loaded config.
    # validate_config will sys.exit on critical failures.
    if not validate_config(config_dict):
        # Message logged by validate_config, sys.exit also handled there.
        # This path should not be reached if validate_config exits.
        logger.critical("Configuration validation failed. Terminating.")
        return None # Should be unreachable if validate_config exits
    return config_dict


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Performs basic structural and type checks on the configuration dictionary.
    Logs errors and calls sys.exit(1) for critical validation failures.

    Args:
        config: The configuration dictionary to validate.

    Returns:
        True if validation passes (though critical errors will exit).
        This return value is somewhat moot if sys.exit is called.
    """
    validation_passed = True

    def _validation_error(message: str, is_critical: bool = True):
        nonlocal validation_passed
        validation_passed = False
        if is_critical:
            logger.critical("Config validation error: %s. Exiting.", message)
            sys.exit(1)
        else:
            logger.error("Config validation error: %s.", message)


    # Mandatory non-empty lists for core variable definitions
    for key in ("input_variables", "target_variables"):
        if not isinstance(config.get(key), list) or not config[key]:
            _validation_error(f"'{key}' must be a non-empty list.")

    # Validate sequence_types structure
    sequence_types_val = config.get("sequence_types")
    if not isinstance(sequence_types_val, dict) or not sequence_types_val:
        _validation_error("'sequence_types' must be a non-empty dictionary.")
    else:
        for st_name, st_vars in sequence_types_val.items():
            if not isinstance(st_vars, list): # Each value must be a list
                 _validation_error(
                    f"Variables for sequence type '{st_name}' must be a list."
                )
            # Allowing empty list for a sequence type if not output_seq_type,
            # create_prediction_model will handle if it's problematic.
            # dataset.py also handles empty var lists for sequence types.

    # Validate output_seq_type
    output_seq_type_val = config.get("output_seq_type")
    if not isinstance(output_seq_type_val, str) or not output_seq_type_val:
        _validation_error("'output_seq_type' must be a non-empty string.")
    elif sequence_types_val and output_seq_type_val not in sequence_types_val:
        _validation_error(
            f"'output_seq_type' ('{output_seq_type_val}') must be a key "
            f"in 'sequence_types'."
        )
    elif sequence_types_val and output_seq_type_val in sequence_types_val and \
         not sequence_types_val.get(output_seq_type_val):
         _validation_error(
            f"'output_seq_type' ('{output_seq_type_val}') refers to an "
            "empty variable list in 'sequence_types'."
        )


    # The 'sequence_lengths' key is no longer used with padding.
    # Its validation has been removed.

    # Validate model dimensions
    d_model_val = config.get("d_model")
    nhead_val = config.get("nhead")
    if not (isinstance(d_model_val, int) and d_model_val > 0):
        _validation_error("'d_model' must be a positive integer.")
    if not (isinstance(nhead_val, int) and nhead_val > 0):
        _validation_error("'nhead' must be a positive integer.")

    # d_model % nhead is handled by create_prediction_model, which adjusts nhead
    # and logs a warning. No need for a critical exit here for that specific check.
    # However, if d_model or nhead are invalid types/values, it's critical.

    # Validate other essential integer/float model parameters
    for key in ("num_encoder_layers", "dim_feedforward"):
        val = config.get(key)
        if not isinstance(val, int) or val <= 0:
            _validation_error(f"'{key}' must be a positive integer.")

    dropout_val = config.get("dropout", 0.1) # Default if not present
    if not isinstance(dropout_val, (float, int)) or not (0.0 <= dropout_val < 1.0):
        _validation_error("'dropout' must be a float or int in the range [0.0, 1.0).")

    # Validate positional_encoding_type if present
    pe_type = config.get("positional_encoding_type")
    if pe_type is not None and not isinstance(pe_type, str):
        _validation_error("'positional_encoding_type' must be a string if provided.")


    if not validation_passed:
        # This path should ideally not be reached if critical errors call sys.exit.
        # It's a fallback.
        logger.critical("One or more configuration validation errors occurred. Please review config. Exiting.")
        sys.exit(1)
    return True


def ensure_dirs(*paths: Union[str, Path]) -> None:
    """
    Create each supplied directory path, including any necessary parent
    directories. Does not raise an error if the directory already exists.

    Args:
        *paths: One or more directory paths (str or Path objects).
    """
    for p_item in paths:
        Path(p_item).mkdir(parents=True, exist_ok=True)


def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for objects not handled by the default encoder.
    Covers numpy types, torch tensors, sets, and Path objects.

    Args:
        obj: The object to serialize.

    Returns:
        A JSON-serializable representation of the object.
    Raises:
        TypeError: If the object type is not supported.
    """
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, set):
        return sorted(list(obj)) # Convert set to sorted list for consistent output
    if isinstance(obj, Path):
        return str(obj)
    # Let the default encoder handle other types or raise TypeError
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable by this custom encoder."
    )


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> bool:
    """
    Write dictionary data to a JSON file with indentation and robust encoding.

    Args:
        data: The dictionary to save.
        path: The file path where the JSON should be saved.

    Returns:
        True if saving was successful, False otherwise.
    """
    file_path = Path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=_json_serializer, ensure_ascii=False)
        return True
    except IOError as exc:
        logger.error("Failed to save JSON to %s (IOError): %s", file_path, exc)
        return False
    except TypeError as exc: # Catch specific serialization errors
        logger.error("Failed to serialize data for JSON %s (TypeError): %s", file_path, exc)
        return False
    except Exception as exc: # Catch any other unexpected errors
        logger.error("Unexpected error saving JSON to %s: %s", file_path, exc, exc_info=True)
        return False


def seed_everything(seed: int = 42) -> None:
    """
    Sets random seeds for Python's `random`, NumPy, and PyTorch (CPU and CUDA)
    to promote reproducibility. Also sets PYTHONHASHSEED environment variable.

    Args:
        seed: The integer seed value to use.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # For all GPUs
        # The following lines can be enabled for further determinism,
        # but may impact performance.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    logger.info("Global random seed set to %d.", seed)


__all__ = [
    "setup_logging",
    "load_config",
    "validate_config",
    "ensure_dirs",
    "save_json",
    "seed_everything",
]
