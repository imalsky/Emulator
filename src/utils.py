#!/usr/bin/env python3
"""
utils.py - Helper functions for configuration, logging, and data handling.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import h5py
import numpy as np
import torch

# Try to import JSON5 for comment support in config files
try:
    import json5 as _json_backend
    _HAS_JSON5 = True
except ImportError:
    _json_backend = json
    _HAS_JSON5 = False

# Global constants
DTYPE = torch.float32
PADDING_VALUE = -9999.0
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
DEFAULT_SEED = 42
METADATA_FILENAME = "normalization_metadata.json"
UTF8_ENCODING = "utf-8"
UTF8_SIG_ENCODING = "utf-8-sig"  # Handle UTF-8 with BOM
HASH_ALGORITHM = "sha256"

logger = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO,
    log_file: Union[str, Path, None] = None,
    force: bool = False,
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (e.g., logging.INFO)
        log_file: Optional file path for logging output
        force: If True, remove existing handlers before setup
    """
    root_logger = logging.getLogger()
    
    # Force reset if requested
    if force:
        while root_logger.handlers:
            handler = root_logger.handlers.pop()
            handler.close()
    
    root_logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Add console handler if none exists
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file:
        try:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_file_path, mode="a", encoding=UTF8_ENCODING
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            print(f"Logging to console and file: {log_file_path.resolve()}")
        except OSError as e:
            print(f"Error setting up file logging for {log_file}: {e}. Using console only.")
    else:
        print("Logging to console only.")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate a configuration file (JSON or JSON5).
    
    Args:
        path: Path to configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        RuntimeError: If config is invalid or malformed
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r", encoding=UTF8_SIG_ENCODING) as f:
            if _HAS_JSON5:
                config_dict = _json_backend.load(f)
            else:
                logger.warning("JSON5 not available; comments in config will cause errors.")
                config_dict = json.load(f)
        
        validate_config(config_dict)
        
        backend = "JSON5" if _HAS_JSON5 else "JSON"
        logger.info(f"Loaded and validated {backend} config from {config_path}.")
        return config_dict
        
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from {config_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load or validate config {config_path}: {e}") from e


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate data specification
    data_spec = config.get("data_specification")
    if not isinstance(data_spec, dict):
        raise ValueError("Config section 'data_specification' is missing or not a dictionary.")
    
    if not data_spec.get("input_variables"):
        raise ValueError("'input_variables' must be a non-empty list.")
    
    if not data_spec.get("target_variables"):
        raise ValueError("'target_variables' must be a non-empty list.")
    
    # Validate model hyperparameters
    model_params = config.get("model_hyperparameters")
    if not isinstance(model_params, dict):
        raise ValueError("Config section 'model_hyperparameters' is missing or not a dictionary.")
    
    d_model = model_params.get("d_model", 0)
    nhead = model_params.get("nhead", 0)
    
    if not isinstance(d_model, int) or d_model <= 0:
        raise ValueError("'d_model' must be a positive integer.")
    
    if not isinstance(nhead, int) or nhead <= 0:
        raise ValueError("'nhead' must be a positive integer.")
    
    if d_model % nhead != 0:
        raise ValueError(f"'d_model' ({d_model}) must be divisible by 'nhead' ({nhead}).")


def ensure_dirs(*paths: Union[str, Path, None]) -> bool:
    """
    Create directories if they don't exist.
    
    Args:
        *paths: Variable number of directory paths
        
    Returns:
        True if successful, False otherwise
    """
    try:
        for path in paths:
            if path is not None:
                Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directories {paths}: {e}")
        return False


def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for NumPy/PyTorch types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable representation
        
    Raises:
        TypeError: If object type is not supported
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj.resolve())
    
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def save_json(data: Dict[str, Any], path: Union[str, Path], compact: bool = False) -> bool:
    """
    Save dictionary to JSON file with custom serialization.

    Args:
        data: Dictionary to save
        path: Output file path
        compact: If True, use minimal formatting for smaller files

    Returns:
        True if successful, False otherwise
    """
    try:
        json_path = Path(path)
        ensure_dirs(json_path.parent)

        with json_path.open("w", encoding=UTF8_ENCODING) as f:
            if compact:
                # Minimal formatting for compact files
                json.dump(data, f, default=_json_serializer,
                          ensure_ascii=False, separators=(',', ':'))
            else:
                # Human-readable formatting
                json.dump(data, f, indent=2, default=_json_serializer,
                          ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        return True

    except (OSError, TypeError) as e:
        logger.error(f"Failed to save JSON to {path}: {e}", exc_info=True)
        return False


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Global random seed set to {seed}.")


def generate_dataset_splits(
    raw_hdf5_paths: List[Path],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_seed: int = DEFAULT_SEED,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Generate train/validation/test splits from HDF5 files.
    
    Args:
        raw_hdf5_paths: List of HDF5 file paths
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        random_seed: Seed for reproducible splits
        
    Returns:
        Dictionary with 'train', 'validation', 'test' splits
        
    Raises:
        ValueError: If fractions are invalid
        RuntimeError: If no data found
    """
    # Validate fractions
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError(f"Invalid split fractions: val={val_frac}, test={test_frac}")
    
    all_indices = []
    total_profiles = 0
    
    # Collect all indices from all files
    for h5_path in raw_hdf5_paths:
        if not h5_path.is_file():
            logger.warning(f"Skipping missing HDF5 file: {h5_path}")
            continue
            
        try:
            with h5py.File(h5_path, "r") as hf:
                if not hf.keys():
                    continue
                
                datasets = [k for k in hf.keys() 
                           if isinstance(hf[k], h5py.Dataset) and hf[k].ndim > 0]
                
                if not datasets:
                    continue
                
                n_profiles = hf[datasets[0]].shape[0]
                
                # Verify consistent dimensions across datasets only
                if not all(hf[k].shape[0] == n_profiles for k in datasets):
                    raise AssertionError("Inconsistent leading dimensions in HDF5")
                
                file_stem = h5_path.stem
                all_indices.extend([(file_stem, i) for i in range(n_profiles)])
                total_profiles += n_profiles
                
        except (OSError, AssertionError) as e:
            logger.warning(f"Failed to read {h5_path}: {e}. Skipping.")
    
    if total_profiles == 0:
        raise RuntimeError("No profiles found across all HDF5 files.")
    
    # Calculate split sizes
    n_val = max(1, int(round(total_profiles * val_frac)))
    n_test = max(1, int(round(total_profiles * test_frac)))
    n_train = total_profiles - n_val - n_test
    
    if n_train <= 0:
        raise ValueError(f"Training split empty. Reduce val/test fractions.")
    
    # Shuffle and split
    rng = random.Random(random_seed)
    rng.shuffle(all_indices)
    
    splits = {
        "train": all_indices[:n_train],
        "validation": all_indices[n_train:n_train + n_val],
        "test": all_indices[n_train + n_val:],
    }
    
    logger.info(
        f"Generated splits from {total_profiles} profiles across {len(raw_hdf5_paths)} files: "
        f"train={len(splits['train'])} ({len(splits['train'])/total_profiles:.1%}), "
        f"val={len(splits['validation'])} ({len(splits['validation'])/total_profiles:.1%}), "
        f"test={len(splits['test'])} ({len(splits['test'])/total_profiles:.1%})"
    )
    
    return splits


def get_config_str(config: Dict[str, Any], section: str, key: str, op_desc: str) -> str:
    """
    Safely extract a string value from nested config dictionary.
    
    Args:
        config: Configuration dictionary
        section: Section name in config
        key: Key within section
        op_desc: Operation description for error messages
        
    Returns:
        The string value
        
    Raises:
        ValueError: If section/key missing or invalid
    """
    if section not in config or not isinstance(config[section], dict):
        raise ValueError(f"Config section '{section}' missing or invalid for {op_desc}.")
    
    path_val = config[section].get(key)
    if not isinstance(path_val, str) or not path_val.strip():
        raise ValueError(f"Config key '{key}' in '{section}' missing or empty for {op_desc}.")
    
    return path_val.strip()


def load_or_generate_splits(
        config: Dict[str, Any],
        data_root_dir: Path,
        raw_hdf5_paths: List[Path],
        model_save_dir: Path,
) -> Tuple[Dict[str, List[Tuple[str, int]]], Path]:
    """
    Load existing dataset splits or generate new ones.

    Args:
        config: Configuration dictionary
        data_root_dir: Root data directory
        raw_hdf5_paths: List of raw HDF5 files
        model_save_dir: Directory to save generated splits

    Returns:
        Tuple of (splits dictionary, splits file path)
    """
    splits_path = None

    try:
        # Try to load existing splits from config
        splits_filename = get_config_str(
            config, "data_paths_config", "dataset_splits_filename", "dataset splits"
        )
        splits_path = data_root_dir / splits_filename

        logger.info(f"Loading dataset splits from: {splits_path}")

        if not splits_path.exists():
            raise FileNotFoundError(f"Splits file not found: {splits_path}")

        with open(splits_path, "r", encoding=UTF8_ENCODING) as f:
            loaded_data = json.load(f)

        # Handle both old and new formats
        if "file_stems" in loaded_data:
            # New compact format - decompress it
            splits = decompress_splits(loaded_data)
        else:
            # Old verbose format
            splits = loaded_data
            # Validate splits structure
            required_keys = {"train", "validation", "test"}
            if not required_keys.issubset(splits.keys()):
                raise ValueError(f"Splits file must contain keys: {required_keys}")

            for key in required_keys:
                if not isinstance(splits[key], list) or not splits[key]:
                    raise ValueError(f"Split '{key}' must be a non-empty list")

            # Convert lists to tuples for consistency
            for split in splits.values():
                for i, item in enumerate(split):
                    if isinstance(item, list) and len(item) == 2:
                        split[i] = tuple(item)

        logger.info(f"Loaded splits from {splits_path}")
        logger.info(
            f"Split sizes: {len(splits['train'])} train, "
            f"{len(splits['validation'])} val, {len(splits['test'])} test."
        )
        return splits, splits_path

    except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        logger.info(f"Could not load splits file. Reason: {e}. Generating new splits.")

    # Generate new splits
    logger.info("Generating new dataset splits...")

    train_params = config.get("training_hyperparameters", {})
    val_frac = train_params.get("val_frac", 0.15)
    test_frac = train_params.get("test_frac", 0.15)

    misc_settings = config.get("miscellaneous_settings", {})
    random_seed = misc_settings.get("random_seed", DEFAULT_SEED)

    splits = generate_dataset_splits(
        raw_hdf5_paths=raw_hdf5_paths,
        val_frac=val_frac,
        test_frac=test_frac,
        random_seed=random_seed,
    )

    # Save generated splits with CONSISTENT naming
    new_splits_path = data_root_dir / "dataset_splits.json"  # Changed from model_save_dir
    compressed = compress_splits(splits)
    if save_json(compressed, new_splits_path, compact=True):
        logger.info(f"Saved compressed splits to {new_splits_path}")
    else:
        logger.warning("Failed to save generated splits")

    return splits, new_splits_path


def compute_data_hash(config: Dict[str, Any], raw_hdf5_paths: List[Path]) -> str:
    """
    Compute hash of ONLY data-relevant configuration.
    
    Only includes:
    - shard_size
    - hdf5_dataset_filename
    - dataset_splits_filename  
    - data_specification (entire section)
    - normalization (entire section)
    """
    hasher = hashlib.new(HASH_ALGORITHM)
    
    # Create filtered config with only data-relevant parts
    data_relevant_config = {}
    
    # Get shard_size from miscellaneous_settings
    if "miscellaneous_settings" in config:
        data_relevant_config["shard_size"] = config["miscellaneous_settings"].get("shard_size")
    
    # Get data paths config
    if "data_paths_config" in config:
        data_relevant_config["hdf5_dataset_filename"] = config["data_paths_config"].get("hdf5_dataset_filename")
        data_relevant_config["dataset_splits_filename"] = config["data_paths_config"].get("dataset_splits_filename")
    
    # Get entire data_specification section
    if "data_specification" in config:
        data_relevant_config["data_specification"] = config["data_specification"]
    
    # Get entire normalization section
    if "normalization" in config:
        data_relevant_config["normalization"] = config["normalization"]
    
    # Hash only the filtered config
    hasher.update(json.dumps(data_relevant_config, sort_keys=True).encode(UTF8_ENCODING))
    
    # Hash file paths (but not timestamps)
    for path in sorted(raw_hdf5_paths):
        hasher.update(str(path.name).encode(UTF8_ENCODING))  # Just filename, not full path
    
    return hasher.hexdigest()


def compute_data_hash_with_stats(config: Dict[str, Any], raw_hdf5_paths: List[Path]) -> str:
    """
    Compute hash including file sizes but NOT modification times.
    Only includes data-relevant configuration sections.
    """
    hasher = hashlib.new(HASH_ALGORITHM)
    
    # Create filtered config with only data-relevant parts
    data_relevant_config = {}
    
    # Get shard_size from miscellaneous_settings
    if "miscellaneous_settings" in config:
        data_relevant_config["shard_size"] = config["miscellaneous_settings"].get("shard_size")
    
    # Get data paths config
    if "data_paths_config" in config:
        data_relevant_config["hdf5_dataset_filename"] = config["data_paths_config"].get("hdf5_dataset_filename")
        data_relevant_config["dataset_splits_filename"] = config["data_paths_config"].get("dataset_splits_filename")
    
    # Get entire data_specification section
    if "data_specification" in config:
        data_relevant_config["data_specification"] = config["data_specification"]
    
    # Get entire normalization section  
    if "normalization" in config:
        data_relevant_config["normalization"] = config["normalization"]
    
    # Hash only the filtered config
    hasher.update(json.dumps(data_relevant_config, sort_keys=True).encode(UTF8_ENCODING))
    
    # Hash file names and sizes (but NOT timestamps or full paths)
    for path in sorted(raw_hdf5_paths):
        hasher.update(str(path.name).encode(UTF8_ENCODING))  # Just filename
        
        if path.is_file():
            stat = path.stat()
            # Include file size only
            hasher.update(str(stat.st_size).encode(UTF8_ENCODING))
            # REMOVED: modification time
        else:
            hasher.update(b"missing")
    
    return hasher.hexdigest()


def compress_splits(splits: Dict[str, List[Tuple[str, int]]]) -> Dict:
    """Convert verbose format to compact format."""
    # Extract unique file stems
    file_stems = sorted(set(stem for split in splits.values()
                            for stem, _ in split))
    stem_to_idx = {stem: i for i, stem in enumerate(file_stems)}

    compressed = {
        "file_stems": file_stems,
        "train": [[stem_to_idx[s], i] for s, i in splits["train"]],
        "validation": [[stem_to_idx[s], i] for s, i in splits["validation"]],
        "test": [[stem_to_idx[s], i] for s, i in splits["test"]]
    }
    return compressed

def decompress_splits(compressed: Dict) -> Dict[str, List[Tuple[str, int]]]:
    """Convert compact format back to verbose format."""
    file_stems = compressed["file_stems"]
    return {
        "train": [(file_stems[s], i) for s, i in compressed["train"]],
        "validation": [(file_stems[s], i) for s, i in compressed["validation"]],
        "test": [(file_stems[s], i) for s, i in compressed["test"]]
    }

__all__ = [
    "DTYPE",
    "PADDING_VALUE",
    "LOG_FORMAT",
    "DEFAULT_SEED",
    "METADATA_FILENAME",
    "setup_logging",
    "load_config",
    "validate_config",
    "ensure_dirs",
    "save_json",
    "seed_everything",
    "generate_dataset_splits",
    "get_config_str",
    "load_or_generate_splits",
    "compute_data_hash",
    "compute_data_hash_with_stats",
    "compress_splits",
    "decompress_splits"

]