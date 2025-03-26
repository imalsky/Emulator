#!/usr/bin/env python3
"""
utils.py

Core utility functions for the atmospheric prediction pipeline:
- Logging configuration
- Configuration loading and validation
- Directory management
- Data handling utilities
- Model loading and initialization
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
import re
import torch
import numpy as np
import json5


# Import hardware module for device handling
from hardware import setup_device

def setup_logging(level=logging.INFO, log_file: Optional[str] = None):
    """
    Configure logging with a clean, minimal format.
    
    Parameters
    ----------
    level : int
        Logging level (default: logging.INFO)
    log_file : str, optional
        If provided, also log to this file
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with clean format
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration from JSON/JSONC file with comment support.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    
    Returns
    -------
    Optional[Dict[str, Any]]
        Configuration dictionary if successful, None otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Try to use json5 for JSONC support if available
        try:
            with open(config_path, 'r') as f:
                config = json5.load(f)
        except ImportError:
            logger.debug("json5 not available, using custom comment stripping")
            
            # Strip comments using regex
            with open(config_path, 'r') as f:
                content = f.read()
                # Remove single-line comments
                content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
                # Remove multi-line comments
                content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                config = json.loads(content)
            
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load configuration: {e}")
        return None


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate model configuration for consistency.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Model configuration dictionary
    
    Returns
    -------
    bool
        True if configuration is valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check essential parameters
        if "input_variables" not in config or not isinstance(config["input_variables"], list) or len(config["input_variables"]) == 0:
            logger.error("Configuration must include non-empty 'input_variables' list")
            return False
            
        if "target_variables" not in config or not isinstance(config["target_variables"], list) or len(config["target_variables"]) == 0:
            logger.error("Configuration must include non-empty 'target_variables' list")
            return False
            
        if "learning_rate" not in config or not isinstance(config["learning_rate"], (int, float)) or config["learning_rate"] <= 0:
            logger.error("Configuration must include positive 'learning_rate'")
            return False
            
        # Check if hidden dimension is divisible by number of attention heads
        hidden_dim = config.get("d_model", 256)
        nhead = config.get("nhead", 8)
        
        if hidden_dim % nhead != 0:
            # Try to adjust nhead to make it work
            old_nhead = nhead
            for divisor in range(nhead, 0, -1):
                if hidden_dim % divisor == 0:
                    nhead = divisor
                    logger.warning(f"Adjusting nhead from {old_nhead} to {nhead} to ensure it divides hidden_dim ({hidden_dim})")
                    config["nhead"] = nhead
                    break
            else:
                logger.error(f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error validating configuration: {str(e)}")
        return False


def ensure_dirs(*dirs: Union[str, Path]) -> None:
    """
    Create directories if they don't exist.
    
    Parameters
    ----------
    *dirs : str or Path
        One or more directory paths to create
    """
    for directory in dirs:
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)


def load_normalization_metadata(metadata_path: str) -> Optional[Dict[str, Any]]:
    """
    Load normalization metadata from a JSON file.
    
    Parameters
    ----------
    metadata_path : str
        Path to the normalization metadata file
    
    Returns
    -------
    Dict[str, Any] or None
        Normalization metadata if successful, None otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        metadata_file = Path(metadata_path)
        if not metadata_file.exists():
            logger.error(f"Normalization metadata not found at {metadata_path}")
            return None
        
        # Use the same loading function for consistency
        return load_config(metadata_path)
    except Exception as e:
        logger.error(f"Error loading normalization metadata: {e}")
        return None


def save_json(data: Dict[str, Any], filepath: str) -> bool:
    """
    Save dictionary to a JSON file with proper formatting.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary to save
    filepath : str
        Path where to save the file
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Handle non-serializable types
        def json_encoder(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, (np.bool_)):
                return bool(obj)
            if torch.is_tensor(obj):
                return obj.cpu().detach().numpy().tolist()
            if isinstance(obj, set):
                return list(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=json_encoder)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        return False


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across libraries.
    
    Parameters
    ----------
    seed : int
        Seed value to use
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior where possible
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger = logging.getLogger(__name__)
    logger.info(f"Random seed set to {seed} for reproducibility")


def save_config(config: Dict[str, Any], filepath: str) -> bool:
    """
    Save configuration to a JSON file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to save
    filepath : str
        Path where to save the configuration file
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Use the existing save_json function for consistency
        success = save_json(config, filepath)
        if success:
            logger.info(f"Configuration saved to {filepath}")
        return success
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        return False