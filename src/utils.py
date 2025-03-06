#!/usr/bin/env python3
"""
utils.py

Utility functions for the prediction pipeline, providing:
- Configuration loading and validation
- Device setup for computation
- Logging configuration
- Directory management
- Data handling utilities
- Evaluation metrics for prediction
- Visualization helpers
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any,  Union, Optional

import torch
import numpy as np


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
    
    # Configure module logger too
    module_logger = logging.getLogger("Prediction")
    module_logger.setLevel(level)


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration from JSON/JSONC file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    
    Returns
    -------
    Optional[Dict[str, Any]]
        Configuration dictionary if successful, None otherwise
    """
    logger = logging.getLogger("Prediction")
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        # Try to import json5 for JSONC support
        try:
            import json5
            json_loader = json5.load
        except ImportError:
            logger.warning("json5 not available, using standard json (comments not supported)")
            
            # Fallback to use custom JSON loader that strips JavaScript style comments
            def json_loader(f):
                content = f.read()
                # Remove both single-line and multi-line comments (basic approach)
                import re
                content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
                content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                return json.loads(content)
            
        with open(config_path, 'r') as f:
            config = json_loader(f)
            
        # Add derived parameters for model
        if "pressure_range" in config and "points" in config["pressure_range"]:
            config["input_seq_length"] = config["pressure_range"]["points"]
            
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return None


def setup_device() -> torch.device:
    """
    Set up the compute device (GPU if available, otherwise CPU).
    
    Returns
    -------
    torch.device
        Selected compute device
    """
    logger = logging.getLogger("Prediction")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Check if MPS is available (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Metal Performance Shaders) is available")
        return device
    else:
        # Check if MPS is available (Apple Silicon)
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon GPU)")
            return device
        logger.info("Using CPU")
        return torch.device("cpu")


def ensure_dirs(*dirs: str) -> None:
    """
    Create directories if they don't exist.
    
    Parameters
    ----------
    *dirs : str
        One or more directory paths to create
    """
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


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
    logger = logging.getLogger("Prediction")
    
    try:
        # Common validation
        if "input_variables" not in config or not isinstance(config["input_variables"], list) or len(config["input_variables"]) == 0:
            logger.error("Configuration must include non-empty 'input_variables' list")
            return False
            
        if "target_variables" not in config or not isinstance(config["target_variables"], list) or len(config["target_variables"]) == 0:
            logger.error("Configuration must include non-empty 'target_variables' list")
            return False
            
        if "output_seq_length" in config and (not isinstance(config["output_seq_length"], int) or config["output_seq_length"] <= 0):
            logger.error("Configuration has invalid 'output_seq_length', must be positive integer")
            return False
            
        if "learning_rate" not in config or not isinstance(config["learning_rate"], (int, float)) or config["learning_rate"] <= 0:
            logger.error("Configuration must include positive 'learning_rate'")
            return False
            
        # Model validation - support both hidden_dim and d_model
        hidden_dim = config.get("hidden_dim", config.get("d_model", 256))
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
                
        # Set default mlp_hidden_dim if missing or invalid
        if "mlp_hidden_dim" not in config or config.get("mlp_hidden_dim", 0) <= 0:
            logger.warning("mlp_hidden_dim not specified or invalid, using hidden_dim instead")
            config["mlp_hidden_dim"] = hidden_dim
            
        return True
    except Exception as e:
        logger.error(f"Error validating configuration: {str(e)}")
        return False


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if torch.is_tensor(obj):
            return obj.cpu().detach().numpy().tolist()
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


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
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyJSONEncoder)
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")
        return False


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
    try:
        metadata_file = Path(metadata_path)
        if not metadata_file.exists():
            raise FileNotFoundError(f"Normalization metadata not found at {metadata_path}")
        
        return load_config(metadata_path)
    except Exception as e:
        logging.error(f"Error loading normalization metadata: {e}")
        return None


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
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic algorithms where available
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # Some operations don't have deterministic implementations
    
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to {seed} for reproducibility")