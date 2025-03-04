#!/usr/bin/env python3
"""
spectral_utils.py

Utility functions for the spectral prediction pipeline, providing:
- Configuration loading and validation
- Device setup for computation
- Logging configuration
- Directory management
- Spectral data handling utilities
- Evaluation metrics for spectral prediction
- Visualization helpers
"""

import logging
import os
import json
from pathlib import Path
import math
from typing import Dict, Any, List, Union, Optional, Callable, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt


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
    module_logger = logging.getLogger("spectral_prediction")
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
    logger = logging.getLogger("spectral_prediction")
    
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
            json_loader = json.load
            
        with open(config_path, 'r') as f:
            config = json_loader(f)
            
        # Add derived parameters for spectral model
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
    logger = logging.getLogger("spectral_prediction")
    
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


def validate_spectral_config(config: Dict[str, Any]) -> bool:
    """
    Validate spectral model configuration for consistency.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Model configuration dictionary
    
    Returns
    -------
    bool
        True if configuration is valid, False otherwise
    """
    logger = logging.getLogger("spectral_prediction")
    
    try:
        # Common validation
        if "input_variables" not in config or not isinstance(config["input_variables"], list) or len(config["input_variables"]) == 0:
            logger.error("Configuration must include non-empty 'input_variables' list")
            return False
            
        if "target_variables" not in config or not isinstance(config["target_variables"], list) or len(config["target_variables"]) == 0:
            logger.error("Configuration must include non-empty 'target_variables' list")
            return False
            
        if "output_seq_length" not in config or not isinstance(config["output_seq_length"], int) or config["output_seq_length"] <= 0:
            logger.error("Configuration must include positive 'output_seq_length'")
            return False
            
        if "learning_rate" not in config or not isinstance(config["learning_rate"], (int, float)) or config["learning_rate"] <= 0:
            logger.error("Configuration must include positive 'learning_rate'")
            return False
            
        # Spectral model validation
        d_model = config.get("d_model", 256)
        nhead = config.get("nhead", 8)
        
        if d_model % nhead != 0:
            logger.error(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
            return False
                
        # Set default mlp_hidden_dim if missing or invalid
        if "mlp_hidden_dim" not in config or config.get("mlp_hidden_dim", 0) <= 0:
            logger.warning("mlp_hidden_dim not specified or invalid, using d_model instead")
            config["mlp_hidden_dim"] = d_model
            
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to {seed} for reproducibility")


# Spectral-specific utility functions

def calculate_spectral_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate multiple metrics for spectral prediction quality.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Predicted spectra of shape [batch_size, seq_length, n_features]
    targets : torch.Tensor
        Target spectra of shape [batch_size, seq_length, n_features]
    
    Returns
    -------
    Dict[str, float]
        Dictionary of metric names and values
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    # Overall metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # Calculate peak error (error at absorption/emission peaks)
    # Find peaks in target spectra (using a simple approach - find local maxima)
    batch_size, seq_length, n_features = targets.shape
    
    peak_errors = []
    for b in range(batch_size):
        for f in range(n_features):
            target_seq = targets[b, :, f]
            # Simple peak detection (local maxima)
            peaks = np.where((target_seq[1:-1] > target_seq[:-2]) & 
                             (target_seq[1:-1] > target_seq[2:]))[0] + 1
            
            if len(peaks) > 0:
                # Calculate error at peaks
                pred_seq = predictions[b, :, f]
                peak_error = np.mean(np.abs(pred_seq[peaks] - target_seq[peaks]))
                peak_errors.append(peak_error)
    
    # Calculate average peak error if any peaks found
    if peak_errors:
        avg_peak_error = np.mean(peak_errors)
    else:
        avg_peak_error = 0.0
    
    # Pearson correlation coefficient (averaged over batches)
    corr_sum = 0.0
    count = 0
    for b in range(batch_size):
        for f in range(n_features):
            pred = predictions[b, :, f]
            targ = targets[b, :, f]
            
            # Skip if no variation in target (would give NaN correlation)
            if np.std(targ) > 1e-10:
                corr = np.corrcoef(pred, targ)[0, 1]
                if not np.isnan(corr):
                    corr_sum += corr
                    count += 1
    
    avg_correlation = corr_sum / max(1, count)
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "peak_error": float(avg_peak_error),
        "correlation": float(avg_correlation)
    }


def plot_spectral_comparison(prediction: np.ndarray, target: np.ndarray, 
                            wavelengths: Optional[np.ndarray] = None,
                            title: str = "Spectral Comparison",
                            save_path: Optional[str] = None,
                            show_metrics: bool = True,
                            feature_idx: int = 0):
    """
    Plot predicted spectrum against target spectrum.
    
    Parameters
    ----------
    prediction : np.ndarray
        Predicted spectrum of shape [seq_length, n_features] or [seq_length]
    target : np.ndarray
        Target spectrum of shape [seq_length, n_features] or [seq_length]
    wavelengths : np.ndarray, optional
        Wavelength values for x-axis
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
    show_metrics : bool, optional
        Whether to show metrics on the plot
    feature_idx : int, optional
        Index of feature to plot if multi-feature data
    
    Returns
    -------
    Figure
        The matplotlib figure object
    """
    plt.figure(figsize=(12, 6))
    
    # Extract the feature to plot
    if prediction.ndim > 1 and prediction.shape[1] > 1:
        pred = prediction[:, feature_idx]
        targ = target[:, feature_idx]
    else:
        pred = prediction.squeeze()
        targ = target.squeeze()
    
    # X-axis values
    if wavelengths is not None:
        x = wavelengths
        xlabel = "Wavelength"
    else:
        x = np.arange(len(pred))
        xlabel = "Wavelength Index"
    
    # Plot data
    plt.plot(x, targ, 'b-', label='Ground Truth', alpha=0.7)
    plt.plot(x, pred, 'r-', label='Prediction', alpha=0.7)
    
    # Calculate metrics if requested
    if show_metrics:
        mse = np.mean((pred - targ) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred - targ))
        
        try:
            corr = np.corrcoef(pred, targ)[0, 1]
        except:
            corr = 0.0
            
        metrics_text = f"MSE: {mse:.4e}, RMSE: {rmse:.4e}\nMAE: {mae:.4e}, Corr: {corr:.4f}"
        plt.figtext(0.15, 0.15, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
    
    # Formatting
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return plt.gcf()


def calculate_spectral_uncertainty(predictions: torch.Tensor, targets: torch.Tensor, 
                                  n_bootstrap: int = 100, confidence: float = 0.95) -> Dict[str, Any]:
    """
    Calculate uncertainty in spectral metrics using bootstrapping.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Predicted spectra of shape [batch_size, seq_length, n_features]
    targets : torch.Tensor
        Target spectra of shape [batch_size, seq_length, n_features]
    n_bootstrap : int, optional
        Number of bootstrap samples
    confidence : float, optional
        Confidence level for intervals
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of metrics with mean and confidence intervals
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    batch_size = predictions.shape[0]
    
    # Function to compute metrics for a sample
    def compute_sample_metrics(sample_indices):
        preds_sample = predictions[sample_indices]
        targets_sample = targets[sample_indices]
        return calculate_spectral_metrics(preds_sample, targets_sample)
    
    # Bootstrap
    np.random.seed(42)  # For reproducibility
    bootstrap_metrics = {
        "mse": [],
        "rmse": [],
        "mae": [],
        "peak_error": [],
        "correlation": []
    }
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(batch_size, batch_size, replace=True)
        sample_metrics = compute_sample_metrics(indices)
        
        for metric_name, value in sample_metrics.items():
            bootstrap_metrics[metric_name].append(value)
    
    # Calculate confidence intervals
    alpha = 1.0 - confidence
    result = {}
    
    for metric_name, values in bootstrap_metrics.items():
        values = np.array(values)
        mean = np.mean(values)
        lower = np.percentile(values, alpha/2 * 100)
        upper = np.percentile(values, (1 - alpha/2) * 100)
        
        result[metric_name] = {
            "mean": float(mean),
            "lower": float(lower),
            "upper": float(upper)
        }
    
    return result


def analyze_prediction_errors(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    """
    Analyze the distribution and patterns of prediction errors.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted spectra
    targets : np.ndarray
        Target spectra
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing error analysis results
    """
    # Calculate errors
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    # Basic statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    max_abs_error = np.max(abs_errors)
    
    # Error distribution characteristics
    skewness = np.mean(((errors - mean_error) / std_error) ** 3)
    kurtosis = np.mean(((errors - mean_error) / std_error) ** 4) - 3
    
    # Percentiles
    percentiles = {
        "5th": float(np.percentile(errors, 5)),
        "25th": float(np.percentile(errors, 25)),
        "50th": float(np.percentile(errors, 50)),
        "75th": float(np.percentile(errors, 75)),
        "95th": float(np.percentile(errors, 95))
    }
    
    # Check for outliers (using 3 sigma rule)
    outlier_mask = abs_errors > (mean_error + 3 * std_error)
    outlier_count = np.sum(outlier_mask)
    outlier_percentage = 100 * outlier_count / errors.size
    
    # Return analysis results
    return {
        "mean_error": float(mean_error),
        "median_error": float(median_error), 
        "std_error": float(std_error),
        "max_abs_error": float(max_abs_error),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "percentiles": percentiles,
        "outlier_count": int(outlier_count),
        "outlier_percentage": float(outlier_percentage)
    }