#!/usr/bin/env python3
"""
spectral_normalizer.py

Data normalization utilities optimized for spectral data with variable sequence lengths.

Supported normalization methods:
- "standard": Z-score normalization using mean and standard deviation
- "minmax": Min-max normalization to [0,1] range 
- "iqr": Normalizes based on median and interquartile range
- "log": Log-scaling followed by normalization to [0,1]
- "signed_log": Preserves sign and applies log scaling to absolute values
"""

import json
import logging
import torch
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class SpectralDataNormalizer:
    """
    A class for normalizing atmospheric profile data and spectral outputs.
    
    Optimized for handling sequences of variable lengths and different normalization methods.
    """
    
    def __init__(self, input_folder, output_folder, batch_size=100):
        """Initialize the normalizer with input and output folders."""
        self.input_folder = Path(input_folder)
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
            
        self.output_folder = Path(output_folder)
        self.batch_size = max(1, batch_size)
        
        logger.info(f"SpectralDataNormalizer initialized with input: {input_folder}, output: {output_folder}")

    def calculate_global_stats(self, variable_methods=None, default_method="standard"):
        """Compute global normalization statistics for all variables."""
        logger.info("Calculating global normalization statistics...")
        variable_methods = variable_methods or {}
        
        # Get all JSON files (except metadata)
        data_files = list(self.input_folder.glob("*.json"))
        if not data_files:
            raise FileNotFoundError(f"No JSON files found in '{self.input_folder}'")
        
        # Initialize storage for values by variable name
        var_values = {}
        sequence_lengths = {}
        
        # Process each file
        for file_path in data_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                for var_name, value in data.items():
                    if var_name not in var_values:
                        var_values[var_name] = []
                        sequence_lengths[var_name] = []
                        
                    if isinstance(value, (int, float)):
                        var_values[var_name].append(torch.tensor([float(value)], dtype=torch.float32))
                    elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                        tensor = torch.tensor(value, dtype=torch.float32)
                        var_values[var_name].append(tensor)
                        sequence_lengths[var_name].append(len(value))
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
        
        # Concatenate tensors and determine normalization methods
        stats = {}
        normalization_methods = {}
        
        for var_name in list(var_values.keys()):
            if not var_values[var_name]:
                continue
                
            # Concatenate values (works for both scalar and sequence variables)
            values = torch.cat(var_values[var_name], dim=0)
            
            # Determine method - use specified method or default
            method = variable_methods.get(var_name, default_method)
            normalization_methods[var_name] = method
            
            # Store sequence length statistics if applicable
            if sequence_lengths[var_name]:
                min_len = min(sequence_lengths[var_name])
                max_len = max(sequence_lengths[var_name])
                mean_len = sum(sequence_lengths[var_name]) / len(sequence_lengths[var_name])
                stats[var_name] = {
                    "is_sequence": True,
                    "min_length": min_len,
                    "max_length": max_len,
                    "mean_length": mean_len
                }
                if min_len != max_len:
                    logger.info(f"Variable '{var_name}' has variable sequence length: min={min_len}, max={max_len}")
            else:
                stats[var_name] = {"is_sequence": False}
            
            # Compute normalization statistics
            try:
                stats[var_name].update(self._compute_stats_for_variable(values, method))
            except ValueError as e:
                logger.warning(f"Error computing {method} stats for '{var_name}': {e}. Using 'standard' instead.")
                method = "standard"
                normalization_methods[var_name] = method
                stats[var_name].update(self._compute_stats_for_variable(values, method))
        
        # Add metadata
        stats["normalization_methods"] = normalization_methods
        logger.info(f"Statistics computed for {len(stats) - 1} variables")
        return stats

    def _compute_stats_for_variable(self, values, method):
        """Compute statistics for a specific variable based on the chosen method."""
        # Always record global min and max
        global_min = values.min().item()
        global_max = values.max().item()
        stats = {"global_min": global_min, "global_max": global_max}
        
        if method == "iqr":
            median = torch.median(values).item()
            q1 = torch.quantile(values, 0.25).item()
            q3 = torch.quantile(values, 0.75).item()
            iqr = max(q3 - q1, 1e-6)  # Prevent division by zero
            stats.update({"median": median, "iqr": iqr})
            
        elif method == "minmax":
            range_val = max(global_max - global_min, 1e-6)  # Prevent division by zero
            stats.update({"range": range_val})
            
        elif method == "standard":
            mean = values.mean().item()
            std = max(values.std().item(), 1e-6)  # Prevent division by zero
            stats.update({"mean": mean, "std": std})
            
        elif method == "log":
            # Ensure all values are positive with offset
            offset = max(0, -global_min + 1e-6)
            log_min = np.log(global_min + offset) if global_min + offset > 0 else 0
            log_max = np.log(global_max + offset) if global_max + offset > 0 else 0
            log_range = max(log_max - log_min, 1e-6)  # Prevent division by zero
            stats.update({"offset": offset, "log_min": log_min, "log_range": log_range})
            
        elif method == "signed_log":
            # Track sign and apply log to absolute values
            # Find minimum absolute non-zero value for offset
            abs_values = torch.abs(values)
            non_zero_min = abs_values[abs_values > 0].min().item() if torch.any(abs_values > 0) else 1e-6
            
            # Offset to ensure log works (avoid log of values near zero)
            offset = max(1e-6, non_zero_min / 10)
            log_abs_max = np.log(global_max + offset) if global_max > 0 else 0
            log_abs_min = np.log(abs(global_min) + offset) if global_min < 0 else 0
            max_log = max(log_abs_max, log_abs_min)
            stats.update({"offset": offset, "max_log": max_log})
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        return stats

    @staticmethod
    def normalize_tensor(data, method, stats):
        """Normalize a tensor using the specified method and statistics."""
        if method == "iqr":
            median = stats["median"]
            iqr = stats["iqr"]
            normalized = (data - median) / iqr
            
        elif method == "minmax":
            min_val = stats["global_min"]
            range_val = stats["range"]
            normalized = (data - min_val) / range_val
            
        elif method == "standard":
            mean = stats["mean"]
            std = stats["std"]
            normalized = (data - mean) / std
            
        elif method == "log":
            # Apply log scaling followed by 0-1 normalization
            offset = stats["offset"]
            log_min = stats["log_min"]
            log_range = stats["log_range"]
            # Add offset, take log, then normalize to 0-1
            normalized = (torch.log(data + offset) - log_min) / log_range
            
        elif method == "signed_log":
            # Preserve sign while applying log to absolute values
            offset = stats["offset"]
            max_log = stats["max_log"]
            
            # Get sign (-1 for negative, 1 for positive)
            sign = torch.sign(data)
            
            # Apply log to absolute values (with offset), then normalize by max_log
            logged = torch.log(torch.abs(data) + offset) / max_log
            
            # Reapply sign
            normalized = sign * logged
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
            
        return normalized

    @staticmethod
    def denormalize(norm_values, metadata, variable_name):
        """Denormalize values using stored normalization metadata."""
        if variable_name not in metadata["normalization_methods"]:
            raise KeyError(f"No normalization method found for '{variable_name}'")
            
        method = metadata["normalization_methods"][variable_name]
        
        if variable_name not in metadata:
            raise KeyError(f"No statistics found for '{variable_name}'")
            
        stats = metadata[variable_name]
        
        # Convert input to tensor if needed
        is_scalar = isinstance(norm_values, (int, float))
        is_list = isinstance(norm_values, list)
        norm_tensor = torch.tensor(norm_values, dtype=torch.float32)
        
        # Apply inverse transformation
        if method == "iqr":
            median = stats["median"]
            iqr = stats["iqr"]
            denorm = norm_tensor * iqr + median
            
        elif method == "minmax":
            min_val = stats["global_min"]
            range_val = stats["range"]
            denorm = norm_tensor * range_val + min_val
            
        elif method == "standard":
            mean = stats["mean"]
            std = stats["std"]
            denorm = norm_tensor * std + mean
            
        elif method == "log":
            offset = stats["offset"]
            log_min = stats["log_min"]
            log_range = stats["log_range"]
            # Inverse of log normalization: unnormalize, exponentiate, subtract offset
            denorm = torch.exp(norm_tensor * log_range + log_min) - offset
            
        elif method == "signed_log":
            offset = stats["offset"]
            max_log = stats["max_log"]
            
            # Get sign from normalized values
            sign = torch.sign(norm_tensor)
            
            # Inverse log operation on absolute values
            denorm = sign * torch.exp(torch.abs(norm_tensor) * max_log) - offset
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # Return in the same format as input
        if is_scalar:
            return denorm.item()
        elif is_list:
            return denorm.tolist()
        else:
            return denorm

    def process_data(self, stats):
        """Normalize all data files using the computed statistics."""
        logger.info(f"Processing data using computed statistics...")
        
        # Create output directory and save metadata
        self.output_folder.mkdir(parents=True, exist_ok=True)
        with open(self.output_folder / "normalization_metadata.json", "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)

        
        # Get normalization methods
        methods = stats["normalization_methods"]
        
        # Process each data file
        data_files = list(self.input_folder.glob("*.json"))
        processed_count = 0
        error_count = 0
        
        for file_path in data_files:
            try:
                if file_path.name == "normalization_metadata.json":
                    continue
                    
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                # Create normalized data
                normalized_data = {}
                
                for var_name, value in data.items():
                    if var_name in stats and var_name != "normalization_methods":
                        method = methods.get(var_name)
                        var_stats = stats[var_name]
                        
                        try:
                            if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                                data_tensor = torch.tensor(value, dtype=torch.float32)
                                norm_data = self.normalize_tensor(data_tensor, method, var_stats)
                                normalized_data[var_name] = norm_data.tolist()
                            elif isinstance(value, (int, float)):
                                data_tensor = torch.tensor([value], dtype=torch.float32)
                                norm_data = self.normalize_tensor(data_tensor, method, var_stats)
                                normalized_data[var_name] = norm_data.item()
                            else:
                                normalized_data[var_name] = value
                        except Exception as e:
                            logger.warning(f"Error normalizing '{var_name}': {e}, keeping original value")
                            normalized_data[var_name] = value
                    else:
                        normalized_data[var_name] = value
                
                # Save normalized data
                with open(self.output_folder / file_path.name, "w") as f:
                    json.dump(normalized_data, f, indent=2, sort_keys=True)
                
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                error_count += 1
        
        logger.info(f"Processed {processed_count} files with {error_count} errors")