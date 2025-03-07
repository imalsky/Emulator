#!/usr/bin/env python3
"""
normalizer.py

Data normalization utilities for atmospheric profiles.

This module provides a DataNormalizer class that:
1. Computes global normalization statistics from raw JSON profile files
2. Applies various normalization methods to the data
3. Saves the normalized profiles with metadata

Supported normalization methods:
- "iqr": Normalizes based on median and interquartile range
- "log-min-max": Applies log transform then min-max scaling
- "arctan-compression": Compresses data using arctan transform
- "max-out": Divides data by absolute global maximum
- "invlogit-compression": Maps data using sigmoid transformation
- "custom": Sign-preserving logarithmic transformation
- "standard": Standard z-score normalization
"""

import os
import json
import math
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    A class for normalizing atmospheric profile data.
    
    Computes global statistics from raw profiles and applies various
    normalization methods to produce normalized profiles for ML training.
    """
    
    def __init__(self, input_folder: str, output_folder: str, batch_size: int = 100):
        """
        Initialize the DataNormalizer.

        Parameters
        ----------
        input_folder : str
            Directory containing raw JSON profile files
        output_folder : str
            Directory where normalized profiles will be saved
        batch_size : int, optional
            Number of profiles to process in one batch (for memory efficiency)
        """
        self.input_folder = Path(input_folder)
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
            
        self.output_folder = Path(output_folder)
        self.batch_size = max(1, batch_size)
        
        logger.info(f"DataNormalizer initialized with input: {input_folder}, output: {output_folder}")

    @staticmethod
    def clip_outliers(values: torch.Tensor, 
                       lower_quantile: float = 0.01, 
                       upper_quantile: float = 0.99) -> torch.Tensor:
        """
        Clip values outside the specified quantile range.
        
        Parameters
        ----------
        values : torch.Tensor
            Tensor containing data to be clipped
        lower_quantile : float, optional
            Lower quantile threshold (default: 0.01)
        upper_quantile : float, optional
            Upper quantile threshold (default: 0.99)
        
        Returns
        -------
        torch.Tensor
            Tensor with values clamped between the quantile bounds
        """
        if not (0 <= lower_quantile < upper_quantile <= 1):
            raise ValueError("Quantiles must satisfy: 0 ≤ lower < upper ≤ 1")
        
        # Handle empty tensor case
        if values.numel() == 0:
            return values
            
        lower_bound = torch.quantile(values, lower_quantile)
        upper_bound = torch.quantile(values, upper_quantile)
        
        return torch.clamp(values, min=lower_bound, max=upper_bound)

    def calculate_global_stats(
        self,
        key_methods: Optional[Dict[str, str]] = None,
        default_method: str = "iqr",
        clip_outliers_before_scaling: bool = False
    ) -> Dict[str, Any]:
        """
        Compute global normalization statistics for all variables.
        
        Parameters
        ----------
        key_methods : Dict[str, str], optional
            Mapping of variable names to normalization methods
        default_method : str, optional
            Default normalization method for variables not in key_methods
        clip_outliers_before_scaling : bool, optional
            Whether to clip outliers before computing statistics
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing computed statistics and configuration
        
        Raises
        ------
        FileNotFoundError
            If no JSON files are found in the input folder
        RuntimeError
            If there's an error reading a profile file
        """
        logger.info("Calculating global normalization statistics...")
        
        # Get all JSON files (except metadata)
        profile_files = list(self.input_folder.glob("*.json"))
        if not profile_files:
            raise FileNotFoundError(f"No JSON files found in '{self.input_folder}'")
        
        # Initialize storage for values by key
        key_values: Dict[str, List[torch.Tensor]] = {}
        
        # Process files in batches for memory efficiency
        total_files = len(profile_files)
        batch_size = min(self.batch_size, total_files)
        num_batches = (total_files + batch_size - 1) // batch_size
        
        logger.info(f"Processing {total_files} files in {num_batches} batches")
        
        # Process each batch
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_files)
            batch_files = profile_files[start_idx:end_idx]
            
            logger.debug(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_files)} files)")
            
            # Process each profile in the batch
            for profile_path in batch_files:
                try:
                    with open(profile_path, "r") as f:
                        profile = json.load(f)
                    
                    for key, value in profile.items():
                        if key not in key_values:
                            key_values[key] = []
                            
                        if isinstance(value, (int, float)):
                            key_values[key].append(torch.tensor([float(value)], dtype=torch.float32))
                        elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                            key_values[key].append(torch.tensor(value, dtype=torch.float32))
                except Exception as e:
                    raise RuntimeError(f"Error reading {profile_path}: {e}")
        
        # Concatenate tensors for each key
        for key in list(key_values.keys()):
            if key_values[key]:  # Check if we have values
                key_values[key] = torch.cat(key_values[key], dim=0)
            else:
                del key_values[key]  # Remove keys with no valid values
        
        # Apply outlier clipping if requested
        if clip_outliers_before_scaling:
            logger.info("Clipping outliers before computing statistics")
            for key, vals in key_values.items():
                if vals.numel() > 0:  # Only clip if tensor is not empty
                    key_values[key] = self.clip_outliers(vals)
        
        # Determine normalization method for each key
        normalization_methods = {}
        for key in key_values.keys():
            if key_methods and key in key_methods:
                normalization_methods[key] = key_methods[key]
            else:
                normalization_methods[key] = default_method
        
        # Compute statistics for each key
        stats = {}
        for key, values in key_values.items():
            if values.numel() == 0:
                logger.warning(f"Skipping statistics computation for '{key}' - no values available")
                continue
                
            method = normalization_methods[key]
            try:
                stats[key] = self._compute_stats_for_key(values, method)
                logger.debug(f"Computed {method} statistics for '{key}'")
            except ValueError as e:
                # Handle case where method is incompatible with data
                logger.warning(f"Error computing {method} statistics for '{key}': {e}. Falling back to 'iqr'.")
                normalization_methods[key] = "iqr"
                try:
                    stats[key] = self._compute_stats_for_key(values, "iqr")
                except ValueError as e2:
                    logger.error(f"Failed to compute fallback statistics for '{key}': {e2}")
                    # Skip this key instead of failing the entire process
                    continue
        
        # Add metadata
        stats["normalization_methods"] = normalization_methods
        stats["config"] = {
            "clip_outliers_before_scaling": clip_outliers_before_scaling,
            "default_method": default_method,
            "key_methods": key_methods if key_methods else {},
        }
        
        logger.info(f"Statistics computed for {len(key_values)} variables")
        return stats

    def _compute_stats_for_key(self, values: torch.Tensor, method: str) -> Dict[str, float]:
        """
        Compute statistics for a specific key based on the chosen normalization method.
        
        Parameters
        ----------
        values : torch.Tensor
            All collected values for a given key
        method : str
            Normalization method to use
        
        Returns
        -------
        Dict[str, float]
            Dictionary of computed statistics
        
        Raises
        ------
        ValueError
            If the normalization method is not supported or incompatible with data
        """
        # Check for empty tensor
        if values.numel() == 0:
            raise ValueError("Cannot compute statistics on empty tensor")
            
        # Always record global min and max
        global_min = values.min().item()
        global_max = values.max().item()
        stats = {"global_min": global_min, "global_max": global_max}
        
        if method == "iqr":
            median = torch.median(values).item()
            q1 = torch.quantile(values, 0.25).item()
            q3 = torch.quantile(values, 0.75).item()
            iqr = q3 - q1
            if iqr < 1e-6:  # Prevent division by zero
                iqr = 1.0
            stats.update({"median": median, "iqr": iqr})
            
        elif method == "log-min-max":
            # Handle negative or zero values
            if global_min <= 0:
                # Add offset to make all values positive
                offset = abs(global_min) + 1e-6
                logger.warning(f"log-min-max requires positive values, found min={global_min}. Adding offset: {offset}")
                values = values + offset
                stats["offset"] = offset
                global_min = values.min().item()  # Update min after offset
                
            log_vals = torch.log(values)
            vmin = log_vals.min().item()
            vmax = log_vals.max().item()
            if abs(vmax - vmin) < 1e-6:
                vmin, vmax = 0.0, 1.0
            stats.update({"min": vmin, "max": vmax})
            
        elif method == "arctan-compression":
            # If data spans zero, force center to zero
            if global_min < 0 and global_max > 0:
                center = 0.0
            else:
                center = (global_min + global_max) / 2.0
                
            scale = (global_max - global_min) / 2.0
            if scale < 1e-6:
                scale = 1.0
                
            alpha = math.tan(0.99 * (math.pi / 2))  # ~18.43
            stats.update({"center": center, "scale": scale, "alpha": alpha})
            
        elif method == "max-out":
            max_val = max(abs(global_min), abs(global_max))
            if max_val < 1e-6:
                max_val = 1.0
            stats.update({"max_val": max_val})
            
        elif method == "invlogit-compression":
            mean = values.mean().item()
            std = values.std().item()
            
            # If data spans zero, force mean to zero for symmetry
            if global_min < 0 and global_max > 0:
                mean = 0.0
                
            if std < 1e-6:  # Prevent division by zero
                std = 1.0
                
            stats.update({"mean": mean, "std": std})
            
        elif method == "custom":
            # Add small epsilon for better numerical stability
            epsilon = 1e-10
            
            # Compute y = sign(x)*log1p(|x| + epsilon) for each element
            y_positive = torch.log1p(torch.clamp(values, min=0) + epsilon)
            y_negative = -torch.log1p(torch.clamp(-values, min=0) + epsilon)
            
            # Combine based on the sign of values
            y = torch.where(values >= 0, y_positive, y_negative)
            
            m_pos = torch.log1p(torch.tensor(global_max) + epsilon).item() if global_max > 0 else 0.0
            m_neg = torch.log1p(torch.tensor(-global_min) + epsilon).item() if global_min < 0 else 0.0
            m = max(m_pos, m_neg)
            if m < 1e-6:
                m = 1.0
            
            stats.update({"m": m, "epsilon": epsilon})
            
        elif method == "standard":
            mean = values.mean().item()
            std = values.std().item()
            
            if std < 1e-6:  # Prevent division by zero
                std = 1.0
                
            stats.update({"mean": mean, "std": std})
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
            
        return stats

    @staticmethod
    def normalize_tensor(data: torch.Tensor, method: str, stats: Dict[str, float]) -> torch.Tensor:
        """
        Normalize a tensor using the specified method and statistics.
        
        Parameters
        ----------
        data : torch.Tensor
            The tensor to normalize
        method : str
            Normalization method to use
        stats : Dict[str, float]
            Normalization statistics for the variable
        
        Returns
        -------
        torch.Tensor
            Normalized tensor
        
        Raises
        ------
        ValueError
            If the normalization method is not supported
        """
        eps = 1e-9  # Small epsilon to prevent division by zero
        
        # Handle empty tensor case
        if data.numel() == 0:
            return data
        
        if method == "iqr":
            median = stats["median"]
            iqr = max(stats["iqr"], eps)
            normalized = (data - median) / iqr
            
        elif method == "log-min-max":
            # Apply offset if present in stats
            if "offset" in stats:
                offset = stats["offset"]
                data = data + offset
                
            # Ensure all values are positive
            if torch.any(data <= 0):
                min_val = torch.min(data).item()
                logger.warning(f"log-min-max found non-positive values (min={min_val}), applying epsilon offset")
                data = torch.clamp(data, min=eps)
                
            log_data = torch.log(data)
            min_val = stats["min"]
            max_val = stats["max"]
            denom = max(max_val - min_val, eps)
            normalized = (log_data - min_val) / denom
            normalized = torch.clamp(normalized, 0.0, 1.0)
            
        elif method == "arctan-compression":
            center = stats["center"]
            scale = max(stats["scale"], eps)
            alpha = stats["alpha"]
            normalized = (2.0 / math.pi) * torch.atan(alpha * (data - center) / scale)
            
        elif method == "max-out":
            max_val = max(stats["max_val"], eps)
            normalized = data / max_val
            
        elif method == "invlogit-compression":
            mean = stats["mean"]
            std = max(stats["std"], eps)
            normalized = 2.0 * torch.sigmoid((data - mean) / std) - 1.0
            
        elif method == "custom":
            m = max(stats["m"], eps)
            # Use epsilon from stats or default to 1e-10
            epsilon = stats.get("epsilon", 1e-10)
            
            # Compute y = sign(x)*log1p(|x| + epsilon) for each element
            y_positive = torch.log1p(torch.clamp(data, min=0) + epsilon)
            y_negative = -torch.log1p(torch.clamp(-data, min=0) + epsilon)
            
            # Combine based on the sign of data
            y = torch.where(data >= 0, y_positive, y_negative)
            normalized = y / m
            
        elif method == "standard":
            mean = stats["mean"]
            std = max(stats["std"], eps)
            normalized = (data - mean) / std
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
            
        return normalized

    @staticmethod
    def denormalize(norm_values: Union[torch.Tensor, List[float], float], 
                   metadata: Dict[str, Any], 
                   variable_name: str) -> Union[torch.Tensor, List[float], float]:
        """
        Denormalize values using stored normalization metadata.
        
        Parameters
        ----------
        norm_values : Union[torch.Tensor, List[float], float]
            Normalized values to be denormalized
        metadata : Dict[str, Any]
            Dictionary containing normalization statistics
        variable_name : str
            Name of the variable
        
        Returns
        -------
        Union[torch.Tensor, List[float], float]
            Denormalized values in the same form as input
        
        Raises
        ------
        KeyError
            If variable or its normalization method is not in metadata
        ValueError
            If the normalization method is not supported
        """
        # Check if we have metadata for this variable
        if "normalization_methods" not in metadata or variable_name not in metadata.get("normalization_methods", {}):
            raise KeyError(f"No normalization method found for '{variable_name}'")
            
        method = metadata["normalization_methods"][variable_name]
        
        if variable_name not in metadata:
            raise KeyError(f"No statistics found for '{variable_name}'")
            
        stats = metadata[variable_name]
        
        # Convert input to tensor
        is_scalar = isinstance(norm_values, (int, float))
        is_list = isinstance(norm_values, list)
        
        norm_tensor = torch.tensor(norm_values, dtype=torch.float32)
        eps = 1e-9  # Small epsilon for numerical stability
        
        # Handle empty tensor case
        if norm_tensor.numel() == 0:
            return norm_values
        
        if method == "iqr":
            median = stats["median"]
            iqr = stats["iqr"]
            denorm = norm_tensor * iqr + median
            
        elif method == "log-min-max":
            min_val = stats["min"]
            max_val = stats["max"]
            log_x = norm_tensor * (max_val - min_val) + min_val
            denorm = torch.exp(log_x)
            
            # Remove offset if it was applied during normalization
            if "offset" in stats:
                denorm = denorm - stats["offset"]
            
        elif method == "arctan-compression":
            center = stats["center"]
            scale = stats["scale"]
            alpha = stats["alpha"]
            # Clamp normalized values to avoid tangent overflow
            safe_norm = torch.clamp(norm_tensor, -0.99999, 0.99999)
            denorm = center + (scale / alpha) * torch.tan((math.pi / 2) * safe_norm)
            
        elif method == "max-out":
            max_val = stats["max_val"]
            denorm = norm_tensor * max_val
            
        elif method == "invlogit-compression":
            mean = stats["mean"]
            std = stats["std"]
            # Handle edge cases to prevent numerical issues
            safe_norm = torch.clamp(norm_tensor, -0.99999, 0.99999)
            logit = torch.log((safe_norm + 1) / (1 - safe_norm + eps))
            denorm = mean + std * logit
            
        elif method == "custom":
            m = stats["m"]
            # Use epsilon from stats or default to 1e-10
            epsilon = stats.get("epsilon", 1e-10)
            y = norm_tensor * m
            # Calculate the exponential minus 1 for positive and negative values separately,
            # then subtract epsilon to reverse the addition during normalization
            denorm = torch.where(y >= 0, 
                               torch.expm1(y) - epsilon, 
                               -torch.expm1(-y) - epsilon)
            
        elif method == "standard":
            mean = stats["mean"]
            std = stats["std"]
            denorm = norm_tensor * std + mean
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # Return in the same format as input
        if is_scalar:
            return denorm.item()
        elif is_list:
            return denorm.tolist()
        else:
            return denorm

    def process_profiles(self, stats: Dict[str, Any]) -> None:
        """
        Normalize all JSON profiles using the computed statistics.
        
        Creates normalized profiles and saves them with metadata to the output folder.
        
        Parameters
        ----------
        stats : Dict[str, Any]
            Dictionary containing normalization statistics and methods
        
        Raises
        ------
        OSError
            If there's an error creating output files
        """
        logger.info(f"Processing profiles using computed statistics...")
        
        # Create output directory
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Save normalization metadata
        metadata_path = self.output_folder / "normalization_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved normalization metadata to: {metadata_path}")
        
        # Get normalization methods for each variable
        methods = stats.get("normalization_methods", {})
        if not methods:
            logger.error("No normalization methods found in stats")
            raise ValueError("Invalid statistics: missing normalization_methods")
        
        # Process each profile file
        profile_files = list(self.input_folder.glob("*.json"))
        total_files = len(profile_files)
        batch_size = min(self.batch_size, total_files)
        num_batches = (total_files + batch_size - 1) // batch_size
        
        logger.info(f"Processing {total_files} files in {num_batches} batches")
        processed_count = 0
        error_count = 0
        
        # Process in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_files)
            batch_files = profile_files[start_idx:end_idx]
            
            logger.debug(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_files)} files)")
            
            for profile_path in batch_files:
                try:
                    with open(profile_path, "r") as f:
                        profile = json.load(f)
                    
                    # Create normalized profile
                    normalized_profile = {}
                    
                    for key, value in profile.items():
                        if key in stats and key not in ["normalization_methods", "config"]:
                            method = methods.get(key)
                            if not method:
                                logger.warning(f"No normalization method found for '{key}', skipping")
                                normalized_profile[key] = value
                                continue
                                
                            key_stats = stats[key]
                            
                            if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                                data = torch.tensor(value, dtype=torch.float32)
                                try:
                                    norm_data = self.normalize_tensor(data, method, key_stats)
                                    normalized_profile[key] = norm_data.tolist()
                                except ValueError as e:
                                    logger.warning(f"Error normalizing '{key}' in {profile_path.name}: {e}")
                                    logger.warning(f"Using unnormalized values for '{key}' which may affect model performance")
                                    normalized_profile[key] = value
                            elif isinstance(value, (int, float)):
                                data = torch.tensor([value], dtype=torch.float32)
                                try:
                                    norm_data = self.normalize_tensor(data, method, key_stats)
                                    normalized_profile[key] = norm_data.item()
                                except ValueError as e:
                                    logger.warning(f"Error normalizing '{key}' in {profile_path.name}: {e}")
                                    logger.warning(f"Using unnormalized values for '{key}' which may affect model performance")
                                    normalized_profile[key] = value
                            else:
                                normalized_profile[key] = value
                        else:
                            normalized_profile[key] = value
                    
                    # Save normalized profile
                    output_path = self.output_folder / profile_path.name
                    with open(output_path, "w") as f:
                        json.dump(normalized_profile, f, indent=2)
                    
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing {profile_path.name}: {e}")
                    error_count += 1
            
            # Progress report
            if batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1:
                progress = (batch_idx + 1) / num_batches * 100
                logger.info(f"Normalization progress: {progress:.1f}% ({processed_count}/{total_files} files)")
        
        if error_count > 0:
            logger.warning(f"Encountered errors in {error_count} files during normalization")
            
        logger.info(f"Processed and saved {processed_count} normalized profiles")