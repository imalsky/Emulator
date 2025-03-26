#!/usr/bin/env python3
"""
normalizer.py

Data normalization utilities for atmospheric profiles using log base 10.

This module provides a DataNormalizer class that:
1. Computes global normalization statistics from raw JSON profile files
2. Applies various normalization methods to the data
3. Saves the normalized profiles with metadata

Supported normalization methods:
- "iqr": Normalizes based on median and interquartile range
- "log-min-max": Applies log base‑10 transform then min‑max scaling
- "arctan-compression": Compresses data using arctan transform
- "max-out": Divides data by absolute global maximum
- "invlogit-compression": Maps data using sigmoid transformation
- "custom": Sign-preserving logarithmic (base‑10) transformation
- "standard": Standard z-score normalization with scaling to [-1, 1] range
- "symlog": Symmetric logarithmic (base‑10) transformation with tunable threshold
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
        """
        self.input_folder = Path(input_folder)
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
            
        self.output_folder = Path(output_folder)
        self.batch_size = max(1, batch_size)
        
        logger.info(f"DataNormalizer initialized with input: {input_folder}, output: {output_folder}")

    @staticmethod
    def clip_outliers(values: torch.Tensor, 
                       lower_quantile: float = 0.001, 
                       upper_quantile: float = 0.999) -> torch.Tensor:
        """
        Clip values outside the specified quantile range.
        """
        if not (0 <= lower_quantile < upper_quantile <= 1):
            raise ValueError("Quantiles must satisfy: 0 ≤ lower < upper ≤ 1")
        
        if values.numel() == 0:
            return values
            
        lower_bound = torch.quantile(values, lower_quantile)
        upper_bound = torch.quantile(values, upper_quantile)
        
        return torch.clamp(values, min=lower_bound, max=upper_bound)

    def calculate_global_stats(
        self,
        key_methods: Optional[Dict[str, str]] = None,
        default_method: str = "iqr",
        clip_outliers_before_scaling: bool = False,
        symlog_percentile: float = 0.5,
        symlog_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Compute global normalization statistics for all variables.
        """
        logger.info("Calculating global normalization statistics...")
        
        profile_files = list(self.input_folder.glob("*.json"))
        if not profile_files:
            raise FileNotFoundError(f"No JSON files found in '{self.input_folder}'")
        
        key_values: Dict[str, List[torch.Tensor]] = {}
        total_files = len(profile_files)
        batch_size = min(self.batch_size, total_files)
        num_batches = (total_files + batch_size - 1) // batch_size
        
        logger.info(f"Processing {total_files} files in {num_batches} batches")
        all_keys = set()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_files)
            batch_files = profile_files[start_idx:end_idx]
            
            for profile_path in batch_files:
                try:
                    with open(profile_path, "r") as f:
                        profile = json.load(f)
                    
                    all_keys.update(profile.keys())
                    
                    for key, value in profile.items():
                        if key not in key_values:
                            key_values[key] = []
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            continue
                        if isinstance(value, (int, float)):
                            key_values[key].append(torch.tensor([float(value)], dtype=torch.float32))
                        elif isinstance(value, list) and all(isinstance(v, (int, float)) or (isinstance(v, float) and np.isnan(v)) for v in value):
                            valid_values = [v for v in value if not (isinstance(v, float) and np.isnan(v))]
                            if valid_values:
                                key_values[key].append(torch.tensor(valid_values, dtype=torch.float32))
                        elif isinstance(value, bool):
                            if key not in key_values:
                                key_values[key] = ['boolean']
                except Exception as e:
                    raise RuntimeError(f"Error reading {profile_path}: {e}")
        
        boolean_keys = set()
        for key in list(key_values.keys()):
            if key_values[key] and key_values[key][0] == 'boolean':
                boolean_keys.add(key)
                del key_values[key]
                continue
            if key_values[key]:
                try:
                    values = torch.cat(key_values[key], dim=0)
                    unique_vals = torch.unique(values)
                    if len(unique_vals) <= 2 and all(val in [0.0, 1.0] for val in unique_vals):
                        boolean_keys.add(key)
                        del key_values[key]
                        continue
                    key_values[key] = values
                except:
                    del key_values[key]
            else:
                del key_values[key]
        
        normalization_methods = {}
        for key in all_keys:
            if key_methods and key in key_methods:
                normalization_methods[key] = key_methods[key]
            else:
                normalization_methods[key] = default_method
                
        for key in boolean_keys:
            normalization_methods[key] = "none"
        
        if clip_outliers_before_scaling:
            logger.info("Clipping outliers before computing statistics")
            for key, vals in key_values.items():
                if vals.numel() > 0:
                    key_values[key] = self.clip_outliers(vals)
        
        stats = {}
        for key, values in key_values.items():
            if values.numel() == 0:
                logger.warning(f"Skipping statistics computation for '{key}' - no values available")
                continue
                
            method = normalization_methods[key]
            try:
                if method == "symlog":
                    threshold = None
                    if symlog_thresholds and key in symlog_thresholds:
                        threshold = symlog_thresholds[key]
                    stats[key] = self._compute_stats_for_key(values, method, 
                                                            symlog_percentile=symlog_percentile,
                                                            symlog_threshold=threshold)
                else:
                    stats[key] = self._compute_stats_for_key(values, method)
                logger.debug(f"Computed {method} statistics for '{key}'")
            except ValueError as e:
                logger.warning(f"Error computing {method} statistics for '{key}': {e}. Falling back to 'max-out'.")
                normalization_methods[key] = "max-out"
                try:
                    stats[key] = self._compute_stats_for_key(values, "max-out")
                except ValueError as e2:
                    logger.error(f"Failed to compute fallback statistics for '{key}': {e2}")
                    continue
        
        stats["normalization_methods"] = normalization_methods
        stats["config"] = {
            "clip_outliers_before_scaling": clip_outliers_before_scaling,
            "default_method": default_method,
            "key_methods": key_methods if key_methods else {},
            "boolean_keys": list(boolean_keys),
            "all_keys": list(all_keys),
            "symlog_percentile": symlog_percentile,
            "symlog_thresholds": symlog_thresholds if symlog_thresholds else {},
        }
        
        logger.info(f"Statistics computed for {len(key_values)} variables")
        return stats

    def _compute_stats_for_key(self, values: torch.Tensor, method: str, 
                              symlog_percentile: float = 0.5,
                              symlog_threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Compute statistics for a specific key based on the chosen normalization method.
        Now using base-10 logarithms.
        """
        if values.numel() == 0:
            raise ValueError("Cannot compute statistics on empty tensor")
            
        global_min = values.min().item()
        global_max = values.max().item()
        stats = {"global_min": global_min, "global_max": global_max}
        is_constant = (global_max - global_min) < 1e-6
        method = method.lower().strip()
        
        if method == "iqr":
            median = torch.median(values).item()
            q1 = torch.quantile(values, 0.25).item()
            q3 = torch.quantile(values, 0.75).item()
            iqr = q3 - q1
            if iqr < 1e-6:
                iqr = 1.0
                stats["is_constant"] = True
            stats.update({"median": median, "iqr": iqr})
            
        elif method == "log-min-max":
            if is_constant:
                stats.update({
                    "min": 0.0,
                    "max": 1.0,
                    "is_constant": True
                })
                return stats
            if global_min <= 0:
                offset = abs(global_min) + 1e-6
                logger.warning(f"log-min-max requires positive values, found min={global_min}. Adding offset: {offset}")
                values = values + offset
                stats["offset"] = offset
                global_min = values.min().item()
            log_vals = torch.log10(values)
            vmin = log_vals.min().item()
            vmax = log_vals.max().item()
            if abs(vmax - vmin) < 1e-6:
                vmin, vmax = 0.0, 1.0
            stats.update({"min": vmin, "max": vmax})
            
        elif method == "arctan-compression":
            if global_min < 0 and global_max > 0:
                center = 0.0
            else:
                center = (global_min + global_max) / 2.0
            scale = (global_max - global_min) / 2.0
            if scale < 1e-6:
                scale = 1.0
                stats["is_constant"] = True
            alpha = math.tan(0.99 * (math.pi / 2))
            stats.update({"center": center, "scale": scale, "alpha": alpha})
            
        elif method == "max-out":
            abs_max = max(abs(global_min), abs(global_max))
            if abs_max < 1e-6:
                abs_max = 1.0
                stats["is_constant"] = True
            stats.update({"max_val": abs_max})
            
        elif method == "invlogit-compression":
            mean = values.mean().item()
            std = values.std().item()
            if global_min < 0 and global_max > 0:
                mean = 0.0
            if std < 1e-6:
                std = 1.0
                stats["is_constant"] = True
            stats.update({"mean": mean, "std": std})
            
        elif method == "custom":
            if is_constant:
                stats.update({
                    "m": 1.0,
                    "epsilon": 1e-10,
                    "is_constant": True
                })
                return stats
            epsilon = 1e-10
            y_positive = torch.log10(torch.clamp(values, min=0) + 1 + epsilon)
            y_negative = -torch.log10(torch.clamp(-values, min=0) + 1 + epsilon)
            y = torch.where(values >= 0, y_positive, y_negative)
            m_pos = torch.log10(torch.tensor(global_max) + 1 + epsilon).item() if global_max > 0 else 0.0
            m_neg = torch.log10(torch.tensor(-global_min) + 1 + epsilon).item() if global_min < 0 else 0.0
            m = max(m_pos, m_neg)
            if m < 1e-6:
                m = 1.0
            stats.update({"m": m, "epsilon": epsilon})
            
        elif method == "symlog":
            if is_constant:
                stats.update({
                    "threshold": 1.0,
                    "scale_factor": 1.0,
                    "epsilon": 1e-10,
                    "is_constant": True
                })
                return stats
            epsilon = 1e-10
            if symlog_threshold is None:
                threshold = torch.quantile(torch.abs(values), symlog_percentile).item()
                threshold = max(threshold, 1e-6)
            else:
                threshold = symlog_threshold
            abs_values = torch.abs(values)
            linear_region = abs_values <= threshold
            log_region = abs_values > threshold
            transformed = torch.zeros_like(values)
            transformed[linear_region] = values[linear_region] / threshold
            log_values = torch.log10(abs_values[log_region] / threshold) + 1
            transformed[log_region] = torch.sign(values[log_region]) * log_values
            max_abs_transformed = torch.max(torch.abs(transformed)).item()
            scale_factor = max(max_abs_transformed, 1.0)
            stats.update({
                "threshold": threshold,
                "scale_factor": scale_factor,
                "epsilon": epsilon
            })
            
        elif method == "standard":
            mean = values.mean().item()
            std = values.std().item()
            if std < 1e-6 or is_constant:
                std = 1.0
                stats["is_constant"] = True
            if is_constant:
                scale_factor = 1.0
            else:
                max_dev = max(abs(global_max - mean), abs(global_min - mean))
                scale_factor = max_dev / std
            stats.update({
                "mean": mean, 
                "std": std,
                "scale_factor": scale_factor
            })
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
            
        return stats

    @staticmethod
    def normalize_tensor(data: torch.Tensor, method: str, stats: Dict[str, float]) -> torch.Tensor:
        """
        Normalize a tensor using the specified method and statistics.
        Uses base-10 logarithms.
        """
        eps = 1e-9
        if data.numel() == 0:
            return data
        if method == "none":
            return data
        if stats.get("is_constant", False):
            return data
        method = method.lower().strip()
        
        if method == "iqr":
            median = stats["median"]
            iqr = max(stats["iqr"], eps)
            normalized = (data - median) / iqr
            
        elif method == "log-min-max":
            if "offset" in stats:
                offset = stats["offset"]
                data = data + offset
            if torch.any(data <= 0):
                min_val = torch.min(data).item()
                logger.warning(f"log-min-max found non-positive values (min={min_val}), applying epsilon offset")
                data = torch.clamp(data, min=eps)
            log_data = torch.log10(data)
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
            epsilon = stats.get("epsilon", 1e-10)
            y_positive = torch.log10(torch.clamp(data, min=0) + 1 + epsilon)
            y_negative = -torch.log10(torch.clamp(-data, min=0) + 1 + epsilon)
            y = torch.where(data >= 0, y_positive, y_negative)
            normalized = y / m
            
        elif method == "symlog":
            threshold = stats["threshold"]
            scale_factor = max(stats["scale_factor"], eps)
            abs_data = torch.abs(data)
            linear_region = abs_data <= threshold
            log_region = abs_data > threshold
            transformed = torch.zeros_like(data)
            transformed[linear_region] = data[linear_region] / threshold
            if torch.any(log_region):
                safe_vals = torch.clamp(abs_data[log_region] / threshold, min=1e-10)
                log_values = torch.log10(safe_vals) + 1
                transformed[log_region] = torch.sign(data[log_region]) * log_values
            normalized = transformed / scale_factor
            normalized = torch.clamp(normalized, -1.0, 1.0)
            
        elif method == "standard":
            mean = stats["mean"]
            std = max(stats["std"], eps)
            scale_factor = max(stats["scale_factor"], eps)
            standardized = (data - mean) / std
            normalized = standardized / scale_factor
            normalized = torch.clamp(normalized, -1.0, 1.0)
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
            
        return normalized

    @staticmethod
    def denormalize(norm_values: Union[torch.Tensor, List[float], float], 
                   metadata: Dict[str, Any], 
                   variable_name: str) -> Union[torch.Tensor, List[float], float]:
        """
        Denormalize values using stored normalization metadata.
        Uses base-10 logarithms where applicable.
        """
        if "normalization_methods" not in metadata or variable_name not in metadata.get("normalization_methods", {}):
            raise KeyError(f"No normalization method found for '{variable_name}'")
            
        method_raw = metadata["normalization_methods"][variable_name]
        method = str(method_raw).lower().strip()
        
        if method == "none":
            return norm_values
            
        if variable_name not in metadata:
            raise KeyError(f"No statistics found for '{variable_name}'")
            
        stats = metadata[variable_name]
        is_scalar = isinstance(norm_values, (int, float))
        is_list = isinstance(norm_values, list)
        
        norm_tensor = torch.tensor(norm_values, dtype=torch.float32)
        eps = 1e-9
        
        if norm_tensor.numel() == 0:
            return norm_values
            
        if stats.get("is_constant", False):
            return norm_values
        
        if method.startswith("iqr"):
            median = stats["median"]
            iqr = stats["iqr"]
            denorm = norm_tensor * iqr + median
            
        elif method.startswith("log-min"):
            min_val = stats["min"]
            max_val = stats["max"]
            log_x = norm_tensor * (max_val - min_val) + min_val
            denorm = torch.pow(10, log_x)
            if "offset" in stats:
                denorm = denorm - stats["offset"]
            
        elif method.startswith("arctan"):
            center = stats["center"]
            scale = stats["scale"]
            alpha = stats["alpha"]
            safe_norm = torch.clamp(norm_tensor, -0.99999, 0.99999)
            denorm = center + (scale / alpha) * torch.tan((math.pi / 2) * safe_norm)
            
        elif method.startswith("max-out") or method.startswith("max"):
            max_val = stats.get("max_val", stats.get("abs_max", 1.0))
            denorm = norm_tensor * max_val
            
        elif method.startswith("invlogit"):
            mean = stats["mean"]
            std = stats["std"]
            safe_norm = torch.clamp(norm_tensor, -0.99999, 0.99999)
            logit = torch.log((safe_norm + 1) / (1 - safe_norm + eps))
            denorm = mean + std * logit
            
        elif method.startswith("custom"):
            m = stats["m"]
            epsilon = stats.get("epsilon", 1e-10)
            y = norm_tensor * m
            denorm = torch.where(y >= 0, 
                               torch.pow(10, y) - 1 - epsilon, 
                               - (torch.pow(10, -y) - 1) - epsilon)
            
        elif method.startswith("symlog"):
            threshold = stats["threshold"]
            scale_factor = stats["scale_factor"]
            unscaled = norm_tensor * scale_factor
            eps_bound = 1e-6
            abs_unscaled = torch.abs(unscaled)
            linear_region = abs_unscaled <= (1.0 + eps_bound)
            log_region = ~linear_region
            denorm = torch.zeros_like(unscaled)
            denorm[linear_region] = unscaled[linear_region] * threshold
            if torch.any(log_region):
                safe_abs_unscaled = torch.clamp(abs_unscaled[log_region] - 1.0, min=-50.0, max=50.0)
                log_values = torch.pow(10, safe_abs_unscaled) * threshold
                denorm[log_region] = torch.sign(unscaled[log_region]) * log_values
            
        elif method.startswith("standard"):
            mean = stats["mean"]
            std = stats["std"]
            scale_factor = stats["scale_factor"]
            unscaled = norm_tensor * scale_factor
            denorm = unscaled * std + mean
            
        else:
            raise ValueError(f"Unsupported normalization method: '{method}' (original: '{method_raw}', type: {type(method_raw)}, repr: {repr(method_raw)})")
        
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
        """
        logger.info("Processing profiles using computed statistics...")
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        metadata_path = self.output_folder / "normalization_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved normalization metadata to: {metadata_path}")
        
        methods = stats.get("normalization_methods", {})
        if not methods:
            logger.error("No normalization methods found in stats")
            raise ValueError("Invalid statistics: missing normalization_methods")
            
        boolean_keys = set(stats.get("config", {}).get("boolean_keys", []))
        
        profile_files = list(self.input_folder.glob("*.json"))
        total_files = len(profile_files)
        batch_size = min(self.batch_size, total_files)
        num_batches = (total_files + batch_size - 1) // batch_size
        
        logger.info(f"Processing {total_files} files in {num_batches} batches")
        processed_count = 0
        error_count = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_files)
            batch_files = profile_files[start_idx:end_idx]
            
            for profile_path in batch_files:
                try:
                    with open(profile_path, "r") as f:
                        profile = json.load(f)
                    
                    normalized_profile = {}
                    
                    for key, value in profile.items():
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            normalized_profile[key] = value
                            continue
                            
                        if isinstance(value, bool) or key in boolean_keys:
                            normalized_profile[key] = value
                            continue
                            
                        method = methods.get(key)
                        if not method or method == "none":
                            normalized_profile[key] = value
                            continue
                        
                        key_stats = stats.get(key)
                        if not key_stats and key in stats.get("config", {}).get("all_keys", []):
                            if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                                data = torch.tensor(value, dtype=torch.float32)
                                key_stats = self._compute_stats_for_key(data, "max-out")
                                stats[key] = key_stats
                                methods[key] = "max-out"
                            elif isinstance(value, (int, float)):
                                data = torch.tensor([float(value)], dtype=torch.float32)
                                key_stats = self._compute_stats_for_key(data, "max-out")
                                stats[key] = key_stats
                                methods[key] = "max-out"
                            else:
                                normalized_profile[key] = value
                                continue
                        elif not key_stats:
                            normalized_profile[key] = value
                            continue
                                
                        if isinstance(value, list) and all(isinstance(v, (int, float)) or (isinstance(v, float) and np.isnan(v)) for v in value):
                            valid_indices = [i for i, v in enumerate(value) if not (isinstance(v, float) and np.isnan(v))]
                            valid_values = [value[i] for i in valid_indices]
                            
                            if valid_values:
                                data = torch.tensor(valid_values, dtype=torch.float32)
                                try:
                                    norm_data = self.normalize_tensor(data, method, key_stats)
                                    result = []
                                    valid_idx = 0
                                    for i in range(len(value)):
                                        if i in valid_indices:
                                            result.append(norm_data[valid_idx].item())
                                            valid_idx += 1
                                        else:
                                            result.append(value[i])
                                    normalized_profile[key] = result
                                except ValueError as e:
                                    logger.warning(f"Error normalizing '{key}': {e}")
                                    normalized_profile[key] = value
                            else:
                                normalized_profile[key] = value
                        elif isinstance(value, (int, float)):
                            data = torch.tensor([float(value)], dtype=torch.float32)
                            try:
                                norm_data = self.normalize_tensor(data, method, key_stats)
                                normalized_profile[key] = norm_data.item()
                            except ValueError as e:
                                logger.warning(f"Error normalizing '{key}': {e}")
                                normalized_profile[key] = value
                        else:
                            normalized_profile[key] = value
                    
                    output_path = self.output_folder / profile_path.name
                    with open(output_path, "w") as f:
                        json.dump(normalized_profile, f, indent=2)
                    
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing {profile_path.name}: {e}")
                    error_count += 1
        
        with open(metadata_path, "w") as f:
            stats["normalization_methods"] = methods
            json.dump(stats, f, indent=2)
            
        if error_count > 0:
            logger.warning(f"Encountered errors in {error_count} files during normalization")
            
        logger.info(f"Processed and saved {processed_count} normalized profiles")
