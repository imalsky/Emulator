#!/usr/bin/env python3
"""normalizer.py – Compute **and** invert eight normalization schemes.

Supported methods (case‑insensitive):
--------------------------------------------------------------
* `iqr`                 – median / IQR scaling → ℝ.
* `log-min-max`         – log₁₀ → min‑max to [0, 1].
* `arctan-compression`  – arctan ‑π↦π → [-1, 1].
* `max-out`             – divide by global |max|  → [-1, 1].
* `invlogit-compression`– sigmoid → [-1, 1].
* `custom`              – sign‑preserving log₁₀ (He et al.).
* `symlog`              – linear near 0, log₁₀ tails  → [-1, 1].
* `standard`            – z‑score then scale to [-1, 1].

The class **guarantees** that for any numeric value `x`,
`denormalize(normalize(x)) == x` (within fp tolerance).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["DataNormalizer"]


# -----------------------------------------------------------------------------
# helper utils
# -----------------------------------------------------------------------------

def _to_tensor(v: Union[List[float], float]) -> Tensor:
    """Converts float or list of floats to a float32 tensor."""
    return torch.tensor(v if isinstance(v, list) else [v], dtype=torch.float32)


# -----------------------------------------------------------------------------
# main class
# -----------------------------------------------------------------------------

class DataNormalizer:
    """Computes statistics and (de)normalizes JSON profile files."""

    METHODS = {
        "iqr",
        "log-min-max",
        "arctan-compression",
        "max-out",
        "invlogit-compression",
        "custom",
        "symlog",
        "standard",
        "none",  # passthrough (for booleans / categorical, etc.)
    }

    def __init__(self, input_folder: Union[str, Path], output_folder: Union[str, Path], batch_size: int = 100):
        """
        Initializes the normalizer.

        Args:
            input_folder: Path to directory with raw JSON profiles.
            output_folder: Path to directory where normalized profiles and metadata will be saved.
            batch_size: Number of files to process per batch when calculating stats (controls memory usage).
        """
        self.input_dir = Path(input_folder)
        self.output_dir = Path(output_folder)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input folder not found: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = max(batch_size, 1)
        logger.info(f"DataNormalizer input: {self.input_dir}, output: {self.output_dir}")

    # ------------------------------------------------------------------
    # statistics calculation
    # ------------------------------------------------------------------

    @staticmethod
    def clip_outliers(t: Tensor, low_quantile: float = 0.001, high_quantile: float = 0.999) -> Tensor:
        """Clips tensor values outside the specified quantile range."""
        if t.numel() == 0:
            return t # Return empty tensor if input is empty
        if not (0 <= low_quantile < high_quantile <= 1):
             raise ValueError("Quantiles must satisfy 0 <= low < high <= 1")
        low_bound = torch.quantile(t, low_quantile)
        high_bound = torch.quantile(t, high_quantile)
        return torch.clamp(t, min=low_bound, max=high_bound)

    def calculate_global_stats(
        self,
        key_methods: Optional[Dict[str, str]] = None,
        default_method: str = "standard", # Changed default to standard
        clip_outliers_before_scaling: bool = False,
        symlog_percentile: float = 0.5,
        symlog_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates global statistics across all JSON profiles for normalization.

        Args:
            key_methods: Dictionary mapping specific variable keys to normalization methods.
            default_method: Default normalization method if not specified in key_methods.
            clip_outliers_before_scaling: If True, clip outliers before computing stats.
            symlog_percentile: Percentile used to determine the linear threshold for 'symlog'.
            symlog_thresholds: Dictionary mapping specific variable keys to fixed thresholds for 'symlog'.

        Returns:
            Dictionary containing computed statistics and metadata.
        """
        logger.info("Calculating global normalization statistics...")
        files = [p for p in self.input_dir.glob("*.json") if p.name != "normalization_metadata.json"]
        if not files:
            raise FileNotFoundError(f"No JSON profiles found in input folder: {self.input_dir}")

        # Accumulate values for each key
        value_buffer: Dict[str, List[Tensor]] = {}
        boolean_keys: set[str] = set()
        processed_files = 0
        num_batches = (len(files) + self.batch_size - 1) // self.batch_size

        for i in range(num_batches):
             batch_files = files[i*self.batch_size:(i+1)*self.batch_size]
             for fpath in batch_files:
                try:
                    prof = json.loads(fpath.read_text())
                    processed_files += 1
                    for k, v in prof.items():
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            continue # Skip null/NaN values
                        if isinstance(v, bool):
                            boolean_keys.add(k)
                        elif isinstance(v, (int, float)):
                            # Convert scalars to tensors
                            value_buffer.setdefault(k, []).append(torch.tensor([float(v)], dtype=torch.float32))
                        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v if not (isinstance(x, float) and np.isnan(x))):
                             # Handle lists, filtering out NaNs before converting to tensor
                             valid_vals = [float(x) for x in v if not (isinstance(x, float) and np.isnan(x))]
                             if valid_vals: # Only append if list contains valid numbers
                                value_buffer.setdefault(k, []).append(torch.tensor(valid_vals, dtype=torch.float32))
                except Exception as e:
                    logger.warning(f"Could not process file {fpath.name}: {e}")
                    continue # Skip malformed files

        logger.info(f"Scanned {processed_files} profiles to gather statistics.")

        # Remove boolean keys from value buffer
        for bk in boolean_keys:
            value_buffer.pop(bk, None)

        # Determine normalization method for each key
        methods = {k: m.lower().strip() for k, m in (key_methods or {}).items()}
        default_method_lower = default_method.lower().strip()
        if default_method_lower not in self.METHODS:
            raise ValueError(f"Unknown default_method '{default_method}'")

        all_numeric_keys = set(value_buffer.keys())
        final_methods = {k: methods.get(k, default_method_lower) for k in all_numeric_keys}
        for k in boolean_keys: # Add boolean keys with 'none' method
             final_methods[k] = 'none'


        # Compute statistics per key
        stats: Dict[str, Any] = {}
        computed_keys = 0
        for key, tensor_list in value_buffer.items():
            if not tensor_list:
                logger.warning(f"No valid numeric data found for key '{key}', skipping stats calculation.")
                continue

            # Concatenate all tensors for the key
            try:
                 full_vector = torch.cat(tensor_list)
            except RuntimeError as e:
                 logger.error(f"Error concatenating tensors for key '{key}': {e}. Sizes: {[t.shape for t in tensor_list]}")
                 continue # Skip this key if concatenation fails

            if full_vector.numel() == 0:
                 logger.warning(f"Empty tensor after concatenation for key '{key}', skipping.")
                 continue

            if clip_outliers_before_scaling:
                full_vector = self.clip_outliers(full_vector)

            method_to_use = final_methods[key]
            if method_to_use not in self.METHODS:
                logger.warning(f"Invalid method '{method_to_use}' specified for '{key}', defaulting to '{default_method_lower}'.")
                method_to_use = default_method_lower
                final_methods[key] = method_to_use # Update the methods dict

            try:
                stats[key] = self._compute_stats_for_key(
                    full_vector,
                    method_to_use,
                    symlog_percentile,
                    (symlog_thresholds or {}).get(key)
                )
                computed_keys += 1
            except Exception as e:
                 logger.error(f"Failed to compute stats for key '{key}' using method '{method_to_use}': {e}")
                 # Optionally, try a fallback method like 'none' or remove key
                 final_methods.pop(key, None) # Remove from methods if stats failed


        # Assemble final metadata dictionary
        metadata = {
            "normalization_methods": final_methods,
            "config": { # Store config used for stat calculation
                "clip_outliers_before_scaling": clip_outliers_before_scaling,
                "default_method": default_method_lower,
                "key_methods_provided": key_methods or {},
                "symlog_percentile": symlog_percentile,
                "symlog_thresholds_provided": symlog_thresholds or {},
                "boolean_keys_detected": list(boolean_keys),
                "numeric_keys_processed": list(stats.keys()),
            },
            **stats # Merge computed stats
        }

        # Save metadata
        meta_path = self.output_dir / "normalization_metadata.json"
        try:
            from utils import save_json # Use save_json for consistency
            save_json(metadata, meta_path)
            logger.info(f"Saved normalization metadata ({computed_keys} numeric keys) to {meta_path}")
        except ImportError:
             logger.warning("Could not import save_json from utils, saving with standard json.")
             meta_path.write_text(json.dumps(metadata, indent=2))
        except Exception as e:
             logger.error(f"Failed to save normalization metadata: {e}")


        return metadata


    def _compute_stats_for_key(
        self,
        data: Tensor,
        method: str,
        symlog_percentile: float = 0.5,
        symlog_threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """Computes statistics needed for a specific normalization method."""
        if data.numel() == 0:
            raise ValueError("Cannot compute statistics on empty tensor.")

        global_min, global_max = float(data.min()), float(data.max())
        stats: Dict[str, float] = {"global_min": global_min, "global_max": global_max}
        # Check if data is effectively constant
        is_constant = (global_max - global_min) < 1e-9 # Use a small epsilon
        if is_constant:
            stats["is_constant"] = 1.0 # Use float for JSON compatibility

        method_lower = method.lower().strip()
        epsilon = 1e-10 # Small value to avoid log(0) or division by zero

        if method_lower == "iqr":
            q1 = float(torch.quantile(data, 0.25))
            q3 = float(torch.quantile(data, 0.75))
            iqr = q3 - q1
            stats.update({
                "median": float(torch.median(data).values), # .values needed for torch >= 1.8
                "iqr": max(iqr, epsilon), # Avoid zero IQR
            })
        elif method_lower == "log-min-max":
            offset = 0.0
            if global_min <= 0:
                offset = abs(global_min) + epsilon
                logger.debug(f"log-min-max requires positive values, adding offset: {offset:.2e} for key")
            # Apply offset before log
            shifted_data = data + offset
            # Clamp potentially very small values after offset before log10
            log_vals = torch.log10(torch.clamp(shifted_data, min=epsilon))
            log_min, log_max = float(log_vals.min()), float(log_vals.max())
            stats.update({"min": log_min, "max": log_max, "offset": offset})
        elif method_lower == "arctan-compression":
            center = 0.0 if (global_min < 0 and global_max > 0) else (global_min + global_max) / 2.0
            scale = max((global_max - global_min) / 2.0, epsilon)
            # alpha determines the steepness, scales input to ~[-1, 1] before atan
            alpha = math.tan(0.99 * math.pi / 2) # Value that maps large inputs close to +/- pi/2
            stats.update({"center": center, "scale": scale, "alpha": alpha})
        elif method_lower == "max-out":
            max_abs_val = max(abs(global_min), abs(global_max), epsilon)
            stats["max_val"] = max_abs_val
        elif method_lower == "invlogit-compression":
            stats.update({"mean": float(data.mean()), "std": max(float(data.std()), epsilon)})
        elif method_lower == "custom": # Sign-preserving log10
            m_pos = math.log10(global_max + 1 + epsilon) if global_max > -epsilon else 0.0
            m_neg = math.log10(abs(global_min) + 1 + epsilon) if global_min < epsilon else 0.0
            stats.update({"m": max(m_pos, m_neg, 1.0), "epsilon": epsilon}) # Use max of magnitude logs
        elif method_lower == "symlog":
            if is_constant:
                 # Handle constant case for symlog gracefully
                 stats.update({"threshold": 1.0, "scale_factor": 1.0, "epsilon": epsilon})
            else:
                abs_data = torch.abs(data)
                if symlog_threshold is None:
                    # Determine threshold from percentile of absolute values
                    threshold = max(float(torch.quantile(abs_data, symlog_percentile)), epsilon)
                else:
                    threshold = max(symlog_threshold, epsilon) # Use provided threshold if valid

                # Transform data to calculate scale_factor
                linear_mask = abs_data <= threshold
                log_mask = ~linear_mask
                transformed = torch.zeros_like(data)
                transformed[linear_mask] = data[linear_mask] / threshold
                # Clamp log argument to avoid log(<=0)
                safe_log_arg = torch.clamp(abs_data[log_mask] / threshold, min=epsilon)
                transformed[log_mask] = torch.sign(data[log_mask]) * (torch.log10(safe_log_arg) + 1)

                scale_factor = max(float(torch.max(torch.abs(transformed))), 1.0) # Ensure scale factor is at least 1
                stats.update({"threshold": threshold, "scale_factor": scale_factor, "epsilon": epsilon})
        elif method_lower == "standard":
            mean = float(data.mean())
            std = max(float(data.std()), epsilon) # Avoid zero std dev
            # Scale factor ensures output is roughly within [-1, 1]
            scale_factor = max(abs(global_max - mean), abs(global_min - mean)) / std if not is_constant else 1.0
            stats.update({"mean": mean, "std": std, "scale_factor": max(scale_factor, 1.0)}) # Ensure scale_factor >= 1
        elif method_lower == "none":
            pass # No stats needed for 'none' method
        else:
            raise ValueError(f"Unsupported normalization method: '{method}'")

        return stats

    # ------------------------------------------------------------------
    # Apply normalization / denormalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_tensor(x: Tensor, method: str, stats: Dict[str, float]) -> Tensor:
        """Normalizes a tensor using the specified method and stats."""
        if x.numel() == 0 or method == "none" or stats.get("is_constant"):
            return x # Passthrough for empty, 'none', or constant data

        method_lower = method.lower().strip()
        epsilon = stats.get("epsilon", 1e-9) # Get epsilon from stats if available

        if method_lower == "iqr":
            normed = (x - stats["median"]) / stats["iqr"]
        elif method_lower == "log-min-max":
            x_shifted = x + stats.get("offset", 0.0)
            log_x = torch.log10(torch.clamp(x_shifted, min=epsilon))
            denom = max(stats["max"] - stats["min"], epsilon)
            normed = torch.clamp((log_x - stats["min"]) / denom, 0.0, 1.0)
        elif method_lower == "arctan-compression":
            normed = (2 / math.pi) * torch.atan(stats["alpha"] * (x - stats["center"]) / stats["scale"])
        elif method_lower == "max-out":
            normed = x / stats["max_val"]
        elif method_lower == "invlogit-compression":
            normed = 2 * torch.sigmoid((x - stats["mean"]) / stats["std"]) - 1
        elif method_lower == "custom":
            pos = torch.log10(torch.clamp(x, min=0) + 1 + epsilon)
            neg = -torch.log10(torch.clamp(-x, min=0) + 1 + epsilon)
            y = torch.where(x >= 0, pos, neg)
            normed = y / stats["m"]
        elif method_lower == "symlog":
            thr, sf = stats["threshold"], stats["scale_factor"]
            abs_x = torch.abs(x)
            linear = abs_x <= thr
            logarithmic = ~linear
            normed = torch.zeros_like(x)
            normed[linear] = x[linear] / thr
            # Clamp log argument safely away from zero
            safe_log_arg = torch.clamp(abs_x[logarithmic] / thr, min=epsilon)
            normed[logarithmic] = torch.sign(x[logarithmic]) * (torch.log10(safe_log_arg) + 1)
            normed = torch.clamp(normed / sf, -1.0, 1.0) # Clamp output to [-1, 1]
        elif method_lower == "standard":
            normed = torch.clamp(((x - stats["mean"]) / stats["std"]) / stats["scale_factor"], -1.0, 1.0)
        else:
            raise ValueError(f"Unsupported normalization method: '{method}'")

        # Ensure output has same dtype as input (should be float32)
        return normed.to(x.dtype)


    @staticmethod
    def denormalize(
        v: Union[Tensor, List[float], float],
        metadata: Dict[str, Any],
        var_name: str,
    ) -> Union[Tensor, List[float], float]:
        """Denormalizes values using stored metadata for a specific variable."""
        # Extract method and stats for the variable
        methods = metadata.get("normalization_methods", {})
        method = methods.get(var_name, "none").lower().strip()
        if method == "none":
            return v # Passthrough

        stats = metadata.get(var_name)
        if stats is None:
            raise KeyError(f"Normalization statistics not found for variable '{var_name}'")
        if stats.get("is_constant"):
            # If original was constant, return the original constant value if possible,
            # otherwise return the input 'v' as denormalized value isn't well-defined.
            # Returning v is safer than trying to guess the constant.
            logger.debug(f"Variable '{var_name}' was constant, returning input value during denormalization.")
            return v


        # Convert input to tensor for processing
        is_scalar = not isinstance(v, (list, Tensor))
        x = v if isinstance(v, Tensor) else _to_tensor(v)
        original_dtype = x.dtype
        x = x.float() # Ensure float for calculations

        epsilon = stats.get("epsilon", 1e-9)

        # Apply inverse transformation based on method
        if method == "iqr":
            denormed = x * stats["iqr"] + stats["median"]
        elif method == "log-min-max":
            log_val = x * (stats["max"] - stats["min"]) + stats["min"]
            denormed = torch.pow(10, log_val) - stats.get("offset", 0.0)
        elif method == "arctan-compression":
            # Clamp input to avoid tan blowing up near +/- 1
            safe_x = torch.clamp(x, -0.999999, 0.999999)
            denormed = stats["center"] + (stats["scale"] / stats["alpha"]) * torch.tan((math.pi / 2) * safe_x)
        elif method == "max-out":
            denormed = x * stats["max_val"]
        elif method == "invlogit-compression":
            # Clamp input to avoid issues with logit of +/- 1
            safe_x = torch.clamp(x, -0.999999, 0.999999)
            # logit(p) = log(p / (1 - p)) where p = (safe_x + 1) / 2
            p = (safe_x + 1) / 2.0
            logit_p = torch.log(p / torch.clamp(1 - p, min=epsilon))
            denormed = stats["mean"] + stats["std"] * logit_p
        elif method == "custom":
            y = x * stats["m"]
            pos = torch.pow(10, y) - 1 - epsilon
            neg = -(torch.pow(10, -y) - 1 - epsilon)
            denormed = torch.where(y >= 0, pos, neg)
        elif method == "symlog":
            thr, sf = stats["threshold"], stats["scale_factor"]
            unscaled = x * sf
            linear = torch.abs(unscaled) <= 1.0 # Linear region is where |transformed| <= 1
            logarithmic = ~linear
            denormed = torch.zeros_like(x)
            denormed[linear] = unscaled[linear] * thr
            # Clamp exponent base to avoid potential overflow/underflow issues
            safe_exponent = torch.clamp(torch.abs(unscaled[logarithmic]) - 1, min=-50, max=50) # Clamp exponent range
            denormed[logarithmic] = torch.sign(unscaled[logarithmic]) * thr * torch.pow(10, safe_exponent)
        elif method == "standard":
            denormed = (x * stats["scale_factor"]) * stats["std"] + stats["mean"]
        else:
            raise ValueError(f"Unsupported denormalization method: '{method}'")

        # Return in original format (scalar, list, or tensor) and dtype
        denormed = denormed.to(original_dtype)
        if is_scalar:
            return denormed.item()
        elif isinstance(v, list):
            return denormed.tolist()
        else:
            return denormed

    # ------------------------------------------------------------------
    # Process all profiles
    # ------------------------------------------------------------------

    def process_profiles(self, stats: Dict[str, Any]) -> None:
        """
        Applies normalization to all profiles in the input directory
        using the provided statistics, saving results to the output directory.
        """
        logger.info(f"Applying normalization and saving profiles to {self.output_dir}...")
        methods = stats.get("normalization_methods", {})
        if not methods:
            logger.error("No normalization methods found in provided statistics.")
            return

        processed_count = 0
        error_count = 0
        files = [p for p in self.input_dir.glob("*.json") if p.name != "normalization_metadata.json"]

        for fpath in files:
            try:
                prof = json.loads(fpath.read_text())
                normalized_profile = {}
                for key, value in prof.items():
                    method = methods.get(key, "none").lower() # Default to 'none' if key missing
                    if method == "none" or isinstance(value, bool) or value is None:
                        normalized_profile[key] = value
                        continue

                    key_stats = stats.get(key)
                    if key_stats is None:
                         logger.warning(f"No stats found for key '{key}' in {fpath.name}, skipping normalization.")
                         normalized_profile[key] = value
                         continue

                    try:
                        is_list = isinstance(value, list)
                        tensor_val = _to_tensor(value)
                        if tensor_val.numel() > 0: # Check if tensor is not empty
                             normed_tensor = self.normalize_tensor(tensor_val, method, key_stats)
                             # Convert back to original type (list or scalar)
                             normalized_profile[key] = normed_tensor.tolist() if is_list else normed_tensor.item()
                        else:
                             normalized_profile[key] = value # Keep empty list as is
                    except Exception as norm_exc:
                         logger.warning(f"Failed to normalize '{key}' in {fpath.name} with method '{method}': {norm_exc}. Keeping original value.")
                         normalized_profile[key] = value

                # Save the normalized profile
                output_path = self.output_dir / fpath.name
                save_json(normalized_profile, output_path) # Use save_json for consistency
                processed_count += 1

            except Exception as file_exc:
                logger.error(f"Error processing file {fpath.name}: {file_exc}")
                error_count += 1

        logger.info(f"Normalization processing complete. Saved {processed_count} normalized profiles.")
        if error_count > 0:
            logger.warning(f"Encountered errors in {error_count} files during processing.")