#!/usr/bin/env python3
"""normalizer.py – Compute and invert eight normalization schemes.

Supported methods (case-insensitive):
--------------------------------------------------------------
* `iqr`                 – median / IQR scaling -> R.
* `log-min-max`         – log10 -> min-max to [0, 1]. **Requires strictly positive data.**
* `arctan-compression`  – arctan -pi maps to pi -> [-1, 1].
* `max-out`             – divide by global |max|  -> [-1, 1].
* `invlogit-compression`– sigmoid -> [-1, 1].
* `custom`              – sign-preserving log10 (He et al.).
* `symlog`              – linear near 0, log10 tails  -> [-1, 1].
* `standard`            – z-score then scale to [-1, 1].

The class guarantees that for any numeric value `x`,
`denormalize(normalize(x)) == x` (within fp tolerance, ignoring clamping effects).
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

# Assuming utils.py is available for save_json
try:
    from utils import save_json
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import save_json from utils, saving metadata with standard json.")
    save_json = None # Define a fallback or handle absence

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
        "none",
    }

    def __init__(
        self,
        input_folder: Union[str, Path],
        output_folder: Union[str, Path],
        batch_size: int = 100,
    ):
        """
        Initializes the normalizer.

        Args:
            input_folder: Path to directory with raw JSON profiles.
            output_folder: Path to directory where normalized profiles and metadata will be saved.
            batch_size: Number of files to process per batch when calculating stats.

        Raises:
            FileNotFoundError: If the input folder does not exist.
        """
        self.input_dir = Path(input_folder)
        self.output_dir = Path(output_folder)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input folder not found: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = max(batch_size, 1)
        logger.info(
            f"DataNormalizer input: {self.input_dir}, output: {self.output_dir}"
        )

    # ------------------------------------------------------------------
    # statistics calculation
    # ------------------------------------------------------------------

    @staticmethod
    def clip_outliers(
        t: Tensor, low_quantile: float = 0.001, high_quantile: float = 0.999
    ) -> Tensor:
        """Clips tensor values outside the specified quantile range."""
        if t.numel() == 0:
            return t
        if not (0 <= low_quantile < high_quantile <= 1):
            raise ValueError("Quantiles must satisfy 0 <= low < high <= 1")
        # Ensure tensor is float for quantile calculation
        t_float = t.float()
        low_bound = torch.quantile(t_float, low_quantile)
        high_bound = torch.quantile(t_float, high_quantile)
        return torch.clamp(t, min=low_bound, max=high_bound)

    def calculate_global_stats(
        self,
        key_methods: Optional[Dict[str, str]] = None,
        default_method: str = "standard",
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

        Raises:
            FileNotFoundError: If no JSON profiles are found in the input directory.
            ValueError: If an invalid normalization method is specified, or if 'log-min-max'
                        is applied to non-positive data.
            RuntimeError: If statistics calculation fails for other reasons.
        """
        logger.info("Calculating global normalization statistics...")
        files = [
            p
            for p in self.input_dir.glob("*.json")
            if p.name != "normalization_metadata.json"
        ]
        if not files:
            raise FileNotFoundError(
                f"No JSON profiles found in input folder: {self.input_dir}"
            )

        value_buffer: Dict[str, List[Tensor]] = {}
        boolean_keys: set[str] = set()
        processed_files = 0
        num_batches = (len(files) + self.batch_size - 1) // self.batch_size

        for i in range(num_batches):
            batch_files = files[i * self.batch_size : (i + 1) * self.batch_size]
            for fpath in batch_files:
                try:
                    # Use utf-8-sig to handle potential BOM (Byte Order Mark)
                    prof = json.loads(fpath.read_text(encoding='utf-8-sig'))
                    processed_files += 1
                    for k, v in prof.items():
                        if v is None or (
                            isinstance(v, float) and np.isnan(v)
                        ):
                            continue
                        if isinstance(v, bool):
                            boolean_keys.add(k)
                        elif isinstance(v, (int, float)):
                            value_buffer.setdefault(k, []).append(
                                torch.tensor([float(v)], dtype=torch.float32)
                            )
                        elif isinstance(v, list) and all(
                            isinstance(x, (int, float))
                            for x in v
                            if not (isinstance(x, float) and np.isnan(x))
                        ):
                            valid_vals = [
                                float(x)
                                for x in v
                                if not (isinstance(x, float) and np.isnan(x))
                            ]
                            if valid_vals:
                                value_buffer.setdefault(k, []).append(
                                    torch.tensor(valid_vals, dtype=torch.float32)
                                )
                except json.JSONDecodeError as e:
                     logger.error(f"JSON decode error in file {fpath.name}: {e}. Stopping normalization.")
                     raise RuntimeError(f"Failed to decode JSON in {fpath.name}") from e
                except Exception as e:
                    logger.warning(f"Could not process file {fpath.name}: {e}")
                    continue # Allow skipping minor processing errors, but decode errors are fatal

        logger.info(f"Scanned {processed_files} profiles to gather statistics.")

        for bk in boolean_keys:
            value_buffer.pop(bk, None)

        methods = {k: m.lower().strip() for k, m in (key_methods or {}).items()}
        default_method_lower = default_method.lower().strip()
        if default_method_lower not in self.METHODS:
            raise ValueError(f"Unknown default_method '{default_method}'")

        all_numeric_keys = set(value_buffer.keys())
        final_methods = {
            k: methods.get(k, default_method_lower) for k in all_numeric_keys
        }
        for k in boolean_keys:
            final_methods[k] = "none"

        stats: Dict[str, Any] = {}
        computed_keys = 0
        for key, tensor_list in value_buffer.items():
            if not tensor_list:
                logger.warning(
                    f"No valid numeric data found for key '{key}', skipping stats calculation."
                )
                continue

            try:
                full_vector = torch.cat(tensor_list)
            except RuntimeError as e:
                logger.error(
                    f"Error concatenating tensors for key '{key}': {e}. Sizes: {[t.shape for t in tensor_list]}"
                )
                continue

            if full_vector.numel() == 0:
                logger.warning(
                    f"Empty tensor after concatenation for key '{key}', skipping."
                )
                continue

            if clip_outliers_before_scaling:
                full_vector = DataNormalizer.clip_outliers(full_vector)

            method_to_use = final_methods[key]
            if method_to_use not in self.METHODS:
                logger.warning(
                    f"Invalid method '{method_to_use}' specified for '{key}', defaulting to '{default_method_lower}'."
                )
                method_to_use = default_method_lower
                final_methods[key] = method_to_use

            try:
                # Compute stats, potentially raising ValueError for log-min-max
                key_stats = self._compute_stats_for_key(
                    full_vector,
                    method_to_use,
                    symlog_percentile,
                    (symlog_thresholds or {}).get(key),
                )
                stats[key] = key_stats
                computed_keys += 1
            except ValueError as ve:
                # Specifically catch ValueErrors from _compute_stats_for_key (e.g., log-min-max)
                logger.error(f"Stats calculation failed for key '{key}' with method '{method_to_use}': {ve}")
                raise # Re-raise to stop the normalization process
            except Exception as e:
                logger.error(
                    f"Unexpected error computing stats for key '{key}' using method '{method_to_use}': {e}"
                )
                # Set method to 'none' if stats fail for other reasons, but log-min-max error is fatal
                final_methods[key] = "none"
                stats[key] = {} # Add empty stats dict for consistency


        metadata = {
            "normalization_methods": final_methods,
            "config": {
                "clip_outliers_before_scaling": clip_outliers_before_scaling,
                "default_method": default_method_lower,
                "key_methods_provided": key_methods or {},
                "symlog_percentile": symlog_percentile,
                "symlog_thresholds_provided": symlog_thresholds or {},
                "boolean_keys_detected": list(boolean_keys),
                "numeric_keys_processed": list(stats.keys()),
            },
            **stats,
        }

        meta_path = self.output_dir / "normalization_metadata.json"
        try:
            if save_json:
                 save_success = save_json(metadata, meta_path)
                 if not save_success:
                     logger.error("Failed to save normalization metadata using utils.save_json.")
                     # Optionally raise an error here if saving metadata is critical
            else:
                 # Fallback to standard json saving
                 meta_path.write_text(json.dumps(metadata, indent=2))
                 logger.info("Saved normalization metadata using standard json.")

            logger.info(
                f"Saved normalization metadata ({computed_keys} numeric keys) to {meta_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save normalization metadata: {e}")
            # Optionally raise an error here

        return metadata

    def _compute_stats_for_key(
        self,
        data: Tensor,
        method: str,
        symlog_percentile: float = 0.5,
        symlog_threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Computes statistics needed for a specific normalization method.

        Raises:
            ValueError: If data is empty, or if 'log-min-max' is applied to non-positive data.
        """
        if data.numel() == 0:
            raise ValueError("Cannot compute statistics on empty tensor.")

        # Ensure data is float for calculations
        data = data.float()
        global_min, global_max = float(data.min()), float(data.max())
        stats: Dict[str, float] = {
            "global_min": global_min,
            "global_max": global_max,
        }
        is_constant = (global_max - global_min) < 1e-9
        if is_constant:
            stats["is_constant"] = 1.0

        method_lower = method.lower().strip()
        epsilon = 1e-10

        if method_lower == "iqr":
            q1 = float(torch.quantile(data, 0.25))
            q3 = float(torch.quantile(data, 0.75))
            iqr = q3 - q1
            stats.update(
                {
                    "median": float(torch.median(data).values),
                    "iqr": max(iqr, epsilon),
                }
            )
        elif method_lower == "log-min-max":
            # --- Strict Positive Check ---
            if global_min <= 0:
                raise ValueError(
                    f"Method 'log-min-max' requires strictly positive data, but found minimum value: {global_min:.4e}. "
                    f"Consider using 'symlog' or check the data source."
                )
            # --- Removed Offset Logic ---
            log_vals = torch.log10(data) # No clamp needed due to check above
            log_min, log_max = float(log_vals.min()), float(log_vals.max())
            stats.update({"min": log_min, "max": log_max}) # Removed offset from stats
        elif method_lower == "arctan-compression":
            center = (
                0.0
                if (global_min < 0 and global_max > 0)
                else (global_min + global_max) / 2.0
            )
            scale = max((global_max - global_min) / 2.0, epsilon)
            alpha = math.tan(0.99 * math.pi / 2)
            stats.update({"center": center, "scale": scale, "alpha": alpha})
        elif method_lower == "max-out":
            max_abs_val = max(abs(global_min), abs(global_max), epsilon)
            stats["max_val"] = max_abs_val
        elif method_lower == "invlogit-compression":
            stats.update(
                {"mean": float(data.mean()), "std": max(float(data.std()), epsilon)}
            )
        elif method_lower == "custom":
            m_pos = (
                math.log10(global_max + 1 + epsilon)
                if global_max > -epsilon
                else 0.0
            )
            m_neg = (
                math.log10(abs(global_min) + 1 + epsilon)
                if global_min < epsilon
                else 0.0
            )
            stats.update({"m": max(m_pos, m_neg, 1.0), "epsilon": epsilon})
        elif method_lower == "symlog":
            if is_constant:
                stats.update(
                    {"threshold": 1.0, "scale_factor": 1.0, "epsilon": epsilon}
                )
            else:
                abs_data = torch.abs(data)
                if symlog_threshold is None:
                    threshold = max(
                        float(torch.quantile(abs_data, symlog_percentile)),
                        epsilon,
                    )
                else:
                    threshold = max(symlog_threshold, epsilon)

                linear_mask = abs_data <= threshold
                log_mask = ~linear_mask
                transformed = torch.zeros_like(data)
                transformed[linear_mask] = data[linear_mask] / threshold
                safe_log_arg = torch.clamp(
                    abs_data[log_mask] / threshold, min=epsilon
                )
                transformed[log_mask] = torch.sign(data[log_mask]) * (
                    torch.log10(safe_log_arg) + 1
                )

                scale_factor = max(
                    float(torch.max(torch.abs(transformed))), 1.0
                )
                stats.update(
                    {
                        "threshold": threshold,
                        "scale_factor": scale_factor,
                        "epsilon": epsilon,
                    }
                )
        elif method_lower == "standard":
            mean = float(data.mean())
            std = max(float(data.std()), epsilon)
            # Calculate scale factor based on z-scores' max absolute deviation
            z_scores = (data - mean) / std if not is_constant else torch.zeros_like(data)
            # Use 3-sigma range for scaling unless max deviation is larger
            scale_factor = max(3.0, float(torch.max(torch.abs(z_scores)))) if not is_constant else 1.0
            stats.update(
                {"mean": mean, "std": std, "scale_factor": max(scale_factor, epsilon)} # Ensure scale factor > 0
            )
        elif method_lower == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalization method: '{method}'")

        return stats

    # ------------------------------------------------------------------
    # Apply normalization / denormalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_tensor(
        x: Tensor, method: str, stats: Dict[str, float]
    ) -> Tensor:
        """Normalizes a tensor using the specified method and stats."""
        if x.numel() == 0 or method == "none" or stats.get("is_constant"):
            return x

        method_lower = method.lower().strip()
        epsilon = stats.get("epsilon", 1e-9)

        if method_lower == "iqr":
            normed = (x - stats["median"]) / stats["iqr"]
        elif method_lower == "log-min-max":
            # --- Removed Offset Logic ---
            # Assumes x is already positive due to check in _compute_stats_for_key
            log_x = torch.log10(x)
            denom = max(stats["max"] - stats["min"], epsilon)
            normed = torch.clamp((log_x - stats["min"]) / denom, 0.0, 1.0)
        elif method_lower == "arctan-compression":
            normed = (2 / math.pi) * torch.atan(
                stats["alpha"] * (x - stats["center"]) / stats["scale"]
            )
        elif method_lower == "max-out":
            normed = x / stats["max_val"]
        elif method_lower == "invlogit-compression":
            # Normalize: z = (x - mean) / std
            z = (x - stats["mean"]) / stats["std"]
            # Apply sigmoid: 2 * sigmoid(z) - 1
            normed = 2 * torch.sigmoid(z) - 1
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
            safe_log_arg = torch.clamp(abs_x[logarithmic] / thr, min=epsilon)
            normed[logarithmic] = torch.sign(x[logarithmic]) * (
                torch.log10(safe_log_arg) + 1
            )
            normed = torch.clamp(normed / sf, -1.0, 1.0)
        elif method_lower == "standard":
            # Clamp output to [-1, 1] after scaling z-score
            normed = torch.clamp(
                ((x - stats["mean"]) / stats["std"]) / stats["scale_factor"],
                -1.0,
                1.0,
            )
        else:
            raise ValueError(f"Unsupported normalization method: '{method}'")

        return normed.to(x.dtype)

    @staticmethod
    def denormalize(
        v: Union[Tensor, List[float], float],
        metadata: Dict[str, Any],
        var_name: str,
    ) -> Union[Tensor, List[float], float]:
        """
        Denormalizes values using stored metadata for a specific variable.

        Note: Inverse transformations might not perfectly recover original values
        if clamping occurred during normalization (e.g., in 'standard', 'symlog',
        'log-min-max').
        """
        methods = metadata.get("normalization_methods", {})
        method = methods.get(var_name, "none").lower().strip()
        if method == "none":
            return v

        stats = metadata.get(var_name)
        if stats is None:
            # Check if stats calculation failed earlier and method was set to none
            if methods.get(var_name) == "none":
                 logger.warning(f"Stats missing for '{var_name}' (likely failed calculation), returning original value.")
                 return v
            raise KeyError(
                f"Normalization statistics not found for variable '{var_name}'"
            )
        if stats.get("is_constant"):
            # Return the original constant value stored during stats calculation
            original_value = stats.get("global_min", stats.get("global_max"))
            if original_value is not None:
                logger.debug(f"Variable '{var_name}' was constant, returning original constant value: {original_value}")
                # Replicate input structure (scalar, list, tensor) with the constant value
                if isinstance(v, Tensor):
                    return torch.full_like(v, fill_value=original_value, dtype=v.dtype)
                elif isinstance(v, list):
                    return [original_value] * len(v)
                else: # Scalar
                    return original_value
            else:
                logger.warning(f"Variable '{var_name}' was constant, but original value not found in stats. Returning input.")
                return v


        is_scalar = not isinstance(v, (list, Tensor))
        x = v if isinstance(v, Tensor) else _to_tensor(v)
        original_dtype = x.dtype
        x = x.float()

        epsilon = stats.get("epsilon", 1e-9)

        if method == "iqr":
            denormed = x * stats["iqr"] + stats["median"]
        elif method == "log-min-max":
            # Note: Clamping in normalize limits the range, inverse might not reach original min/max
            log_val = x * (stats["max"] - stats["min"]) + stats["min"]
            # --- Removed Offset Logic ---
            denormed = torch.pow(10, log_val)
        elif method == "arctan-compression":
            safe_x = torch.clamp(x, -0.999999, 0.999999)
            denormed = stats["center"] + (
                stats["scale"] / stats["alpha"]
            ) * torch.tan((math.pi / 2) * safe_x)
        elif method == "max-out":
            denormed = x * stats["max_val"]
        elif method == "invlogit-compression":
            # --- CORRECTED INVERSE LOGIC ---
            # Input x is in range [-1, 1] (approx)
            # 1. Inverse the sigmoid scaling: p = (x + 1) / 2
            safe_x = torch.clamp(x, -0.999999, 0.999999) # Ensure x is in valid range for p
            p = (safe_x + 1) / 2.0
            # 2. Calculate the inverse sigmoid (logit): logit_p = log(p / (1-p))
            logit_p = torch.log(p / torch.clamp(1 - p, min=epsilon))
            # 3. Inverse the z-score scaling: original = mean + std * logit_p
            denormed = stats["mean"] + stats["std"] * logit_p
        elif method == "custom":
            y = x * stats["m"]
            pos = torch.pow(10, y) - 1 - epsilon
            neg = -(torch.pow(10, -y) - 1 - epsilon)
            denormed = torch.where(y >= 0, pos, neg)
        elif method == "symlog":
            # Note: Clamping in normalize limits the range
            thr, sf = stats["threshold"], stats["scale_factor"]
            unscaled = x * sf
            linear = torch.abs(unscaled) <= 1.0
            logarithmic = ~linear
            denormed = torch.zeros_like(x)
            denormed[linear] = unscaled[linear] * thr
            safe_exponent = torch.clamp(
                torch.abs(unscaled[logarithmic]) - 1, min=-50, max=50 # Avoid extreme exponents
            )
            denormed[logarithmic] = (
                torch.sign(unscaled[logarithmic])
                * thr
                * torch.pow(10, safe_exponent)
            )
        elif method == "standard":
            # --- CORRECTED INVERSE LOGIC ---
            # Input x is in range [-1, 1] (approx)
            # 1. Inverse the clamping/scaling factor: z_approx = x * scale_factor
            z_approx = x * stats["scale_factor"]
            # 2. Inverse the z-score: original = mean + std * z_approx
            denormed = (z_approx * stats["std"]) + stats["mean"]
        else:
            raise ValueError(f"Unsupported denormalization method: '{method}'")

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

        Raises:
            RuntimeError: If processing fails for a file after stats calculation.
        """
        logger.info(
            f"Applying normalization and saving profiles to {self.output_dir}..."
        )
        methods = stats.get("normalization_methods", {})
        if not methods:
            logger.error("No normalization methods found in provided statistics.")
            raise RuntimeError("Cannot process profiles without normalization methods.")

        processed_count = 0
        error_count = 0
        files = [
            p
            for p in self.input_dir.glob("*.json")
            if p.name != "normalization_metadata.json"
        ]

        for fpath in files:
            try:
                # Use utf-8-sig to handle potential BOM
                prof = json.loads(fpath.read_text(encoding='utf-8-sig'))
                normalized_profile = {}
                for key, value in prof.items():
                    method = methods.get(key, "none").lower()
                    if method == "none" or isinstance(value, bool) or value is None:
                        normalized_profile[key] = value
                        continue

                    key_stats = stats.get(key)
                    if key_stats is None:
                         logger.error(f"Logic error: Stats missing for key '{key}' with method '{method}' in {fpath.name}. Skipping normalization.")
                         normalized_profile[key] = value # Keep original if stats somehow missing
                         continue

                    try:
                        is_list = isinstance(value, list)
                        tensor_val = _to_tensor(value)
                        if tensor_val.numel() > 0:
                            normed_tensor = self.normalize_tensor(
                                tensor_val, method, key_stats
                            )
                            # Convert back to list/scalar and handle potential NaN/Inf after normalization
                            if is_list:
                                normed_list = normed_tensor.tolist()
                                # Replace non-finite values with None or raise error? Decide policy.
                                # For now, keep them as they might be handled later or indicate issues.
                                normalized_profile[key] = [v if np.isfinite(v) else None for v in normed_list]
                            else:
                                normed_item = normed_tensor.item()
                                normalized_profile[key] = normed_item if np.isfinite(normed_item) else None
                        else:
                            normalized_profile[key] = value # Keep original empty list/value
                    except Exception as norm_exc:
                        logger.warning(
                            f"Failed to normalize '{key}' in {fpath.name} with method '{method}': {norm_exc}. Keeping original value."
                        )
                        normalized_profile[key] = value

                output_path = self.output_dir / fpath.name
                try:
                    if save_json:
                         save_success = save_json(normalized_profile, output_path)
                         if not save_success:
                              logger.error(f"Failed to save normalized profile {output_path.name} using utils.save_json.")
                              # Decide if this should be a fatal error
                    else:
                         # Fallback to standard json saving
                         output_path.write_text(json.dumps(normalized_profile, indent=2))

                except Exception as save_exc:
                     logger.error(f"Failed to save normalized profile {output_path.name}: {save_exc}")
                     # Decide if this should be a fatal error

                processed_count += 1

            except json.JSONDecodeError as file_exc:
                 logger.error(f"JSON decode error processing file {fpath.name}: {file_exc}. Stopping.")
                 raise RuntimeError(f"Fatal JSON decode error in {fpath.name}") from file_exc
            except Exception as file_exc:
                logger.error(f"Error processing file {fpath.name}: {file_exc}")
                error_count += 1 # Count non-fatal errors per file

        logger.info(
            f"Normalization processing complete. Saved {processed_count} normalized profiles."
        )
        if error_count > 0:
            logger.warning(
                f"Encountered non-fatal errors in {error_count} files during processing."
            )

