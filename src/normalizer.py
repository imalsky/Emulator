#!/usr/bin/env python3
"""
normalizer.py -- compute *and invert* normalization schemes,
plus a simple boolean converter, using a memory-efficient streaming approach
for global statistics calculation.
Keys for processing can be filtered based on a configuration dictionary.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set

import numpy as np
import torch
from torch import Tensor

try:
    from utils import save_json
except ImportError:
    save_json = None

logger = logging.getLogger(__name__)

__all__ = ["DataNormalizer"]

# --- Helper Utilities ---

def _to_tensor(v: Union[List[float], float]) -> Tensor:
    """Converts a scalar or a Python list of floats to a 1-D float32 tensor."""
    if isinstance(v, list) and any(x is None for x in v):
        raise ValueError("Cannot convert list containing None to tensor. Pre-filter Nones or handle them before this step.")
    return torch.tensor(v if isinstance(v, list) else [v], dtype=torch.float32)


def _np_quantile(t: Tensor, q: float) -> float:
    """
    Calculates the q-th quantile of a tensor using NumPy with float64 precision.
    This is generally safer for large tensors or tensors with extreme values
    to avoid precision issues.
    """
    if t.numel() == 0:
        logger.warning("Attempted to calculate quantile on an empty tensor. Returning 0.0.")
        return 0.0
    # Convert tensor to a NumPy array on CPU with float64 type for quantile calculation
    return float(np.quantile(t.cpu().numpy().astype(np.float64), q))


# --- Main DataNormalizer Class ---

class DataNormalizer:
    """
    Computes global statistics by streaming data on a key-by-key basis
    to conserve memory. It normalizes data using various methods and
    can invert these transformations.

    If a configuration dictionary is provided during initialization, only keys
    specified in 'input_variables', 'target_variables', or 'global_variables'
    will be considered for statistics calculation and normalization. Other keys
    from input profiles will be omitted from the normalized output.

    Mathematical Formulas for Normalization Methods:
    (eps denotes a small epsilon value for numerical stability)

    1.  IQR (Interquartile Range)
        - Stats: Q1, Q3, median, iqr_val = max(Q3 - Q1, eps)
        - Normalize: y = (x - median) / iqr_val
        - Denormalize: x_orig = (y * iqr_val) + median

    2.  Log-Min-Max
        - Pre-condition: All input values x must be > 0 (or > eps after filtering).
        - Stats: log_min, log_max (of log10(data))
        - Normalize: y = clamp((log10(clamp(x,min=eps)) - log_min) / max(log_max - log_min, eps), 0.0, 1.0)
        - Denormalize: x_orig = 10^(y * (log_max - log_min) + log_min)

    3.  Max-Out
        - Stats: max_abs_val = max(|global_min_data|, |global_max_data|, eps)
        - Normalize: y = x / max_abs_val
        - Denormalize: x_orig = y * max_abs_val

    4.  Scaled Signed Offset Log (`scaled_signed_offset_log`)
        - Stats: m_pos = log10(max(0, global_max_data) + 1 + eps), m_neg = log10(max(0, -global_min_data) + 1 + eps), m = max(m_pos, m_neg, 1.0)
        - Normalize: y_intermediate = where(x >= 0, log10(clamp(x,0)+1+eps), -log10(clamp(-x,0)+1+eps)); y = y_intermediate / max(m, eps)
        - Denormalize: ytmp = y*m; x_orig = where(ytmp >= 0, 10^ytmp-1-eps, -(10^(-ytmp)-1-eps))

    5.  Symlog (Symmetric Logarithm)
        - Stats: thr (threshold), sf (scale_factor)
        - Normalize: y_calc = (x/thr if |x|<=thr else sign(x)*(log10(clamp(|x|/thr,eps))+1)); y = clamp(y_calc/max(sf,eps), -1, 1)
        - Denormalize: unscaled=y*sf; x_orig = (unscaled*thr if |unscaled|<=1+eps else sign(unscaled)*thr*10^clamp(|unscaled|-1,max=(30-log10(thr) if thr > 1 else 30)))


    6.  Standard (Standardization / Z-score based scaling)
        - Stats: mean, std, scale_factor = max(3.0, max(|(data-mean)/std|))
        - Normalize: y = clamp(((x - mean) / max(std, eps)) / max(scale_factor, eps), -1.0, 1.0)
        - Denormalize: x_orig = (y * scale_factor * std) + mean

    7.  Bool
        - Normalize: y = int(bool(x)) if x is not None else None
        - Denormalize: x_orig = bool(round(abs(float(y)))) if y is not None else None

    8.  None
        - Normalize: y = x
        - Denormalize: x_orig = y
    """

    METHODS = {
        "iqr",
        "log-min-max",
        "max-out",
        "scaled_signed_offset_log",
        "symlog",
        "standard",
        "bool",
        "none",
    }

    def __init__(
        self,
        input_folder: Union[str, Path],
        output_folder: Union[str, Path],
        *,
        config_data: Optional[Dict[str, Any]] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """
        Initializes the DataNormalizer.
        If config_data is provided, it's used to determine allowed keys and epsilon.
        """
        self.input_dir = Path(input_folder)
        self.output_dir = Path(output_folder)
        if not self.input_dir.is_dir():
            logger.error(f"Input folder not found: {self.input_dir}")
            raise FileNotFoundError(f"Input folder not found: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.allowed_keys: Optional[Set[str]] = None # None means process all keys

        if config_data:
            norm_config = config_data.get("normalization", {})
            effective_epsilon = norm_config.get("epsilon", epsilon)

            input_vars = set(config_data.get("input_variables", []))
            target_vars = set(config_data.get("target_variables", []))
            global_vars = set(config_data.get("global_variables", []))
            self.allowed_keys = input_vars.union(target_vars).union(global_vars)

            if not self.allowed_keys:
                 logger.warning(
                    "Config data provided, but 'input_variables', 'target_variables', and 'global_variables' "
                    "resulted in an empty set of allowed keys. Consequently, no keys will be processed for statistics "
                    "or included in the normalized profiles based on these lists. If this is unintended, please check the config."
                 )
        else: 
            effective_epsilon = epsilon
            self.allowed_keys = None 

        self.eps = float(effective_epsilon if effective_epsilon is not None else 1e-10)
        if self.eps <= 0:
            logger.warning(f"Final epsilon is {self.eps}, which is not > 0. Using hard default 1e-10.")
            self.eps = 1e-10

        logger.info(
            "DataNormalizer initialised: input=%s, output=%s, effective_epsilon=%.2e",
            self.input_dir,
            self.output_dir,
            self.eps,
        )
        if self.allowed_keys is not None:
            logger.info(f"Processing will be restricted to {len(self.allowed_keys)} allowed keys from config (if found in data).")
        else:
            logger.info("No specific key filtering from config; all discovered keys will be considered.")


    @staticmethod
    def clip_outliers(t: Tensor, low: float = 0.001, high: float = 0.999) -> Tensor:
        """
        Clips tensor values that fall outside the specified lower (low) and
        upper (high) quantiles.
        """
        if t.numel() == 0:
            logger.debug("Clip_outliers called on an empty tensor. Returning as is.")
            return t
        if not (0 <= low < high <= 1):
            logger.error(f"Invalid quantiles for clipping: low={low}, high={high}")
            raise ValueError("Quantiles must satisfy 0 <= low < high <= 1")

        q_low = _np_quantile(t, low)
        q_high = _np_quantile(t, high)

        if q_low >= q_high:
            logger.warning(f"Quantile bounds collapsed or invalid (q_low={q_low}, q_high={q_high}); skipping clipping.")
            return t

        return torch.clamp(t, min=q_low, max=q_high)

    def calculate_global_stats(
        self,
        *,
        key_methods: Optional[Dict[str, str]] = None,
        default_method: str = "standard",
        clip_outliers_before_scaling: bool = False,
        symlog_percentile: float = 0.5,
        symlog_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates global statistics for allowed keys found in the input JSON files.
        """
        logger.info("Starting calculation of global statistics...")
        json_files = [
            p for p in self.input_dir.glob("*.json")
            if p.name != "normalization_metadata.json"
        ]
        if not json_files:
            raise FileNotFoundError(f"No JSON profiles found in {self.input_dir}")
        logger.info(f"Found {len(json_files)} JSON files to process for statistics.")

        discovered_boolean_keys: Set[str] = set()
        discovered_potential_numeric_keys: Set[str] = set()

        for fpath in json_files:
            try:
                prof_data = json.loads(fpath.read_text(encoding="utf-8-sig"))
                for k, v in prof_data.items():
                    if self.allowed_keys is not None and k not in self.allowed_keys:
                        continue 

                    if isinstance(v, bool): discovered_boolean_keys.add(k)
                    elif isinstance(v, (int, float, list)): discovered_potential_numeric_keys.add(k)
            except json.JSONDecodeError as exc:
                logger.error(f"Bad JSON in {fpath.name} during key discovery: {exc}")
            except Exception as exc:
                logger.error(f"Unexpected error reading {fpath.name} during key discovery: {exc}")
        
        final_numeric_keys_to_consider = discovered_potential_numeric_keys - discovered_boolean_keys
        final_boolean_keys_to_consider = discovered_boolean_keys

        if self.allowed_keys is not None:
            logger.info(f"Discovered {len(final_numeric_keys_to_consider)} numeric and {len(final_boolean_keys_to_consider)} boolean keys among allowed keys to process.")
        else:
            logger.info(f"Discovered {len(final_numeric_keys_to_consider)} numeric and {len(final_boolean_keys_to_consider)} boolean keys (all keys processed).")


        actual_default_method = default_method.lower().strip()
        if actual_default_method not in self.METHODS:
            raise ValueError(f"Unknown default method '{actual_default_method}'")

        final_methods: Dict[str, str] = {}
        user_key_methods = {k: m.lower().strip() for k, m in (key_methods or {}).items()}

        for k_val in final_numeric_keys_to_consider:
            method_to_assign = user_key_methods.get(k_val, actual_default_method)
            if method_to_assign not in self.METHODS:
                logger.warning(f"Key '{k_val}' has unknown method '{method_to_assign}'. Using default '{actual_default_method}'.")
                method_to_assign = actual_default_method
            final_methods[k_val] = method_to_assign

        for k_val in final_boolean_keys_to_consider:
            method_to_assign = user_key_methods.get(k_val, "bool")
            if method_to_assign != "bool":
                logger.warning(f"Key '{k_val}' detected as boolean but method set to '{method_to_assign}'. Forcing to 'bool'.")
                method_to_assign = "bool"
            final_methods[k_val] = method_to_assign
        
        logger.info("Computing statistics for numeric keys (Pass 2)...")
        computed_stats: Dict[str, Any] = {}
        processed_numeric_keys: List[str] = []
        
        for key_name in final_numeric_keys_to_consider:
            logger.info(f"Processing statistics for numeric key: '{key_name}'")
            key_is_actually_numeric = False
            key_values_accumulator: List[float] = []

            for fpath_inner in json_files:
                try:
                    prof_data = json.loads(fpath_inner.read_text(encoding="utf-8-sig"))
                    if key_name not in prof_data: continue
                    
                    val = prof_data[key_name]
                    if val is None: continue
                    
                    current_key_method = final_methods.get(key_name)
                    if isinstance(val, bool):
                        if current_key_method != "bool":
                            raise ValueError(
                                f"Key '{key_name}' (intended for numeric processing) in file {fpath_inner.name} "
                                f"has boolean value, but its assigned method is '{current_key_method}', not 'bool'. "
                            )
                        continue

                    if isinstance(val, list):
                        num_list_elements = []
                        for x_idx, x_val in enumerate(val):
                            if x_val is None: continue
                            if isinstance(x_val, bool):
                                if current_key_method != "bool":
                                    raise ValueError(
                                        f"Key '{key_name}' in {fpath_inner.name} list element [{x_idx}] "
                                        f"is boolean, but method is '{current_key_method}', not 'bool'."
                                    )
                            elif isinstance(x_val, (int, float)) and not np.isnan(x_val):
                                if current_key_method == "log-min-max" and float(x_val) <= self.eps:
                                    logger.warning(
                                        f"Skipping non-positive value {x_val} for log-min-max key "
                                        f"'{key_name}' in list element [{x_idx}] of {fpath_inner.name}"
                                    )
                                    continue
                                num_list_elements.append(float(x_val))
                        if num_list_elements:
                            key_values_accumulator.extend(num_list_elements)
                            key_is_actually_numeric = True
                    elif isinstance(val, (int, float)) and not np.isnan(val):
                        if current_key_method == "log-min-max" and float(val) <= self.eps:
                            logger.warning(
                                f"Skipping non-positive value {val} for log-min-max key "
                                f"'{key_name}' in {fpath_inner.name}"
                            )
                            continue
                        key_values_accumulator.append(float(val))
                        key_is_actually_numeric = True
                except json.JSONDecodeError as exc:
                    logger.warning(f"Bad JSON in {fpath_inner.name} while processing key '{key_name}': {exc}.")
                except Exception as exc: # pylint: disable=broad-except
                    logger.warning(f"Error processing key '{key_name}' in {fpath_inner.name}: {exc}")
            
            if not self._handle_key_data_collection_result(
                key_name, key_is_actually_numeric, key_values_accumulator,
                user_key_methods, final_methods, computed_stats,
                processed_numeric_keys, final_numeric_keys_to_consider, 
                clip_outliers_before_scaling, symlog_percentile, symlog_thresholds
            ):
                continue

        final_boolean_keys_processed_for_stats = []
        for key_name in final_boolean_keys_to_consider:
            if final_methods.get(key_name) == "bool":
                computed_stats[key_name] = {"epsilon": self.eps}
                final_boolean_keys_processed_for_stats.append(key_name)

        metadata = self._assemble_metadata(
            final_methods, clip_outliers_before_scaling, default_method,
            key_methods, symlog_percentile, symlog_thresholds,
            final_boolean_keys_processed_for_stats, 
            processed_numeric_keys, computed_stats
        )
        self._save_metadata(metadata)
        return metadata

    def _handle_key_data_collection_result(
        self, key_name: str, key_is_actually_numeric: bool,
        key_values_accumulator: List[float],
        user_key_methods: Dict[str, str], final_methods: Dict[str, str],
        computed_stats: Dict[str, Any], processed_numeric_keys: List[str],
        active_numeric_keys_set: Set[str], 
        clip_outliers_flag: bool,
        symlog_perc: float, symlog_thresh_map: Optional[Dict[str, float]]
    ) -> bool:
        """
        Helper function to process aggregated data for a key and compute stats.
        Modifies `final_methods`, `computed_stats`, `processed_numeric_keys`, `active_numeric_keys_set`.
        """
        if not key_is_actually_numeric:
            logger.warning(f"No valid numeric data found for key '{key_name}'. Reclassifying or setting method to 'none'.")
            if key_name not in user_key_methods:
                final_methods[key_name] = "none"
            computed_stats[key_name] = {"epsilon": self.eps, "is_constant": 0.0, "global_min": 0.0, "global_max": 0.0}
            if key_name in active_numeric_keys_set: active_numeric_keys_set.remove(key_name)
            if final_methods.get(key_name) != "bool": 
                final_methods[key_name] = "none"
            return False

        if not key_values_accumulator:
            logger.warning(f"Data list for numeric key '{key_name}' is empty. Setting method to 'none'.")
            computed_stats[key_name] = {"epsilon": self.eps, "is_constant": 0.0, "global_min": 0.0, "global_max": 0.0}
            final_methods[key_name] = "none"
            if key_name in active_numeric_keys_set: active_numeric_keys_set.remove(key_name)
            return False

        vec = torch.tensor(key_values_accumulator, dtype=torch.float32)
        del key_values_accumulator

        if clip_outliers_flag and vec.numel() > 0:
            vec = self.clip_outliers(vec)

        method_for_key = final_methods[key_name]
        try:
            if vec.numel() == 0:
                 logger.warning(f"Tensor for key '{key_name}' became empty after clipping. Setting method to 'none'.")
                 current_key_stats = {"is_constant": 0.0, "global_min": 0.0, "global_max": 0.0, "epsilon": self.eps}
                 final_methods[key_name] = "none"
                 if key_name in active_numeric_keys_set: active_numeric_keys_set.remove(key_name)
            else:
                current_key_stats = self._compute_stats_for_key(
                    data=vec, method=method_for_key, key_epsilon=self.eps,
                    symlog_percentile_val=symlog_perc,
                    symlog_threshold_val=(symlog_thresh_map or {}).get(key_name)
                )
                computed_stats[key_name] = current_key_stats
                processed_numeric_keys.append(key_name)
        except ValueError as e:
             logger.error(f"Error computing stats for key '{key_name}' with method '{method_for_key}': {e}. Setting method to 'none'.")
             computed_stats[key_name] = {"epsilon": self.eps, "is_constant": 0.0, "global_min": 0.0, "global_max": 0.0}
             final_methods[key_name] = "none"
             if key_name in active_numeric_keys_set: active_numeric_keys_set.remove(key_name)
        del vec
        return True

    def _assemble_metadata(self, final_methods, clip_flag, default_meth,
                           key_meth_prov, sym_perc, sym_thresh_prov,
                           bool_keys_det, num_keys_proc, comp_stats) -> Dict[str, Any]:
        """Helper function to assemble the final metadata dictionary."""
        logger.info("Assembling normalization metadata...")
        metadata_content = {
            "global_processing_epsilon": self.eps,
            "normalization_methods": final_methods,
            "per_key_stats": comp_stats,
            "config": {
                "clip_outliers_before_scaling": clip_flag,
                "default_method": default_meth,
                "key_methods_provided": key_meth_prov or {},
                "symlog_percentile": sym_perc,
                "symlog_thresholds_provided": sym_thresh_prov or {},
                "boolean_keys_detected": bool_keys_det,
                "numeric_keys_processed": num_keys_proc,
            },
        }
        return metadata_content


    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Saves the metadata dictionary to 'normalization_metadata.json'."""
        meta_path = self.output_dir / "normalization_metadata.json"
        try:
            if save_json:
                save_json(metadata, meta_path)
            else:
                meta_path.write_text(json.dumps(metadata, indent=2))
            logger.info(f"Saved normalization metadata: {meta_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata JSON to {meta_path}: {e}")

    def _compute_stats_for_key(
        self,
        data: Tensor,
        method: str,
        key_epsilon: float,
        symlog_percentile_val: float,
        symlog_threshold_val: Optional[float],
    ) -> Dict[str, Any]:
        """
        Computes normalization-specific statistics for a single key's data tensor.
        """
        if data.numel() == 0:
            return {"is_constant": 0.0, "global_min": 0.0, "global_max": 0.0, "epsilon": key_epsilon}

        data_float = data.float()
        gmin_val, gmax_val = float(data_float.min()), float(data_float.max())
        
        stats: Dict[str, Any] = {
            "global_min": gmin_val,
            "global_max": gmax_val,
            "epsilon": key_epsilon,
        }

        is_constant_val = 0.0
        if data_float.numel() > 0:
            valid_data = data_float[~torch.isnan(data_float)]
            if valid_data.numel() > 0:
                if float(valid_data.min()) == float(valid_data.max()):
                    is_constant_val = 1.0
        
        stats["is_constant"] = is_constant_val
        if is_constant_val == 1.0:
            logger.debug(f"Data for key is constant (all valid values are {gmin_val}).")

        current_method = method.lower().strip()
        if current_method in {"none", "bool"}:
            return stats

        arr64 = None
        def get_arr64():
            nonlocal arr64
            if arr64 is None:
                arr64 = data_float.cpu().numpy().astype(np.float64)
            return arr64
            
        if is_constant_val == 1.0 and current_method not in {"max-out", "scaled_signed_offset_log", "symlog"}:
             logger.debug(f"Method {current_method} with constant data; specific stats might be trivial.")

        if current_method == "iqr":
            if is_constant_val == 1.0:
                stats.update(median=gmin_val, iqr=key_epsilon)
            else:
                q1 = _np_quantile(data_float, 0.25)
                q3 = _np_quantile(data_float, 0.75)
                median_val = _np_quantile(data_float, 0.50)
                stats.update(median=median_val, iqr=max(q3 - q1, key_epsilon))

        elif current_method == "log-min-max":
            if data_float.numel() == 0 or gmin_val <= key_epsilon:
                raise ValueError(
                    f"'log-min-max' requires strictly positive data (> eps={key_epsilon}). "
                    f"Received data with min_val={gmin_val} or empty data for stats computation."
                )
            if is_constant_val == 1.0:
                 log_gmin = math.log10(max(gmin_val, key_epsilon))
                 stats.update(min=log_gmin, max=log_gmin)
            else:
                arr_for_log = get_arr64()
                if np.any(arr_for_log <= 0):
                    raise ValueError("Internal error: Non-positive values found in data for log-min-max despite filtering.")
                log_vals = np.log10(arr_for_log)
                stats.update(min=float(log_vals.min()), max=float(log_vals.max()))

        elif current_method == "max-out":
            stats["max_val"] = max(abs(gmin_val), abs(gmax_val), key_epsilon)

        elif current_method == "scaled_signed_offset_log":
            m_pos = math.log10(max(0, gmax_val) + 1 + key_epsilon)
            m_neg = math.log10(max(0, -gmin_val) + 1 + key_epsilon)
            stats.update(m=max(m_pos, m_neg, 1.0))

        elif current_method == "symlog":
            if is_constant_val == 1.0:
                stats.update(threshold=max(abs(gmin_val), key_epsilon), scale_factor=1.0)
            else:
                abs_arr_np = np.abs(get_arr64())
                threshold_candidate = symlog_threshold_val if symlog_threshold_val is not None \
                                      else float(np.quantile(abs_arr_np, symlog_percentile_val))
                thr = max(threshold_candidate, key_epsilon)
                
                abs_data_tensor = torch.abs(data_float)
                linear_part_mask = abs_data_tensor <= thr
                transformed_tensor = torch.zeros_like(data_float)
                transformed_tensor[linear_part_mask] = data_float[linear_part_mask] / thr
                
                safe_log_arg = torch.clamp(abs_data_tensor[~linear_part_mask] / thr, min=key_epsilon)
                transformed_tensor[~linear_part_mask] = torch.sign(data_float[~linear_part_mask]) * \
                                                     (torch.log10(safe_log_arg) + 1)
                scale_f = float(torch.max(torch.abs(transformed_tensor))) if transformed_tensor.numel() > 0 else 1.0
                stats.update(threshold=thr, scale_factor=max(scale_f, 1.0))

        elif current_method == "standard":
            mu = float(get_arr64().mean())
            sigma = float(get_arr64().std()) if is_constant_val == 0.0 else key_epsilon
            
            scale_val = 1.0
            if is_constant_val == 0.0 and sigma > key_epsilon:
                z_scores = (data_float - mu) / sigma
                scale_val = max(3.0, float(torch.max(torch.abs(z_scores)))) if z_scores.numel() > 0 else 3.0
            
            stats.update(mean=mu, std=max(sigma,key_epsilon), scale_factor=max(scale_val, 1.0))
        else:
            raise ValueError(f"Unsupported method '{current_method}' in _compute_stats_for_key.")
        return stats

    @staticmethod
    def normalize_tensor(x: Tensor, method: str, stats: Dict[str, Any]) -> Tensor:
        """
        Normalizes a tensor `x` using the specified `method` and pre-computed `stats`.
        """
        eps = stats.get("epsilon", 1e-10)
        if x.numel() == 0:
            return x
        if stats.get("is_constant") == 1.0:
            return x

        current_method = method.lower().strip()
        if current_method in {"none", "bool"}:
            return x

        y: Tensor

        if current_method == "iqr":
            median, iqr_val = stats["median"], stats["iqr"]
            y = (x - median) / max(iqr_val, eps)
        elif current_method == "log-min-max":
            log_min, log_max = stats["min"], stats["max"]
            x_safe = torch.clamp(x, min=eps)
            denom = max(log_max - log_min, eps)
            y = torch.clamp((torch.log10(x_safe) - log_min) / denom, 0.0, 1.0)
        elif current_method == "max-out":
            y = x / max(stats["max_val"], eps)
        elif current_method == "scaled_signed_offset_log":
            m_val = stats["m"]
            x_pos = torch.log10(torch.clamp(x, min=0) + 1 + eps)
            x_neg = -torch.log10(torch.clamp(-x, min=0) + 1 + eps)
            y_intermediate = torch.where(x >= 0, x_pos, x_neg)
            y = y_intermediate / max(m_val, eps)
        elif current_method == "symlog":
            thr, sf = stats["threshold"], stats["scale_factor"]
            abs_x = torch.abs(x)
            linear_mask = abs_x <= thr
            y_calc = torch.zeros_like(x, dtype=torch.float32)
            y_calc[linear_mask] = x[linear_mask] / max(thr, eps)
            
            non_linear_abs_x_scaled = abs_x[~linear_mask] / max(thr, eps)
            safe_log_arg = torch.clamp(non_linear_abs_x_scaled, min=eps)
            y_calc[~linear_mask] = torch.sign(x[~linear_mask]) * (torch.log10(safe_log_arg) + 1)
            y = torch.clamp(y_calc / max(sf, eps), -1.0, 1.0)
        elif current_method == "standard":
            mean_val, std_val, scale_factor = stats["mean"], stats["std"], stats["scale_factor"]
            y = torch.clamp(((x - mean_val) / max(std_val, eps)) / max(scale_factor, eps), -1.0, 1.0)
        else:
            raise ValueError(f"Unsupported method '{current_method}' in normalize_tensor.")
        return y.to(x.dtype)

    @staticmethod
    def denormalize(
        v: Union[Tensor, List[float], float, None], 
        metadata: Dict[str, Any],
        var_name: str,
    ) -> Union[Tensor, List[float], float, bool, None]:
        """
        Inverts the normalization for a given variable `var_name`.
        """
        methods_map = metadata.get("normalization_methods", {})
        method = methods_map.get(var_name, "none").lower().strip()

        if method == "bool":
            if isinstance(v, Tensor):
                return (v.float().round().abs() > 0.5).to(dtype=torch.bool)
            if isinstance(v, list):
                return [
                    bool(round(abs(float(x_val)))) if x_val is not None else None
                    for x_val in v
                ]
            return bool(round(abs(float(v)))) if v is not None else None


        if method == "none":
            return v

        if v is None:
            logger.warning(f"Denormalize called with None value for non-bool/non-none key '{var_name}'. Returning None.")
            return None

        all_key_stats = metadata.get("per_key_stats", {})
        key_stats = all_key_stats.get(var_name)

        if not isinstance(key_stats, dict):
            raise KeyError(f"Normalization stats not found or invalid for '{var_name}' in 'per_key_stats'.")

        eps = key_stats.get("epsilon", metadata.get("global_processing_epsilon", 1e-10))

        if key_stats.get("is_constant") == 1.0:
            const_val = key_stats.get("global_min") 
            if const_val is None: return v
            if isinstance(v, Tensor): return torch.full_like(v, const_val, dtype=v.dtype)
            if isinstance(v, list): return [const_val] * len(v)
            return const_val

        is_scalar_input = not isinstance(v, (Tensor, list))
        x_tensor = v if isinstance(v, Tensor) else _to_tensor(v)
        original_dtype = x_tensor.dtype
        x_float = x_tensor.float()
        y_denorm: Tensor

        if method == "iqr":
            median, iqr_val = key_stats["median"], key_stats["iqr"]
            y_denorm = x_float * iqr_val + median
        elif method == "log-min-max":
            log_min, log_max = key_stats["min"], key_stats["max"]
            log_val_reconstructed = x_float * (log_max - log_min) + log_min
            y_denorm = torch.pow(10, log_val_reconstructed)
        elif method == "max-out":
            y_denorm = x_float * key_stats["max_val"]
        elif method == "scaled_signed_offset_log":
            m_val = key_stats["m"]
            ytmp = x_float * m_val
            pow10_ytmp_pos = torch.pow(10, ytmp)
            pow10_ytmp_neg = torch.pow(10, -ytmp)
            val_if_orig_pos = pow10_ytmp_pos - 1.0 - eps
            val_if_orig_neg = -(pow10_ytmp_neg - 1.0 - eps)
            y_denorm = torch.where(ytmp >= 0, val_if_orig_pos, val_if_orig_neg)
        elif method == "symlog":
            thr, sf = key_stats["threshold"], key_stats["scale_factor"]
            unscaled_val = x_float * sf
            abs_unscaled_val = torch.abs(unscaled_val)
            y_calc_denorm = torch.zeros_like(x_float)
            
            linear_mask_denorm = abs_unscaled_val <= (1.0 + eps)
            y_calc_denorm[linear_mask_denorm] = unscaled_val[linear_mask_denorm] * thr
            
            non_linear_mask_denorm = ~linear_mask_denorm
            max_exponent_for_thr = 37.0 
            if thr > 1e-30: 
                 max_exponent_for_thr -= math.log10(thr)

            exponent_val = torch.clamp(
                abs_unscaled_val[non_linear_mask_denorm] - 1.0,
                max=max_exponent_for_thr
            )
            term_pow10 = torch.pow(10, exponent_val)
            y_calc_denorm[non_linear_mask_denorm] = \
                torch.sign(unscaled_val[non_linear_mask_denorm]) * thr * term_pow10
            y_denorm = y_calc_denorm
        elif method == "standard":
            mean_val, std_val, scale_factor = key_stats["mean"], key_stats["std"], key_stats["scale_factor"]
            z_reconstructed = x_float * scale_factor
            y_denorm = mean_val + z_reconstructed * std_val
        else:
            raise ValueError(f"Unsupported method '{method}' for denormalization of '{var_name}'.")

        final_y = y_denorm.to(original_dtype if isinstance(v, Tensor) else torch.float32)
        if is_scalar_input: return final_y.item()
        if isinstance(v, list): return final_y.tolist()
        return final_y

    def process_profiles(self, stats_metadata: Dict[str, Any]) -> None:
        """
        Normalizes all JSON profiles in the input directory.
        If config-based key filtering was active during __init__, only allowed keys
        will be included in the output profiles.
        """
        logger.info(f"Starting profile processing. Output to: {self.output_dir}")
        methods_map = stats_metadata.get("normalization_methods", {})
        if not isinstance(methods_map, dict):
            raise RuntimeError("Stats metadata missing or invalid 'normalization_methods'")

        processed_files, skipped_files = 0, 0
        json_files_to_process = [
            p for p in self.input_dir.glob("*.json") if p.name != "normalization_metadata.json"
        ]
        if not json_files_to_process:
            logger.warning(f"No JSON files found in {self.input_dir} to process.")
            return

        for fpath in json_files_to_process:
            try:
                profile_content = json.loads(fpath.read_text(encoding="utf-8-sig"))
            except Exception as exc: # pylint: disable=broad-except
                logger.error(f"Skipping file {fpath.name} due to read/JSON error: {exc}")
                skipped_files += 1
                continue

            output_profile: Dict[str, Any] = {}
            for key, value in profile_content.items():
                # --- MODIFICATION START ---
                # If config-based key filtering is active (self.allowed_keys is a set),
                # only process and include keys that are in self.allowed_keys.
                if self.allowed_keys is not None and key not in self.allowed_keys:
                    continue # Skip this key entirely from the output profile
                # --- MODIFICATION END ---

                method_for_key = methods_map.get(key, "none").lower().strip()

                if method_for_key == "bool":
                    if value is None: output_profile[key] = None
                    elif isinstance(value, list):
                        output_profile[key] = [
                            int(bool(item)) if item is not None else None
                            for item in value
                        ]
                    else: output_profile[key] = int(bool(value))
                    continue

                if isinstance(value, bool): # This check is after the allowed_keys filter
                    raise ValueError(
                        f"Key '{key}' in file {fpath.name} has a boolean value "
                        f"but its normalization method is '{method_for_key}', not 'bool'."
                    )
                
                # If key was not in methods_map (e.g., it was an allowed key but no stats computed,
                # or no config was used and it's an unassigned key), method_for_key defaults to 'none'.
                # Also, if method is explicitly 'none', or value is None, pass through.
                if method_for_key == "none" or value is None:
                    output_profile[key] = value
                    continue
                
                all_key_stats = stats_metadata.get("per_key_stats", {})
                key_specific_stats = all_key_stats.get(key)

                if not isinstance(key_specific_stats, dict):
                    logger.warning(f"Stats not found for key '{key}' (method='{method_for_key}') in {fpath.name}. Passing raw value.")
                    output_profile[key] = value
                    continue
                
                if "epsilon" not in key_specific_stats:
                    key_specific_stats["epsilon"] = stats_metadata.get("global_processing_epsilon", self.eps)

                try:
                    if not isinstance(value, (int, float, list)):
                        output_profile[key] = value; continue
                    
                    value_for_tensor = value
                    if isinstance(value, list):
                        value_for_tensor = [item for item in value if item is not None]
                        if not value_for_tensor and value: 
                            output_profile[key] = [] 
                            continue
                    
                    if not value_for_tensor and isinstance(value, list):
                         output_profile[key] = []
                         continue
                    if value_for_tensor is None and not isinstance(value, list):
                         output_profile[key] = None
                         continue

                    tensor_value = _to_tensor(value_for_tensor)
                except ValueError as ve_tensor: 
                    logger.warning(f"Could not convert value for key '{key}' in {fpath.name} to tensor: {ve_tensor}. Keeping raw.")
                    output_profile[key] = value; continue
                except Exception as e: # pylint: disable=broad-except
                    logger.warning(f"Could not convert value for key '{key}' in {fpath.name} to tensor: {e}. Keeping raw.")
                    output_profile[key] = value; continue

                if tensor_value.numel() == 0:
                    output_profile[key] = [] if isinstance(value, list) else value 
                    continue

                try:
                    normalized_tensor = self.normalize_tensor(tensor_value, method_for_key, key_specific_stats)
                    if isinstance(value, list):
                        output_profile[key] = normalized_tensor.tolist()
                    else:
                        output_profile[key] = normalized_tensor.item()
                except Exception as exc_norm: # pylint: disable=broad-except
                    logger.warning(f"Failed to normalize key '{key}' in {fpath.name} (method: {method_for_key}): {exc_norm}. Keeping raw.")
                    output_profile[key] = value
            
            output_file_path = self.output_dir / fpath.name
            try:
                if save_json: save_json(output_profile, output_file_path)
                else: output_file_path.write_text(json.dumps(output_profile, indent=2))
                processed_files +=1
            except Exception as e: # pylint: disable=broad-except
                logger.error(f"Failed to write normalized profile {output_file_path.name}: {e}")
                skipped_files +=1
        
        logger.info(f"Profile processing complete. {processed_files} files normalized. {skipped_files} files skipped.")

