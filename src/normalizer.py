#!/usr/bin/env python3
"""
normalizer.py - GPU-aware data normalization with numerical stability.

Calculates global statistics from training samples across multiple HDF5 files
using memory-efficient batch processing with online algorithms.
All logarithmic operations are performed in base 10.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import time

from hardware import setup_device
from utils import DTYPE

logger = logging.getLogger(__name__)

DEFAULT_EPSILON = 1e-9
DEFAULT_QUANTILE_MEMORY_LIMIT = 1_000_000
DEFAULT_SYMLOG_PERCENTILE = 0.5
STATS_CHUNK_SIZE = 8192
NORMALIZED_VALUE_CLAMP = 50.0


class DataNormalizer:
    METHODS = {
        "iqr",
        "log-min-max",
        "max-out",
        "signed-log",
        "scaled_signed_offset_log",
        "symlog",
        "standard",
        "log-standard",
        "bool",
        "none",
    }
    QUANTILE_METHODS = {"iqr", "symlog", "log-min-max"}

    def __init__(self, *, config_data: Dict[str, Any]):
        self.config = config_data
        self.device = setup_device()
        self.norm_config = self.config.get("normalization", {})
        self.eps = float(self.norm_config.get("epsilon", DEFAULT_EPSILON))
        self.keys_to_process, self.key_methods = self._get_keys_and_methods()
        self._approximated_quantile_keys = set()
        logger.info(f"DataNormalizer initialized on device '{self.device}'.")

    def _get_keys_and_methods(self) -> Tuple[Set[str], Dict[str, str]]:
        spec = self.config.get("data_specification", {})
        all_vars = set()
        for key in ["input_variables", "global_variables", "target_variables"]:
            all_vars.update(spec.get(key, []))

        user_key_methods = self.norm_config.get("key_methods", {})
        default_method = self.norm_config.get("default_method", "standard")

        key_methods = {}
        for key in all_vars:
            method = user_key_methods.get(key, default_method).lower()
            if method not in self.METHODS:
                raise ValueError(f"Unsupported method '{method}' for key '{key}'.")
            key_methods[key] = method

        return all_vars, key_methods

    def calculate_stats(
        self, raw_hdf5_paths: List[Path], train_indices: List[Tuple[str, int]]
    ) -> Dict[str, Any]:
        logger.info(
            f"Starting statistics calculation from {len(train_indices)} training samples..."
        )
        start_time = time.time()

        if not train_indices:
            raise ValueError(
                "Cannot calculate statistics from empty training indices. Exiting."
            )

        file_map = {path.stem: path for path in raw_hdf5_paths if path.is_file()}
        if not file_map:
            raise RuntimeError("No valid HDF5 files found. Exiting.")

        available_keys = self._get_available_keys(file_map)
        keys_to_load = self.keys_to_process.intersection(available_keys)

        if len(keys_to_load) != len(self.keys_to_process):
            missing = self.keys_to_process - available_keys
            logger.warning(f"Keys not found in HDF5 and will be skipped: {missing}")

        accumulators = self._initialize_accumulators(keys_to_load)

        dataset_metadata = {
            "total_train_profiles": len(train_indices),
            "variables": list(keys_to_load),
            "sequence_lengths": {
                key: {"min": float("inf"), "max": float("-inf")} for key in keys_to_load
            },
        }

        grouped_indices = self._group_indices_by_file(train_indices)
        for file_stem, indices in grouped_indices.items():
            if file_stem not in file_map:
                logger.warning(f"Skipping unknown file stem '{file_stem}' in indices.")
                continue

            h5_path = file_map[file_stem]
            with h5py.File(h5_path, "r", swmr=True, libver="latest") as hf:
                num_chunks = (len(indices) + STATS_CHUNK_SIZE - 1) // STATS_CHUNK_SIZE

                for i in tqdm(
                    range(0, len(indices), STATS_CHUNK_SIZE),
                    desc=f"Stats for {file_stem}",
                    total=num_chunks,
                    leave=False,
                ):
                    chunk_indices = indices[i : i + STATS_CHUNK_SIZE]

                    # Sort indices for efficient h5py reading
                    chunk_indices_np = np.array(chunk_indices)
                    sorter = np.argsort(chunk_indices_np)
                    sorted_indices = chunk_indices_np[sorter].tolist()

                    batch_of_tensors = {}
                    for key in keys_to_load:
                        if key in hf:
                            data_chunk_np = hf[key][sorted_indices]
                            # For stats calculation, order doesn't matter
                            batch_of_tensors[key] = torch.from_numpy(data_chunk_np).to(
                                device=self.device, dtype=DTYPE
                            )

                    self._update_accumulators_with_batch(
                        batch_of_tensors, accumulators, dataset_metadata
                    )

        logger.info(f"Finished statistics calculation in {time.time() - start_time:.2f}s.")

        computed_stats = self._finalize_stats(accumulators)
        if not computed_stats:
            raise RuntimeError("No statistics computed due to invalid data. Exiting.")

        metadata = {
            "normalization_methods": self.key_methods,
            "per_key_stats": computed_stats,
        }

        for key in dataset_metadata["sequence_lengths"]:
            min_len = dataset_metadata["sequence_lengths"][key]["min"]
            max_len = dataset_metadata["sequence_lengths"][key]["max"]
            if min_len == float("inf"):
                dataset_metadata["sequence_lengths"][key] = None
            else:
                dataset_metadata["sequence_lengths"][key] = {
                    "min": int(min_len),
                    "max": int(max_len),
                }

        metadata["dataset_metadata"] = dataset_metadata

        logger.info("Statistics calculation complete.")
        return metadata

    def _get_available_keys(self, file_map: Dict[str, Path]) -> Set[str]:
        available = set()
        for path in file_map.values():
            with h5py.File(path, "r") as hf:
                available.update(hf.keys())
        return available

    def _group_indices_by_file(
        self, indices: List[Tuple[str, int]]
    ) -> Dict[str, List[int]]:
        grouped = {}
        for file_stem, idx in indices:
            grouped.setdefault(file_stem, []).append(idx)
        return grouped

    def _initialize_accumulators(
        self, keys_to_process: Set[str]
    ) -> Dict[str, Dict[str, Any]]:
        accumulators = {}
        for key in keys_to_process:
            method = self.key_methods[key]
            if method in ("none", "bool"):
                continue

            acc: Dict[str, Any] = {}
            if method in ("standard", "log-standard", "signed-log"):
                acc.update(
                    {
                        "count": 0,
                        "mean": torch.tensor(0.0, dtype=DTYPE, device=self.device),
                        "m2": torch.tensor(0.0, dtype=DTYPE, device=self.device),
                    }
                )

            if method in self.QUANTILE_METHODS:
                acc["values"] = torch.empty(0, dtype=DTYPE, device=self.device)
                acc["total_values_seen"] = 0

            if method in ("max-out", "scaled_signed_offset_log", "log-min-max"):
                acc.update(
                    {
                        "min": torch.tensor(
                            float("inf"), dtype=DTYPE, device=self.device
                        ),
                        "max": torch.tensor(
                            float("-inf"), dtype=DTYPE, device=self.device
                        ),
                    }
                )

            if acc:
                accumulators[key] = acc
        return accumulators

    def _update_accumulators_with_batch(
        self, batch: Dict[str, Tensor], accumulators: Dict, dataset_metadata: Dict
    ) -> None:
        memory_limit = self.norm_config.get(
            "quantile_max_values_in_memory", DEFAULT_QUANTILE_MEMORY_LIMIT
        )
        for key, data_batch in batch.items():
            if key not in accumulators:
                continue

            method = self.key_methods[key]
            key_acc = accumulators[key]

            data = data_batch.flatten()
            valid_data = data[torch.isfinite(data)]
            if valid_data.numel() == 0:
                continue

            data_for_stats = valid_data
            if method == "log-standard":
                valid_data = torch.clamp(valid_data, min=self.eps)
                data_for_stats = torch.log10(valid_data)
            elif method == "signed-log":
                data_for_stats = torch.sign(valid_data) * torch.log10(
                    torch.abs(valid_data) + 1.0
                )

            if "count" in key_acc:
                n_new = data_for_stats.numel()
                if n_new > 0:
                    count_old = key_acc["count"]
                    mean_old = key_acc["mean"]
                    m2_old = key_acc["m2"]

                    batch_mean = data_for_stats.mean()
                    batch_var = torch.var(data_for_stats, unbiased=False)
                    batch_m2 = batch_var * n_new

                    delta = batch_mean - mean_old
                    count_new = count_old + n_new
                    mean_new = mean_old + delta * (n_new / count_new)
                    m2_new = (
                        m2_old + batch_m2 + delta**2 * count_old * n_new / count_new
                    )

                    key_acc["count"] = count_new
                    key_acc["mean"] = mean_new
                    key_acc["m2"] = m2_new

            if "values" in key_acc:
                key_acc["total_values_seen"] += valid_data.numel()
                current_stored_size = key_acc["values"].numel()

                if current_stored_size + valid_data.numel() <= memory_limit:
                    key_acc["values"] = torch.cat([key_acc["values"], valid_data])
                else:
                    self._approximated_quantile_keys.add(key)
                    combined_data = torch.cat([key_acc["values"], valid_data])
                    perm = torch.randperm(combined_data.numel(), device=self.device)[
                        :memory_limit
                    ]
                    key_acc["values"] = combined_data[perm]

            if "max" in key_acc:
                if method == "scaled_signed_offset_log":
                    log_vals = torch.sign(valid_data) * torch.log10(
                        torch.abs(valid_data) + 1.0
                    )
                    key_acc["min"] = torch.min(key_acc["min"], log_vals.min())
                    key_acc["max"] = torch.max(key_acc["max"], log_vals.max())
                elif method == "log-min-max":
                    log_vals = torch.log10(torch.clamp(valid_data, min=self.eps))
                    key_acc["min"] = torch.min(key_acc["min"], log_vals.min())
                    key_acc["max"] = torch.max(key_acc["max"], log_vals.max())
                else:
                    key_acc["min"] = torch.min(key_acc["min"], valid_data.min())
                    key_acc["max"] = torch.max(key_acc["max"], valid_data.max())

            if data_batch.ndim == 2:
                current_len = data_batch.shape[1]
                dataset_metadata["sequence_lengths"][key]["min"] = min(
                    dataset_metadata["sequence_lengths"][key]["min"], current_len
                )
                dataset_metadata["sequence_lengths"][key]["max"] = max(
                    dataset_metadata["sequence_lengths"][key]["max"], current_len
                )

    def _finalize_stats(self, accumulators: Dict) -> Dict[str, Any]:
        final_stats = {}
        for key, method in self.key_methods.items():
            stats: Dict[str, Any] = {"method": method, "epsilon": self.eps}
            if key not in accumulators:
                if method not in ("none", "bool"):
                    stats["method"] = "none"
                final_stats[key] = stats
                continue

            key_acc = accumulators[key]

            if "count" in key_acc and key_acc["count"] > 1:
                mean = key_acc["mean"].item()
                variance = key_acc["m2"].item() / (key_acc["count"] - 1)
                std = max(math.sqrt(variance), self.eps)

                if method == "standard":
                    stats.update({"mean": mean, "std": std})
                elif method == "log-standard":
                    stats.update({"log_mean": mean, "log_std": std})
                elif method == "signed-log":
                    stats.update({"mean": mean, "std": std})

            if "values" in key_acc and key_acc["values"].numel() > 0:
                if key in self._approximated_quantile_keys:
                    logger.info(
                        f"Approximating quantiles for '{key}' using sample of "
                        f"{key_acc['values'].numel():,} values (out of {key_acc['total_values_seen']:,} total)."
                    )

                all_values = key_acc["values"]
                stats.update(self._compute_quantile_stats(all_values, key, method))

            if method == "max-out":
                max_val = max(abs(key_acc["min"].item()), abs(key_acc["max"].item()))
                stats["max_val"] = max(max_val, self.eps)

            elif method == "scaled_signed_offset_log":
                m = max(
                    abs(key_acc["min"].item()), abs(key_acc["max"].item()), self.eps
                )
                stats["m"] = m

            final_stats[key] = stats
        return final_stats

    def _compute_quantile_stats(self, values: Tensor, key: str, method: str) -> dict:
        """Compute quantile-based statistics with improved numerical stability."""
        stats: Dict[str, float] = {}

        def _robust_quantile(tensor: Tensor, q_values: Union[float, Tensor]) -> Tensor:
            try:
                return torch.quantile(tensor, q_values)
            except RuntimeError as e:
                if "too large" in str(e).lower() or "out of memory" in str(e).lower():
                    fallback_size = 1_000_000
                    if tensor.numel() <= fallback_size:
                        raise e
                    logger.warning(
                        f"Quantile failed for '{key}' on {tensor.numel():,} elements. "
                        f"Subsampling to {fallback_size:,}."
                    )
                    perm = torch.randperm(tensor.numel(), device=tensor.device)[
                        :fallback_size
                    ]
                    subsampled = tensor.flatten()[perm]
                    return torch.quantile(subsampled, q_values)
                raise e

        if method == "iqr":
            q_tensor = torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE, device=values.device)
            q_vals = _robust_quantile(values, q_tensor)
            q1, med, q3 = q_vals[0].item(), q_vals[1].item(), q_vals[2].item()
            iqr = max(q3 - q1, self.eps)
            stats.update({"median": med, "iqr": iqr})

        elif method == "log-min-max":
            log_vals = torch.log10(torch.clamp(values, min=self.eps))
            min_v, max_v = log_vals.min().item(), log_vals.max().item()
            stats.update({"min": min_v, "max": max(max_v, min_v + self.eps)})

        elif method == "symlog":
            percentile = self.norm_config.get(
                "symlog_percentile", DEFAULT_SYMLOG_PERCENTILE
            )
            thr = _robust_quantile(torch.abs(values), percentile).item()
            thr = max(thr, self.eps * 100)
            
            abs_v = torch.abs(values)
            mask = abs_v > thr
            transformed = torch.zeros_like(values)
            
            # Safe division with larger threshold ensures numerical stability
            transformed[mask] = torch.sign(values[mask]) * (torch.log10(abs_v[mask] / thr) + 1)
            transformed[~mask] = values[~mask] / thr

            sf = transformed.abs().max().item() if transformed.numel() > 0 else 1.0
            stats.update({"threshold": thr, "scale_factor": max(sf, 1.0)})

        return stats

    @staticmethod
    def normalize_tensor(x: Tensor, method: str, stats: Dict[str, Any]) -> Tensor:
        x = x.to(DTYPE)

        if method in ("none", "bool") or not stats:
            return x

        eps = stats.get("epsilon", DEFAULT_EPSILON)
        result = x

        try:
            if method == "standard":
                result = (x - stats["mean"]) / stats["std"]
            elif method == "log-standard":
                x_safe = torch.log10(torch.clamp(x, min=eps))
                result = (x_safe - stats["log_mean"]) / stats["log_std"]
            elif method == "signed-log":
                y = torch.sign(x) * torch.log10(torch.abs(x) + 1.0)
                result = (y - stats["mean"]) / stats["std"]
            elif method == "log-min-max":
                log_x = torch.log10(torch.clamp(x, min=eps))
                denom = stats["max"] - stats["min"]
                if denom <= 0:
                    raise ValueError("log-min-max stats have zero range")
                normed = (log_x - stats["min"]) / denom
                result = torch.clamp(normed, 0.0, 1.0)
            elif method == "max-out":
                result = x / stats["max_val"]
            elif method == "iqr":
                result = (x - stats["median"]) / stats["iqr"]
            elif method == "scaled_signed_offset_log":
                y = torch.sign(x) * torch.log10(torch.abs(x) + 1)
                result = y / stats["m"]
            elif method == "symlog":
                thr, sf = stats["threshold"], stats["scale_factor"]
                abs_x = torch.abs(x)
                linear_mask = abs_x <= thr
                y = torch.zeros_like(x)
                y[linear_mask] = x[linear_mask] / thr
                y[~linear_mask] = torch.sign(x[~linear_mask]) * (
                    torch.log10(abs_x[~linear_mask] / thr) + 1.0
                )
                result = y / sf
            else:
                logger.warning(f"Unsupported method '{method}'. Returning raw tensor.")
        except KeyError as e:
            logger.error(f"Missing stat '{e}' for '{method}'. Returning raw tensor.")
            return x

        if method in ("standard", "log-standard", "signed-log", "iqr"):
            result = torch.clamp(
                result, -NORMALIZED_VALUE_CLAMP, NORMALIZED_VALUE_CLAMP
            )

        return result

    @staticmethod
    def normalize_array(x: np.ndarray, method: str, stats: Dict[str, Any]) -> None:
        if method in ("none", "bool") or not stats:
            return

        eps = stats.get("epsilon", DEFAULT_EPSILON)

        try:
            if method == "standard":
                x[:] = (x - stats["mean"]) / stats["std"]
            elif method == "log-standard":
                x_safe = np.log10(np.maximum(x, eps))
                x[:] = (x_safe - stats["log_mean"]) / stats["log_std"]
            elif method == "signed-log":
                y = np.sign(x) * np.log10(np.abs(x) + 1.0)
                x[:] = (y - stats["mean"]) / stats["std"]
            elif method == "log-min-max":
                log_x = np.log10(np.maximum(x, eps))
                denom = stats["max"] - stats["min"]
                if denom <= 0:
                    raise ValueError("log-min-max stats have zero range")
                normed = (log_x - stats["min"]) / denom
                x[:] = np.clip(normed, 0.0, 1.0)
            elif method == "max-out":
                x[:] = x / stats["max_val"]
            elif method == "iqr":
                x[:] = (x - stats["median"]) / stats["iqr"]
            elif method == "scaled_signed_offset_log":
                y = np.sign(x) * np.log10(np.abs(x) + 1)
                x[:] = y / stats["m"]
            elif method == "symlog":
                thr, sf = stats["threshold"], stats["scale_factor"]
                abs_x = np.abs(x)
                linear_mask = abs_x <= thr
                y = np.zeros_like(x)
                y[linear_mask] = x[linear_mask] / thr
                y[~linear_mask] = np.sign(x[~linear_mask]) * (
                    np.log10(abs_x[~linear_mask] / thr) + 1.0
                )
                x[:] = y / sf
            else:
                logger.warning(f"Unsupported method '{method}'. Array unchanged.")
        except KeyError as e:
            logger.error(f"Missing stat '{e}' for '{method}'. Array unchanged.")

        if method in ("standard", "log-standard", "signed-log", "iqr"):
            np.clip(x, -NORMALIZED_VALUE_CLAMP, NORMALIZED_VALUE_CLAMP, out=x)

    @staticmethod
    def denormalize_tensor(x: Tensor, method: str, stats: Dict[str, Any]) -> Tensor:
        x = x.to(DTYPE)
        if method in ("none", "bool"):
            return x
        if not stats:
            raise ValueError(f"No stats for denormalization with method '{method}'")

        dtype, device = x.dtype, x.device
        eps = stats.get("epsilon", DEFAULT_EPSILON)

        def to_t(val: float) -> Tensor:
            return torch.as_tensor(val, dtype=dtype, device=device)

        if method == "standard":
            return x.mul(to_t(stats["std"])).add(to_t(stats["mean"]))
        elif method == "log-standard":
            return 10 ** (x.mul(to_t(stats["log_std"])).add(to_t(stats["log_mean"])))
        elif method == "signed-log":
            unscaled_log = x.mul(to_t(stats["std"])).add(to_t(stats["mean"]))
            return torch.sign(unscaled_log) * (10 ** torch.abs(unscaled_log) - 1.0)
        elif method == "log-min-max":
            unscaled = (
                torch.clamp(x, 0, 1)
                .mul(to_t(stats["max"] - stats["min"]))
                .add(to_t(stats["min"]))
            )
            return 10**unscaled
        elif method == "max-out":
            return x.mul(to_t(stats["max_val"]))
        elif method == "iqr":
            return x.mul(to_t(stats["iqr"])).add(to_t(stats["median"]))
        elif method == "scaled_signed_offset_log":
            ytmp = x.mul(to_t(stats["m"]))
            return torch.sign(ytmp) * (10 ** torch.abs(ytmp) - 1)
        elif method == "symlog":
            unscaled = x.mul(to_t(stats["scale_factor"]))
            abs_unscaled = torch.abs(unscaled)
            linear_mask = abs_unscaled <= 1.0
            thr = to_t(stats["threshold"])
            y = torch.zeros_like(x)
            y[linear_mask] = unscaled[linear_mask].mul(thr)
            y[~linear_mask] = (
                torch.sign(unscaled[~linear_mask])
                * thr
                * (10 ** (abs_unscaled[~linear_mask] - 1.0))
            )
            return y
        else:
            raise ValueError(f"Unsupported denormalization method '{method}'")

    @staticmethod
    def denormalize(
        v: Union[Tensor, List, float, bool, None],
        metadata: Dict[str, Any],
        var_name: str,
    ) -> Union[Tensor, List, float, bool, None]:
        if v is None:
            return None

        method = metadata["normalization_methods"].get(var_name, "none")
        if method in ("none", "bool"):
            return v

        stats = metadata["per_key_stats"].get(var_name)
        if not stats:
            raise ValueError(f"No stats for '{var_name}' in metadata.")

        is_scalar = not isinstance(v, (torch.Tensor, list))
        is_list = isinstance(v, list)

        tensor_v = (
            torch.as_tensor(v, dtype=DTYPE)
            if not isinstance(v, torch.Tensor)
            else v.to(DTYPE)
        )

        denorm_tensor = DataNormalizer.denormalize_tensor(tensor_v, method, stats)

        if is_scalar:
            return denorm_tensor.item()
        elif is_list:
            return denorm_tensor.tolist()
        else:
            return denorm_tensor


__all__ = ["DataNormalizer"]
