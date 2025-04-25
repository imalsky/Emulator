#!/usr/bin/env python3
"""
normalizer.py – Compute *and invert* eight normalization schemes.

Supported methods (case-insensitive)
--------------------------------------------------------------
* `iqr`                 – median / IQR scaling   → ℝ
* `log-min-max`         – log10 → min-max to [0, 1]   **requires strictly positive data**
* `arctan-compression`  – arctan (-π … π) → [-1, 1]
* `max-out`             – divide by global |max| → [-1, 1]
* `invlogit-compression`– sigmoid → [-1, 1]
* `custom`              – sign-preserving log10 (He et al.)
* `symlog`              – linear near 0, log10 tails → [-1, 1]
* `standard`            – z-score then scale to [-1, 1]

The implementation guarantees that, for any numeric value *x*,

    denormalize(normalize(x)) == x                 (within FP tolerance,
                                                    ignoring intentional clamping)

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

# -----------------------------------------------------------------------------  
# optional dependency for robust JSON writing                                   #
# -----------------------------------------------------------------------------  
try:
    from utils import save_json          # robust numpy / torch encoder
except ImportError:                       # utils.py not on PYTHONPATH
    save_json = None                      # fallback to plain json.dump

logger = logging.getLogger(__name__)

__all__ = ["DataNormalizer"]

# -----------------------------------------------------------------------------  
# helper utilities                                                              
# -----------------------------------------------------------------------------


def _to_tensor(v: Union[List[float], float]) -> Tensor:
    """Convert a scalar or list to a 1-D float32 tensor."""
    return torch.tensor(v if isinstance(v, list) else [v], dtype=torch.float32)


# -----------------------------------------------------------------------------  
# main class                                                                    
# -----------------------------------------------------------------------------


class DataNormalizer:
    """Compute global statistics, normalise data, and invert the transform."""

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

    # ------------------------------------------------------------------  
    # construction                                                      
    # ------------------------------------------------------------------
    def __init__(
        self,
        input_folder: Union[str, Path],
        output_folder: Union[str, Path],
        batch_size: int = 100,
    ) -> None:
        self.input_dir = Path(input_folder)
        self.output_dir = Path(output_folder)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input folder not found: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = max(int(batch_size), 1)
        logger.info("DataNormalizer: input=%s  output=%s", self.input_dir, self.output_dir)

    # ------------------------------------------------------------------  
    # statistics helpers                                                
    # ------------------------------------------------------------------
    @staticmethod
    def clip_outliers(t: Tensor, low: float = 0.001, high: float = 0.999) -> Tensor:
        """Clip tensor values outside the (low, high) quantiles."""
        if t.numel() == 0:
            return t
        if not (0 <= low < high <= 1):
            raise ValueError("Quantiles must satisfy 0 ≤ low < high ≤ 1")
        lo = torch.quantile(t.float(), low)
        hi = torch.quantile(t.float(), high)
        return torch.clamp(t, min=lo, max=hi)

    # ------------------------------------------------------------------  
    # global statistics                                                  
    # ------------------------------------------------------------------
    def calculate_global_stats(
        self,
        key_methods: Optional[Dict[str, str]] = None,
        default_method: str = "standard",
        clip_outliers_before_scaling: bool = False,
        symlog_percentile: float = 0.5,
        symlog_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Scan every JSON profile, compute per-key statistics for the chosen
        normalisation schemes, and return a metadata dict (also saved to disk).
        """
        files = [
            p for p in self.input_dir.glob("*.json")
            if p.name != "normalization_metadata.json"
        ]
        if not files:
            raise FileNotFoundError(f"No JSON profiles in {self.input_dir}")

        value_buf: Dict[str, List[Tensor]] = {}
        bool_keys: set[str] = set()

        # --- gather data -------------------------------------------------
        for fpath in files:
            try:
                prof = json.loads(fpath.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON {fpath.name}: {e}") from e

            for k, v in prof.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                if isinstance(v, bool):
                    bool_keys.add(k)
                    continue
                if isinstance(v, (int, float)):
                    value_buf.setdefault(k, []).append(torch.tensor([float(v)]))
                    continue
                if isinstance(v, list) and all(
                    isinstance(x, (int, float)) and not np.isnan(x) for x in v
                ):
                    value_buf.setdefault(k, []).append(torch.tensor(v, dtype=torch.float32))

        # --- map keys to methods ----------------------------------------
        default_method = default_method.lower().strip()
        if default_method not in self.METHODS:
            raise ValueError(f"Unknown default method '{default_method}'")

        key_methods = {k: m.lower().strip() for k, m in (key_methods or {}).items()}
        methods_final = {
            k: key_methods.get(k, default_method) for k in value_buf.keys()
        }
        for bk in bool_keys:
            methods_final[bk] = "none"

        # --- compute stats ----------------------------------------------
        stats: Dict[str, Any] = {}
        for key, tensors in value_buf.items():
            vec = torch.cat(tensors)
            if clip_outliers_before_scaling:
                vec = self.clip_outliers(vec)
            method = methods_final[key]
            stats[key] = self._compute_stats_for_key(
                vec, method,
                symlog_percentile,
                (symlog_thresholds or {}).get(key),
            )

        meta: Dict[str, Any] = {
            "normalization_methods": methods_final,
            "config": {
                "clip_outliers_before_scaling": clip_outliers_before_scaling,
                "default_method": default_method,
                "key_methods_provided": key_methods or {},
                "symlog_percentile": symlog_percentile,
                "symlog_thresholds_provided": symlog_thresholds or {},
                "boolean_keys_detected": list(bool_keys),
                "numeric_keys_processed": list(stats.keys()),
            },
            **stats,
        }

        # --- persist -----------------------------------------------------
        meta_path = self.output_dir / "normalization_metadata.json"
        if save_json:
            save_json(meta, meta_path)
        else:
            meta_path.write_text(json.dumps(meta, indent=2))
        logger.info("Saved normalization metadata → %s", meta_path.name)
        return meta

    # ------------------------------------------------------------------  
    # per-key stats generator                                            
    # ------------------------------------------------------------------
    def _compute_stats_for_key(
        self,
        data: Tensor,
        method: str,
        symlog_percentile: float,
        symlog_threshold: Optional[float],
    ) -> Dict[str, float]:
        if data.numel() == 0:
            raise ValueError("empty tensor")

        data = data.float()
        gmin, gmax = float(data.min()), float(data.max())
        stats: Dict[str, float] = {"global_min": gmin, "global_max": gmax}
        is_constant = (gmax - gmin) < 1e-9
        if is_constant:
            stats["is_constant"] = 1.0

        eps = 1e-10
        method = method.lower().strip()

        if method == "iqr":
            q1, q3 = float(torch.quantile(data, 0.25)), float(torch.quantile(data, 0.75))
            stats.update(
                median=float(torch.median(data).values),
                iqr=max(q3 - q1, eps),
            )
        elif method == "log-min-max":
            if gmin <= 0:
                raise ValueError("'log-min-max' requires strictly positive data")
            log_vals = torch.log10(data)
            stats.update(min=float(log_vals.min()), max=float(log_vals.max()))
        elif method == "arctan-compression":
            center = 0.0 if (gmin < 0 < gmax) else (gmin + gmax) / 2.0
            scale = max((gmax - gmin) / 2.0, eps)
            stats.update(center=center, scale=scale, alpha=math.tan(0.99 * math.pi / 2))
        elif method == "max-out":
            stats["max_val"] = max(abs(gmin), abs(gmax), eps)
        elif method == "invlogit-compression":
            stats.update(mean=float(data.mean()), std=max(float(data.std()), eps))
        elif method == "custom":
            m_pos = math.log10(gmax + 1 + eps) if gmax > -eps else 0.0
            m_neg = math.log10(abs(gmin) + 1 + eps) if gmin < eps else 0.0
            stats.update(m=max(m_pos, m_neg, 1.0), epsilon=eps)
        elif method == "symlog":
            if is_constant:
                stats.update(threshold=1.0, scale_factor=1.0, epsilon=eps)
            else:
                abs_data = torch.abs(data)
                thr = max(
                    float(torch.quantile(abs_data, symlog_percentile))
                    if symlog_threshold is None
                    else symlog_threshold,
                    eps,
                )
                lin = abs_data <= thr
                trans = torch.zeros_like(data)
                trans[lin] = data[lin] / thr
                safe_arg = torch.clamp(abs_data[~lin] / thr, min=eps)
                trans[~lin] = torch.sign(data[~lin]) * (torch.log10(safe_arg) + 1)
                stats.update(
                    threshold=thr,
                    scale_factor=max(float(torch.max(torch.abs(trans))), 1.0),
                    epsilon=eps,
                )
        elif method == "standard":
            mu, sigma = float(data.mean()), max(float(data.std()), eps)
            z = (data - mu) / sigma if not is_constant else torch.zeros_like(data)
            scale = max(3.0, float(torch.max(torch.abs(z)))) if not is_constant else 1.0
            stats.update(mean=mu, std=sigma, scale_factor=scale)
        elif method == "none":
            pass
        else:
            raise ValueError(f"Unsupported method '{method}'")

        return stats

    # ------------------------------------------------------------------  
    # normalisation / denormalisation                                   
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_tensor(x: Tensor, method: str, stats: Dict[str, float]) -> Tensor:
        """Return *new* tensor – does **not** modify *x* in-place."""
        if x.numel() == 0 or method == "none" or stats.get("is_constant"):
            return x

        eps = stats.get("epsilon", 1e-9)
        method = method.lower().strip()

        if method == "iqr":
            y = (x - stats["median"]) / stats["iqr"]
        elif method == "log-min-max":
            denom = max(stats["max"] - stats["min"], eps)
            y = torch.clamp((torch.log10(x) - stats["min"]) / denom, 0.0, 1.0)
        elif method == "arctan-compression":
            y = (2 / math.pi) * torch.atan(
                stats["alpha"] * (x - stats["center"]) / stats["scale"]
            )
        elif method == "max-out":
            y = x / stats["max_val"]
        elif method == "invlogit-compression":
            z = (x - stats["mean"]) / stats["std"]
            y = 2 * torch.sigmoid(z) - 1
        elif method == "custom":
            pos = torch.log10(torch.clamp(x, min=0) + 1 + eps)
            neg = -torch.log10(torch.clamp(-x, min=0) + 1 + eps)
            y = torch.where(x >= 0, pos, neg) / stats["m"]
        elif method == "symlog":
            thr, sf = stats["threshold"], stats["scale_factor"]
            abs_x = torch.abs(x)
            lin = abs_x <= thr
            y = torch.zeros_like(x)
            y[lin] = x[lin] / thr
            safe_arg = torch.clamp(abs_x[~lin] / thr, min=eps)
            y[~lin] = torch.sign(x[~lin]) * (torch.log10(safe_arg) + 1)
            y = torch.clamp(y / sf, -1.0, 1.0)
        elif method == "standard":
            y = torch.clamp(
                ((x - stats["mean"]) / stats["std"]) / stats["scale_factor"],
                -1.0, 1.0,
            )
        else:
            raise ValueError(f"Unsupported method '{method}'")

        return y.to(x.dtype)

    # ------------------------------------------------------------------  
    # full inverse transform                                            
    # ------------------------------------------------------------------
    @staticmethod
    def denormalize(
        v: Union[Tensor, List[float], float],
        metadata: Dict[str, Any],
        var_name: str,
    ) -> Union[Tensor, List[float], float]:
        """
        Invert the normalisation for variable *var_name* using *metadata*.

        Parameters
        ----------
        v
            Value(s) to invert – scalar, list, or tensor.
        metadata
            Dict returned by :py:meth:`calculate_global_stats`.
        var_name
            Key to look up in ``metadata``.
        """
        methods = metadata.get("normalization_methods", {})
        method = methods.get(var_name, "none").lower().strip()
        if method == "none":
            return v

        stats = metadata.get(var_name)
        if stats is None:
            if methods.get(var_name) == "none":  # stats failed at calc time
                logger.warning(
                    "Stats missing for '%s' (method forced to 'none' during calc) – returning raw value.",
                    var_name,
                )
                return v
            raise KeyError(f"Normalisation stats not found for '{var_name}'")

        if stats.get("is_constant"):
            const_val = stats.get("global_min", stats.get("global_max"))
            if const_val is None:
                return v
            if isinstance(v, Tensor):
                return torch.full_like(v, const_val, dtype=v.dtype)
            if isinstance(v, list):
                return [const_val] * len(v)
            return const_val

        is_scalar = not isinstance(v, (list, Tensor))
        x = v if isinstance(v, Tensor) else _to_tensor(v)
        orig_dtype = x.dtype
        x = x.float()

        eps = stats.get("epsilon", 1e-9)

        if method == "iqr":
            y = x * stats["iqr"] + stats["median"]
        elif method == "log-min-max":
            log_val = x * (stats["max"] - stats["min"]) + stats["min"]
            y = torch.pow(10, log_val)
        elif method == "arctan-compression":
            safe_x = torch.clamp(x, -0.999999, 0.999999)
            y = stats["center"] + (
                stats["scale"] / stats["alpha"]
            ) * torch.tan((math.pi / 2) * safe_x)
        elif method == "max-out":
            y = x * stats["max_val"]
        elif method == "invlogit-compression":
            safe_x = torch.clamp(x, -0.999999, 0.999999)
            p = (safe_x + 1) / 2.0
            logit_p = torch.log(p / torch.clamp(1 - p, min=eps))
            y = stats["mean"] + stats["std"] * logit_p
        elif method == "custom":
            ytmp = x * stats["m"]
            pos = torch.pow(10, ytmp) - 1 - eps
            neg = -(torch.pow(10, -ytmp) - 1 - eps)
            y = torch.where(ytmp >= 0, pos, neg)
        elif method == "symlog":
            thr, sf = stats["threshold"], stats["scale_factor"]
            unscaled = x * sf
            lin = torch.abs(unscaled) <= 1.0
            y = torch.zeros_like(x)
            y[lin] = unscaled[lin] * thr
            safe_exp = torch.clamp(torch.abs(unscaled[~lin]) - 1, min=-50, max=50)
            y[~lin] = torch.sign(unscaled[~lin]) * thr * torch.pow(10, safe_exp)
        elif method == "standard":
            z = x * stats["scale_factor"]
            y = stats["mean"] + stats["std"] * z
        else:
            raise ValueError(f"Unsupported method '{method}'")

        y = y.to(orig_dtype)
        if is_scalar:
            return y.item()
        if isinstance(v, list):
            return y.tolist()
        return y

    # ------------------------------------------------------------------  
    # file processing                                                   
    # ------------------------------------------------------------------
    def process_profiles(self, stats: Dict[str, Any]) -> None:
        """
        Normalise every profile under *input_dir* and write to *output_dir*.
        """
        methods = stats.get("normalization_methods", {})
        if not methods:
            raise RuntimeError("Stats dict missing 'normalization_methods'")

        for fpath in self.input_dir.glob("*.json"):
            if fpath.name == "normalization_metadata.json":
                continue

            try:
                prof = json.loads(fpath.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as e:
                logger.error("Skipping bad JSON %s: %s", fpath.name, e)
                continue

            out_prof: Dict[str, Any] = {}
            for k, v in prof.items():
                meth = methods.get(k, "none").lower()
                if meth == "none" or v is None or isinstance(v, bool):
                    out_prof[k] = v
                    continue

                key_stats = stats.get(k)
                if key_stats is None:
                    logger.warning("No stats for key '%s' (method=%s); keeping raw", k, meth)
                    out_prof[k] = v
                    continue

                tensor_val = _to_tensor(v)
                if tensor_val.numel() == 0:
                    out_prof[k] = v
                    continue

                try:
                    norm = self.normalize_tensor(tensor_val, meth, key_stats)
                    out_prof[k] = norm.tolist() if isinstance(v, list) else norm.item()
                except Exception as e:
                    logger.warning("Failed to normalize %s in %s: %s; keeping raw", k, fpath.name, e)
                    out_prof[k] = v

            out_path = self.output_dir / fpath.name
            if save_json:
                save_json(out_prof, out_path)
            else:
                out_path.write_text(json.dumps(out_prof, indent=2))

        logger.info("Normalization complete → %s", self.output_dir)
