#!/usr/bin/env python3
"""
utils.py – core helpers for the atmospheric-profile pipeline.

Highlights
----------
* JSON / JSONC loading (optional json5)
* Rigorous config validation
* Simple logging bootstrap
* Robust JSON serialisation for numpy / torch / Path
* One-call reproducibility seeding
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# optional json5 for “// comments” & trailing commas                          #
# --------------------------------------------------------------------------- #

try:
    import json5 as _json_backend  # type: ignore
    _HAS_JSON5 = True
except ImportError:  # pragma: no cover
    _json_backend = json  # fall back to stdlib
    _HAS_JSON5 = False

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# logging                                                                     #
# --------------------------------------------------------------------------- #


def setup_logging(level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None) -> None:
    """Initialise root logger with console (and optional file) output."""
    root = logging.getLogger()
    root.setLevel(level)

    # remove stale handlers (idempotent re-calls)
    while root.handlers:
        h = root.handlers.pop()
        h.close()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        fp = Path(log_file)
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            file_h = logging.FileHandler(fp, mode="a", encoding="utf-8")
            file_h.setFormatter(fmt)
            root.addHandler(file_h)
        except Exception as exc:  # pragma: no cover
            root.error("File-logging setup failed: %s. Falling back to console only.", exc)


# --------------------------------------------------------------------------- #
# config loading / validation                                                 #
# --------------------------------------------------------------------------- #

# Regexes for stripping JSONC comments and trailing commas
_JSONC_COM_RE = re.compile(r"//.*?$|/\\*.*?\\*/", re.DOTALL | re.MULTILINE)
_JSONC_TRAILING_COMMA_RE = re.compile(r",\\s*([}\\]])")


def _strip_jsonc(text: str) -> str:
    text = _JSONC_COM_RE.sub("", text)
    return _JSONC_TRAILING_COMMA_RE.sub(r"\\1", text)


def load_config(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load JSON / JSONC config and validate; returns dict or *None* on failure."""
    p = Path(path)
    if not p.is_file():
        logger.error("Config not found: %s", p)
        return None

    raw = p.read_text(encoding="utf-8-sig")  # handle BOM
    try:
        cfg = _json_backend.loads(raw) if _HAS_JSON5 else json.loads(_strip_jsonc(raw))
    except Exception as exc:
        logger.error("Failed to parse config %s: %s", p, exc)
        return None

    logger.info("Loaded config from %s", p)
    if not validate_config(cfg):
        logger.error("Configuration validation failed – aborting.")
        return None
    return cfg


def validate_config(cfg: Dict[str, Any]) -> bool:
    """Basic shape/type checks so downstream modules can assume consistency."""
    ok = True

    def _err(msg: str) -> None:
        nonlocal ok
        ok = False
        logger.error(msg)

    # mandatory non-empty lists
    for key in ("input_variables", "target_variables"):
        if not isinstance(cfg.get(key), list) or not cfg[key]:
            _err(f"'{key}' must be a non-empty list.")

    seq_types = cfg.get("sequence_types")
    if not isinstance(seq_types, dict) or not seq_types:
        _err("'sequence_types' must be a non-empty dict.")
    else:
        if any(not isinstance(v, list) or not v for v in seq_types.values()):
            _err("Each entry in 'sequence_types' must be a non-empty list.")

    out_seq = cfg.get("output_seq_type")
    if not isinstance(out_seq, str) or out_seq not in seq_types:
        _err("'output_seq_type' must name one of the keys in 'sequence_types'.")

    seq_len = cfg.get("sequence_lengths")
    if (
        not isinstance(seq_len, dict)
        or out_seq not in seq_len
        or not isinstance(seq_len[out_seq], int)
        or seq_len[out_seq] <= 0
    ):
        _err("'sequence_lengths' must map each sequence type to a positive int; invalid for output sequence.")

    # model dims
    d_model, nhead = cfg.get("d_model"), cfg.get("nhead")
    if not (isinstance(d_model, int) and d_model > 0):
        _err("'d_model' must be a positive int.")
    if not (isinstance(nhead, int) and nhead > 0):
        _err("'nhead' must be a positive int.")
    if isinstance(d_model, int) and isinstance(nhead, int) and d_model % nhead != 0:
        _err("'d_model' must be divisible by 'nhead'.")

    # a few other ints/floats
    for k in ("num_encoder_layers", "dim_feedforward"):
        if not isinstance(cfg.get(k), int) or cfg[k] <= 0:
            _err(f"'{k}' must be a positive int.")
    drop = cfg.get("dropout", 0.1)
    if not isinstance(drop, (float, int)) or not 0.0 <= drop < 1.0:
        _err("'dropout' must be a float in [0,1).")

    return ok


# --------------------------------------------------------------------------- #
# misc helpers                                                                #
# --------------------------------------------------------------------------- #

def ensure_dirs(*paths: Union[str, Path]) -> None:
    """Create each supplied directory (recursively); ignore if already exists."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def _json_encoder(obj: Any):
    """Safe encoder for json.dump covering numpy, torch, and Path objects."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON-serialisable")


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> bool:
    """Write *data* to *path* with indent + robust encoding; return success flag."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=_json_encoder, ensure_ascii=False)
        return True
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to save JSON %s: %s", p, exc)
        return False


def seed_everything(seed: int = 42) -> None:
    """Reproducibility helper – seeds python/random/numpy/torch (+CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Global seed set to %d", seed)


__all__ = [
    "setup_logging",
    "load_config",
    "validate_config",
    "ensure_dirs",
    "save_json",
    "seed_everything",
]
