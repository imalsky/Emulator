#!/usr/bin/env python3
"""
hardware.py – Device detection and DataLoader configuration.

Focus
-----
* Robustly detect CUDA / MPS / CPU even on odd builds where a backend is
  compiled in but unavailable at runtime.
* Expose a single helper (`configure_dataloader_settings`) that returns a
  dictionary of kwargs you can pass straight into `torch.utils.data.DataLoader`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------  
# Device detection  
# -----------------------------------------------------------------------------


def _has_mps() -> bool:
    """True if the torch build supports MPS and it is available."""
    return getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()


def get_device_type() -> str:
    """
    Decide on the best acceleration backend present on this system.

    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    if _has_mps():
        return "mps"
    return "cpu"


def setup_device() -> torch.device:
    """
    Return a `torch.device` corresponding to the best backend and log the choice.
    """
    dev_type = get_device_type()
    device = torch.device(dev_type)

    if dev_type == "cuda":
        try:
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            logger.info("Using CUDA device: %s", name)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not query CUDA device name (%s)", exc)
            logger.info("Using CUDA device.")
    elif dev_type == "mps":
        logger.info("Using Apple Silicon MPS device.")
    else:
        logger.info("Using CPU device.")

    return device


# -----------------------------------------------------------------------------  
# Device-capability helper  
# -----------------------------------------------------------------------------


def get_device_properties() -> Dict[str, Any]:
    """
    Return a small dict with information about the selected backend.

    Notes
    -----
    * `supports_amp` is True for CUDA (bfloat16/FP16) and False otherwise.
    * Memory is the *total* GPU memory in bytes; call `torch.cuda.mem_get_info`
      yourself if you want the free amount.
    """
    dev_type = get_device_type()
    props: Dict[str, Any] = {"type": dev_type, "supports_amp": dev_type == "cuda"}

    if dev_type == "cuda":
        try:
            idx = torch.cuda.current_device()
            spec = torch.cuda.get_device_properties(idx)
            props.update(
                {
                    "name": spec.name,
                    "memory": spec.total_memory,
                    "capability": (spec.major, spec.minor),
                }
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not read CUDA properties: %s", exc)

    return props


# -----------------------------------------------------------------------------  
# DataLoader tuning helper  
# -----------------------------------------------------------------------------


def configure_dataloader_settings() -> Dict[str, Any]:
    """
    Return recommended kwargs for a `DataLoader` tuned to the current backend.

    Environment variable override
    -----------------------------
    Set `NUM_WORKERS=<int>` in your shell to force a different worker count.
    """
    dev_type = get_device_type()

    # Default recommendations
    settings: Dict[str, Any] = {
        "pin_memory": dev_type == "cuda",
        "persistent_workers": dev_type == "cuda",
        "num_workers": 4,
    }

    # Apple MPS is sensitive to multiprocessing – keep it simple
    if dev_type == "mps":
        settings.update({"persistent_workers": False, "num_workers": 1})

    # Let the user override num_workers for profiling without touching code
    env_override = os.getenv("NUM_WORKERS")
    if env_override is not None:
        try:
            settings["num_workers"] = max(0, int(env_override))
            logger.info("NUM_WORKERS override -> %d", settings["num_workers"])
        except ValueError:
            logger.warning("Invalid NUM_WORKERS value '%s' – ignoring", env_override)

    # If num_workers == 0 we must disable persistent_workers
    if settings["num_workers"] == 0:
        settings["persistent_workers"] = False

    return settings


# -----------------------------------------------------------------------------  
# Public exports  
# -----------------------------------------------------------------------------


__all__ = [
    "get_device_type",
    "setup_device",
    "get_device_properties",
    "configure_dataloader_settings",
]
