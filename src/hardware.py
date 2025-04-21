#!/usr/bin/env python3
"""
hardware.py – Device detection and DataLoader configuration.

Provides utilities to detect the available compute hardware (CUDA, MPS, CPU)
and configure optimal settings for PyTorch DataLoaders based on the device.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Device Detection and Setup
# =============================================================================


def get_device_type() -> str:
    """
    Detects available hardware acceleration and returns the device type string.

    Checks for CUDA, then MPS (Apple Silicon), falling back to CPU.

    Returns:
        One of "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def setup_device() -> torch.device:
    """
    Selects the best available compute device (CUDA > MPS > CPU) and logs the choice.

    Returns:
        A torch.device object representing the selected device.
    """
    device_type = get_device_type()
    device = torch.device(device_type)

    if device_type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            logger.info("Using CUDA device: %s", gpu_name)
        except Exception as e:
            logger.warning("Could not get CUDA device name: %s", e)
            logger.info("Using CUDA device.")
    elif device_type == "mps":
        logger.info("Using Apple Silicon MPS device.")
    else:
        logger.info("Using CPU device.")

    return device


def get_device_properties() -> Dict[str, Any]:
    """
    Returns a dictionary containing properties of the selected compute device.

    Includes device type, AMP support status, and GPU-specific details
    (name, memory, compute capability) if CUDA is available.

    Returns:
        A dictionary containing device properties.
    """
    device_type = get_device_type()
    properties: Dict[str, Any] = {
        "type": device_type,
        "supports_amp": device_type == "cuda",
    }

    if device_type == "cuda":
        try:
            idx = torch.cuda.current_device()
            cuda_props = torch.cuda.get_device_properties(idx)
            properties.update(
                {
                    "name": cuda_props.name,
                    "memory": cuda_props.total_memory,  # Total memory in bytes
                    "capability": (cuda_props.major, cuda_props.minor),
                }
            )
        except Exception as e:
            logger.warning(
                "Could not retrieve detailed CUDA device properties: %s", e
            )

    return properties


# =============================================================================
# DataLoader Configuration
# =============================================================================


def configure_dataloader_settings() -> Dict[str, Any]:
    """
    Recommends optimal DataLoader keyword arguments based on the detected device.

    Adjusts `pin_memory`, `num_workers`, and `persistent_workers` for potentially
    better performance depending on the hardware (CUDA, MPS, CPU). These are
    recommendations and might require further tuning based on the specific
    system's CPU cores, RAM, and dataset I/O characteristics.

    Returns:
        A dictionary with recommended settings for DataLoader arguments:
        `pin_memory`, `num_workers`, `persistent_workers`.
    """
    device_type = get_device_type()

    settings: Dict[str, Any] = {
        "pin_memory": False,
        "persistent_workers": True,
        "num_workers": 4,
    }

    if device_type == "cuda":
        settings["pin_memory"] = True
        logger.debug("DataLoader: Enabled pin_memory for CUDA device.")

    elif device_type == "mps":
        settings["persistent_workers"] = False
        settings["num_workers"] = 1
        logger.debug(
            "DataLoader: Adjusted settings for MPS (num_workers=1, persistent_workers=False)."
        )

    if settings["num_workers"] == 0:
        settings["persistent_workers"] = False

    return settings


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "get_device_type",
    "setup_device",
    "get_device_properties",
    "configure_dataloader_settings",
]
