#!/usr/bin/env python3
"""
hardware.py – Device detection and DataLoader configuration.

This module provides utilities to detect the best available PyTorch device
(CUDA, MPS, CPU) and to configure device-specific DataLoader settings
(excluding num_workers, which is now handled elsewhere).
"""

from __future__ import annotations

import logging
import os # Still needed if other env vars were to be used in future
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def _has_mps() -> bool:
    """Checks if the current PyTorch build supports MPS and if it's available."""
    return getattr(torch.backends, "mps", None) is not None and \
           torch.backends.mps.is_available()


def get_device_type() -> str:
    """
    Determines the best available hardware acceleration backend.

    The priority for device selection is: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        A string representing the selected device type ("cuda", "mps", or "cpu").
    """
    if torch.cuda.is_available():
        return "cuda"
    if _has_mps(): # Checks for MPS availability (Apple Silicon GPU)
        return "mps"
    return "cpu"


def setup_device() -> torch.device:
    """
    Selects and returns a `torch.device` object for the best backend.

    Logs the chosen device and, if CUDA is selected, attempts to log the
    CUDA device name.

    Returns:
        A `torch.device` object (e.g., torch.device('cuda:0')).
    """
    selected_device_type = get_device_type()
    device_instance = torch.device(selected_device_type)

    if selected_device_type == "cuda":
        try:
            # Attempt to get and log the name of the current CUDA device.
            cuda_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            logger.info("Using CUDA device: %s", cuda_device_name)
        except Exception as exc: # pragma: no cover
            # Fallback if device name cannot be queried.
            logger.warning(
                "Could not query CUDA device name (Exception: %s). "
                "Proceeding with CUDA device.", exc
            )
            logger.info("Using CUDA device.")
    elif selected_device_type == "mps":
        logger.info("Using Apple Silicon MPS device.")
    else: # CPU
        logger.info("Using CPU device.")

    return device_instance


def get_device_properties() -> Dict[str, Any]:
    """
    Retrieves a dictionary of properties for the selected backend.

    Properties include type, AMP support, and (for CUDA) name, memory,
    and compute capability.

    Returns:
        A dictionary containing properties of the selected device.
    """
    selected_device_type = get_device_type()
    properties: Dict[str, Any] = {
        "type": selected_device_type,
        # AMP (Automatic Mixed Precision) is primarily beneficial and stable on CUDA.
        "supports_amp": selected_device_type == "cuda"
    }

    if selected_device_type == "cuda":
        try:
            current_cuda_device_idx = torch.cuda.current_device()
            cuda_spec = torch.cuda.get_device_properties(current_cuda_device_idx)
            properties.update(
                {
                    "name": cuda_spec.name,
                    "memory": cuda_spec.total_memory, # Total GPU memory in bytes
                    "capability": (cuda_spec.major, cuda_spec.minor),
                }
            )
        except Exception as exc: # pragma: no cover
            logger.warning(
                "Could not read CUDA device properties (Exception: %s).", exc
            )
    return properties


def configure_dataloader_settings() -> Dict[str, Any]:
    """
    Returns recommended keyword arguments for a PyTorch DataLoader,
    tuned to the current backend (excluding 'num_workers').

    This function configures 'pin_memory' and 'persistent_workers'
    based on the device type. 'num_workers' is now handled separately
    (e.g., as a constant in the training script).

    Returns:
        A dictionary with recommended DataLoader settings.
    """
    selected_device_type = get_device_type()

    settings: Dict[str, Any] = {
        "pin_memory": selected_device_type == "cuda",
        "persistent_workers": selected_device_type == "cuda",
    }

    if selected_device_type == "mps":
        settings["persistent_workers"] = False

    return settings


__all__ = [
    "get_device_type",
    "setup_device",
    "get_device_properties",
    "configure_dataloader_settings",
]
