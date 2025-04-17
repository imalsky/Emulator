#!/usr/bin/env python3
"""
hardware.py – Device detection and DataLoader configuration.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def get_device_type() -> str:
    """
    Detect available hardware and return the device type.
    Returns one of: "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_device() -> torch.device:
    """
    Select the compute device and log the choice.
    Returns a torch.device.
    """
    device_type = get_device_type()
    device = torch.device(device_type)
    if device_type == "cuda":
        name = torch.cuda.get_device_name(0)
        logger.info("Using CUDA GPU: %s", name)
    elif device_type == "mps":
        logger.info("Using Apple Silicon MPS device")
    else:
        logger.info("Using CPU")
    return device


def get_device_properties() -> Dict[str, Any]:
    """
    Return a dictionary of properties for the current device.
    Includes type, AMP support, and on CUDA devices, name, memory, and compute capability.
    """
    dtype = get_device_type()
    props: Dict[str, Any] = {"type": dtype, "supports_amp": dtype == "cuda"}
    if dtype == "cuda":
        idx = torch.cuda.current_device()
        props.update({
            "name": torch.cuda.get_device_name(idx),
            "memory": torch.cuda.get_device_properties(idx).total_memory,
            "capability": torch.cuda.get_device_capability(idx),
        })
    return props


def configure_dataloader_settings() -> Dict[str, Any]:
    """
    Recommend DataLoader settings based on the selected device.
    Returns a dict with keys: pin_memory, persistent_workers, num_workers.
    """
    dtype = get_device_type()
    settings: Dict[str, Any] = {
        "pin_memory": False,
        "persistent_workers": True,
        "num_workers": 4,
    }
    if dtype == "cuda":
        settings["pin_memory"] = True
    elif dtype == "mps":
        # MPS often works better with fewer workers
        settings["persistent_workers"] = False
        settings["num_workers"] = 1
    return settings


__all__ = [
    "get_device_type",
    "setup_device",
    "get_device_properties",
    "configure_dataloader_settings",
]
