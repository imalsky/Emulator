#!/usr/bin/env python3
"""
hardware.py – Device detection and DataLoader helpers.
"""
from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """
    Choose the best available device (CUDA ▸ MPS ▸ CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            logger.info(f"Using CUDA device: {name}")
        except Exception:
            logger.info("Using CUDA device.")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple‑Silicon MPS device (may be slower than CUDA).")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")
    return device


def should_pin_memory() -> bool:
    """Pin memory only when CUDA is present."""
    return torch.cuda.is_available()


__all__ = ["setup_device", "should_pin_memory"]
