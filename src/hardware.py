#!/usr/bin/env python3
"""Device detection and DataLoader optimization helpers."""

import logging
import torch

logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            props = torch.cuda.get_device_properties(0)
            logger.info(
                f"Using CUDA: {torch.cuda.get_device_name(0)} "
                f"({props.total_memory / 1024**3:.1f}GB)"
            )
        except (RuntimeError, AssertionError) as e:
            logger.warning(f"CUDA details unavailable: {e}. Check drivers.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")
    
    return device


def should_pin_memory() -> bool:
    """Determine if memory pinning should be enabled."""
    return torch.cuda.is_available()


__all__ = ["setup_device", "should_pin_memory"]