#!/usr/bin/env python3
"""
hardware.py - Hardware detection and configuration

Centralizes device detection and configuration to ensure consistent behavior
across different hardware platforms (CPU, NVIDIA GPU, Apple Silicon).
"""

import logging
import torch

logger = logging.getLogger(__name__)

def get_device_type():
    """
    Detect available hardware and return the appropriate device type.
    
    Returns
    -------
    str
        Device type: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def setup_device():
    """
    Set up the compute device (GPU if available, otherwise CPU).
    
    Returns
    -------
    torch.device
        Selected compute device
    """
    device_type = get_device_type()
    device = torch.device(device_type)
    
    if device_type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif device_type == "mps":
        logger.info("Using MPS (Apple Silicon GPU)")
    else:
        logger.info("Using CPU")
    
    return device

def get_device_properties():
    """
    Get properties of the current device.
    
    Returns
    -------
    dict
        Dictionary of device properties
    """
    device_type = get_device_type()
    properties = {
        "type": device_type,
        "supports_amp": device_type == "cuda",
    }
    
    if device_type == "cuda":
        properties.update({
            "name": torch.cuda.get_device_name(0),
            "memory": torch.cuda.get_device_properties(0).total_memory,
            "compute_capability": torch.cuda.get_device_capability(0)
        })
    
    return properties

def configure_dataloader_settings():
    """
    Configure DataLoader settings based on available hardware.
    
    Returns
    -------
    dict
        Dictionary of recommended DataLoader settings
    """
    device_type = get_device_type()
    
    # Default settings
    settings = {
        "pin_memory": False,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "num_workers": 4
    }
    
    if device_type == "cuda":
        settings["pin_memory"] = True
    elif device_type == "mps":
        # MPS works better with fewer workers
        settings["persistent_workers"] = False
        settings["num_workers"] = 1
    
    return settings