#!/usr/bin/env python3
"""
PADDING CONVENTION:
- Padding value: -9999.0 (defined in config)
- Mask convention: True = padding position, False = valid position
- This follows PyTorch's convention for key_padding_mask
- All features at a timestep must equal padding_value for it to be considered padding

DATA ASSUMPTIONS:
- Shards contain consistent data types within each split
- All shards in a split have the same feature dimensions
- Shard metadata accurately reflects the contained data
- Indices provided are non-negative integers
- HDF5 files have been preprocessed and normalized appropriately
"""
from __future__ import annotations

import json
import logging
import psutil
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils import DTYPE, PADDING_VALUE

logger = logging.getLogger(__name__)

FLOAT_COMPARISON = 1e-6

class AtmosphericDataset(Dataset):
    """
    Dataset for loading preprocessed atmospheric profile data.
    
    Automatically chooses between RAM loading and disk caching based on
    available memory. Uses memory mapping for large files to reduce I/O.
    
    Padding Convention:
    - Sequences are right-padded (padding at the end)
    - Padding positions have all features set to padding_value
    """
    
    def __init__(
        self,
        dir_path: Path,
        config: Dict[str, Any],
        indices: List[int],
        force_disk_loading: bool = False,
    ) -> None:
        """
        Initialize dataset with automatic memory management.
        
        Args:
            dir_path: Directory containing processed data shards
            config: Configuration dictionary
            indices: List of sample indices to use (0-based within split)
            force_disk_loading: If True, force disk-based loading
        """
        super().__init__()
        
        if not dir_path.is_dir():
            raise RuntimeError(f"Directory not found: {dir_path}")
        
        self.dir_path = dir_path
        self.config = config

        self.indices = list(indices)
        self.force_disk_loading = force_disk_loading
        
        # Extract data specification
        data_spec = self.config["data_specification"]
        self.input_variables = data_spec["input_variables"]
        self.target_variables = data_spec["target_variables"]
        self.global_variables = data_spec.get("global_variables", [])
        
        # Padding configuration for safe comparison
        self.padding_value = float(data_spec.get("padding_value", PADDING_VALUE))

        # Tolerance for floating-point comparison
        self.padding_epsilon = FLOAT_COMPARISON
        
        # Load shard metadata first to determine structure
        self._load_metadata()
        
        # Validate directory structure based on metadata
        self._validate_structure()
        
        # Decide loading strategy and load data
        self._estimate_memory_and_load()
        
        logger.info(f"AtmosphericDataset initialized: {len(self)} samples from {dir_path} ")
    
    def _validate_structure(self) -> None:
        """Validate that required directories exist based on metadata."""
        required_dirs = ["sequence_inputs", "targets"]
        # Check metadata instead of config for globals
        if self.has_globals:
            required_dirs.append("globals")
        
        missing = [d for d in required_dirs if not (self.dir_path / d).exists()]
        if missing:
            raise RuntimeError(f"Missing directories in {self.dir_path}: {missing}")
    
    def _load_metadata(self) -> None:
        """Load metadata about shard structure."""
        metadata_path = self.dir_path / "metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(f"Missing metadata.json in {self.dir_path}")
        
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        self.shard_size = self.metadata["shard_size"]
        self.total_samples = self.metadata["total_samples"]
        self.num_shards = self.metadata["num_shards"]
        self.has_globals = self.metadata["has_globals"]
        self.sequence_length = self.metadata["sequence_length"]
    
    def _estimate_memory_and_load(self) -> None:
        """
        Estimate memory requirements and choose loading strategy.
        
        If dataset fits in available RAM, load everything for fast access.
        Otherwise, use disk caching with memory mapping.
        """
        # Get available shard files
        seq_dir = self.dir_path / "sequence_inputs"
        self.seq_shards = sorted(seq_dir.glob("shard_*.npy"))
        
        if len(self.seq_shards) == 0:
            raise RuntimeError(f"No shard files found in {seq_dir}")
        
        # Peek at first shard to get dtype and dimensions
        seq_shard = np.load(self.seq_shards[0], mmap_mode='r')
        seq_dtype = seq_shard.dtype
        seq_bytes_per_sample = seq_shard.itemsize * int(np.prod(seq_shard.shape[1:]))
        
        tgt_path0 = self.dir_path / "targets" / self.seq_shards[0].name
        if not tgt_path0.exists():
            raise RuntimeError(f"Missing target shard file: {tgt_path0}")
        tgt_shard = np.load(tgt_path0, mmap_mode='r')
        tgt_dtype = tgt_shard.dtype
        tgt_bytes_per_sample = tgt_shard.itemsize * int(np.prod(tgt_shard.shape[1:]))
        
        total_bytes_per_sample = seq_bytes_per_sample + tgt_bytes_per_sample
        
        # Add globals if present
        glb_dtype = None
        if self.has_globals:
            glb_path0 = self.dir_path / "globals" / self.seq_shards[0].name
            if not glb_path0.exists():
                raise RuntimeError(f"Missing globals shard file: {glb_path0}")
            glb_shard = np.load(glb_path0, mmap_mode='r')
            glb_dtype = glb_shard.dtype
            glb_bytes_per_sample = glb_shard.itemsize * int(glb_shard.shape[1])
            total_bytes_per_sample += glb_bytes_per_sample
        
        # Store dtypes for later use
        self._seq_dtype = seq_dtype
        self._tgt_dtype = tgt_dtype
        self._glb_dtype = glb_dtype
        
        # Validate and filter indices against total_samples for *both* modes
        self._valid_indices = [i for i in self.indices if 0 <= i < self.total_samples]
        n_invalid = len(self.indices) - len(self._valid_indices)
        if n_invalid:
            logger.warning(f"{n_invalid} indices out of range [0,{self.total_samples}) were dropped.")
        
        # Calculate total memory needed
        total_bytes_needed = total_bytes_per_sample * len(self._valid_indices)
        total_gb_needed = total_bytes_needed / (1024**3)
        
        # Get available memory (use 80% as safety margin)
        available_memory = psutil.virtual_memory().available
        available_gb = available_memory / (1024**3)
        safe_available_gb = available_gb * 0.8
        
        logger.info(
            f"Memory estimate: {total_gb_needed:.2f} GB needed for {len(self._valid_indices)} samples, "
            f"{safe_available_gb:.2f} GB safely available"
        )
        
        # Choose loading strategy
        if self.force_disk_loading:
            logger.info("Force disk loading enabled - using disk cache mode")
            self._setup_disk_cache()
            self.ram_mode = False
        elif total_gb_needed < safe_available_gb:
            logger.info("Loading entire dataset into RAM...")
            self._load_all_to_ram()
            self.ram_mode = True
        else:
            logger.warning(
                f"Dataset too large for RAM ({total_gb_needed:.2f} GB > "
                f"{safe_available_gb:.2f} GB available). Using disk cache mode."
            )
            self._setup_disk_cache()
            self.ram_mode = False
    
    def _load_all_to_ram(self) -> None:
        """Load entire dataset into RAM for fast access."""
        n_samples = len(self._valid_indices)
        
        # Early exit if nothing to load
        if n_samples == 0:
            self.ram_sequences = np.zeros(
                (0, self.sequence_length, len(self.input_variables)), dtype=self._seq_dtype
            )
            self.ram_targets = np.zeros(
                (0, self.sequence_length, len(self.target_variables)), dtype=self._tgt_dtype
            )
            self.ram_globals = (
                np.zeros((0, len(self.global_variables)), dtype=self._glb_dtype)
                if self.has_globals else None
            )
            self.ram_index_map = {}
            return
        
        # Allocate arrays with correct dtype
        seq_shape = (n_samples, self.sequence_length, len(self.input_variables))
        tgt_shape = (n_samples, self.sequence_length, len(self.target_variables))
        
        self.ram_sequences = np.zeros(seq_shape, dtype=self._seq_dtype)
        self.ram_targets = np.zeros(tgt_shape, dtype=self._tgt_dtype)
        
        if self.has_globals:
            glb_shape = (n_samples, len(self.global_variables))
            self.ram_globals = np.zeros(glb_shape, dtype=self._glb_dtype)
        
        # Create index mapping
        self.ram_index_map = {}
        
        # Group validated indices by shard for efficient loading
        indices_by_shard = {}
        for idx, original_idx in enumerate(self._valid_indices):
            shard_idx = original_idx // self.shard_size
            within_shard_idx = original_idx % self.shard_size
            
            if shard_idx not in indices_by_shard:
                indices_by_shard[shard_idx] = []
            indices_by_shard[shard_idx].append((idx, within_shard_idx, original_idx))
        
        # Load data shard by shard
        processed = 0
        for shard_idx in sorted(indices_by_shard.keys()):
            shard_name = f"shard_{shard_idx:06d}.npy"
            
            # Load entire shard at once
            seq_data = np.load(self.dir_path / "sequence_inputs" / shard_name)
            tgt_data = np.load(self.dir_path / "targets" / shard_name)
            
            if self.has_globals:
                glb_data = np.load(self.dir_path / "globals" / shard_name)
            
            # Extract samples from this shard
            for idx, within_shard_idx, original_idx in indices_by_shard[shard_idx]:
                self.ram_sequences[idx] = seq_data[within_shard_idx]
                self.ram_targets[idx] = tgt_data[within_shard_idx]
                
                if self.has_globals:
                    self.ram_globals[idx] = glb_data[within_shard_idx]
                
                self.ram_index_map[idx] = original_idx
                processed += 1
            
            if processed % 10000 == 0 or processed == n_samples:
                logger.info(f"Loaded {processed}/{n_samples} samples into RAM")
    
    def _setup_disk_cache(self) -> None:
        """Setup disk caching with memory mapping for large files."""
        # Map validated indices to shards
        self.effective_indices = []
        for idx in self._valid_indices:
            shard_idx = idx // self.shard_size
            within_shard_idx = idx % self.shard_size
            self.effective_indices.append((idx, shard_idx, within_shard_idx))
        
        # Initialize caches
        self._shard_cache = {}  # For small shards loaded fully
        self._mmap_cache = {}   # For memory-mapped large shards
        self._cache_size = min(len(self.seq_shards), 200)
        self._cache_order = []
        
        # Pre-load frequently accessed shards
        logger.info(f"Pre-loading up to {self._cache_size} shards into cache...")
        unique_shard_indices = sorted(set(idx[1] for idx in self.effective_indices))
        
        for i, shard_idx in enumerate(unique_shard_indices[:self._cache_size]):
            self._load_shard(shard_idx)
            if (i + 1) % 20 == 0:
                logger.info(f"Pre-loaded {i + 1} shards")
    
    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        """Load a shard with intelligent caching strategy."""
        if shard_idx in self._shard_cache:
            # Thread-safe LRU update
            try:
                self._cache_order.remove(shard_idx)
            except ValueError:
                pass
            self._cache_order.append(shard_idx)
            return self._shard_cache[shard_idx]
        
        if shard_idx in self._mmap_cache:
            return self._mmap_cache[shard_idx]
        
        shard_name = f"shard_{shard_idx:06d}.npy"
        seq_path = self.dir_path / "sequence_inputs" / shard_name
        tgt_path = self.dir_path / "targets" / shard_name
        
        if not (seq_path.exists() and tgt_path.exists()):
            raise RuntimeError(f"Missing shard files for shard {shard_idx}")
        
        # Memory map large files (>50MB)
        if seq_path.stat().st_size > 50 * 1024 * 1024:
            shard_data = {
                "sequence_inputs": np.load(seq_path, mmap_mode='r'),
                "targets": np.load(tgt_path, mmap_mode='r'),
            }
            if self.has_globals:
                glb_path = self.dir_path / "globals" / shard_name
                if not glb_path.exists():
                    raise RuntimeError(f"Missing globals for shard {shard_idx}")
                shard_data["globals"] = np.load(glb_path, mmap_mode='r')
            
            self._mmap_cache[shard_idx] = shard_data
        else:
            shard_data = {
                "sequence_inputs": np.load(seq_path),
                "targets": np.load(tgt_path),
            }
            if self.has_globals:
                glb_path = self.dir_path / "globals" / shard_name
                if not glb_path.exists():
                    raise RuntimeError(f"Missing globals for shard {shard_idx}")
                shard_data["globals"] = np.load(glb_path)
            
            # LRU eviction
            if len(self._cache_order) >= self._cache_size:
                oldest = self._cache_order.pop(0)
                del self._shard_cache[oldest]
            
            self._shard_cache[shard_idx] = shard_data
            self._cache_order.append(shard_idx)
        
        return shard_data
    
    def __len__(self) -> int:
        """Return number of loadable samples in dataset."""
        if getattr(self, "ram_mode", False):
            return len(self.ram_sequences) if hasattr(self, "ram_sequences") else 0
        return len(self.effective_indices) if hasattr(self, "effective_indices") else 0
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_dict, target_tensor)
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        if self.ram_mode:
            # Fast path: direct RAM access
            seq_in_np = self.ram_sequences[idx]
            tgt_np = self.ram_targets[idx]
            
            # Create input dictionary
            inputs = {"sequence": torch.from_numpy(seq_in_np.copy()).to(DTYPE)}
            
            if self.has_globals:
                glb_np = self.ram_globals[idx]
                inputs["global_features"] = torch.from_numpy(glb_np.copy()).to(DTYPE)
            
            targets = torch.from_numpy(tgt_np.copy()).to(DTYPE)
            
        else:
            # Disk cache path
            global_idx, shard_idx, within_shard_idx = self.effective_indices[idx]
            
            # Load the shard
            shard_data = self._load_shard(shard_idx)
            
            # Extract the specific sample
            seq_in_np = shard_data["sequence_inputs"][within_shard_idx]
            tgt_np = shard_data["targets"][within_shard_idx]
            
            # Copy from memory-mapped arrays to ensure tensor owns memory
            if hasattr(seq_in_np, 'base') and seq_in_np.base is not None:
                seq_in_np = seq_in_np.copy()
                tgt_np = tgt_np.copy()
            
            inputs = {"sequence": torch.from_numpy(seq_in_np).to(DTYPE)}
            
            if self.has_globals and "globals" in shard_data:
                glb_np = shard_data["globals"][within_shard_idx]
                if hasattr(glb_np, 'base') and glb_np.base is not None:
                    glb_np = glb_np.copy()
                inputs["global_features"] = torch.from_numpy(glb_np).to(DTYPE)
            
            targets = torch.from_numpy(tgt_np).to(DTYPE)
        
        return inputs, targets


def create_dataset(
    dir_path: Path,
    config: Dict[str, Any],
    indices: List[int],
) -> AtmosphericDataset:
    """
    Create a dataset instance.
    
    Args:
        dir_path: Directory containing processed data
        config: Configuration dictionary
        indices: List of sample indices to use
        
    Returns:
        AtmosphericDataset instance
    """
    logger.info(f"Creating dataset from {dir_path}...")
    return AtmosphericDataset(
        dir_path=dir_path,
        config=config,
        indices=indices,
    )


def pad_collate(
    batch: List[Tuple[Dict[str, Tensor], Tensor]],
    padding_value: float = PADDING_VALUE,
    padding_epsilon: float = 1e-6,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor, Tensor]:
    """
    Collate function with safe padding detection.
    
    Creates padding masks by comparing values to padding sentinel
    within an epsilon tolerance to handle floating point precision.
    
    This follows PyTorch's convention for key_padding_mask in attention.
    
    Args:
        batch: List of (inputs, targets) tuples
        padding_value: Sentinel value for padding
        padding_epsilon: Tolerance for padding comparison
        
    Returns:
        Tuple of (batched_inputs, masks, batched_targets, target_masks)
        where masks have True for padding positions
    """
    inputs, targets = zip(*batch)
    
    # Stack sequences
    seq = torch.stack([d["sequence"] for d in inputs])
    
    # Create padding mask (True = padding position)
    # A timestep is considered padding if ALL features equal padding_value
    seq_mask = (torch.abs(seq - padding_value) < padding_epsilon).all(dim=-1)
    
    # Build batched inputs and masks
    batched = {"sequence": seq}
    masks = {"sequence": seq_mask}  # True = padding
    
    # Handle global features if present
    if "global_features" in inputs[0]:
        batched["global_features"] = torch.stack([d["global_features"] for d in inputs])
    
    # Stack targets and create masks
    tgt = torch.stack(targets)
    
    # Target mask: True = padding position
    # All target features at a timestep should be padding_value if it's a padding position
    tgt_mask = (torch.abs(tgt - padding_value) < padding_epsilon).all(dim=-1)
    
    # Validate that sequence and target masks match
    # (padding positions should be the same for inputs and targets)
    if not torch.equal(seq_mask, tgt_mask):
        logger.warning("Sequence and target padding masks don't match!")
    
    return batched, masks, tgt, tgt_mask


def create_collate_fn(padding_value: float) -> Callable:
    """
    Create a collate function with specified padding value.
    
    Args:
        padding_value: Sentinel value for padding
        
    Returns:
        Partial collate function
    """
    return partial(pad_collate, padding_value=padding_value)


__all__ = ["AtmosphericDataset", "create_dataset", "create_collate_fn"]