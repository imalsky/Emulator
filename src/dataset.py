#!/usr/bin/env python3
"""
dataset.py - Optimized data loader with full RAM loading support.
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


class AtmosphericDataset(Dataset):
    def __init__(
        self,
        dir_path: Path,
        config: Dict[str, Any],
        indices: List[int],
        force_disk_loading: bool = False,
    ) -> None:
        super().__init__()
        if not dir_path.is_dir():
            raise RuntimeError(f"Directory not found: {dir_path}")

        self.dir_path = dir_path
        self.config = config
        self.indices = indices
        self.force_disk_loading = force_disk_loading

        data_spec = self.config["data_specification"]
        self.input_variables = data_spec["input_variables"]
        self.target_variables = data_spec["target_variables"]
        self.global_variables = data_spec.get("global_variables", [])
        
        # Use safe padding comparison
        self.padding_value = float(data_spec.get("padding_value", PADDING_VALUE))
        self.padding_epsilon = 1e-6

        self._validate_structure()
        self._load_metadata()
        
        # Estimate memory requirements and decide loading strategy
        self._estimate_memory_and_load()

        logger.info(
            f"AtmosphericDataset initialized: {len(self.indices)} samples from {dir_path}"
            f" (mode: {'RAM' if self.ram_mode else 'disk cache'})"
        )

    def _validate_structure(self) -> None:
        required_dirs = ["sequence_inputs", "targets"]
        if self.global_variables:
            required_dirs.append("globals")

        missing = [d for d in required_dirs if not (self.dir_path / d).exists()]
        if missing:
            raise RuntimeError(f"Missing directories in {self.dir_path}: {missing}")

    def _load_metadata(self) -> None:
        """Load split metadata to understand shard structure."""
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
        """Estimate memory requirements and load data accordingly."""
        # Get shard files
        seq_dir = self.dir_path / "sequence_inputs"
        self.seq_shards = sorted(seq_dir.glob("shard_*.npy"))
        
        if len(self.seq_shards) == 0:
            raise RuntimeError(f"No shard files found in {seq_dir}")
        
        # Estimate memory requirements
        sample_shard = np.load(self.seq_shards[0], mmap_mode='r')
        bytes_per_sample = sample_shard.itemsize * np.prod(sample_shard.shape[1:])
        
        # Account for all data types (seq + targets + globals)
        total_bytes_per_sample = bytes_per_sample * 2  # seq + targets
        if self.has_globals:
            total_bytes_per_sample += bytes_per_sample / self.sequence_length  # globals are smaller
        
        total_bytes_needed = total_bytes_per_sample * len(self.indices)
        total_gb_needed = total_bytes_needed / (1024**3)
        
        # Get available memory
        available_memory = psutil.virtual_memory().available
        available_gb = available_memory / (1024**3)
        
        # Safety margin - use only 80% of available memory
        safe_available_gb = available_gb * 0.8
        
        logger.info(
            f"Memory estimate: {total_gb_needed:.2f} GB needed, "
            f"{safe_available_gb:.2f} GB safely available"
        )
        
        # Decide loading strategy
        if not self.force_disk_loading and total_gb_needed < safe_available_gb:
            logger.info("Loading entire dataset into RAM...")
            self._load_all_to_ram()
            self.ram_mode = True
        else:
            if total_gb_needed >= safe_available_gb:
                logger.warning(
                    f"Dataset too large for RAM ({total_gb_needed:.2f} GB > "
                    f"{safe_available_gb:.2f} GB available). Using disk cache mode."
                )
            logger.info("Using disk cache mode with aggressive caching...")
            self._setup_disk_cache()
            self.ram_mode = False
    
    def _load_all_to_ram(self) -> None:
        """Load entire dataset into RAM for maximum performance."""
        # Pre-allocate arrays
        n_samples = len(self.indices)
        seq_shape = (n_samples, self.sequence_length, len(self.input_variables))
        tgt_shape = (n_samples, self.sequence_length, len(self.target_variables))
        
        self.ram_sequences = np.zeros(seq_shape, dtype=np.float32)
        self.ram_targets = np.zeros(tgt_shape, dtype=np.float32)
        
        if self.has_globals:
            glb_shape = (n_samples, len(self.global_variables))
            self.ram_globals = np.zeros(glb_shape, dtype=np.float32)
        
        # Create index mapping
        self.ram_index_map = {}
        
        # Load all data
        for idx, original_idx in enumerate(self.indices):
            shard_idx = original_idx // self.shard_size
            within_shard_idx = original_idx % self.shard_size
            
            # Load shard if not already loaded
            shard_name = f"shard_{shard_idx:06d}.npy"
            seq_data = np.load(self.dir_path / "sequence_inputs" / shard_name)
            tgt_data = np.load(self.dir_path / "targets" / shard_name)
            
            self.ram_sequences[idx] = seq_data[within_shard_idx]
            self.ram_targets[idx] = tgt_data[within_shard_idx]
            
            if self.has_globals:
                glb_data = np.load(self.dir_path / "globals" / shard_name)
                self.ram_globals[idx] = glb_data[within_shard_idx]
            
            self.ram_index_map[idx] = original_idx
            
            if (idx + 1) % 10000 == 0:
                logger.info(f"Loaded {idx + 1}/{n_samples} samples into RAM")
    
    def _setup_disk_cache(self) -> None:
        """Setup disk caching with proper memory mapping."""
        # Map indices to shards
        self.effective_indices = []
        for idx in self.indices:
            if idx >= self.total_samples:
                logger.warning(f"Index {idx} exceeds total samples {self.total_samples}")
                continue
            shard_idx = idx // self.shard_size
            within_shard_idx = idx % self.shard_size
            self.effective_indices.append((idx, shard_idx, within_shard_idx))
        
        self._shard_cache = {}
        self._mmap_cache = {} 
        self._cache_size = min(len(self.seq_shards), 200)
        self._cache_order = []
        
        # Pre-load first N shards
        logger.info(f"Pre-loading up to {self._cache_size} shards into cache...")
        unique_shard_indices = sorted(set(idx[1] for idx in self.effective_indices))
        for i, shard_idx in enumerate(unique_shard_indices[:self._cache_size]):
            self._load_shard(shard_idx)
            if (i + 1) % 20 == 0:
                logger.info(f"Pre-loaded {i + 1} shards")

    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        """Load a shard with proper memory mapping for efficient disk access."""
        # Check if already in memory cache
        if shard_idx in self._shard_cache:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._shard_cache[shard_idx]
        
        # Check if already memory-mapped
        if shard_idx in self._mmap_cache:
            return self._mmap_cache[shard_idx]

        # Load shard files
        shard_name = f"shard_{shard_idx:06d}.npy"
        seq_path = self.dir_path / "sequence_inputs" / shard_name
        tgt_path = self.dir_path / "targets" / shard_name

        if not seq_path.exists() or not tgt_path.exists():
            raise RuntimeError(f"Missing shard files for shard {shard_idx}")

        seq_size = seq_path.stat().st_size
        use_mmap = seq_size > 50 * 1024 * 1024  # Memory map if > 50MB
        
        if use_mmap:
            # Use memory mapping for large files
            shard_data = {
                "sequence_inputs": np.load(seq_path, mmap_mode='r'),
                "targets": np.load(tgt_path, mmap_mode='r'),
            }
            
            if self.has_globals:
                glb_path = self.dir_path / "globals" / shard_name
                if glb_path.exists():
                    shard_data["globals"] = np.load(glb_path, mmap_mode='r')
                else:
                    raise RuntimeError(f"Missing globals shard for shard {shard_idx}")
            
            # Store in mmap cache (no LRU needed as these are lightweight)
            self._mmap_cache[shard_idx] = shard_data
            return shard_data
        
        else:
            # Load small files fully into memory with caching
            shard_data = {
                "sequence_inputs": np.load(seq_path),
                "targets": np.load(tgt_path),
            }

            if self.has_globals:
                glb_path = self.dir_path / "globals" / shard_name
                if glb_path.exists():
                    shard_data["globals"] = np.load(glb_path)
                else:
                    raise RuntimeError(f"Missing globals shard for shard {shard_idx}")

            # Add to cache with LRU eviction
            if len(self._cache_order) >= self._cache_size:
                # Evict oldest
                oldest = self._cache_order.pop(0)
                del self._shard_cache[oldest]

            self._shard_cache[shard_idx] = shard_data
            self._cache_order.append(shard_idx)

            return shard_data

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """Get item with proper handling of memory-mapped arrays."""
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        if self.ram_mode:
            # Fast path: direct RAM access
            seq_in_np = self.ram_sequences[idx]
            tgt_np = self.ram_targets[idx]
            
            inputs = {"sequence": torch.from_numpy(seq_in_np.copy()).to(DTYPE)}
            
            if self.has_globals:
                glb_np = self.ram_globals[idx]
                inputs["global_features"] = torch.from_numpy(glb_np.copy()).to(DTYPE)
            
            targets = torch.from_numpy(tgt_np.copy()).to(DTYPE)
        else:
            # Disk cache path
            global_idx, shard_idx, within_shard_idx = self.effective_indices[idx]
            
            # Load the shard (may be memory-mapped)
            shard_data = self._load_shard(shard_idx)
            
            # Extract the specific sample
            # IMPORTANT: For memory-mapped arrays, indexing returns a view
            # We need to copy to ensure the tensor owns its memory
            seq_in_np = shard_data["sequence_inputs"][within_shard_idx]
            tgt_np = shard_data["targets"][within_shard_idx]
            
            # FIXED: Always copy from memory-mapped arrays to ensure tensor owns memory
            if hasattr(seq_in_np, 'base') and seq_in_np.base is not None:
                # This is a view into a memory-mapped array
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
):
    """FIX: Use safe padding comparison instead of exact equality."""
    inputs, targets = zip(*batch)

    seq = torch.stack([d["sequence"] for d in inputs])
    # Safe padding comparison
    seq_mask = (torch.abs(seq - padding_value) < padding_epsilon).all(dim=-1)

    batched = {"sequence": seq}
    masks = {"sequence": seq_mask}

    if "global_features" in inputs[0]:
        batched["global_features"] = torch.stack([d["global_features"] for d in inputs])

    tgt = torch.stack(targets)
    # Safe padding comparison
    tgt_mask = (torch.abs(tgt - padding_value) < padding_epsilon).all(dim=-1)

    return batched, masks, tgt, tgt_mask


def create_collate_fn(padding_value: float) -> Callable:
    return partial(pad_collate, padding_value=padding_value)


__all__ = ["AtmosphericDataset", "create_dataset", "create_collate_fn"]