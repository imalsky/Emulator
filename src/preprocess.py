#!/usr/bin/env python3
"""
preprocess.py - Preprocess raw HDF5 data into normalized NPY shards.

This module handles:
- Loading raw atmospheric profile data from HDF5 files
- Applying normalization using computed statistics
- Padding sequences to maximum length
- Saving processed data as NPY shards for efficient loading
"""
from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

from normalizer import DataNormalizer
from utils import compute_data_hash_with_stats, ensure_dirs, save_json

logger = logging.getLogger(__name__)

# Processing constants
HDF5_READ_CHUNK_SIZE = 524288


def _group_indices_by_file(indices: List[Tuple[str, int]]) -> Dict[str, List[int]]:
    """
    Group indices by file stem for efficient batch loading.
    
    Args:
        indices: List of (file_stem, index) tuples
        
    Returns:
        Dictionary mapping file_stem to list of indices
    """
    grouped = {}
    for file_stem, idx in indices:
        grouped.setdefault(file_stem, []).append(idx)
    return grouped


def _load_and_restore_chunk(
    hf_file: h5py.File,
    variables: List[str],
    indices: np.ndarray,
    max_seq_len: Optional[int] = None
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load a chunk of data for specified variables and indices.
    
    Uses sort/unsort technique for efficient HDF5 reading while preserving order.
    
    Args:
        hf_file: Open HDF5 file handle
        variables: List of variable names to load
        indices: Array of indices to load
        max_seq_len: Optional maximum sequence length for truncation
        
    Returns:
        Dictionary of variable_name -> data array, or None if error
    """
    if not indices.size:
        return {}
    
    # Sort indices for efficient HDF5 reading
    sorter = np.argsort(indices)
    sorted_indices = indices[sorter]
    
    # Compute inverse permutation to restore original order
    inverse_sorter = np.argsort(sorter)
    
    data_chunk = {}
    for var in variables:
        if var not in hf_file:
            logger.error(f"Critical error: Variable '{var}' not found in HDF5 file.")
            return None
        
        # Load data in sorted order (efficient for HDF5)
        data_sorted = hf_file[var][sorted_indices]
        
        # Restore original order
        data_orig_order = data_sorted[inverse_sorter]
        
        # Apply truncation if needed
        if max_seq_len is not None and data_orig_order.ndim == 2:
            if data_orig_order.shape[1] > max_seq_len:
                logger.warning(
                    f"Truncating sequence from {data_orig_order.shape[1]} to {max_seq_len} "
                    f"for variable '{var}'"
                )
                data_orig_order = data_orig_order[:, :max_seq_len]
        
        data_chunk[var] = data_orig_order
    
    return data_chunk


def _save_shard(
    shard_buffers: Dict[str, List[np.ndarray]],
    seq_dir: Path,
    tgt_dir: Path,
    glb_dir: Optional[Path],
    shard_idx: int,
) -> None:
    """
    Save a shard of data to NPY files.
    
    Args:
        shard_buffers: Dictionary containing buffered data arrays
        seq_dir: Directory for sequence input shards
        tgt_dir: Directory for target shards
        glb_dir: Directory for global feature shards (optional)
        shard_idx: Index of this shard
    """
    # Skip empty shards
    if (
        not shard_buffers["sequence_inputs"]
        or not shard_buffers["sequence_inputs"][0].size
    ):
        return
    
    shard_name = f"shard_{shard_idx:06d}.npy"
    
    # Save sequence inputs
    seq_data = np.concatenate(shard_buffers["sequence_inputs"], axis=0)
    np.save(seq_dir / shard_name, seq_data)
    
    # Save targets
    tgt_data = np.concatenate(shard_buffers["targets"], axis=0)
    np.save(tgt_dir / shard_name, tgt_data)
    
    # Save globals if present
    if (
        glb_dir is not None
        and shard_buffers.get("globals")
        and shard_buffers["globals"]
    ):
        glb_data = np.concatenate(shard_buffers["globals"], axis=0)
        np.save(glb_dir / shard_name, glb_data)


def _save_preprocessing_summary(
    output_dir: Path,
    config: Dict[str, Any],
    all_splits: Dict[str, list],
    norm_metadata: Dict[str, Any],
    data_hash: str,
) -> None:
    """
    Save a human-readable summary of preprocessing results.
    
    Args:
        output_dir: Output directory for summary
        config: Configuration dictionary
        all_splits: Dictionary of data splits
        norm_metadata: Normalization metadata
        data_hash: Hash of data configuration
    """
    summary_path = output_dir / "preprocessing_summary.txt"
    logger.info(f"Saving preprocessing summary to {summary_path}")
    
    try:
        # Build summary content
        lines = [
            "# Preprocessing Summary",
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Hash: {data_hash}",
            "",
            "--- Data Specification ---",
            f"Input Variables: {config['data_specification']['input_variables']}",
            f"Global Variables: {config['data_specification'].get('global_variables', 'None')}",
            f"Target Variables: {config['data_specification']['target_variables']}",
            f"Padding Value: {config['data_specification']['padding_value']}",
            f"Max Sequence Length: {config['model_hyperparameters']['max_sequence_length']}",
            "",
            "--- Data Splits ---",
        ]
        
        shard_size = config.get("miscellaneous_settings", {}).get("shard_size", 1000)
        for name, indices in all_splits.items():
            count = len(indices)
            num_shards = (count + shard_size - 1) // shard_size if count > 0 else 0
            lines.extend([
                f"{name.capitalize()} Split:",
                f"  Total Samples: {count:,}",
                f"  Shards Created: {num_shards}",
            ])
        
        lines.extend(["", "--- Normalization Metadata ---"])
        methods = norm_metadata.get("normalization_methods", {})
        for var, stats in norm_metadata.get("per_key_stats", {}).items():
            lines.extend([
                f"Variable: '{var}'",
                f"  Method: {methods.get(var, 'N/A')}",
            ])
        
        content = "\n".join(lines)
        summary_path.write_text(content)
        
    except Exception as e:
        logger.error(f"Could not write preprocessing summary: {e}")

def preprocess_data(
    config: Dict[str, Any],
    raw_hdf5_paths: List[Path],
    splits: Dict[str, List[Tuple[str, int]]],
    processed_dir: Path,
) -> bool:
    """
    Main preprocessing pipeline with efficient sharding.
    """
    ensure_dirs(processed_dir)
    
    # Compute hash for cache invalidation
    current_hash = compute_data_hash_with_stats(config, raw_hdf5_paths)
    hash_path = processed_dir / "data_hash.txt"
    metadata_path = processed_dir / "normalization_metadata.json"
    
    shard_size = config.get("miscellaneous_settings", {}).get("shard_size", 4096)
    all_splits = {
        "train": splits["train"],
        "val": splits["validation"],
        "test": splits["test"],
    }
    
    # Check if preprocessing is needed
    if (
        hash_path.exists()
        and hash_path.read_text().strip() == current_hash
        and metadata_path.exists()
    ):
        logger.info(
            "Processed data is up-to-date based on configuration and file stats. "
            "Skipping preprocessing."
        )
        return True
    
    logger.info("Configuration or source files have changed. Starting preprocessing...")
    preprocessing_start_time = time.time()
    
    # Step 1: Calculate normalization statistics
    start_time = time.time()
    normalizer = DataNormalizer(config_data=config)
    norm_metadata = normalizer.calculate_stats(raw_hdf5_paths, splits["train"])
    
    if not save_json(norm_metadata, metadata_path):
        logger.error("Failed to save normalization metadata.")
        return False
    
    logger.info(
        f"Normalization metadata computed and saved in {time.time() - start_time:.2f}s."
    )
    
    # Prepare for processing
    file_map = {path.stem: path for path in raw_hdf5_paths if path.is_file()}
    data_spec = config["data_specification"]
    input_vars = data_spec["input_variables"]
    global_vars = data_spec.get("global_variables", [])
    target_vars = data_spec["target_variables"]
    padding_value = data_spec["padding_value"]
    max_seq_len = config["model_hyperparameters"]["max_sequence_length"]
    
    device = torch.device("cpu")
    logger.info(f"Using device: {device} for normalization.")
    
    # Step 2: Process each split
    for split_name, split_indices in all_splits.items():
        if not split_indices:
            logger.warning(f"No indices for '{split_name}' split. Skipping.")
            continue
        
        split_start_time = time.time()
        grouped_indices = _group_indices_by_file(split_indices)
        
        # Create output directories
        split_dir = processed_dir / split_name
        seq_dir = split_dir / "sequence_inputs"
        tgt_dir = split_dir / "targets"
        glb_dir = split_dir / "globals" if global_vars else None
        ensure_dirs(seq_dir, tgt_dir, glb_dir)
        
        # Save split metadata
        num_shards = (len(split_indices) + shard_size - 1) // shard_size
        split_metadata = {
            "total_samples": len(split_indices),
            "shard_size": shard_size,
            "num_shards": num_shards,
            "sequence_length": max_seq_len,
            "has_globals": bool(global_vars),
        }
        save_json(split_metadata, split_dir / "metadata.json")
        
        pending_chunks = {
            "sequence_inputs": [],
            "targets": [],
            "globals": [] if global_vars else None,
        }
        samples_in_pending = 0
        current_shard_idx = 0
        
        logger.info(f"Processing {split_name} split ({len(split_indices)} profiles)...")
        
        with tqdm(total=len(split_indices), desc=f"Processing {split_name}") as pbar:
            for file_stem, indices in grouped_indices.items():
                if file_stem not in file_map:
                    logger.warning(f"Skipping unknown file '{file_stem}'.")
                    pbar.update(len(indices))
                    continue
                
                with h5py.File(file_map[file_stem], "r") as hf_raw:
                    # Process in chunks for memory efficiency
                    for i in range(0, len(indices), HDF5_READ_CHUNK_SIZE):
                        chunk_idx = np.array(indices[i:i + HDF5_READ_CHUNK_SIZE])
                        
                        # Load data
                        input_data = _load_and_restore_chunk(
                            hf_raw, input_vars, chunk_idx, max_seq_len
                        )
                        target_data = _load_and_restore_chunk(
                            hf_raw, target_vars, chunk_idx, max_seq_len
                        )
                        global_data = (
                            _load_and_restore_chunk(hf_raw, global_vars, chunk_idx)
                            if global_vars
                            else {}
                        )
                        
                        # Check for errors
                        if (
                            input_data is None
                            or target_data is None
                            or (global_vars and global_data is None)
                        ):
                            logger.error(
                                "A required variable was missing from HDF5 file. Aborting."
                            )
                            return False
                        
                        # Stack variables into arrays
                        seq_in_np = np.stack(
                            [input_data[var] for var in input_vars], axis=-1
                        )
                        tgt_np = np.stack(
                            [target_data[var] for var in target_vars], axis=-1
                        )
                        glb_np = (
                            np.stack([global_data[var] for var in global_vars], axis=-1)
                            if global_vars
                            else None
                        )
                        
                        # Convert to tensors for normalization
                        seq_in = torch.from_numpy(seq_in_np).to(device).float()
                        tgt = torch.from_numpy(tgt_np).to(device).float()
                        glb = (
                            torch.from_numpy(glb_np).to(device).float()
                            if global_vars and glb_np is not None
                            else None
                        )
                        
                        # Apply normalization to each variable
                        for j, var in enumerate(input_vars):
                            method = normalizer.key_methods.get(var, "none")
                            stats = norm_metadata.get("per_key_stats", {}).get(var)
                            if stats:
                                seq_in[:, :, j] = normalizer.normalize_tensor(
                                    seq_in[:, :, j], method, stats
                                )
                        
                        for j, var in enumerate(target_vars):
                            method = normalizer.key_methods.get(var, "none")
                            stats = norm_metadata.get("per_key_stats", {}).get(var)
                            if stats:
                                tgt[:, :, j] = normalizer.normalize_tensor(
                                    tgt[:, :, j], method, stats
                                )
                        
                        if global_vars and glb is not None:
                            for j, var in enumerate(global_vars):
                                method = normalizer.key_methods.get(var, "none")
                                stats = norm_metadata.get("per_key_stats", {}).get(var)
                                if stats:
                                    glb[:, j] = normalizer.normalize_tensor(
                                        glb[:, j], method, stats
                                    )
                        
                        # Apply padding if sequences are shorter than max_seq_len
                        current_seq_len = seq_in.shape[1]
                        pad_width = max_seq_len - current_seq_len
                        
                        if pad_width > 0:
                            # Pad sequences and targets
                            pad_spec = ((0, 0), (0, pad_width), (0, 0))
                            seq_in_np = np.pad(
                                seq_in.cpu().numpy(),
                                pad_spec,
                                constant_values=padding_value,
                            )
                            tgt_np = np.pad(
                                tgt.cpu().numpy(),
                                pad_spec,
                                constant_values=padding_value,
                            )
                        else:
                            seq_in_np = seq_in.cpu().numpy()
                            tgt_np = tgt.cpu().numpy()
                        
                        glb_np = (
                            glb.cpu().numpy()
                            if global_vars and glb is not None
                            else None
                        )
                        
                        chunk_samples = len(chunk_idx)
                        pending_chunks["sequence_inputs"].append(seq_in_np.astype(np.float32))
                        pending_chunks["targets"].append(tgt_np.astype(np.float32))
                        
                        if global_vars and glb_np is not None:
                            pending_chunks["globals"].append(glb_np.astype(np.float32))
                        
                        samples_in_pending += chunk_samples
                        pbar.update(chunk_samples)
                        
                        while samples_in_pending >= shard_size:
                            # Calculate how many samples we need from pending chunks
                            samples_needed = shard_size
                            samples_collected = 0
                            chunks_for_shard = {
                                "sequence_inputs": [],
                                "targets": [],
                                "globals": [] if global_vars else None,
                            }
                            
                            # Collect chunks for this shard
                            while samples_collected < samples_needed and pending_chunks["sequence_inputs"]:
                                next_chunk_seq = pending_chunks["sequence_inputs"][0]
                                next_chunk_tgt = pending_chunks["targets"][0]
                                chunk_size = next_chunk_seq.shape[0]
                                
                                if samples_collected + chunk_size <= samples_needed:
                                    # Use entire chunk
                                    chunks_for_shard["sequence_inputs"].append(
                                        pending_chunks["sequence_inputs"].pop(0)
                                    )
                                    chunks_for_shard["targets"].append(
                                        pending_chunks["targets"].pop(0)
                                    )
                                    if global_vars and pending_chunks["globals"]:
                                        chunks_for_shard["globals"].append(
                                            pending_chunks["globals"].pop(0)
                                        )
                                    samples_collected += chunk_size
                                else:
                                    # Split chunk
                                    split_point = samples_needed - samples_collected
                                    
                                    # Take what we need
                                    chunks_for_shard["sequence_inputs"].append(
                                        next_chunk_seq[:split_point]
                                    )
                                    chunks_for_shard["targets"].append(
                                        next_chunk_tgt[:split_point]
                                    )
                                    
                                    # Put remainder back
                                    pending_chunks["sequence_inputs"][0] = next_chunk_seq[split_point:]
                                    pending_chunks["targets"][0] = next_chunk_tgt[split_point:]
                                    
                                    if global_vars and pending_chunks["globals"]:
                                        next_chunk_glb = pending_chunks["globals"][0]
                                        chunks_for_shard["globals"].append(
                                            next_chunk_glb[:split_point]
                                        )
                                        pending_chunks["globals"][0] = next_chunk_glb[split_point:]
                                    
                                    samples_collected += split_point
                            
                            # Save the shard
                            if chunks_for_shard["sequence_inputs"]:
                                shard_data = {
                                    "sequence_inputs": chunks_for_shard["sequence_inputs"],
                                    "targets": chunks_for_shard["targets"],
                                    "globals": chunks_for_shard["globals"],
                                }
                                _save_shard(
                                    shard_data,
                                    seq_dir,
                                    tgt_dir,
                                    glb_dir,
                                    current_shard_idx,
                                )
                                current_shard_idx += 1
                            
                            samples_in_pending -= samples_collected
        
        # Save any remaining samples as final partial shard
        if samples_in_pending > 0 and pending_chunks["sequence_inputs"]:
            logger.info(
                f"Saving final shard for '{split_name}' with {samples_in_pending} samples."
            )
            _save_shard(
                pending_chunks,
                seq_dir,
                tgt_dir,
                glb_dir,
                current_shard_idx
            )
        
        logger.info(
            f"Completed {split_name} split in {time.time() - split_start_time:.2f}s."
        )
    
    # Save summary and hash
    _save_preprocessing_summary(
        processed_dir, config, all_splits, norm_metadata, current_hash
    )
    hash_path.write_text(current_hash)
    
    total_time = time.time() - preprocessing_start_time
    logger.info(f"Preprocessing completed successfully in {total_time:.2f}s.")
    
    return True

__all__ = ["preprocess_data"]