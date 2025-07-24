#!/usr/bin/env python3
"""
preprocess.py - Preprocess raw HDF5 data into normalized NPY shards.

This module reads raw HDF5 files, computes normalization stats from the training
split, normalizes all data, and saves it into NPY files in subdirectories
for efficient loading.
"""
from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from normalizer import DataNormalizer
from utils import compute_data_hash, ensure_dirs, save_json

logger = logging.getLogger(__name__)


HDF5_READ_CHUNK_SIZE = 131072

def _group_indices_by_file(indices: List[Tuple[str, int]]) -> Dict[str, List[int]]:
    """Groups a list of (file_stem, index) tuples by file_stem."""
    grouped = {}
    for file_stem, idx in indices:
        grouped.setdefault(file_stem, []).append(idx)
    return grouped


def _load_and_restore_chunk(
    hf_file: h5py.File, variables: List[str], indices: np.ndarray, max_seq_len: int = None
) -> Dict[str, np.ndarray] | None:
    """
    Loads a chunk of data for specified variables and indices, using a
    sort/unsort method for efficient HDF5 reading while preserving order.
    """
    if not indices.size:
        return {}

    # Sort indices for efficient h5py reading
    sorter = np.argsort(indices)
    sorted_indices = indices[sorter]

    # Compute inverse permutation to restore original order later
    inverse_sorter = np.argsort(sorter)

    data_chunk = {}
    for var in variables:
        if var not in hf_file:
            logger.error(f"Critical error: Variable '{var}' not found in HDF5 file.")
            return None
        
        # 1. Load data in sorted order (efficient)
        data_sorted = hf_file[var][sorted_indices]
        # 2. Restore original order (correctness)
        data_orig_order = data_sorted[inverse_sorter]
        
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
    glb_dir: Path | None,
    shard_idx: int,
) -> None:
    """Saves a shard of data to NPY files using efficient concatenation."""
    if (
        not shard_buffers["sequence_inputs"]
        or not shard_buffers["sequence_inputs"][0].size
    ):
        return  # Do not save empty shards

    shard_name = f"shard_{shard_idx:06d}.npy"

    # Use np.concatenate for performance, as buffers may contain multiple arrays
    seq_data = np.concatenate(shard_buffers["sequence_inputs"], axis=0)
    np.save(seq_dir / shard_name, seq_data)

    tgt_data = np.concatenate(shard_buffers["targets"], axis=0)
    np.save(tgt_dir / shard_name, tgt_data)

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
    Saves a human-readable summary of the preprocessing results.

    Args:
        output_dir: The root directory for processed data (e.g., 'data/processed').
        config: The main configuration dictionary.
        all_splits: Dictionary containing the sample indices for each split.
        norm_metadata: The computed normalization statistics.
        data_hash: The hash of the data configuration.
    """
    summary_path = output_dir / "preprocessing_summary.txt"
    logger.info(f"Saving preprocessing summary to {summary_path}")

    try:
        content = f"""# Preprocessing Summary
                    - Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    - Data Hash: {data_hash}

                    --- Data Specification ---
                    - Input Variables: {config['data_specification']['input_variables']}
                    - Global Variables: {config['data_specification'].get('global_variables', 'None')}
                    - Target Variables: {config['data_specification']['target_variables']}
                    - Padding Value: {config['data_specification']['padding_value']}
                    - Max Sequence Length: {config['model_hyperparameters']['max_sequence_length']}

                    --- Data Splits ---
                    """
        shard_size = config.get("miscellaneous_settings", {}).get("shard_size", 1000)
        for name, indices in all_splits.items():
            count = len(indices)
            num_shards = (count + shard_size - 1) // shard_size if count > 0 else 0
            content += f"- {name.capitalize()} Split:\n"
            content += f"  - Total Samples: {count:,}\n"
            content += f"  - Shards Created: {num_shards}\n"

        content += "\n--- Normalization Metadata ---\n"
        for var, stats in norm_metadata.items():
            content += f"- Variable: '{var}'\n"
            content += f"  - Method: {stats.get('method', 'N/A')}\n"
            content += f"  - Stats:\n"
            if "params" in stats:
                for key, value in stats["params"].items():
                    if isinstance(value, (int, float)):
                        content += f"    - {key}: {value:g}\n"
                    else:
                        content += f"    - {key}: {value}\n"
            else:
                content += "    - No parameters found.\n"

        summary_path.write_text(content)

    except Exception as e:
        logger.error(f"Could not write preprocessing summary: {e}")


def preprocess_data(
    config: Dict[str, Any],
    raw_hdf5_paths: List[Path],
    splits: Dict[str, List[Tuple[str, int]]],
    processed_dir: Path,
) -> bool:
    """Main function to orchestrate the preprocessing of raw HDF5 data."""
    ensure_dirs(processed_dir)
    
    max_seq_len = config["model_hyperparameters"]["max_sequence_length"]
    
    current_hash = compute_data_hash(config, raw_hdf5_paths)
    hash_path = processed_dir / "data_hash.txt"
    metadata_path = processed_dir / "normalization_metadata.json"

    shard_size = config.get("miscellaneous_settings", {}).get("shard_size", 4096)
    all_splits = {
        "train": splits["train"],
        "val": splits["validation"],
        "test": splits["test"],
    }

    if (
        hash_path.exists()
        and hash_path.read_text().strip() == current_hash
        and metadata_path.exists()
    ):
        logger.info(
            "Processed data is up-to-date based on configuration hash. Skipping preprocessing."
        )
        return True

    logger.info(
        "Configuration has changed or processed data not found. Starting preprocessing..."
    )
    preprocessing_start_time = time.time()

    start_time = time.time()
    normalizer = DataNormalizer(config_data=config)
    norm_metadata = normalizer.calculate_stats(raw_hdf5_paths, splits["train"])
    if not save_json(norm_metadata, metadata_path):
        logger.error("Failed to save normalization metadata. Exiting.")
        return False
    logger.info(
        f"Normalization metadata computed and saved in {time.time() - start_time:.2f}s."
    )

    file_map = {path.stem: path for path in raw_hdf5_paths if path.is_file()}
    data_spec = config["data_specification"]
    input_vars = data_spec["input_variables"]
    global_vars = data_spec.get("global_variables", [])
    target_vars = data_spec["target_variables"]
    padding_value = data_spec["padding_value"]
    max_seq_len = config["model_hyperparameters"]["max_sequence_length"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for normalization.")

    for split_name, split_indices in all_splits.items():
        if not split_indices:
            logger.warning(f"No indices for '{split_name}' split. Skipping.")
            continue

        split_start_time = time.time()
        grouped_indices = _group_indices_by_file(split_indices)
        split_dir = processed_dir / split_name
        seq_dir = split_dir / "sequence_inputs"
        tgt_dir = split_dir / "targets"
        glb_dir = split_dir / "globals" if global_vars else None
        ensure_dirs(seq_dir, tgt_dir, glb_dir)

        num_shards = (len(split_indices) + shard_size - 1) // shard_size
        split_metadata = {
            "total_samples": len(split_indices),
            "shard_size": shard_size,
            "num_shards": num_shards,
            "sequence_length": max_seq_len,
            "has_globals": bool(global_vars),
        }
        save_json(split_metadata, split_dir / "metadata.json")

        shard_buffers = {
            "sequence_inputs": [],
            "targets": [],
            "globals": [] if global_vars else None,
        }
        num_samples_in_buffer = 0
        current_shard_idx = 0

        logger.info(f"Processing {split_name} split ({len(split_indices)} profiles)...")

        with tqdm(total=len(split_indices), desc=f"Processing {split_name}") as pbar:
            for file_stem, indices in grouped_indices.items():
                if file_stem not in file_map:
                    logger.warning(f"Skipping unknown file '{file_stem}'.")
                    pbar.update(len(indices))
                    continue

                with h5py.File(file_map[file_stem], "r") as hf_raw:
                    for i in range(0, len(indices), HDF5_READ_CHUNK_SIZE):
                        chunk_idx = np.array(indices[i : i + HDF5_READ_CHUNK_SIZE])
                        
                        # Pass max_seq_len to enforce truncation
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

                        if (
                            input_data is None
                            or target_data is None
                            or global_data is None
                        ):
                            logger.error(
                                "A required variable was missing from HDF5 file. Aborting."
                            )
                            return False

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

                        seq_in = torch.from_numpy(seq_in_np).to(device).float()
                        tgt = torch.from_numpy(tgt_np).to(device).float()
                        glb = (
                            torch.from_numpy(glb_np).to(device).float()
                            if global_vars and glb_np is not None
                            else None
                        )

                        for j, var in enumerate(input_vars):
                            method = normalizer.key_methods.get(var, "none")
                            stats = norm_metadata.get("per_key_stats", {}).get(var)
                            if stats:
                                seq_in[..., j] = normalizer.normalize_tensor(
                                    seq_in[..., j], method, stats
                                )
                        for j, var in enumerate(target_vars):
                            method = normalizer.key_methods.get(var, "none")
                            stats = norm_metadata.get("per_key_stats", {}).get(var)
                            if stats:
                                tgt[..., j] = normalizer.normalize_tensor(
                                    tgt[..., j], method, stats
                                )
                        if global_vars and glb is not None:
                            for j, var in enumerate(global_vars):
                                method = normalizer.key_methods.get(var, "none")
                                stats = norm_metadata.get("per_key_stats", {}).get(var)
                                if stats:
                                    glb[..., j] = normalizer.normalize_tensor(
                                        glb[..., j], method, stats
                                    )

                        pad_width = max_seq_len - seq_in.shape[1]
                        if pad_width > 0:
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

                        shard_buffers["sequence_inputs"].append(
                            seq_in_np.astype(np.float32)
                        )
                        shard_buffers["targets"].append(tgt_np.astype(np.float32))
                        if global_vars and glb_np is not None:
                            shard_buffers["globals"].append(glb_np.astype(np.float32))

                        num_samples_in_buffer += len(chunk_idx)
                        pbar.update(len(chunk_idx))

                        while num_samples_in_buffer >= shard_size:
                            full_seq_data = np.concatenate(
                                shard_buffers["sequence_inputs"], axis=0
                            )
                            full_tgt_data = np.concatenate(
                                shard_buffers["targets"], axis=0
                            )
                            seq_to_save = full_seq_data[:shard_size]
                            tgt_to_save = full_tgt_data[:shard_size]
                            temp_shard_buffer = {
                                "sequence_inputs": [seq_to_save],
                                "targets": [tgt_to_save],
                            }

                            if global_vars and shard_buffers.get("globals"):
                                full_glb_data = np.concatenate(
                                    shard_buffers["globals"], axis=0
                                )
                                glb_to_save = full_glb_data[:shard_size]
                                temp_shard_buffer["globals"] = [glb_to_save]
                                shard_buffers["globals"] = (
                                    [full_glb_data[shard_size:]]
                                    if full_glb_data.shape[0] > shard_size
                                    else []
                                )

                            _save_shard(
                                temp_shard_buffer,
                                seq_dir,
                                tgt_dir,
                                glb_dir,
                                current_shard_idx,
                            )
                            current_shard_idx += 1

                            shard_buffers["sequence_inputs"] = (
                                [full_seq_data[shard_size:]]
                                if full_seq_data.shape[0] > shard_size
                                else []
                            )
                            shard_buffers["targets"] = (
                                [full_tgt_data[shard_size:]]
                                if full_tgt_data.shape[0] > shard_size
                                else []
                            )
                            num_samples_in_buffer -= shard_size

        if num_samples_in_buffer > 0:
            final_seq = np.concatenate(shard_buffers["sequence_inputs"], axis=0)
            final_tgt = np.concatenate(shard_buffers["targets"], axis=0)

            num_to_pad = shard_size - num_samples_in_buffer
            if num_to_pad > 0:
                seq_pad_shape = (num_to_pad, final_seq.shape[1], final_seq.shape[2])
                seq_padding = np.full(seq_pad_shape, padding_value, dtype=np.float32)
                final_seq = np.concatenate([final_seq, seq_padding], axis=0)

                tgt_pad_shape = (num_to_pad, final_tgt.shape[1], final_tgt.shape[2])
                tgt_padding = np.full(tgt_pad_shape, padding_value, dtype=np.float32)
                final_tgt = np.concatenate([final_tgt, tgt_padding], axis=0)

            final_buffers = {"sequence_inputs": [final_seq], "targets": [final_tgt]}

            if global_vars and shard_buffers["globals"]:
                final_glb = np.concatenate(shard_buffers["globals"], axis=0)
                if num_to_pad > 0:
                    glb_pad_shape = (num_to_pad, final_glb.shape[1])
                    glb_padding = np.full(
                        glb_pad_shape, padding_value, dtype=np.float32
                    )
                    final_glb = np.concatenate([final_glb, glb_padding], axis=0)
                final_buffers["globals"] = [final_glb]

            _save_shard(final_buffers, seq_dir, tgt_dir, glb_dir, current_shard_idx)

        logger.info(
            f"Completed {split_name} split in {time.time() - split_start_time:.2f}s."
        )

    _save_preprocessing_summary(
        processed_dir, config, all_splits, norm_metadata, current_hash
    )
    hash_path.write_text(current_hash)
    total_time = time.time() - preprocessing_start_time
    logger.info(f"Preprocessing completed successfully in {total_time:.2f}s.")
    return True


__all__ = ["preprocess_data"]
