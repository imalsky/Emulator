#!/usr/bin/env python3
"""
dataset.py – Atmospheric profile dataset with strict fixed sequence lengths.

Enforces that sequence lengths in profiles exactly match those provided
in the 'sequence_lengths' dictionary during initialization.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AtmosphericDataset(Dataset):
    """
    Dataset of atmospheric profiles, requiring sequence lengths specified during init.
    """
    def __init__(
        self,
        data_folder: Union[str, Path],
        input_variables: List[str],
        target_variables: List[str],
        global_variables: List[str],
        sequence_types: Dict[str, List[str]],
        sequence_lengths: Dict[str, int], # Expects lengths from config
        output_seq_type: str, # Expects type key from config
        cache_size: int = 1024,
    ) -> None:
        """
        Initialize dataset with strict sequence length validation based on config.

        Parameters:
        ----------
            data_folder : Path to normalized JSON profile files.
            input_variables : Names of input variables.
            target_variables : Names of target variables.
            global_variables : Names of global (scalar) features.
            sequence_types : Mapping of sequence type names to variable lists.
            sequence_lengths : Dict mapping sequence type keys to required lengths.
            output_seq_type : Key indicating which sequence determines target length.
            cache_size : Max number of profiles to cache in memory.
        """
        self.data_folder = Path(data_folder)
        if not self.data_folder.is_dir():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        # Store config
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.global_variables = global_variables
        self.sequence_types = {k: v for k, v in sequence_types.items() if v} # Filter empty
        self.sequence_lengths = sequence_lengths
        self.output_seq_type = output_seq_type

        # --- Validation based on provided lengths ---
        # Ensure all defined sequence types have a length provided
        missing_seq_lens = set(self.sequence_types.keys()) - set(self.sequence_lengths.keys())
        if missing_seq_lens:
            raise ValueError(f"Missing sequence_lengths entries for: {missing_seq_lens}")

        # Ensure output_seq_type is valid and has a length
        if self.output_seq_type not in self.sequence_lengths:
            raise ValueError(f"output_seq_type '{self.output_seq_type}' not found in sequence_lengths keys")
        self.target_length = self.sequence_lengths[self.output_seq_type]
        # --- End Validation ---

        # Find and validate profile files
        all_files = sorted(self.data_folder.glob("*.json"))
        candidate_files = [p for p in all_files if p.name != "normalization_metadata.json"]
        if not candidate_files:
            raise FileNotFoundError(f"No JSON profiles found in {self.data_folder}")

        self.valid_files: List[Path] = []
        invalid_count = 0
        for fpath in candidate_files:
            try:
                prof = self._load_json(fpath)
                if prof is None: # Skip files that fail to load
                    invalid_count += 1
                    continue
                self._validate_profile(prof, fpath.name) # Raises ValueError on failure
                self.valid_files.append(fpath)
            except ValueError as e:
                logger.debug(f"Skipping invalid profile {fpath.name}: {e}")
                invalid_count += 1
            except Exception as e:
                 logger.error(f"Unexpected error processing {fpath.name}: {e}", exc_info=True)
                 invalid_count +=1


        if not self.valid_files:
            raise ValueError("No valid profiles found after validation")

        logger.info(f"Loaded {len(self.valid_files)} valid profiles "
                    f"(skipped {invalid_count} invalid/mismatched) from {self.data_folder}")
        logger.info(f"Strict lengths required: {self.sequence_lengths} (Targets match '{self.output_seq_type}')")


        # Simple LRU cache
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[int, Tuple[Dict[str, Tensor], Tensor]] = OrderedDict()


    def _load_json(self, path: Path) -> Optional[Dict]:
        """Loads a single JSON file, returning None on error."""
        try:
            return json.loads(path.read_text())
        except Exception as e:
            logger.warning(f"Failed to load JSON {path.name}: {e}")
            return None

    def _validate_profile(self, prof: Dict, name: str) -> None:
        """Ensure profile matches required keys and exact lengths."""
        # Check global scalars
        for gv in self.global_variables:
            if gv not in prof or not isinstance(prof[gv], (int, float)):
                raise ValueError(f"Global var '{gv}' missing or not scalar")

        # Check each sequence type
        for seq_key, vars_list in self.sequence_types.items():
            expected_len = self.sequence_lengths[seq_key]
            for var in vars_list:
                val = prof.get(var)
                if not isinstance(val, list):
                    raise ValueError(f"Sequence var '{var}' not list in '{seq_key}'")
                if len(val) != expected_len:
                    raise ValueError(
                        f"Sequence '{seq_key}' var '{var}' length {len(val)} "
                        f"!= expected {expected_len}"
                    )

        # Check targets match the specified target_length
        for tv in self.target_variables:
            val = prof.get(tv)
            if not isinstance(val, list):
                raise ValueError(f"Target var '{tv}' not list")
            if len(val) != self.target_length:
                raise ValueError(
                    f"Target '{tv}' length {len(val)} != expected {self.target_length} "
                    f"(from output_seq_type '{self.output_seq_type}')"
                )

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor]:
        # Return from cache if available
        if idx in self._cache:
            # Move to end for LRU
            self._cache.move_to_end(idx)
            return self._cache[idx]

        if not 0 <= idx < len(self.valid_files):
             raise IndexError("Dataset index out of range")

        fpath = self.valid_files[idx]
        prof = self._load_json(fpath)
        if prof is None:
            # Should not happen after initial validation, but handle defensively
            raise RuntimeError(f"Failed to load previously validated file: {fpath.name}")


        # Build inputs dictionary
        inputs: Dict[str, Tensor] = {}

        # Global features tensor [num_globals]
        if self.global_variables:
            gv_vals = [float(prof[v]) for v in self.global_variables]
            inputs['global'] = torch.tensor(gv_vals, dtype=torch.float32)

        # Sequence features: each [seq_len, num_vars_in_seq]
        for seq_key, vars_list in self.sequence_types.items():
            # Stack each variable's list into a column tensor
            cols = [torch.tensor(prof[v], dtype=torch.float32) for v in vars_list]
            # cols is list of [seq_len], stack to [seq_len, num_vars_in_seq]
            inputs[seq_key] = torch.stack(cols, dim=1)

        # Targets: [tgt_len, num_targets]
        tgt_cols = [
            torch.tensor(prof[tv], dtype=torch.float32)
            for tv in self.target_variables
        ]
        targets = torch.stack(tgt_cols, dim=1)

        # Cache and maybe evict oldest
        self._cache[idx] = (inputs, targets)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False) # pop oldest

        return inputs, targets


class StrictCollate:
    """
    Pickle‑able collate for strict‑length multi‑source batches.
    Stacks each key along a new 0th (batch) dimension.
    """
    def __call__(
        self, batch: List[Tuple[Dict[str, Tensor], Tensor]]
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        if not batch:
            return {}, torch.empty(0)

        # Assumes all items in batch have the same dict keys for inputs
        # and consistent tensor shapes (guaranteed by AtmosphericDataset)
        batched_inputs: Dict[str, List[Tensor]] = {}
        batched_targets: List[Tensor] = []

        # Gather tensors for each key
        for inp_dict, tgt_tensor in batch:
            for key, tensor in inp_dict.items():
                batched_inputs.setdefault(key, []).append(tensor)
            batched_targets.append(tgt_tensor)

        # Stack tensors along new batch dimension (dim=0)
        final_inputs = {key: torch.stack(tensors) for key, tensors in batched_inputs.items()}
        final_targets = torch.stack(batched_targets)

        return final_inputs, final_targets


def create_multi_source_collate_fn() -> StrictCollate:
    """
    Factory returning a StrictCollate instance (needed for pickling).
    """
    return StrictCollate()


__all__ = [
    "AtmosphericDataset",
    "StrictCollate",
    "create_multi_source_collate_fn",
]