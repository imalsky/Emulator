#!/usr/bin/env python3
"""
Dataset utilities for the atmospheric-flux transformer.

Key features
------------
* Strict configuration validation: every sequence type must be defined and the
  `output_seq_type` must have its length declared in *sequence_lengths*.
* End-to-end profile validation at start-up - you fail fast if even **one** JSON
  file breaks the contract (missing key, heterogeneous sequence length, etc.).
* LRU cache so repeatedly sampling from the same few thousand profiles during
  training doesn't thrash the file-system.
* A pickle-friendly StrictCollate that enforces shape consistency and
  handles the common corner-case where there is only **one** target variable.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dataset implementation
# -----------------------------------------------------------------------------

class AtmosphericDataset(Dataset):
    """Loads normalised JSON profiles and returns tensors ready for the model."""

    # ------------------------------------------------------------------
    # Constructor & top-level validation
    # ------------------------------------------------------------------
    def __init__(
        self,
        data_folder: Union[str, Path],
        input_variables: List[str],
        target_variables: List[str],
        global_variables: List[str],
        sequence_types: Dict[str, List[str]],
        sequence_lengths: Dict[str, int],
        output_seq_type: str,
        cache_size: int = 1024,
    ) -> None:
        super().__init__()

        # Path handling --------------------------------------------------
        self.data_dir = Path(data_folder)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(self.data_dir)

        # Store config ----------------------------------------------------
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.global_variables = global_variables or []
        self.sequence_types = {k: v for k, v in sequence_types.items() if v}
        self.sequence_lengths = sequence_lengths
        self.output_seq_type = output_seq_type

        # Sanity checks ---------------------------------------------------
        self._check_cfg()

        # Validate every file once ---------------------------------------
        self.valid_files = self._scan_and_validate()
        logger.info("%d profiles validated in %s", len(self.valid_files), self.data_dir)

        # LRU cache -------------------------------------------------------
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[int, Tuple[Dict[str, Tensor], Tensor]] = OrderedDict()

    # ------------------------------------------------------------------
    # Helper: configuration sanity check
    # ------------------------------------------------------------------
    def _check_cfg(self) -> None:
        """Ensure the configuration dictionaries are self-consistent."""
        if not self.sequence_types:
            raise ValueError("sequence_types is empty - nothing to load")
        if self.output_seq_type not in self.sequence_types:
            raise ValueError("output_seq_type not found in sequence_types")
        try:
            self.target_len = int(self.sequence_lengths[self.output_seq_type])
        except (KeyError, ValueError):
            raise ValueError("Missing or invalid sequence_lengths for output_seq_type")
        if self.target_len <= 0:
            raise ValueError("target_len must be positive")

    # ------------------------------------------------------------------
    # Helper: full profile validation
    # ------------------------------------------------------------------
    def _scan_and_validate(self) -> List[Path]:
        """Iterate through every JSON file once and validate its structure."""
        files = [p for p in sorted(self.data_dir.glob("*.json")) if p.name != "normalization_metadata.json"]
        if not files:
            raise FileNotFoundError("No JSON profiles found")
        for fp in files:
            prof = json.loads(fp.read_text(encoding="utf-8-sig"))
            self._validate_profile(prof, fp.name)
        return files

    def _validate_profile(self, prof: Dict[str, Any], name: str) -> None:
        """Raise immediately if anything is wrong with this profile."""
        # Global scalar sanity -------------------------------------------
        for gv in self.global_variables:
            if gv not in prof:
                raise ValueError(f"{gv} missing in {name}")
            if not isinstance(prof[gv], (int, float)):
                raise ValueError(f"{gv} must be numeric in {name}")

        # Sequence length consistency -----------------------------------
        for seq_name, vars_ in self.sequence_types.items():
            lengths = {len(prof[v]) for v in vars_}
            if len(lengths) != 1:
                raise ValueError(f"Length mismatch in {seq_name} ({name})")
        # Target length match -------------------------------------------
        for tv in self.target_variables:
            if len(prof[tv]) != self.target_len:
                raise ValueError(f"Target length mismatch for {tv} in {name}")

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor]:
        # LRU cache lookup -------------------------------------------------
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        fp = self.valid_files[idx]
        prof = json.loads(fp.read_text(encoding="utf-8-sig"))

        # Build inputs dictionary -----------------------------------------
        sample: Dict[str, Tensor] = {}
        if self.global_variables:
            sample["global"] = torch.tensor([prof[g] for g in self.global_variables], dtype=torch.float32)
        for seq_name, vars_ in self.sequence_types.items():
            seq_tensor = torch.stack([torch.tensor(prof[v], dtype=torch.float32) for v in vars_], dim=1)
            sample[seq_name] = seq_tensor

        # Build targets tensor -------------------------------------------
        target = torch.stack([torch.tensor(prof[tv], dtype=torch.float32) for tv in self.target_variables], dim=1)

        # Update LRU cache ------------------------------------------------
        self._cache[idx] = (sample, target)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return sample, target

# -----------------------------------------------------------------------------
# Collate function (pickle safe)
# -----------------------------------------------------------------------------

class StrictCollate:
    """Guarantees identical key set and shape across the batch before stacking."""

    def __call__(self, batch: List[Tuple[Dict[str, Tensor], Tensor]]) -> Tuple[Dict[str, Tensor], Tensor]:
        if not batch:
            return {}, torch.empty(0)

        keys = list(batch[0][0].keys())
        if any(list(s[0].keys()) != keys for s in batch):
            raise ValueError("Input key mismatch inside batch")

        inputs: Dict[str, Tensor] = {k: torch.stack([s[0][k] for s in batch], dim=0) for k in keys}
        targets = torch.stack([s[1] for s in batch], dim=0)
        return inputs, targets

# Convenience factory ---------------------------------------------------------

def create_multi_source_collate_fn() -> Callable:
    return StrictCollate()

__all__ = [
    "AtmosphericDataset",
    "StrictCollate",
    "create_multi_source_collate_fn",
]
