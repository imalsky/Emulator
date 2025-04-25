#!/usr/bin/env python3
"""
dataset.py – Dataset utilities for the atmospheric‑flux transformer.

Updates
-------
* `_validate_profile` now checks key presence first and raises **clear** `ValueError`s instead
  of uncaught `KeyError`s.
* `StrictCollate` verifies *shape consistency* across the batch after stacking to catch
  rogue sequence‑length mismatches early.
* Minor type‑hint & log‑message polish; functional behaviour otherwise unchanged.
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
    """Load **normalised** JSON profiles and return tensors ready for the model."""

    # ------------------------------------------------------------------
    # Construction & configuration validation
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

        # --- paths -----------------------------------------------------
        self.data_dir = Path(data_folder)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(self.data_dir)

        # --- store config ---------------------------------------------
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.global_variables = global_variables or []
        self.sequence_types = {k: v for k, v in sequence_types.items() if v}
        self.sequence_lengths = sequence_lengths
        self.output_seq_type = output_seq_type

        # --- sanity checks --------------------------------------------
        self._check_cfg()

        # --- validate every file once ---------------------------------
        self.valid_files = self._scan_and_validate()
        logger.info("%d profiles validated in %s", len(self.valid_files), self.data_dir)

        # --- simple LRU cache -----------------------------------------
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[int, Tuple[Dict[str, Tensor], Tensor]] = OrderedDict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_cfg(self) -> None:  # noqa: C901  (simple but long)
        """Ensure configuration dictionaries are self‑consistent."""
        if not self.sequence_types:
            raise ValueError("sequence_types is empty – nothing to load")
        if self.output_seq_type not in self.sequence_types:
            raise ValueError("output_seq_type not found in sequence_types")
        try:
            self.target_len = int(self.sequence_lengths[self.output_seq_type])
        except (KeyError, ValueError):
            raise ValueError("Missing or invalid sequence_lengths for output_seq_type")
        if self.target_len <= 0:
            raise ValueError("target_len must be positive")

    # .................................................................
    # Full profile validation – runs once at construction time
    # .................................................................

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
        # --- global scalars ------------------------------------------
        for gv in self.global_variables:
            if gv not in prof:
                raise ValueError(f"{gv} missing in {name}")
            if not isinstance(prof[gv], (int, float)):
                raise ValueError(f"{gv} must be numeric in {name}")

        # --- sequence length consistency -----------------------------
        for seq_name, vars_ in self.sequence_types.items():
            for v in vars_:
                if v not in prof:
                    raise ValueError(f"{v} missing in {name}")
            lengths = {len(prof[v]) for v in vars_}
            if len(lengths) != 1:
                raise ValueError(f"Length mismatch in {seq_name} ({name})")

        # --- target length match -------------------------------------
        for tv in self.target_variables:
            if tv not in prof:
                raise ValueError(f"{tv} missing in {name}")
            if len(prof[tv]) != self.target_len:
                raise ValueError(f"Target length mismatch for {tv} in {name}")

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor]:
        # --- LRU cache lookup ----------------------------------------
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        fp = self.valid_files[idx]
        prof: Dict[str, Any] = json.loads(fp.read_text(encoding="utf-8-sig"))

        # Build inputs dictionary ------------------------------------
        sample: Dict[str, Tensor] = {}
        if self.global_variables:
            sample["global"] = torch.tensor(
                [prof[g] for g in self.global_variables], dtype=torch.float32
            )
        for seq_name, vars_ in self.sequence_types.items():
            seq_tensor = torch.stack(
                [torch.tensor(prof[v], dtype=torch.float32) for v in vars_], dim=1
            )
            sample[seq_name] = seq_tensor

        # Build targets tensor ---------------------------------------
        target = torch.stack(
            [torch.tensor(prof[tv], dtype=torch.float32) for tv in self.target_variables], dim=1
        )

        # Update LRU cache -------------------------------------------
        self._cache[idx] = (sample, target)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return sample, target

# -----------------------------------------------------------------------------
# Collate function (pickle‑safe)
# -----------------------------------------------------------------------------

class StrictCollate:
    """Guarantee identical key set *and tensor shapes* across the batch."""

    def __call__(
        self, batch: List[Tuple[Dict[str, Tensor], Tensor]]
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        if not batch:
            return {}, torch.empty(0)

        keys = list(batch[0][0].keys())
        if any(list(sample[0].keys()) != keys for sample in batch):
            raise ValueError("Input key mismatch inside batch")

        # stack & shape‑check
        inputs: Dict[str, Tensor] = {}
        for k in keys:
            stacked = torch.stack([sample[0][k] for sample in batch], dim=0)
            shapes = {sample[0][k].shape for sample in batch}
            if len(shapes) != 1:
                raise ValueError(f"Shape mismatch for key '{k}' in collate: {shapes}")
            inputs[k] = stacked

        targets = torch.stack([sample[1] for sample in batch], dim=0)
        return inputs, targets

# Convenience factory ---------------------------------------------------------

def create_multi_source_collate_fn() -> Callable:
    return StrictCollate()

__all__ = [
    "AtmosphericDataset",
    "create_multi_source_collate_fn",
    "StrictCollate",
]
