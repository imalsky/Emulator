#!/usr/bin/env python3
"""
dataset.py – dataset utilities for the atmospheric-flux transformer.

Updates (2025-04-28)
--------------------
* **Optional validation** – constructor accepts `validate_profiles: bool` (default
  *True*).  When *False* the expensive one-off JSON scan is skipped and every
  ``*.json`` file (except the metadata) is assumed valid.
* **StrictCollate** – shape consistency is now checked *before* stacking to avoid
  wasted work on malformed batches.
* Backwards-compatible: existing call-sites need not pass the new argument.
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

# --------------------------------------------------------------------------- #
# Dataset implementation                                                       #
# --------------------------------------------------------------------------- #


class AtmosphericDataset(Dataset):
    """Load *normalised* JSON profiles and return tensors for the model."""

    # .....................................................................
    # construction / config validation
    # .....................................................................
    def __init__(
        self,
        data_folder: Union[str, Path],
        input_variables: List[str],
        target_variables: List[str],
        global_variables: List[str],
        sequence_types: Dict[str, List[str]],
        sequence_lengths: Dict[str, int],
        output_seq_type: str,
        *,
        cache_size: int = 1_024,
        validate_profiles: bool = True,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_folder)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(self.data_dir)

        # store cfg
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.global_variables = global_variables or []
        self.sequence_types = {k: v for k, v in sequence_types.items() if v}
        self.sequence_lengths = sequence_lengths
        self.output_seq_type = output_seq_type
        self.validate_profiles = bool(validate_profiles)

        # sanity-check config itself (cheap)
        self._check_cfg()

        # optionally scan / validate every JSON file once
        if self.validate_profiles:
            self.valid_files = self._scan_and_validate()
        else:
            self.valid_files = [
                p for p in sorted(self.data_dir.glob("*.json")) if p.name != "normalization_metadata.json"
            ]
            if not self.valid_files:
                raise FileNotFoundError("No JSON profiles found")
            logger.info("%d profiles discovered (validation skipped)", len(self.valid_files))

        # simple LRU cache ⟨idx → (inputs, target)⟩
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, Tuple[Dict[str, Tensor], Tensor]] = OrderedDict()

    # .....................................................................
    # private helpers
    # .....................................................................
    def _check_cfg(self) -> None:
        """Validate internal dictionaries for self-consistency."""
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
    # full profile validation (optional)                                
    # .................................................................
    def _scan_and_validate(self) -> List[Path]:
        files = [p for p in sorted(self.data_dir.glob("*.json")) if p.name != "normalization_metadata.json"]
        if not files:
            raise FileNotFoundError("No JSON profiles found")
        for fp in files:
            prof = json.loads(fp.read_text(encoding="utf-8-sig"))
            self._validate_profile(prof, fp.name)
        logger.info("%d profiles validated in %s", len(files), self.data_dir)
        return files

    def _validate_profile(self, prof: Dict[str, Any], name: str) -> None:
        # --- global scalars -----------------------------------------
        for gv in self.global_variables:
            if gv not in prof:
                raise ValueError(f"{gv} missing in {name}")
            if not isinstance(prof[gv], (int, float)):
                raise ValueError(f"{gv} must be numeric in {name}")

        # --- sequence length consistency ----------------------------
        for seq_name, vars_ in self.sequence_types.items():
            for v in vars_:
                if v not in prof:
                    raise ValueError(f"{v} missing in {name}")
            lengths = {len(prof[v]) for v in vars_}
            if len(lengths) != 1:
                raise ValueError(f"Length mismatch in {seq_name} ({name})")

        # --- target length match ------------------------------------
        for tv in self.target_variables:
            if tv not in prof:
                raise ValueError(f"{tv} missing in {name}")
            if len(prof[tv]) != self.target_len:
                raise ValueError(f"Target length mismatch for {tv} in {name}")

    # ------------------------------------------------------------------
    # PyTorch Dataset interface                                          
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor]:
        # LRU cache lookup
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        fp = self.valid_files[idx]
        prof: Dict[str, Any] = json.loads(fp.read_text(encoding="utf-8-sig"))

        # build inputs
        sample: Dict[str, Tensor] = {}
        if self.global_variables:
            sample["global"] = torch.tensor([prof[g] for g in self.global_variables], dtype=torch.float32)
        for seq_name, vars_ in self.sequence_types.items():
            seq_tensor = torch.stack(
                [torch.tensor(prof[v], dtype=torch.float32) for v in vars_], dim=1
            )
            sample[seq_name] = seq_tensor

        # targets
        target = torch.stack(
            [torch.tensor(prof[tv], dtype=torch.float32) for tv in self.target_variables], dim=1
        )

        # update cache
        self._cache[idx] = (sample, target)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return sample, target


# --------------------------------------------------------------------------- #
# Collate function (pickle-safe)                                              #
# --------------------------------------------------------------------------- #


class StrictCollate:
    """Guarantee identical keys *and* tensor shapes across the batch."""

    def __call__(
        self, batch: List[Tuple[Dict[str, Tensor], Tensor]]
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        if not batch:
            return {}, torch.empty(0)

        keys = list(batch[0][0].keys())
        if any(list(sample[0].keys()) != keys for sample in batch):
            raise ValueError("Input key mismatch inside batch")

        inputs: Dict[str, Tensor] = {}
        for k in keys:
            shapes = [sample[0][k].shape for sample in batch]
            if len(set(shapes)) != 1:
                raise ValueError(f"Shape mismatch for key '{k}' in collate: {set(shapes)}")
            inputs[k] = torch.stack([sample[0][k] for sample in batch], dim=0)

        targets = torch.stack([sample[1] for sample in batch], dim=0)
        return inputs, targets


# --------------------------------------------------------------------------- #
# Convenience factory                                                         #
# --------------------------------------------------------------------------- #

def create_multi_source_collate_fn() -> Callable:
    return StrictCollate()


__all__ = [
    "AtmosphericDataset",
    "StrictCollate",
    "create_multi_source_collate_fn",
]