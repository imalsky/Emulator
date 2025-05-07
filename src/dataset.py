#!/usr/bin/env python3
"""
dataset.py – dataset utilities for the atmospheric-flux transformer.
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class AtmosphericDataset(Dataset):
    """
    Loads *normalised* JSON profiles and returns tensors for the model.
    This class handles the discovery, validation (optional), and loading of
    atmospheric profile data stored in JSON files. It also implements an
    optional LRU cache for frequently accessed profiles.
    """

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
        """
        Initializes the AtmosphericDataset.

        Args:
            data_folder: Path to the directory containing JSON profile files.
            input_variables: List of variable names to be used as model inputs.
            target_variables: List of variable names to be used as model targets.
            global_variables: List of variable names representing global scalar inputs.
            sequence_types: Maps sequence type names (e.g., 'flux', 'profile')
                            to lists of variable names belonging to that type.
            sequence_lengths: Maps sequence type names to their expected lengths.
            output_seq_type: The sequence type name of the primary target variable.
            cache_size: Maximum number of profiles to keep in the LRU cache.
                        Set to 0 or less to disable caching.
            validate_profiles: If True, validates each profile's structure and
                               content upon dataset initialization.
        """
        super().__init__()

        self.data_dir = Path(data_folder)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(self.data_dir)

        self.input_variables = input_variables
        self.target_variables = target_variables
        self.global_variables = global_variables or []
        self.sequence_types = {k: v for k, v in sequence_types.items() if v}
        self.sequence_lengths = sequence_lengths
        self.output_seq_type = output_seq_type
        self.validate_profiles = bool(validate_profiles)

        self._check_cfg()

        if self.validate_profiles:
            self.valid_files = self._scan_and_validate()
        else:
            self.valid_files = [
                p for p in sorted(self.data_dir.glob("*.json")) if p.name != "normalization_metadata.json"
            ]
            if not self.valid_files:
                raise FileNotFoundError(f"No JSON profiles found in {self.data_dir}")
            logger.info("%d profiles discovered (validation skipped)", len(self.valid_files))

        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, Tuple[Dict[str, Tensor], Tensor]] = OrderedDict()

    def _check_cfg(self) -> None:
        """
        Validates the internal configuration dictionaries for self-consistency.
        Ensures that sequence types and lengths are properly defined, especially
        for the output sequence.
        """
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

    def _scan_and_validate(self) -> List[Path]:
        """
        Scans the data directory for JSON files and validates each one.
        This method is called during initialization if `validate_profiles` is True.

        Returns:
            A list of Path objects for valid profile files.
        """
        files = [p for p in sorted(self.data_dir.glob("*.json")) if p.name != "normalization_metadata.json"]
        if not files:
            raise FileNotFoundError(f"No JSON profiles found in {self.data_dir}")
        for fp in files:
            prof = json.loads(fp.read_text(encoding="utf-8-sig"))
            self._validate_profile(prof, fp.name)
        logger.info("%d profiles validated in %s", len(files), self.data_dir)
        return files

    def _validate_profile(self, prof: Dict[str, Any], name: str) -> None:
        """
        Validates a single profile dictionary against the dataset configuration.
        Checks for missing variables, correct types, and consistent sequence lengths.

        Args:
            prof: The profile data loaded from a JSON file.
            name: The filename of the profile, for logging purposes.
        """
        for gv in self.global_variables:
            if gv not in prof:
                raise ValueError(f"{gv} missing in {name}")
            if not isinstance(prof[gv], (int, float)):
                raise ValueError(f"{gv} must be numeric in {name}")

        for seq_name, vars_ in self.sequence_types.items():
            for v in vars_:
                if v not in prof:
                    raise ValueError(f"{v} missing in {name}")
            lengths = {len(prof[v]) for v in vars_}
            if len(lengths) != 1:
                raise ValueError(f"Length mismatch in {seq_name} ({name})")

        for tv in self.target_variables:
            if tv not in prof:
                raise ValueError(f"{tv} missing in {name}")
            if len(prof[tv]) != self.target_len:
                raise ValueError(f"Target length mismatch for {tv} in {name}")

    def __len__(self) -> int:
        """Returns the total number of valid profiles in the dataset."""
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Retrieves a single profile by its index, loads it, converts to tensors,
        and applies caching.

        Args:
            idx: The index of the profile to retrieve.

        Returns:
            A tuple containing:
                - A dictionary of input tensors, keyed by sequence/global names.
                - A tensor (or stack of tensors) representing the target(s).
        """
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        fp = self.valid_files[idx]
        prof: Dict[str, Any] = json.loads(fp.read_text(encoding="utf-8-sig"))

        sample: Dict[str, Tensor] = {}
        if self.global_variables:
            sample["global"] = torch.tensor([prof[g] for g in self.global_variables], dtype=torch.float32)
        for seq_name, vars_ in self.sequence_types.items():
            seq_tensor = torch.stack([torch.tensor(prof[v], dtype=torch.float32) for v in vars_], dim=1)
            sample[seq_name] = seq_tensor

        target = torch.stack([torch.tensor(prof[tv], dtype=torch.float32) for tv in self.target_variables], dim=1)

        if len(self._cache) >= self.cache_size and self.cache_size > 0:
            self._cache.popitem(last=False)
        if self.cache_size > 0 :
             self._cache[idx] = (sample, target) # Store in cache only if cache_size > 0
        return sample, target

    def get_profile_filenames_by_indices(self, indices: List[int]) -> List[str]:
        """
        Retrieves the filenames for a given list of profile indices.

        Args:
            indices: A list of integer indices corresponding to self.valid_files.

        Returns:
            A list of profile filenames (e.g., ["profile_001.json", "profile_ABC.json"]).
        """
        if not all(0 <= i < len(self.valid_files) for i in indices):
            invalid_indices = [i for i in indices if not (0 <= i < len(self.valid_files))]
            raise IndexError(
                f"Invalid indices found: {invalid_indices}. "
                f"Dataset size is {len(self.valid_files)}."
            )
        return [self.valid_files[i].name for i in indices]


class StrictCollate:
    """
    A collate function that ensures all samples in a batch have identical input keys
    and that tensors for each key have identical shapes across the batch before stacking.
    """

    def __call__(
        self, batch: List[Tuple[Dict[str, Tensor], Tensor]]
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Collates a list of samples into a batch.

        Args:
            batch: A list of (inputs_dict, target_tensor) tuples.

        Returns:
            A tuple containing:
                - A dictionary of batched input tensors.
                - A batched tensor of targets.
        Raises:
            ValueError: If input keys or tensor shapes are inconsistent within the batch.
        """
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


def create_multi_source_collate_fn() -> StrictCollate:
    """
    Factory function that returns an instance of StrictCollate.
    This maintains a consistent interface for creating the collate function.
    """
    return StrictCollate()

__all__ = [
    "AtmosphericDataset",
    "StrictCollate",
    "create_multi_source_collate_fn",
]
