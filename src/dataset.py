#!/usr/bin/env python3
"""
dataset.py – Atmospheric profile dataset with dynamic sequence length handling.

This module defines the `AtmosphericDataset` class, which loads atmospheric
profile data from JSON files. It validates that variables within the same
sequence type have consistent lengths within each profile and that target
variables match the length specified by the config's `output_seq_type` length.
It raises errors if validation fails. It also provides a `StrictCollate` helper
class for batching, which now explicitly checks for consistent tensor shapes
across the batch before stacking.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Class
# =============================================================================


class AtmosphericDataset(Dataset):
    """
    Dataset of atmospheric profiles from individual JSON files.

    Validates profiles for internal consistency (variables within a sequence type
    must have the same length) and target length consistency (target variables
    must match the length specified for the `output_seq_type` in the config's
    `sequence_lengths`). Raises errors on validation failure.
    Includes an LRU cache for faster access during training.
    """

    def __init__(
        self,
        data_folder: Union[str, Path],
        input_variables: List[str],
        target_variables: List[str],
        global_variables: List[str],
        sequence_types: Dict[str, List[str]],
        sequence_lengths: Dict[str, int], # Still needed for target validation
        output_seq_type: str,
        cache_size: int = 1024,
    ) -> None:
        """
        Initializes the dataset and performs strict validation.

        Args:
            data_folder: Path to the directory containing normalized JSON profiles.
            input_variables: List of variable names expected as input to the model.
            target_variables: List of variable names expected as targets.
            global_variables: List of variable names representing scalar global features.
            sequence_types: Dictionary mapping sequence type names (e.g., "atmosphere")
                            to lists of variable names belonging to that sequence type.
            sequence_lengths: Dictionary mapping sequence type names to their required
                              integer length. **Crucially used for target validation.**
            output_seq_type: The key from `sequence_types` whose length (defined in
                             `sequence_lengths`) determines the expected length of
                             the target variables.
            cache_size: Maximum number of loaded profiles to keep in the LRU cache.

        Raises:
            FileNotFoundError: If the data_folder does not exist or contains no profiles.
            ValueError: If configuration is inconsistent, a profile fails validation,
                        or no valid profiles are found.
        """
        super().__init__()
        self.data_folder = Path(data_folder)
        if not self.data_folder.is_dir():
            raise FileNotFoundError(f"Data folder not found: {self.data_folder}")

        self.input_variables = input_variables
        self.target_variables = target_variables
        self.global_variables = global_variables or []
        # Filter out empty sequence types
        self.sequence_types = {
            k: v for k, v in (sequence_types or {}).items() if v
        }
        self.sequence_lengths = sequence_lengths # Keep for target length validation
        self.output_seq_type = output_seq_type

        self._validate_configuration() # Validates config consistency

        # This now raises errors immediately if any file fails validation
        self.valid_files = self._index_and_validate_profiles()

        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[int, Tuple[Dict[str, Tensor], Tensor]] = (
            OrderedDict()
        )

    def _validate_configuration(self) -> None:
        """Performs initial checks on the provided configuration arguments."""
        if not self.sequence_types:
            raise ValueError(
                "Configuration must define at least one non-empty 'sequence_types'."
            )

        # Check if the output sequence type is defined and has a length specified
        if self.output_seq_type not in self.sequence_types:
             raise ValueError(
                f"Configuration 'output_seq_type' ('{self.output_seq_type}') "
                f"is not defined as a key in 'sequence_types': {list(self.sequence_types.keys())}"
            )
        if self.output_seq_type not in self.sequence_lengths:
            raise ValueError(
                f"Configuration 'output_seq_type' ('{self.output_seq_type}') "
                f"must have a corresponding length defined in 'sequence_lengths'. "
                f"Keys available: {list(self.sequence_lengths.keys())}"
            )

        try:
            # Store the expected target length based on the output sequence type's length
            self.target_length = self.sequence_lengths[self.output_seq_type]
            if not isinstance(self.target_length, int) or self.target_length <= 0:
                 raise ValueError(f"Length for output sequence type '{self.output_seq_type}' in 'sequence_lengths' must be a positive integer.")
        except KeyError:
            # This case should be caught above, but safeguard
            raise ValueError(
                f"Output sequence type '{self.output_seq_type}' defined but lacks length in 'sequence_lengths'."
            )

        logger.info(
            f"Dataset configured. Target length validation based on '{self.output_seq_type}' requires length {self.target_length}."
        )
        logger.info(
            f"Input sequence lengths will be checked for consistency within each profile per sequence type."
        )


    def _index_and_validate_profiles(self) -> List[Path]:
        """
        Scans the data folder, validates profiles strictly, and returns list of valid file paths.

        Raises:
            FileNotFoundError: If no JSON files (excluding metadata) are found.
            ValueError: If any profile fails validation checks (missing keys, wrong types,
                        inconsistent sequence lengths, incorrect target length).
            RuntimeError: If JSON decoding fails for any file.
        """
        logger.info(
            "Indexing and strictly validating profile files in %s...", self.data_folder
        )
        all_files = sorted(self.data_folder.glob("*.json"))
        candidate_files = [
            p for p in all_files if p.name != "normalization_metadata.json"
        ]

        if not candidate_files:
            raise FileNotFoundError(
                f"No JSON profile files (excluding metadata) found in {self.data_folder}"
            )

        valid_files: List[Path] = []
        processed_count = 0

        for fpath in candidate_files:
            processed_count += 1
            try:
                # Use utf-8-sig to handle potential BOM
                prof = self._load_json(fpath)
                if prof is None:
                    # _load_json now raises RuntimeError on decode failure
                    # This path might be reached if _load_json is modified to return None on other errors
                    raise RuntimeError(f"Failed to load JSON from {fpath.name}")

                # This call will raise ValueError if validation fails
                self._validate_profile_contents(prof, fpath.name)
                valid_files.append(fpath)

            except (ValueError, RuntimeError) as e:
                # Catch validation errors (ValueError) or loading errors (RuntimeError)
                logger.error(f"Validation failed for profile {fpath.name}: {e}")
                logger.error("Stopping dataset initialization due to validation failure.")
                raise # Re-raise the caught error to stop the process
            except Exception as e:
                # Catch unexpected errors during processing of a single file
                logger.error(
                    f"Unexpected error processing {fpath.name}: {e}",
                    exc_info=True, # Log traceback for unexpected errors
                )
                raise RuntimeError(f"Unexpected error processing {fpath.name}") from e

        # If loop completes without raising errors, all files are valid
        logger.info(
            f"Successfully validated {len(valid_files)} profiles out of {processed_count} candidates."
        )
        return valid_files

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """
        Loads a single JSON file.

        Raises:
            RuntimeError: If JSON decoding or file reading fails.
        """
        try:
            # Use utf-8-sig to handle potential BOM
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {path.name}: {e}")
            raise RuntimeError(f"Failed to decode JSON in {path.name}") from e
        except OSError as e:
            logger.error(f"OS error reading file {path.name}: {e}")
            raise RuntimeError(f"Failed to read file {path.name}") from e
        except Exception as e:
            logger.error(f"Unexpected error loading JSON {path.name}: {e}")
            raise RuntimeError(f"Unexpected error loading JSON from {path.name}") from e

    def _validate_profile_contents(
        self, prof: Dict[str, Any], filename: str
    ) -> None:
        """
        Ensures profile dictionary matches required keys, types, and length consistencies.

        - Checks global variables are present, scalar, and finite.
        - Checks sequence variables are present and are lists.
        - Checks variables within the *same* sequence type have the *same* length.
        - Checks target variables are present, are lists, and match the `self.target_length`.

        Raises:
            ValueError: If any validation check fails.
        """
        # --- Global Variable Validation ---
        for gv in self.global_variables:
            val = prof.get(gv)
            if val is None:
                raise ValueError(f"Global variable '{gv}' is missing.")
            if not isinstance(val, (int, float)):
                raise ValueError(f"Global variable '{gv}' is not scalar (int/float), found {type(val).__name__}.")
            if not torch.isfinite(torch.tensor(val)):
                raise ValueError(f"Global variable '{gv}' is not finite ({val}).")

        # --- Input Sequence Variable Validation (Internal Consistency) ---
        for seq_key, vars_list in self.sequence_types.items():
            expected_len = -1 # Sentinel value: length not yet determined for this sequence type
            for i, var in enumerate(vars_list):
                val = prof.get(var)
                if val is None:
                    raise ValueError(f"Sequence variable '{var}' (type '{seq_key}') is missing.")
                if not isinstance(val, list):
                    raise ValueError(f"Sequence variable '{var}' (type '{seq_key}') is not a list, found {type(val).__name__}.")

                actual_len = len(val)
                if i == 0:
                    expected_len = actual_len # Set expected length based on the first variable in the list
                elif actual_len != expected_len:
                    raise ValueError(
                        f"Inconsistent lengths within sequence type '{seq_key}'. "
                        f"Variable '{vars_list[0]}' has length {expected_len}, but "
                        f"variable '{var}' has length {actual_len}."
                    )
                # Check for non-finite values within sequences (optional but recommended)
                # if not all(torch.isfinite(torch.tensor(x)) for x in val if isinstance(x, (int, float))):
                #     raise ValueError(f"Sequence variable '{var}' (type '{seq_key}') contains non-finite values.")


        # --- Target Variable Validation (Against Configured Target Length) ---
        for tv in self.target_variables:
            val = prof.get(tv)
            if val is None:
                raise ValueError(f"Target variable '{tv}' is missing.")
            if not isinstance(val, list):
                raise ValueError(f"Target variable '{tv}' is not a list, found {type(val).__name__}.")

            actual_len = len(val)
            if actual_len != self.target_length:
                raise ValueError(
                    f"Target variable '{tv}' length ({actual_len}) "
                    f"does not match expected target length ({self.target_length}) "
                    f"derived from output_seq_type '{self.output_seq_type}' config."
                )
            # Check for non-finite values within targets (optional but recommended)
            # if not all(torch.isfinite(torch.tensor(x)) for x in val if isinstance(x, (int, float))):
            #     raise ValueError(f"Target variable '{tv}' contains non-finite values.")

    def __len__(self) -> int:
        """Returns the number of valid profiles found."""
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Loads, processes, and returns a single profile by index.

        Uses an LRU cache for efficiency. Assumes file has passed validation.

        Args:
            idx: The index of the profile to retrieve.

        Returns:
            A tuple containing:
            - inputs (Dict[str, Tensor]): Dictionary mapping 'global' or sequence type keys
              to their corresponding tensors. Global tensor shape: [num_globals]. Sequence
              tensor shape: [seq_len, num_vars_in_seq].
            - targets (Tensor): Target tensor of shape [tgt_len, num_targets].

        Raises:
            IndexError: If the index is out of bounds.
            RuntimeError: If loading or processing a previously validated file fails unexpectedly.
        """
        if not 0 <= idx < len(self.valid_files):
            raise IndexError(
                f"Dataset index {idx} out of range (0 to {len(self.valid_files)-1})."
            )

        # --- Cache Logic ---
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        # --- Load and Process ---
        fpath = self.valid_files[idx]
        try:
            prof = self._load_json(fpath) # Should not fail if validation passed, but handle defensively
            if prof is None:
                raise RuntimeError(f"Failed to load previously validated file: {fpath.name}")

            inputs: Dict[str, Tensor] = {}
            # Process Global Variables
            if self.global_variables:
                gv_vals = [float(prof[v]) for v in self.global_variables]
                inputs["global"] = torch.tensor(gv_vals, dtype=torch.float32)

            # Process Sequence Variables
            for seq_key, vars_list in self.sequence_types.items():
                # Stack variables for this sequence type: result shape [seq_len, num_vars_in_seq]
                cols = [
                    torch.tensor(prof[v], dtype=torch.float32)
                    for v in vars_list
                ]
                inputs[seq_key] = torch.stack(cols, dim=1)

            # Process Target Variables
            # Stack target variables: result shape [tgt_len, num_targets]
            tgt_cols = [
                torch.tensor(prof[tv], dtype=torch.float32)
                for tv in self.target_variables
            ]
            targets = torch.stack(tgt_cols, dim=1)

        except KeyError as e:
            # This indicates an issue if validation passed but key is missing now
            raise RuntimeError(
                f"Missing expected variable '{e}' during processing of validated file {fpath.name}. Validation logic might be incomplete."
            ) from e
        except (ValueError, TypeError) as e:
            # Catch errors during tensor conversion (e.g., non-numeric data somehow missed by validation)
            raise RuntimeError(
                f"Non-numeric or invalid value found processing validated file {fpath.name}: {e}"
            ) from e
        except Exception as e:
            # Catch any other unexpected errors during processing
            raise RuntimeError(
                f"Unexpected error processing data from validated file {fpath.name}: {e}"
            ) from e

        # --- Update Cache ---
        self._cache[idx] = (inputs, targets)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False) # Remove least recently used

        return inputs, targets


# =============================================================================
# Collate Function
# =============================================================================


class StrictCollate:
    """
    Pickle-able collate function for multi-source batches.

    Stacks tensors for each corresponding key ('global', sequence types, targets)
    along a new batch dimension (dim=0). Explicitly checks for consistent tensor
    shapes across all samples in the batch before stacking. Assumes all samples
    have identical dictionary keys.
    """

    def __call__(
        self, batch: List[Tuple[Dict[str, Tensor], Tensor]]
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Collates a list of samples into a single batch.

        Args:
            batch: A list where each element is a tuple returned by
                   `AtmosphericDataset.__getitem__`.

        Returns:
            A tuple containing:
            - A dictionary mapping input keys ('global' or sequence types)
              to batched tensors.
            - A batched tensor for the targets.

        Raises:
            ValueError: If tensors for the same key across different samples
                        in the batch have inconsistent shapes.
        """
        if not batch:
            return {}, torch.empty(0)

        # --- Check Input Key Consistency (Optional but Recommended) ---
        first_input_keys = batch[0][0].keys()
        if not all(sample[0].keys() == first_input_keys for sample in batch):
             raise ValueError("Inconsistent input keys found within a batch. Check dataset processing.")

        batched_inputs: Dict[str, List[Tensor]] = {
            key: [] for key in first_input_keys
        }
        batched_targets: List[Tensor] = []

        # --- Collect Tensors ---
        for inp_dict, tgt_tensor in batch:
            for key, tensor in inp_dict.items():
                batched_inputs[key].append(tensor)
            batched_targets.append(tgt_tensor)

        # --- Validate Shapes and Stack ---
        final_inputs: Dict[str, Tensor] = {}
        try:
            # Validate Input Shapes
            for key, tensors in batched_inputs.items():
                first_shape = tensors[0].shape
                if not all(t.shape == first_shape for t in tensors):
                    shapes = [t.shape for t in tensors]
                    raise ValueError(
                        f"Inconsistent shapes found for input key '{key}' in batch. "
                        f"Expected shape {first_shape}, but found shapes: {shapes}"
                    )
                final_inputs[key] = torch.stack(tensors, dim=0)

            # Validate Target Shapes
            first_target_shape = batched_targets[0].shape
            if not all(t.shape == first_target_shape for t in batched_targets):
                 shapes = [t.shape for t in batched_targets]
                 raise ValueError(
                    f"Inconsistent shapes found for targets in batch. "
                    f"Expected shape {first_target_shape}, but found shapes: {shapes}"
                 )
            final_targets = torch.stack(batched_targets, dim=0)

        except RuntimeError as e:
            # Catch potential low-level stacking errors (though shape check should prevent most)
            logger.error("RuntimeError during batch collation (stacking): %s", e)
            raise ValueError("Failed to stack tensors during collation. Check tensor shapes and types.") from e

        return final_inputs, final_targets


def create_multi_source_collate_fn() -> Callable:
    """
    Factory function returning a StrictCollate instance.

    This wrapper helps ensure the collate function is pickle-able,
    which is sometimes required by multiprocessing DataLoaders.
    """
    return StrictCollate()


__all__ = [
    "AtmosphericDataset",
    "StrictCollate",
    "create_multi_source_collate_fn",
]
