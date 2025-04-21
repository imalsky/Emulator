#!/usr/bin/env python3
"""
dataset.py – Atmospheric profile dataset with strict fixed sequence lengths.

This module defines the `AtmosphericDataset` class, which loads atmospheric
profile data from JSON files. It enforces strict sequence length requirements
as defined in the configuration, ensuring that all profiles conform to expected
dimensions before training begins. It also provides a `StrictCollate` helper
class for batching fixed-length sequences without padding.
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

    Requires sequence lengths to be specified during initialization via the
    `sequence_lengths` argument (typically derived from the main configuration).
    It validates every profile against these lengths, skipping invalid files.
    Includes an LRU cache for faster access during training.
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
            sequence_lengths: Dictionary mapping sequence type names to their required integer length.
            output_seq_type: The key from `sequence_types` whose length determines the
                             expected length of the target variables.
            cache_size: Maximum number of loaded profiles to keep in the LRU cache.

        Raises:
            FileNotFoundError: If the data_folder does not exist or contains no profiles.
            ValueError: If configuration is inconsistent or no valid profiles are found.
        """
        super().__init__()
        self.data_folder = Path(data_folder)
        if not self.data_folder.is_dir():
            raise FileNotFoundError(f"Data folder not found: {self.data_folder}")

        self.input_variables = input_variables
        self.target_variables = target_variables
        self.global_variables = global_variables or []
        self.sequence_types = {
            k: v for k, v in (sequence_types or {}).items() if v
        }
        self.sequence_lengths = sequence_lengths
        self.output_seq_type = output_seq_type

        self._validate_configuration()

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

        missing_seq_lens = set(self.sequence_types.keys()) - set(
            self.sequence_lengths.keys()
        )
        if missing_seq_lens:
            raise ValueError(
                f"Configuration missing 'sequence_lengths' entries for sequence types: {missing_seq_lens}"
            )

        if self.output_seq_type not in self.sequence_lengths:
            raise ValueError(
                f"Configuration 'output_seq_type' ('{self.output_seq_type}') "
                f"not found in 'sequence_lengths' keys: {list(self.sequence_lengths.keys())}"
            )

        try:
            self.target_length = self.sequence_lengths[self.output_seq_type]
        except KeyError:
            raise ValueError(
                f"Output sequence type '{self.output_seq_type}' defined but lacks length in 'sequence_lengths'."
            )

        logger.info(
            f"Dataset configured with strict lengths: {self.sequence_lengths}"
        )
        logger.info(
            f"Target length set to {self.target_length} (based on '{self.output_seq_type}')"
        )

    def _index_and_validate_profiles(self) -> List[Path]:
        """Scans the data folder, validates profiles, and returns list of valid file paths."""
        logger.info(
            "Indexing and validating profile files in %s...", self.data_folder
        )
        all_files = sorted(self.data_folder.glob("*.json"))
        candidate_files = [
            p for p in all_files if p.name != "normalization_metadata.json"
        ]

        if not candidate_files:
            raise FileNotFoundError(
                f"No JSON profile files found in {self.data_folder}"
            )

        valid_files: List[Path] = []
        invalid_files_summary: Dict[str, int] = {}
        processed_count = 0

        for fpath in candidate_files:
            processed_count += 1
            reason = None
            try:
                prof = self._load_json(fpath)
                if prof is None:
                    reason = "Failed to load/decode JSON"
                else:
                    self._validate_profile_contents(prof, fpath.name)
                    valid_files.append(fpath)

            except ValueError as ve:
                reason = str(ve)
                logger.debug(f"Skipping invalid profile {fpath.name}: {reason}")
            except Exception as e:
                reason = f"Unexpected error ({type(e).__name__})"
                logger.error(
                    f"Unexpected error processing {fpath.name}: {e}",
                    exc_info=False,
                )
                logger.debug("Detailed error trace:", exc_info=True)

            if reason:
                invalid_files_summary[reason] = (
                    invalid_files_summary.get(reason, 0) + 1
                )

        if not valid_files:
            logger.error(
                "No valid profiles found after validation in %s.",
                self.data_folder,
            )
            self._log_invalid_summary(invalid_files_summary)
            raise ValueError(
                f"No valid profiles found in {self.data_folder}. Check logs for details."
            )

        logger.info(
            f"Found {len(valid_files)} valid profiles out of {processed_count} candidates."
        )
        if invalid_files_summary:
            skipped_count = sum(invalid_files_summary.values())
            logger.warning(
                f"Skipped {skipped_count} invalid/mismatched profile(s)."
            )
            self._log_invalid_summary(invalid_files_summary)

        return valid_files

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Loads a single JSON file, returning None on error."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in {path.name}: {e}")
            return None
        except OSError as e:
            logger.warning(f"OS error reading file {path.name}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error loading JSON {path.name}: {e}")
            return None

    def _validate_profile_contents(
        self, prof: Dict[str, Any], filename: str
    ) -> None:
        """
        Ensures profile dictionary matches required keys and exact lengths.

        Raises:
            ValueError: If any validation check fails.
        """
        for gv in self.global_variables:
            val = prof.get(gv)
            if val is None:
                raise ValueError(
                    f"File '{filename}': Global variable '{gv}' is missing."
                )
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"File '{filename}': Global variable '{gv}' is not scalar (int/float), found {type(val).__name__}."
                )
            if not torch.isfinite(torch.tensor(val)):
                raise ValueError(
                    f"File '{filename}': Global variable '{gv}' is not finite ({val})."
                )

        for seq_key, vars_list in self.sequence_types.items():
            expected_len = self.sequence_lengths[seq_key]
            for var in vars_list:
                val = prof.get(var)
                if val is None:
                    raise ValueError(
                        f"File '{filename}': Sequence variable '{var}' (type '{seq_key}') is missing."
                    )
                if not isinstance(val, list):
                    raise ValueError(
                        f"File '{filename}': Sequence variable '{var}' (type '{seq_key}') is not a list, found {type(val).__name__}."
                    )

                actual_len = len(val)
                if actual_len != expected_len:
                    raise ValueError(
                        f"File '{filename}': Sequence '{seq_key}' variable '{var}' length {actual_len} "
                        f"does not match expected length {expected_len}."
                    )

        for tv in self.target_variables:
            val = prof.get(tv)
            if val is None:
                raise ValueError(
                    f"File '{filename}': Target variable '{tv}' is missing."
                )
            if not isinstance(val, list):
                raise ValueError(
                    f"File '{filename}': Target variable '{tv}' is not a list, found {type(val).__name__}."
                )

            actual_len = len(val)
            if actual_len != self.target_length:
                raise ValueError(
                    f"File '{filename}': Target variable '{tv}' length {actual_len} "
                    f"does not match expected target length {self.target_length} "
                    f"(derived from output_seq_type '{self.output_seq_type}')."
                )

    def _log_invalid_summary(self, summary: Dict[str, int]) -> None:
        """Logs a summary of reasons why files were skipped."""
        if not summary:
            return
        logger.warning("Summary of reasons for skipping profiles:")
        sorted_reasons = sorted(
            summary.items(), key=lambda item: item[1], reverse=True
        )
        for reason, count in sorted_reasons[:10]:
            logger.warning(f"  - {count:>5d} file(s): {reason}")
        if len(sorted_reasons) > 10:
            logger.warning(
                f"  - ... and {len(sorted_reasons) - 10} other reasons."
            )

    def __len__(self) -> int:
        """Returns the number of valid profiles found."""
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Loads, processes, and returns a single profile by index.

        Uses an LRU cache for efficiency.

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
            RuntimeError: If loading or processing a previously validated file fails.
        """
        if not 0 <= idx < len(self.valid_files):
            raise IndexError(
                f"Dataset index {idx} out of range (0 to {len(self.valid_files)-1})."
            )

        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        fpath = self.valid_files[idx]
        prof = self._load_json(fpath)
        if prof is None:
            raise RuntimeError(
                f"Failed to load previously validated file: {fpath.name}"
            )

        inputs: Dict[str, Tensor] = {}
        try:
            if self.global_variables:
                gv_vals = [float(prof[v]) for v in self.global_variables]
                inputs["global"] = torch.tensor(gv_vals, dtype=torch.float32)

            for seq_key, vars_list in self.sequence_types.items():
                cols = [
                    torch.tensor(prof[v], dtype=torch.float32)
                    for v in vars_list
                ]
                inputs[seq_key] = torch.stack(cols, dim=1)

            tgt_cols = [
                torch.tensor(prof[tv], dtype=torch.float32)
                for tv in self.target_variables
            ]
            targets = torch.stack(tgt_cols, dim=1)

        except KeyError as e:
            raise RuntimeError(
                f"Missing expected variable '{e}' in validated file {fpath.name}"
            )
        except (ValueError, TypeError) as e:
            raise RuntimeError(
                f"Non-numeric or invalid value found processing validated file {fpath.name}: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error processing data from validated file {fpath.name}: {e}"
            )

        self._cache[idx] = (inputs, targets)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return inputs, targets


# =============================================================================
# Collate Function
# =============================================================================


class StrictCollate:
    """
    Pickle-able collate function for fixed-length multi-source batches.

    Stacks tensors for each corresponding key ('global', sequence types, targets)
    along a new batch dimension (dim=0). Assumes all samples in the batch
    have identical dictionary keys and tensor shapes.
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
        """
        if not batch:
            return {}, torch.empty(0)

        first_input_keys = batch[0][0].keys()
        batched_inputs: Dict[str, List[Tensor]] = {
            key: [] for key in first_input_keys
        }
        batched_targets: List[Tensor] = []

        for inp_dict, tgt_tensor in batch:
            # Optional: Add check for key consistency if needed
            # if inp_dict.keys() != first_input_keys:
            #     raise ValueError("Inconsistent input keys found within a batch.")
            for key, tensor in inp_dict.items():
                if key in batched_inputs:
                    batched_inputs[key].append(tensor)

            batched_targets.append(tgt_tensor)

        try:
            final_inputs = {
                key: torch.stack(tensors)
                for key, tensors in batched_inputs.items()
            }
            final_targets = torch.stack(batched_targets)
        except RuntimeError as e:
            logger.error("Error during batch collation (stacking): %s", e)
            for key, tensors in batched_inputs.items():
                shapes = [t.shape for t in tensors]
                if len(set(shapes)) > 1:
                    logger.error(
                        "Inconsistent shapes for key '%s': %s", key, shapes
                    )
            tgt_shapes = [t.shape for t in batched_targets]
            if len(set(tgt_shapes)) > 1:
                logger.error("Inconsistent target shapes: %s", tgt_shapes)
            raise

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