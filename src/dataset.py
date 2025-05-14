#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AtmosphericDataset(Dataset):
    """
    Loads normalised JSON profiles and returns tensors for the model.

    This class handles the discovery, validation (optional), and loading of
    atmospheric profile data stored in JSON files. It supports variable
    sequence lengths and prepares data for padding. An optional LRU cache
    is implemented for frequently accessed profiles.
    """

    def __init__(
        self,
        data_folder: Union[str, Path],
        input_variables: List[str],
        target_variables: List[str],
        global_variables: List[str],
        sequence_types: Dict[str, List[str]],
        output_seq_type: str,
        padding_value: float = 0.0,
        *,
        cache_size: int = 1_024,
        validate_profiles: bool = True,
    ) -> None:
        """
        Initializes the AtmosphericDataset.

        Args:
            data_folder: Path to the directory containing JSON profile files.
            input_variables: List of variable names for model inputs.
            target_variables: List of variable names for model targets.
            global_variables: List of global scalar input variable names.
            sequence_types: Maps sequence type names to lists of variable
                            names belonging to that type.
            output_seq_type: The sequence type name of the primary target.
            padding_value: Value used for padding sequences.
            cache_size: Max profiles in LRU cache. 0 or less disables.
            validate_profiles: If True, validates profile structure/content.
        """
        super().__init__()

        self.data_dir = Path(data_folder)
        if not self.data_dir.is_dir():
            # Critical error, execution cannot continue.
            logger.critical("Data directory not found: %s. Exiting.", self.data_dir)
            sys.exit(1)


        self.input_variables = input_variables
        self.target_variables = target_variables
        self.global_variables = global_variables or []
        self.sequence_types = {k: v for k, v in sequence_types.items() if v}
        self.output_seq_type = output_seq_type
        self.padding_value = padding_value
        self.validate_profiles = bool(validate_profiles)

        self._check_cfg()

        if self.validate_profiles:
            self.valid_files = self._scan_and_validate()
        else:
            self.valid_files = [
                p
                for p in sorted(self.data_dir.glob("*.json"))
                if p.name != "normalization_metadata.json"
            ]
            if not self.valid_files:
                logger.critical(
                    "No JSON profiles found in %s (validation skipped). Exiting.",
                    self.data_dir
                )
                sys.exit(1)
            logger.info(
                "%d profiles discovered (validation skipped)",
                len(self.valid_files)
            )

        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[
            int, Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor, Tensor]
        ] = OrderedDict()

    def _check_cfg(self) -> None:
        """Validates internal configuration for self-consistency."""
        if not self.sequence_types:
            logger.critical("Config 'sequence_types' is empty. Exiting.")
            sys.exit(1)
        if self.output_seq_type not in self.sequence_types:
            logger.critical(
                "Config 'output_seq_type' ('%s') not found as a key "
                "in 'sequence_types'. Exiting.",
                self.output_seq_type
            )
            sys.exit(1)
        if not self.sequence_types[self.output_seq_type]:
            logger.critical(
                "Config 'output_seq_type' ('%s') lists no variables "
                "in 'sequence_types'. Exiting.",
                self.output_seq_type
            )
            sys.exit(1)

    def _scan_and_validate(self) -> List[Path]:
        """
        Scans data directory for JSON files and validates each one.

        Returns:
            A list of Path objects for valid profile files.
        """
        files = [
            p
            for p in sorted(self.data_dir.glob("*.json"))
            if p.name != "normalization_metadata.json"
        ]
        if not files:
            logger.critical("No JSON profiles found in %s for validation. Exiting.", self.data_dir)
            sys.exit(1)

        validated_file_paths = []
        for fp in files:
            try:
                prof_text = fp.read_text(encoding="utf-8-sig")
                prof = json.loads(prof_text)
                self._validate_profile(prof, fp.name)
                validated_file_paths.append(fp)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON in %s: %s. Skipping file.", fp.name, e)
            except ValueError as e: # Catch validation errors from _validate_profile
                logger.error("Validation failed for profile %s: %s. Skipping file.", fp.name, e)
            except Exception as e: # Catch other unexpected errors during file processing
                logger.error("Unexpected error processing file %s: %s. Skipping file.", fp.name, e, exc_info=True)

        if not validated_file_paths:
            logger.critical("No profiles passed validation in %s. Exiting.", self.data_dir)
            sys.exit(1)

        logger.info("%d profiles validated in %s", len(validated_file_paths), self.data_dir)
        return validated_file_paths


    def _validate_profile(self, prof: Dict[str, Any], name: str) -> None:
        """
        Validates a single profile dict against the dataset configuration.
        Raises ValueError if validation fails.

        Args:
            prof: The profile data loaded from a JSON file.
            name: The filename of the profile, for logging.
        """
        for gv in self.global_variables:
            if gv not in prof:
                raise ValueError(f"Global variable '{gv}' missing.")
            if not isinstance(prof[gv], (int, float)):
                raise ValueError(
                    f"Global variable '{gv}' must be numeric."
                )

        for seq_name, vars_in_seq_type in self.sequence_types.items():
            if not vars_in_seq_type: # Should be caught by _check_cfg for output_seq_type
                continue
            first_var_len = -1
            for v_idx, var_name_in_seq in enumerate(vars_in_seq_type):
                if var_name_in_seq not in prof:
                    raise ValueError(
                        f"Sequence variable '{var_name_in_seq}' for type "
                        f"'{seq_name}' missing."
                    )
                var_data = prof[var_name_in_seq]
                if not isinstance(var_data, list):
                    raise ValueError(
                        f"Sequence variable '{var_name_in_seq}' "
                        f"must be a list."
                    )
                if not var_data: # Check for empty list
                    raise ValueError(
                        f"Sequence variable '{var_name_in_seq}' for type "
                        f"'{seq_name}' is an empty list. Empty sequences are not allowed."
                    )

                current_var_len = len(var_data)
                if v_idx == 0:
                    first_var_len = current_var_len
                elif current_var_len != first_var_len:
                    raise ValueError(
                        f"Length mismatch in sequence type '{seq_name}'. "
                        f"Var '{vars_in_seq_type[0]}' (len "
                        f"{first_var_len}) vs '{var_name_in_seq}' (len "
                        f"{current_var_len})."
                    )

        output_vars_in_output_seq_type = self.sequence_types[
            self.output_seq_type
        ]
        # This length is already validated to be non-empty and consistent above
        expected_target_len = len(prof[output_vars_in_output_seq_type[0]])


        for tv in self.target_variables:
            if tv not in prof:
                raise ValueError(f"Target variable '{tv}' missing.")
            if not isinstance(prof[tv], list):
                raise ValueError(
                    f"Target variable '{tv}' must be a list."
                )
            if not prof[tv] and tv in output_vars_in_output_seq_type:
                 raise ValueError(
                    f"Target variable '{tv}' (part of output sequence type) "
                    "is an empty list. Empty sequences are not allowed."
                )

            if tv in output_vars_in_output_seq_type:
                if len(prof[tv]) != expected_target_len:
                    raise ValueError(
                        f"Target var '{tv}' length mismatch. Expected "
                        f"{expected_target_len} (from type "
                        f"'{self.output_seq_type}'), got {len(prof[tv])}."
                    )

    def __len__(self) -> int:
        """Returns the total number of valid profiles."""
        return len(self.valid_files)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor, Tensor]:
        """
        Retrieves, converts to tensors, and caches a profile.

        Args:
            idx: The index of the profile to retrieve.

        Returns:
            A tuple: (inputs_dict, input_masks_dict, target_tensor,
            target_mask_tensor). Masks indicate valid data points (True).
        """
        if self.cache_size > 0 and idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        fp = self.valid_files[idx]
        # Validation should prevent loading invalid JSON or profiles with empty sequences
        prof: Dict[str, Any] = json.loads(fp.read_text(encoding="utf-8-sig"))

        inputs: Dict[str, Tensor] = {}
        input_masks: Dict[str, Tensor] = {}

        if self.global_variables:
            inputs["global"] = torch.tensor(
                [prof[g] for g in self.global_variables], dtype=torch.float32
            )

        for seq_name, vars_in_seq_type in self.sequence_types.items():
            if not vars_in_seq_type:
                continue
            seq_tensor_list = []
            # Validation ensures all vars_in_seq_type exist and are non-empty lists of same length
            original_length = len(prof[vars_in_seq_type[0]])

            for var_name_in_seq in vars_in_seq_type:
                var_data = prof[var_name_in_seq]
                seq_tensor_list.append(
                    torch.tensor(var_data, dtype=torch.float32)
                )

            inputs[seq_name] = torch.stack(seq_tensor_list, dim=1)
            input_masks[seq_name] = torch.ones(
                original_length, dtype=torch.bool
            )


        target_tensors_list = []
        # Validation ensures target_variables exist, are lists, and non-empty if part of output_seq_type
        # and have consistent length if part of output_seq_type
        first_target_var_name = self.target_variables[0]
        target_original_length = len(prof[first_target_var_name])


        for tv in self.target_variables:
            tv_data = prof[tv]
            target_tensors_list.append(
                torch.tensor(tv_data, dtype=torch.float32)
            )

        targets = torch.stack(target_tensors_list, dim=1)
        # Create mask based on the (consistent) length of target variables
        target_mask = torch.ones(target_original_length, dtype=torch.bool)


        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            self._cache[idx] = (inputs, input_masks, targets, target_mask)
        return inputs, input_masks, targets, target_mask

    def get_profile_filenames_by_indices(
        self, indices: List[int]
    ) -> List[str]:
        """
        Retrieves filenames for a list of profile indices.

        Args:
            indices: List of integer indices for self.valid_files.

        Returns:
            A list of profile filenames.
        """
        if not all(0 <= i < len(self.valid_files) for i in indices):
            invalid_indices = [
                i for i in indices if not (0 <= i < len(self.valid_files))
            ]
            # This is a programming error if it occurs.
            logger.error("Invalid indices requested: %s. Dataset size: %d",
                         invalid_indices, len(self.valid_files))
            raise IndexError(
                f"Invalid indices found: {invalid_indices}. "
                f"Dataset size is {len(self.valid_files)}."
            )
        return [self.valid_files[i].name for i in indices]


class PadCollate:
    """
    Collates samples into a batch, padding sequences to the max length
    in the batch. Also prepares corresponding padding masks.
    Masks from this collate function will have True for VALID data points
    and False for PADDED data points.
    """

    def __init__(self, padding_value: float = 0.0):
        """
        Args:
            padding_value: Value to use for padding sequences.
        """
        self.padding_value = padding_value

    def __call__(
        self,
        batch: List[
            Tuple[
                Dict[str, Tensor], Dict[str, Tensor], Tensor, Tensor
            ]
        ],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor, Tensor]:
        """
        Collates a list of samples into a batch.

        Each sample is (inputs_dict, input_masks_dict, target_tensor,
        target_mask_tensor). Input masks from dataset have True for valid data.

        Returns:
            Tuple of (batched_inputs, batched_input_masks,
            batched_targets, batched_target_masks).
            Output masks have True for valid data, False for padded data.
        """
        if not batch:
            return {}, {}, torch.empty(0), torch.empty(0)

        # Structure of each item in batch:
        # item[0]: inputs_dict (e.g., {'seq1': tensor_data1, 'global': tensor_global})
        # item[1]: input_masks_dict (e.g., {'seq1': tensor_mask1})
        # item[2]: targets_tensor
        # item[3]: target_mask_tensor

        elem_inputs = batch[0][0]
        input_keys = list(elem_inputs.keys())

        # Basic check for consistent keys across the batch
        if any(list(sample[0].keys()) != input_keys for sample in batch):
            # This indicates a severe issue, likely in dataset generation.
            logger.critical("Input key mismatch inside batch. Exiting.")
            sys.exit(1)


        batched_inputs: Dict[str, Tensor] = {}
        batched_input_masks: Dict[str, Tensor] = {} # True means valid

        for key in input_keys:
            tensors_this_key = [sample[0][key] for sample in batch]
            # input_masks_dict from __getitem__ (True means valid)
            masks_this_key_from_getitem = [sample[1].get(key) for sample in batch]

            first_tensor_dim = tensors_this_key[0].ndim
            if first_tensor_dim == 2:  # Sequence Data (Original_L x F)
                if any(t.ndim != 2 for t in tensors_this_key):
                    logger.critical("Tensor dimension mismatch for input key '%s'. Expected 2D. Exiting.", key)
                    sys.exit(1)

                num_features = tensors_this_key[0].size(1)
                if any(t.size(1) != num_features for t in tensors_this_key):
                    logger.critical("Feature dimension mismatch for input key '%s'. Exiting.", key)
                    sys.exit(1)


                batched_inputs[key] = pad_sequence(
                    tensors_this_key,
                    batch_first=True,
                    padding_value=self.padding_value,
                )

                # Ensure all samples provided a mask for this sequence key
                if not all(m is not None and m.ndim == 1 for m in masks_this_key_from_getitem):
                    logger.critical("Inconsistent or missing 1D masks for sequence input key '%s'. Exiting.", key)
                    sys.exit(1)

                # Pad the original validity masks (True for valid)
                # Padded regions in the mask will be False (0)
                batched_input_masks[key] = pad_sequence(
                    [m for m in masks_this_key_from_getitem if m is not None], # Filter Nones just in case, though logic above should prevent
                    batch_first=True,
                    padding_value=0,
                ).bool()

            elif first_tensor_dim == 1:  # Global Data (Original_F)
                if any(t.ndim != 1 for t in tensors_this_key):
                    logger.critical("Tensor dimension mismatch for input key '%s'. Expected 1D. Exiting.", key)
                    sys.exit(1)
                batched_inputs[key] = torch.stack(tensors_this_key, dim=0)
                # No separate padding mask for global features; they are assumed complete.
            else:
                logger.critical("Unsupported tensor dimension %d for input key '%s'. Exiting.", first_tensor_dim, key)
                sys.exit(1)


        raw_targets = [sample[2] for sample in batch]
        raw_target_masks_from_getitem = [sample[3] for sample in batch] # True means valid

        first_target_dim = raw_targets[0].ndim
        if first_target_dim == 2:  # Sequence Targets (Original_L_target x F_target)
            if any(t.ndim != 2 for t in raw_targets):
                logger.critical("Target tensor dimension mismatch. Expected 2D. Exiting.")
                sys.exit(1)

            num_target_features = raw_targets[0].size(1)
            if any(t.size(1) != num_target_features for t in raw_targets):
                logger.critical("Target feature dimension mismatch. Exiting.")
                sys.exit(1)


            batched_targets = pad_sequence(
                raw_targets,
                batch_first=True,
                padding_value=self.padding_value,
            )
            if not all(m is not None and m.ndim == 1 for m in raw_target_masks_from_getitem):
                logger.critical("Inconsistent or missing 1D masks for target sequences. Exiting.")
                sys.exit(1)

            batched_target_masks = pad_sequence(
                [m for m in raw_target_masks_from_getitem if m is not None],
                batch_first=True,
                padding_value=0,
            ).bool()

        elif first_target_dim == 1:  # Scalar/Global Targets (Original_F_target)
            if any(t.ndim != 1 for t in raw_targets):
                logger.critical("Target tensor dimension mismatch. Expected 1D. Exiting.")
                sys.exit(1)
            batched_targets = torch.stack(raw_targets, dim=0)

            if not all(m is not None and m.ndim == 1 for m in raw_target_masks_from_getitem):
                logger.critical("Inconsistent or missing 1D masks for 1D targets. Exiting.")
                sys.exit(1)
            batched_target_masks = torch.stack(
                [m for m in raw_target_masks_from_getitem if m is not None] # Should be (B, F_target_mask_len)
            ).bool()


        else:
            logger.critical("Unsupported target tensor dimension: %d. Exiting.", first_target_dim)
            sys.exit(1)


        return batched_inputs, batched_input_masks, batched_targets, batched_target_masks


def create_multi_source_collate_fn(
    padding_value: float = 0.0,
) -> PadCollate:
    """
    Factory for PadCollate, allowing consistent interface.

    Args:
        padding_value: Value for padding sequences.

    Returns:
        An instance of PadCollate.
    """
    return PadCollate(padding_value=padding_value)


__all__ = [
    "AtmosphericDataset",
    "PadCollate",
    "create_multi_source_collate_fn",
]