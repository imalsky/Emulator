#!/usr/bin/env python3
"""
dataset.py - Handles data processing for the atmospheric emulator

Loads atmospheric profile data from JSON files with strict sequence length enforcement.
Supports organizing data into global features and sequence types.
"""

import json
import logging
from pathlib import Path
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
from hardware import configure_dataloader_settings

logger = logging.getLogger(__name__)

class AtmosphericDataset(Dataset):
    """Dataset for atmospheric profile data with strict sequence length enforcement."""

    def __init__(
        self,
        data_folder,
        input_variables,
        target_variables,
        global_variables=None,
        sequence_types=None,
        cache_size=1024
    ):
        """
        Initialize dataset with strict sequence length validation.
        
        Parameters:
            data_folder: Path to JSON profile files
            input_variables: Names of input variables 
            target_variables: Names of target variables
            global_variables: Names of global (non-sequence) features
            sequence_types: Mapping of sequence type names to variable lists
            cache_size: Maximum number of profiles to cache
        """
        super().__init__()
        self.data_folder = Path(data_folder)
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        # Store variable configurations
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.all_variables = list(set(input_variables + target_variables))
        self.global_variables = global_variables or []
        self.sequence_types = sequence_types or {}
        
        # If no sequence types provided throw an error
        if not self.sequence_types:
            raise ValueError("Sequence types must be provided. Please provide a sequence_types dictionary.")

        # Verify all variables are assigned
        self._validate_variable_assignments()

        # File caching and tracking
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.valid_files = []
        self.invalid_files = []
        self.sequence_lengths = {}
        
        # Process files and build dataset
        self._initialize_dataset()
        
        # Log initialization details
        logger.info(f"Dataset initialized with {len(self.valid_files)} valid profiles")
        logger.info(f"Input variables: {self.input_variables}")
        logger.info(f"Target variables: {self.target_variables}")
        
        if self.global_variables:
            logger.info(f"Global variables: {self.global_variables}")
        
        for seq_type, var_list in self.sequence_types.items():
            seq_length = self.sequence_lengths.get(seq_type, 0)
            logger.info(f"Sequence type '{seq_type}': {var_list} (length: {seq_length})")

    def _validate_variable_assignments(self):
        """Ensure all variables are assigned to a sequence type or global."""
        # Collect all assigned variables
        assigned_vars = set(self.global_variables)
        for seq_vars in self.sequence_types.values():
            assigned_vars.update(seq_vars)
        
        # Check for unassigned variables
        unassigned_inputs = set(self.input_variables) - assigned_vars
        if unassigned_inputs:
            raise ValueError(f"Unassigned input variables: {unassigned_inputs}")
        
        unassigned_targets = set(self.target_variables) - assigned_vars
        if unassigned_targets:
            raise ValueError(f"Unassigned target variables: {unassigned_targets}")

    def _initialize_dataset(self):
        """Process JSON files and build the dataset with strict validation."""
        # Get all JSON files except metadata
        json_files = [f for f in self.data_folder.glob("*.json") if f.name != "normalization_metadata.json"]
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.data_folder}")

        # First determine sequence lengths from first valid file
        for file_path in json_files:
            try:
                # Load profile
                with open(file_path, "r") as f:
                    profile = json.load(f)
                
                # Skip if profile doesn't have all required variables
                if not set(self.all_variables).issubset(profile.keys()):
                    continue
                
                # Get sequence lengths for each type
                for seq_type, var_list in self.sequence_types.items():
                    if not var_list:
                        continue
                    
                    # Get length from first variable
                    first_var = var_list[0]
                    if not isinstance(profile[first_var], list):
                        continue
                        
                    first_length = len(profile[first_var])
                    self.sequence_lengths[seq_type] = first_length
                    
                    # Verify all variables in this type have the same length
                    for var_name in var_list:
                        if not isinstance(profile[var_name], list):
                            continue
                            
                        if len(profile[var_name]) != first_length:
                            raise ValueError(
                                f"Inconsistent sequence lengths in {file_path.name}: "
                                f"variable '{var_name}' has length {len(profile[var_name])}, "
                                f"expected {first_length}"
                            )
                
                # If we found sequence lengths, we can stop
                if self.sequence_lengths:
                    logger.info(f"Sequence lengths determined from {file_path.name}: {self.sequence_lengths}")
                    break
                    
            except Exception as e:
                logger.warning(f"Error analyzing {file_path.name}: {e}")
                continue
        
        # Validate all files against determined sequence lengths
        for file_path in json_files:
            try:
                # Load profile
                try:
                    with open(file_path, "r") as f:
                        profile = json.load(f)
                except Exception as e:
                    self.invalid_files.append((file_path.name, f"Failed to load: {e}"))
                    continue
                
                # Check if profile has all required variables
                if not set(self.all_variables).issubset(profile.keys()):
                    missing = set(self.all_variables) - set(profile.keys())
                    self.invalid_files.append((file_path.name, f"Missing variables: {missing}"))
                    continue
                
                # Validate all variables
                valid = True
                error_reason = ""
                
                for var in self.all_variables:
                    val = profile[var]
                    
                    # Validate global variables
                    if var in self.global_variables:
                        if not isinstance(val, (int, float)):
                            valid = False
                            error_reason = f"Global variable '{var}' is not numeric"
                            break
                        if not torch.isfinite(torch.tensor(float(val))):
                            valid = False
                            error_reason = f"Global variable '{var}' is not finite"
                            break
                    
                    # Validate sequence variables
                    else:
                        # Find which sequence type this variable belongs to
                        for seq_type, var_list in self.sequence_types.items():
                            if var in var_list:
                                if not isinstance(val, list):
                                    valid = False
                                    error_reason = f"Sequence variable '{var}' is not a list"
                                    break
                                
                                if not all(isinstance(x, (int, float)) for x in val):
                                    valid = False
                                    error_reason = f"Non-numeric values in '{var}'"
                                    break
                                
                                if not all(torch.isfinite(torch.tensor(float(x))) for x in val):
                                    valid = False
                                    error_reason = f"Non-finite values in '{var}'"
                                    break
                                
                                # Check sequence length if determined
                                expected_length = self.sequence_lengths.get(seq_type)
                                if expected_length is not None and len(val) != expected_length:
                                    valid = False
                                    error_reason = f"'{var}' has length {len(val)}, expected {expected_length}"
                                    break
                
                if valid:
                    self.valid_files.append(file_path)
                else:
                    self.invalid_files.append((file_path.name, error_reason))
                    
            except Exception as e:
                self.invalid_files.append((file_path.name, f"Error: {str(e)}"))
        
        if not self.valid_files:
            if self.invalid_files:
                # Log some examples of why files were rejected
                examples = self.invalid_files[:5]
                error_msg = "\n".join([f"  - {name}: {reason}" for name, reason in examples])
                raise ValueError(f"No valid profiles found. Examples of rejected files:\n{error_msg}")
            else:
                raise ValueError(f"No valid profiles found in {self.data_folder}")
        
        logger.info(f"Found {len(self.valid_files)} valid profiles and {len(self.invalid_files)} invalid profiles")

    def __len__(self):
        """Return the number of valid profiles."""
        return len(self.valid_files)

    def __getitem__(self, idx):
        """Return inputs and targets for the given profile index."""
        # Check cache first
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
            
        # Validate index
        if idx < 0 or idx >= len(self.valid_files):
            raise IndexError(f"Index {idx} out of range")
            
        file_path = self.valid_files[idx]
        
        # Load profile
        with open(file_path, "r") as f:
            profile = json.load(f)
        
        # Process global features
        inputs = {}
        if self.global_variables:
            global_features = torch.tensor(
                [float(profile[var]) for var in self.global_variables], 
                dtype=torch.float32
            )
            inputs["global"] = global_features
        
        # Process each sequence type
        for seq_type, var_list in self.sequence_types.items():
            if not var_list:
                continue
                
            # Create tensor for each variable in this sequence type
            seq_tensors = []
            for var_name in var_list:
                tensor = torch.tensor(profile[var_name], dtype=torch.float32)
                seq_tensors.append(tensor)
            
            # Stack tensors (seq_length, num_variables)
            inputs[seq_type] = torch.stack(seq_tensors, dim=1)
        
        # Process target variables
        target_tensors = []
        for var_name in self.target_variables:
            val = profile[var_name]
            
            # Handle both scalar and sequence targets
            if isinstance(val, list):
                tensor = torch.tensor(val, dtype=torch.float32)
            else:
                tensor = torch.tensor([float(val)], dtype=torch.float32)
                
            target_tensors.append(tensor)
        
        # Stack target tensors
        targets = torch.stack(target_tensors, dim=1)
        
        # Cache result
        result = (inputs, targets)
        self.cache[idx] = result
        
        # Manage cache size
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        
        return result

    def clear_cache(self):
        """Clear the file cache."""
        self.cache.clear()

    def get_invalid_files(self):
        """Return list of invalid files with reasons."""
        return self.invalid_files


class MultiSourceCollate:
    """Collates samples with separate handling for different data types."""
    
    def __call__(self, batch):
        """Collate a batch of samples into structured batch data."""
        if not batch:
            return {}, torch.tensor([])
            
        # Initialize containers
        inputs_by_type = {}
        targets_list = []
        
        # Collect inputs by type and targets
        for inputs, targets in batch:
            # Process inputs
            for key, value in inputs.items():
                if key not in inputs_by_type:
                    inputs_by_type[key] = []
                inputs_by_type[key].append(value)
            
            # Process targets
            targets_list.append(targets)
        
        # Collate inputs by type
        collated_inputs = {}
        
        # Process global features - simple stack
        if "global" in inputs_by_type:
            collated_inputs["global"] = torch.stack(inputs_by_type["global"])
        
        # Process sequence types - stack along batch dimension
        for seq_type, tensors in inputs_by_type.items():
            if seq_type == "global":
                continue
                
            # Verify all sequences have same length
            seq_lengths = [t.size(0) for t in tensors]
            if not all(length == seq_lengths[0] for length in seq_lengths):
                raise ValueError(f"Inconsistent sequence lengths in batch for {seq_type}")
                
            collated_inputs[seq_type] = torch.stack(tensors)
        
        # Collate targets
        if targets_list:
            # Verify all target sequences have same length
            target_lengths = [t.size(0) for t in targets_list]
            if not all(length == target_lengths[0] for length in target_lengths):
                raise ValueError(f"Inconsistent target lengths in batch")
                
            collated_targets = torch.stack(targets_list)
        else:
            collated_targets = torch.tensor([])
        
        return collated_inputs, collated_targets


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=None,
                      pin_memory=None, drop_last=False, persistent_workers=None, 
                      collate_fn=None):
    """Create an optimized DataLoader for the atmospheric dataset."""
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    # Get hardware-specific dataloader settings
    hardware_settings = configure_dataloader_settings()
    
    # Use provided values or fall back to hardware-specific defaults
    if num_workers is None:
        num_workers = hardware_settings["num_workers"]
    if pin_memory is None:
        pin_memory = hardware_settings["pin_memory"]
    if persistent_workers is None:
        persistent_workers = hardware_settings["persistent_workers"] and num_workers > 0
        
    # If no collate function provided, use MultiSourceCollate
    if collate_fn is None:
        collate_fn = MultiSourceCollate()
            
    logger.info(f"Creating DataLoader with batch_size={batch_size}, num_workers={num_workers}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=drop_last,
        collate_fn=collate_fn
    )