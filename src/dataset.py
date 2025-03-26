#!/usr/bin/env python3
"""
dataset.py - Atmospheric profile dataset for multi-source transformer model

Handles atmospheric profile data with support for:
- Separate processing of different data types (global, sequence)
- Variable-length sequences without excessive padding
- Automatic detection of feature types
- Efficient data caching
- Custom collation for structured batch processing
"""

import json
import logging
from pathlib import Path
from collections import OrderedDict, defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import hardware utilities
from hardware import configure_dataloader_settings

logger = logging.getLogger(__name__)


class AtmosphericDataset(Dataset):
    """
    Dataset for atmospheric profile data with support for multiple data sources.
    
    Separates data into different types (global, sequence) to allow
    efficient processing by specialized encoders without excessive padding.
    """

    def __init__(
        self,
        data_folder,
        input_variables=None,
        target_variables=None,
        global_variables=None,
        sequence_types=None,
        cache_size=1024,
        allow_variable_length=True
    ):
        """
        Initialize dataset with multi-source configuration.
        
        Parameters
        ----------
        data_folder : str or Path
            Path to the folder containing profile data
        input_variables : list of str, optional
            Names of input variables to use
        target_variables : list of str, optional
            Names of target variables to predict
        global_variables : list of str, optional
            Names of global (non-sequential) features
        sequence_types : dict, optional
            Mapping of sequence type names to lists of variable names
        cache_size : int, optional
            Maximum number of profiles to cache in memory
        allow_variable_length : bool, optional
            Whether to allow variable-length sequences
        """
        super().__init__()
        self.data_folder = Path(data_folder)
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        # Variables configuration
        self.input_variables = input_variables or ["pressure", "temperature"]
        self.target_variables = target_variables or ["transit_depth"]
        self.all_variables = list(set(self.input_variables + self.target_variables))
        self.required_keys = set(self.all_variables)

        # Variable classification
        self.global_variables = global_variables or []
        self.sequence_types = sequence_types or {}

        # If sequence types not specified, create a default structure
        if not self.sequence_types:
            # Assign all non-global variables to a default sequence type
            seq_vars = [var for var in self.input_variables if var not in self.global_variables]
            self.sequence_types = {"sequence1": seq_vars}

        # Validate that all input variables are assigned
        self._validate_variable_assignments()

        # Tracking sequence lengths for each type
        self.sequence_lengths = defaultdict(int)
        self.allow_variable_length = allow_variable_length

        # File caching
        self.cache_size = cache_size
        self.cache = OrderedDict()

        # Initialize dataset from files
        self._initialize_dataset()
        self._log_initialization_info()

    def _validate_variable_assignments(self):
        """
        Validate that all input and target variables are assigned to a sequence type or global.
        """
        # Collect all assigned variables
        assigned_vars = set(self.global_variables)
        for seq_vars in self.sequence_types.values():
            assigned_vars.update(seq_vars)
        
        # Check for unassigned input variables
        unassigned_inputs = set(self.input_variables) - assigned_vars
        if unassigned_inputs:
            raise ValueError(f"The following input variables are not assigned to any sequence type or global: {unassigned_inputs}")
        
        # Check for unassigned target variables
        unassigned_targets = set(self.target_variables) - assigned_vars
        if unassigned_targets:
            raise ValueError(f"The following target variables are not assigned to any sequence type: {unassigned_targets}")

    def _log_initialization_info(self):
        """Log information about the initialized dataset."""
        logger.info(f"Initialized dataset with {len(self.valid_files)} valid profiles")
        logger.info(f"Input variables: {self.input_variables}")
        logger.info(f"Target variables: {self.target_variables}")
        
        if self.global_variables:
            logger.info(f"Global variables: {self.global_variables}")
        
        for seq_type, var_list in self.sequence_types.items():
            logger.info(f"Sequence type '{seq_type}': {var_list}")
            logger.info(f"  - Sequence length: {self.sequence_lengths[seq_type]}")
        
        for var in self.target_variables:
            logger.info(f"Target '{var}' sequence length: {self.sequence_lengths[var]}")

    def _initialize_dataset(self):
        """Analyze JSON files, detect variable types, and build the dataset."""
        # Get all JSON files except the normalization metadata file
        json_files = [f for f in self.data_folder.glob("*.json") if f.name != "normalization_metadata.json"]
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.data_folder}")

        # Initialize storage for dataset index
        self.valid_files = []
        self.metadata = {}
        self.filenames = []

        # Validate files and detect sequence lengths
        self._detect_sequence_lengths(json_files[:min(50, len(json_files))])
        self._validate_and_build_dataset(json_files)

    def _detect_sequence_lengths(self, sample_files):
        """Detect typical sequence lengths for each data type."""
        # Initialize storage for sequence length statistics
        seq_length_stats = defaultdict(list)
        target_length_stats = defaultdict(list)
        
        logger.info(f"Analyzing {len(sample_files)} files to detect sequence lengths")
        
        # Process each sample file
        for file_path in sample_files:
            try:
                profile = self._load_json_safely(file_path)
                if not profile:
                    continue
                    
                # Check each sequence type
                for seq_type, var_list in self.sequence_types.items():
                    # Get typical length from the first variable in this sequence type
                    if var_list:
                        var_name = var_list[0]
                        if var_name in profile and isinstance(profile[var_name], list):
                            seq_length_stats[seq_type].append(len(profile[var_name]))
                
                # Check target variables
                for var in self.target_variables:
                    if var in profile and isinstance(profile[var], list):
                        target_length_stats[var].append(len(profile[var]))
                        
            except Exception as e:
                logger.warning(f"Error analyzing {file_path.name}: {e}")
        
        # Set typical sequence lengths based on median values
        for seq_type, lengths in seq_length_stats.items():
            if lengths:
                self.sequence_lengths[seq_type] = int(np.median(lengths))
        
        # Set target sequence lengths
        for var, lengths in target_length_stats.items():
            if lengths:
                self.sequence_lengths[var] = int(np.median(lengths))
                self.sequence_lengths["output"] = int(np.median(lengths))  # Use for convenience

    def _validate_and_build_dataset(self, json_files):
        """Validate JSON files and build the dataset index."""
        logger.info(f"Validating {len(json_files)} files")
        invalid_count = 0
        
        for file_path in json_files:
            try:
                profile = self._load_json_safely(file_path)
                if not profile:
                    invalid_count += 1
                    continue
                    
                if self._validate_profile(profile):
                    self.valid_files.append(file_path)
                    
                    # Collect sequence lengths for this file
                    seq_lengths = {}
                    for seq_type, var_list in self.sequence_types.items():
                        if var_list:
                            var_name = var_list[0]
                            if isinstance(profile.get(var_name), list):
                                seq_lengths[seq_type] = len(profile[var_name])
                    
                    # Collect target lengths
                    target_lengths = {}
                    for var in self.target_variables:
                        if isinstance(profile.get(var), list):
                            target_lengths[var] = len(profile[var])
                    
                    # Store metadata
                    self.metadata[str(file_path)] = {
                        "sequence_lengths": seq_lengths,
                        "target_lengths": target_lengths
                    }
                    
                    self.filenames.append(file_path.name)
                else:
                    invalid_count += 1
            except Exception as e:
                logger.warning(f"Skipping {file_path.name}: {e}")
                invalid_count += 1
                
        if not self.valid_files:
            raise ValueError(f"No valid profiles found in {self.data_folder}")
            
        if invalid_count:
            logger.warning(f"Found {invalid_count} invalid files")

    def _load_json_safely(self, file_path):
        """Load a JSON file with error handling."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
            logger.warning(f"Error loading JSON from {file_path}: {e}")
            return None

    def _validate_profile(self, profile):
        """Validate a profile by ensuring required keys exist and values are numeric."""
        if not isinstance(profile, dict):
            return False
        if not self.required_keys.issubset(profile.keys()):
            return False
        
        for var in self.all_variables:
            val = profile.get(var)
            if isinstance(val, list):
                if not val or not all(isinstance(x, (int, float)) and np.isfinite(x) for x in val):
                    return False
            else:
                if not isinstance(val, (int, float)) or not np.isfinite(val):
                    return False
        return True

    def __len__(self):
        """Return the number of valid profiles."""
        return len(self.valid_files)

    def _process_variable(self, profile, var_name, seq_length=None):
        """
        Return a tensor for a variable, optionally resizing to specified length.
        
        Parameters
        ----------
        profile : dict
            Profile data
        var_name : str
            Variable name to process
        seq_length : int, optional
            Target sequence length (if None, uses natural length)
            
        Returns
        -------
        torch.Tensor
            Processed variable as tensor
        """
        val = profile.get(var_name)
        
        if val is None:
            return torch.zeros(1 if seq_length is None else seq_length, dtype=torch.float32)
            
        if isinstance(val, list):
            if not val:
                return torch.zeros(1 if seq_length is None else seq_length, dtype=torch.float32)
                
            data = torch.tensor(val, dtype=torch.float32)
            
            # If seq_length is specified and differs from data length, adjust
            if seq_length is not None and len(data) != seq_length:
                return self._adjust_length(data, seq_length)
            
            return data
        else:
            # For scalars, either return as scalar or create constant sequence
            if seq_length is None:
                return torch.tensor(float(val), dtype=torch.float32).reshape(1)
            else:
                return torch.full((seq_length,), float(val), dtype=torch.float32)

    def __getitem__(self, idx):
        """
        Return inputs and targets for the given profile index.
        
        Returns inputs grouped by data type for the multi-source transformer.
        
        Parameters
        ----------
        idx : int
            Profile index
            
        Returns
        -------
        tuple
            (inputs, targets) where inputs is a dictionary mapping data type
            to tensor and targets is a tensor
        """
        # Use cached data if available
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
            
        if idx < 0 or idx >= len(self.valid_files):
            raise IndexError(f"Index {idx} out of range")
            
        file_path = self.valid_files[idx]
        
        try:
            profile = self._load_json_safely(file_path)
            if not profile:
                raise ValueError(f"Failed to load profile from {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            # Return empty data on error
            global_features = torch.zeros(len(self.global_variables), dtype=torch.float32)
            inputs = {"global": global_features} if self.global_variables else {}
            for seq_type, var_list in self.sequence_types.items():
                if var_list:
                    inputs[seq_type] = torch.zeros(
                        self.sequence_lengths.get(seq_type, 1), 
                        len(var_list), 
                        dtype=torch.float32
                    )
            targets = torch.zeros(
                self.sequence_lengths.get("output", 1), 
                len(self.target_variables), 
                dtype=torch.float32
            )
            return (inputs, targets)
        
        # Process global features
        if self.global_variables:
            global_features = torch.stack([
                self._process_variable(profile, var_name) 
                for var_name in self.global_variables
            ], dim=0).squeeze(1)  # Remove singleton dimension for scalar globals
            
            inputs = {"global": global_features}
        else:
            inputs = {}
        
        # Process each sequence type
        for seq_type, var_list in self.sequence_types.items():
            if not var_list:
                continue
                
            # Determine sequence length for this type
            if self.allow_variable_length:
                # Use natural length
                seq_length = None
            else:
                # Use predetermined length
                seq_length = self.sequence_lengths.get(seq_type, 1)
            
            # Create a tensor for each variable in this sequence type
            seq_features = []
            for var_name in var_list:
                tensor = self._process_variable(profile, var_name, seq_length)
                seq_features.append(tensor)
            
            # If variable length, ensure all features have the same length
            if self.allow_variable_length and seq_features:
                if not seq_features:
                    # Handle case with no valid features
                    inputs[seq_type] = torch.tensor([])
                    continue
                    
                # Find max length, checking for empty tensors
                max_len = max((t.shape[0] for t in seq_features if t.numel() > 0), default=1)
                seq_features = [
                    self._adjust_length(t, max_len) for t in seq_features
                ]
            
            # Stack features along last dimension
            inputs[seq_type] = torch.stack(seq_features, dim=1) if seq_features else torch.tensor([])
        
        # Process target variables
        target_tensors = []
        for var in self.target_variables:
            # For targets, always use their natural length
            tensor = self._process_variable(profile, var)
            target_tensors.append(tensor)
        
        # Ensure all targets have the same length
        if target_tensors:
            # Find max length, checking for empty tensors
            max_len = max((t.shape[0] for t in target_tensors if t.numel() > 0), default=1)
            target_tensors = [
                self._adjust_length(t, max_len) for t in target_tensors
            ]
            targets = torch.stack(target_tensors, dim=1)
        else:
            targets = torch.tensor([])
        
        # Add to cache efficiently
        result = (inputs, targets)
        self.cache[idx] = result
        # Manage cache size after adding the new item
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        
        return result
    
    def _adjust_length(self, tensor, target_length):
        """Adjust tensor to target length by truncating or padding."""
        if tensor.shape[0] == target_length:
            return tensor
            
        if tensor.shape[0] > target_length:
            return tensor[:target_length]
        else:
            pad_val = tensor[-1] if tensor.numel() > 0 else 0.0
            padding = torch.full((target_length - tensor.shape[0],), pad_val, dtype=tensor.dtype)
            return torch.cat([tensor, padding])

    def clear_cache(self):
        """Clear the file cache."""
        self.cache.clear()


# ------------------------------------------------------------------------------
# Collation functions for batching
# ------------------------------------------------------------------------------
class MultiSourceCollate:
    """
    Collates samples with separate handling for different data types.
    
    Each data type (global, sequence) is processed separately
    and padded only within its group, avoiding excessive padding of
    short sequences to match long ones.
    """
    
    def __call__(self, batch):
        """
        Collate a batch of samples into structured batch data.
        
        Parameters
        ----------
        batch : list of tuples
            Each tuple contains (inputs, targets) where inputs is a dictionary
            
        Returns
        -------
        tuple
            (inputs, targets) where inputs is a dictionary and targets is a tensor
        """
        if not batch:
            return {}, torch.tensor([])
            
        # Initialize containers for each data type
        inputs_dict = defaultdict(list)
        targets_list = []
        
        # Collect inputs by type and targets
        for sample_inputs, sample_targets in batch:
            # Collect inputs by type
            for key, value in sample_inputs.items():
                inputs_dict[key].append(value)
            
            # Collect targets
            targets_list.append(sample_targets)
        
        # Process each input type appropriately
        collated_inputs = {}
        
        # Global features (simple stack)
        if "global" in inputs_dict and inputs_dict["global"]:
            collated_inputs["global"] = torch.stack(inputs_dict["global"])
        
        # Process each sequence type
        for seq_type, tensors in inputs_dict.items():
            if seq_type == "global" or not tensors:
                continue
            
            # Pad sequences to the max length in this batch
            collated_inputs[seq_type] = self._pad_and_stack(tensors)
        
        # Process targets
        if targets_list:
            collated_targets = self._pad_and_stack(targets_list)
        else:
            collated_targets = torch.tensor([])
        
        return collated_inputs, collated_targets
    
    def _pad_and_stack(self, tensors):
        """
        Pad tensors to max length in batch and stack them.
        
        Parameters
        ----------
        tensors : list of torch.Tensor
            List of tensors to pad and stack
            
        Returns
        -------
        torch.Tensor
            Batched tensor with consistent shapes
        """
        if not tensors:
            return torch.tensor([])
            
        # Find max sequence length in this batch, handling empty tensors
        max_len = max((t.shape[0] for t in tensors if t.numel() > 0), default=1)
        
        # Pad tensors to max length
        padded = []
        for t in tensors:
            if t.numel() == 0:
                # Handle empty tensor
                shape = list(t.shape)
                shape[0] = max_len
                padded.append(torch.zeros(*shape, dtype=t.dtype))
            elif t.shape[0] < max_len:
                # Pad with the last value of each sequence
                pad_values = t[-1:].expand(max_len - t.shape[0], *t.shape[1:])
                padded.append(torch.cat([t, pad_values], dim=0))
            else:
                padded.append(t)
        
        # Stack along batch dimension
        return torch.stack(padded)


def create_multi_source_collate_fn():
    """Return a collate function for multi-source transformer data."""
    return MultiSourceCollate()


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=None,
                      pin_memory=None, drop_last=False, persistent_workers=None, collate_fn=None):
    """
    Create an optimized DataLoader for the atmospheric dataset.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to create loader for
    batch_size : int
        Batch size
    shuffle : bool, optional
        Whether to shuffle the data
    num_workers : int, optional
        Number of worker processes
    pin_memory : bool, optional
        Whether to pin memory (good for GPU training)
    drop_last : bool, optional
        Whether to drop the last incomplete batch
    persistent_workers : bool, optional
        Whether to keep worker processes alive between epochs
    collate_fn : callable, optional
        Custom batch collation function
        
    Returns
    -------
    torch.utils.data.DataLoader
        Configured data loader
    """
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
        
    # If no collate function provided, use MultiSourceCollate for multi-source data
    if collate_fn is None:
        collate_fn = create_multi_source_collate_fn()
            
    logger.info(f"Creating DataLoader with batch_size={batch_size}, num_workers={num_workers}, persistent_workers={persistent_workers}")
    
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