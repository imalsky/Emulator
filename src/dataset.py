#!/usr/bin/env python3
"""
spectral_dataset.py

Enhanced PyTorch Dataset implementation for atmospheric profiles paired with spectral data.
Features:
- Automatic detection of sequence vs. global variables
- Improved handling of coordinate variables for large spectral outputs
- Optimized for ultra-large sequences (40,000+ points)
- Efficient LRU caching for file access
- Validation for profiles and spectra
"""

import json
import logging
from pathlib import Path
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)


class Dataset(Dataset):
    """
    PyTorch Dataset for atmospheric profile data paired with spectral outputs.
    
    Handles the case where input profiles and output spectra have different sequence lengths.
    Optimized for ultra-large spectral outputs (up to 40,000+ points).
    Uses an LRU cache to speed up repeated file access.
    """
    
    def __init__(
        self,
        data_folder,
        input_seq_length=None,
        output_seq_length=None,  # Optional, will auto-detect if None
        input_variables=None,
        target_variables=None,
        coordinate_variable=None,
        cache_size=1024
    ):
        """Initialize the dataset with the given configuration."""
        super().__init__()
        self.data_folder = Path(data_folder)
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")
            
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length  # Will be auto-detected if None
        self.input_variables = input_variables or ["pressure", "temperature"]
        self.target_variables = target_variables or ["transit_depth"]
        self.coordinate_variable = coordinate_variable or []
        
        # Validate coordinate variable is a list
        if isinstance(self.coordinate_variable, str):
            self.coordinate_variable = [self.coordinate_variable]
        
        self.all_variables = self.input_variables + self.target_variables + self.coordinate_variable
        self.required_keys = set(self.input_variables + self.target_variables)
        self.cache_size = cache_size
        self.cache = OrderedDict()
        
        # Additional properties for variable type tracking
        self.global_variables = []  # Will store scalar variables
        self.sequence_variables = []  # Will store sequence variables
        
        # Initialize dataset will auto-detect output_seq_length and variable types
        self._initialize_dataset()
        
        logger.info(f"Initialized dataset with {len(self.valid_files)} valid profiles")
        logger.info(f"Input variables: {self.input_variables} (seq length: {self.input_seq_length})")
        logger.info(f"Target variables: {self.target_variables} (seq length: {self.output_seq_length})")
        
        # Log detected variable types
        if self.global_variables:
            logger.info(f"Detected global (scalar) variables: {self.global_variables}")
        if self.sequence_variables:
            logger.info(f"Detected sequence variables: {self.sequence_variables}")
        if self.coordinate_variable:
            logger.info(f"Coordinate variable(s): {self.coordinate_variable}")

    def _initialize_dataset(self):
        """Scan data folder for JSON files and build the dataset index with enhanced auto-detection."""
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {self.data_folder}")
            
        json_files = [
            f for f in self.data_folder.glob("*.json")
            if f.name != "normalization_metadata.json"
        ]
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.data_folder}")

        self.valid_files = []
        self.metadata = {}
        self.filenames = []
        invalid_count = 0
        
        # For auto-detecting output sequence length and variable types
        coordinate_lengths = {}  # Dictionary to store lengths for each coordinate variable
        variable_types = {var: {"is_sequence": False, "count": 0, "lengths": []} for var in self.all_variables}
        
        # First pass: detect variable types and sequence lengths
        for file_path in json_files[:min(50, len(json_files))]:  # Sample up to 50 files for detection
            try:
                profile = self._safe_load_json(file_path)
                
                # Check for any coordinate variables
                for coord_var in self.coordinate_variable:
                    if coord_var in profile and isinstance(profile[coord_var], list):
                        if coord_var not in coordinate_lengths:
                            coordinate_lengths[coord_var] = len(profile[coord_var])
                            logger.info(f"Detected {coord_var} sequence length: {coordinate_lengths[coord_var]}")
                
                # Analyze variable types
                for var in self.all_variables:
                    if var in profile:
                        if isinstance(profile[var], list):
                            variable_types[var]["is_sequence"] = True
                            variable_types[var]["count"] += 1
                            variable_types[var]["lengths"].append(len(profile[var]))
                        else:
                            variable_types[var]["count"] += 1
            except Exception as e:
                logger.warning(f"Error analyzing file {file_path.name}: {str(e)}")
        
        # Categorize variables as global or sequence
        for var, info in variable_types.items():
            if info["count"] > 0:
                if info["is_sequence"]:
                    self.sequence_variables.append(var)
                    # Log median sequence length for this variable
                    if info["lengths"]:
                        median_length = np.median(info["lengths"]).astype(int)
                        logger.debug(f"Variable {var} has median sequence length: {median_length}")
                else:
                    self.global_variables.append(var)
        
        # Choose output sequence length from coordinate lengths if available
        detected_output_seq_length = None
        if coordinate_lengths:
            # Use the first coordinate's length as our reference
            coord_var = next(iter(coordinate_lengths))
            detected_output_seq_length = coordinate_lengths[coord_var]
            logger.info(f"Using {coord_var} length ({detected_output_seq_length}) as output sequence length")
        
        # Second pass: validate and build dataset
        for file_path in json_files:
            try:
                profile = self._safe_load_json(file_path)
                
                if self._validate_profile(profile):
                    self.valid_files.append(file_path)
                    
                    # Get sequence lengths for input and output
                    input_seq_length = self._get_sequence_length(profile, self.input_variables)
                    
                    # Determine output sequence length
                    output_seq_length = self.output_seq_length
                    if output_seq_length is None:
                        if detected_output_seq_length is not None:
                            output_seq_length = detected_output_seq_length
                        else:
                            output_seq_length = self._get_sequence_length(profile, self.target_variables)
                    
                    self.metadata[str(file_path)] = {
                        "input_seq_length": input_seq_length,
                        "output_seq_length": output_seq_length
                    }
                    self.filenames.append(file_path.name)
                else:
                    invalid_count += 1
            except Exception as e:
                logger.warning(f"Skipping invalid file {file_path.name}: {str(e)}")
                invalid_count += 1
                continue

        if not self.valid_files:
            raise ValueError(f"No valid profiles found in {self.data_folder}")
        
        # Set the output_seq_length based on what we found
        if self.output_seq_length is None:
            if detected_output_seq_length is not None:
                self.output_seq_length = detected_output_seq_length
            else:
                # Use the first valid file's output sequence length as default
                first_file = str(self.valid_files[0])
                self.output_seq_length = self.metadata[first_file]["output_seq_length"]
        
        if invalid_count:
            logger.warning(f"Found {invalid_count} invalid files")
            
        logger.info(f"Found {len(self.valid_files)} valid profile files")
        
        # Warn if dealing with ultra-large sequences
        if self.output_seq_length > 30000:
            logger.warning(f"Ultra-large output sequence detected ({self.output_seq_length} points)")
            logger.warning("Memory usage may be high, consider using optimized architecture")

    @staticmethod
    def _safe_load_json(file_path):
        """Safely load and parse a JSON file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in {file_path.name}: {e.msg}", e.doc, e.pos)
        except OSError as e:
            raise OSError(f"Error reading file {file_path.name}: {e}")

    def _validate_profile(self, profile):
        """
        Validate a profile by checking required keys, data types, and sequence lengths.
        """
        # Check profile type
        if not isinstance(profile, dict):
            logger.warning("Profile is not a dictionary")
            return False
                
        # Check required keys
        if not self.required_keys.issubset(profile.keys()):
            missing = self.required_keys - set(profile.keys())
            logger.warning(f"Missing required keys: {missing}")
            return False

        # Check input sequence length if specified
        try:
            input_seq_length = self._get_sequence_length(profile, self.input_variables)
            if self.input_seq_length and input_seq_length != self.input_seq_length:
                logger.warning(f"Expected input sequence length {self.input_seq_length}, got {input_seq_length}")
                return False
        except Exception as e:
            logger.warning(f"Error determining input sequence length: {str(e)}")
            return False
            
        # Check values are numeric and finite
        try:
            for var in self.all_variables:
                if var not in profile:
                    continue
                    
                val = profile[var]
                if isinstance(val, list):
                    if not all(isinstance(x, (int, float)) for x in val):
                        logger.warning(f"Non-numeric values in {var}")
                        return False
                    if not all(np.isfinite(x) for x in val):
                        logger.warning(f"Non-finite values in {var}")
                        return False
                else:
                    if not isinstance(val, (int, float)):
                        logger.warning(f"{var} is not numeric: {type(val).__name__}")
                        return False
                    if not np.isfinite(val):
                        logger.warning(f"{var} is not finite: {val}")
                        return False
            return True
        except Exception as e:
            logger.warning(f"Error validating values: {str(e)}")
            return False

    def _get_sequence_length(self, profile, variables):
        """
        Determine the maximum sequence length from a list of variables.
        """
        max_length = 1  # Default for scalar values
        
        for var in variables:
            # Skip if variable doesn't exist in this profile
            if var not in profile:
                continue
                
            val = profile[var]
            if isinstance(val, list):
                max_length = max(max_length, len(val))
                
        return max_length

    def __len__(self):
        """Return the number of valid profiles in the dataset."""
        return len(self.valid_files)

    def _process_variable(self, profile, var_name, seq_length):
        """
        Handle both sequence and global (scalar) variables with improved null handling.
        """
        # Check if variable exists in profile
        if var_name not in profile:
            # Handle missing variable by creating a tensor of zeros
            return torch.zeros(seq_length, dtype=torch.float32)
        
        val = profile[var_name]
        
        if val is None:
            # Handle null value by creating a tensor of zeros
            return torch.zeros(seq_length, dtype=torch.float32)
        
        if isinstance(val, list):
            # Handle list (sequence) variable
            if not val:  # Empty list
                return torch.zeros(seq_length, dtype=torch.float32)
                
            data = torch.tensor(val, dtype=torch.float32)
            
            # Handle sequence length mismatch
            if len(data) != seq_length:
                if len(data) > seq_length:
                    # Truncate
                    data = data[:seq_length]
                else:
                    # Pad with last value or zeros
                    padding = torch.full((seq_length - len(data),), data[-1] if len(data) > 0 else 0.0, 
                                        dtype=torch.float32)
                    data = torch.cat([data, padding])
            return data
        else:
            # For global (scalar) variables, create a constant sequence
            return torch.full((seq_length,), float(val), dtype=torch.float32)

    def __getitem__(self, idx):
        """
        Get the input and target tensors for a profile with efficient memory handling.
        """
        # Check if item is in cache
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]

        # Load and process the profile
        file_path = self.valid_files[idx]
        profile = self._safe_load_json(file_path)
        
        # Get sequence lengths from metadata or compute them
        metadata = self.metadata[str(file_path)]
        input_seq_length = metadata.get("input_seq_length", self.input_seq_length or 1)
        output_seq_length = metadata.get("output_seq_length", self.output_seq_length or 1)

        # Process input variables
        input_tensors = []
        for var in self.input_variables:
            input_tensors.append(self._process_variable(profile, var, input_seq_length))
            
        # Process target variables
        target_tensors = []
        for var in self.target_variables:
            target_tensors.append(self._process_variable(profile, var, output_seq_length))
        
        # Stack tensors along feature dimension
        input_tensor = torch.stack(input_tensors, dim=1)
        target_tensor = torch.stack(target_tensors, dim=1)
        
        # Process coordinates if provided
        coordinates = None
        if self.coordinate_variable:
            for coord_var in self.coordinate_variable:
                if coord_var in profile and isinstance(profile[coord_var], list):
                    coord_data = torch.tensor(profile[coord_var], dtype=torch.float32)
                    
                    # Ensure correct length
                    if len(coord_data) != output_seq_length:
                        if len(coord_data) > output_seq_length:
                            coord_data = coord_data[:output_seq_length]
                        else:
                            # Extend coordinates by linear interpolation if possible
                            orig_len = len(coord_data)
                            if orig_len > 1:
                                # Create interpolated values
                                step = (coord_data[-1] - coord_data[0]) / (orig_len - 1)
                                extended = torch.tensor([coord_data[-1] + step * (i+1) for i in range(output_seq_length - orig_len)])
                                coord_data = torch.cat([coord_data, extended])
                            else:
                                # If only one value, just repeat it
                                coord_data = coord_data.repeat(output_seq_length)
                    
                    coordinates = coord_data
                    break  # Use the first available coordinate variable
            
            # If no coordinate variable was found in the profile but it was requested,
            # generate a default set of coordinates
            if coordinates is None and self.coordinate_variable:
                logger.debug(f"No coordinate variable found in profile for {file_path.name}, generating default coordinates")
                coordinates = torch.linspace(0, 1, output_seq_length, dtype=torch.float32)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)  # Remove oldest item
            
        # Include coordinates in cache if available
        result = (input_tensor, target_tensor, coordinates) if coordinates is not None else (input_tensor, target_tensor)
        self.cache[idx] = result
        
        return result

    def clear_cache(self):
        """Clear the LRU cache to free memory."""
        self.cache.clear()

    @classmethod
    def create_dataloader(
        cls,
        dataset,
        batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    ):
        """Create an optimized DataLoader for the dataset."""
        # Auto-adjust batch size for very large sequences
        if hasattr(dataset, 'output_seq_length'):
            if dataset.output_seq_length > 40000 and batch_size > 2:
                logger.warning(f"Reducing batch size from {batch_size} to 2 due to extremely large sequence length")
                batch_size = 2
            elif dataset.output_seq_length > 20000 and batch_size > 4:
                logger.warning(f"Reducing batch size from {batch_size} to 4 due to large sequence length")
                batch_size = 4
        
        # Check if dataset is empty
        if len(dataset) == 0:
            logger.error("Dataset is empty - cannot create DataLoader")
            raise ValueError("Cannot create DataLoader from empty dataset")
            
        # Check persistent_workers setting with num_workers
        if persistent_workers and num_workers == 0:
            logger.warning("persistent_workers=True has no effect when num_workers=0, setting to False")
            persistent_workers = False
        
        # Set prefetch_factor only if num_workers > 0
        prefetch_factor = 2 if num_workers > 0 else None
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor,
            drop_last=drop_last
        )