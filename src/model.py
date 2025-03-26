#!/usr/bin/env python3
"""
model.py - Bidirectional encoder-only transformer for atmospheric data prediction

Features:
- Supports up to 2 sequence types for input.
- Global feature integration with FiLM conditioning.
- Cross-attention for inter-sequence information exchange (when multiple sequence types exist).
- Clean, robust error handling and clear documentation.

The model is designed for regression tasks on 1D atmospheric profiles. The output sequence length is
determined by the target variable specification.
"""

import math
import logging
from typing import Dict, Optional, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Positional Encoding
# ------------------------------------------------------------------------------
class SinePositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding module.

    This module generates a fixed sinusoidal positional encoding and adds it to the input.
    Expected input shape: [batch_size, seq_length, d_model]
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, d_model].

        Returns:
            torch.Tensor: Positional encoded tensor.
        """
        return x + self.pe[:, :x.size(1), :]

# ------------------------------------------------------------------------------
# Sequence Encoder
# ------------------------------------------------------------------------------
class SequenceEncoder(nn.Module):
    """
    Bidirectional encoder for sequence data.

    Projects input features to d_model dimensions, adds positional encodings, and processes
    the sequence using a TransformerEncoder.
    
    Expected input shape: [batch_size, seq_length, input_dim]
    """
    def __init__(
        self, 
        input_dim: int, 
        d_model: int, 
        nhead: int, 
        num_layers: int, 
        dim_feedforward: int, 
        dropout: float = 0.1, 
        norm_first: bool = False
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = SinePositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the sequence encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, input_dim].

        Returns:
            torch.Tensor: Encoded sequence tensor of shape [batch_size, seq_length, d_model].
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        return self.encoder(x)

# ------------------------------------------------------------------------------
# Global Features Encoder
# ------------------------------------------------------------------------------
class GlobalEncoder(nn.Module):
    """
    Encodes global features using a multi-layer perceptron (MLP).

    Expected input shape: [batch_size, global_dim]
    Output shape: [batch_size, d_model]
    """
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the global encoder.

        Args:
            x (torch.Tensor): Global features tensor of shape [batch_size, global_dim].

        Returns:
            torch.Tensor: Encoded global features of shape [batch_size, d_model].
        """
        return self.encoder(x)

# ------------------------------------------------------------------------------
# Feature Integration with FiLM
# ------------------------------------------------------------------------------
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Conditions a sequence of features with global features by applying a learned scale and shift.
    
    Args:
        d_model (int): The number of features in the sequence.
        
    Expected x shape: [batch_size, seq_length, d_model]
    Expected condition shape: [batch_size, d_model]
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.film_proj = nn.Linear(d_model, d_model * 2)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying FiLM conditioning.

        Args:
            x (torch.Tensor): Input sequence tensor [batch_size, seq_length, d_model].
            condition (torch.Tensor): Global features [batch_size, d_model].

        Returns:
            torch.Tensor: Conditioned sequence tensor [batch_size, seq_length, d_model].
        """
        film_params = self.film_proj(condition)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.unsqueeze(1).expand(-1, x.size(1), -1)
        beta = beta.unsqueeze(1).expand(-1, x.size(1), -1)
        return gamma * x + beta

# ------------------------------------------------------------------------------
# Cross-Attention Module
# ------------------------------------------------------------------------------
class CrossAttentionModule(nn.Module):
    """
    Cross-attention module that allows one sequence to attend to another.

    Args:
        d_model (int): Dimensionality of the input and output features.
        nhead (int): Number of attention heads.

    Expected query shape: [batch_size, query_len, d_model]
    Expected key_value shape: [batch_size, kv_len, d_model]
    Returns a tensor with shape [batch_size, query_len, d_model].
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention from the query sequence to the key/value sequence.

        Args:
            query (torch.Tensor): Query tensor [batch_size, query_len, d_model].
            key_value (torch.Tensor): Key/Value tensor [batch_size, kv_len, d_model].

        Returns:
            torch.Tensor: Tensor of shape [batch_size, query_len, d_model] after attention.
        """
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_output))

# ------------------------------------------------------------------------------
# Multi-Encoder Transformer
# ------------------------------------------------------------------------------
class MultiEncoderTransformer(nn.Module):
    """
    Encoder-only transformer for multi-sequence data with cross-attention.

    This model processes up to 2 different sequence types and integrates global features 
    via FiLM conditioning. Cross-attention is applied between the two sequence types 
    (if both are provided). The output sequence is determined by the output_seq_type 
    provided during initialization.

    Args:
        global_variables (List[str]): List of global variable names.
        sequence_dims (Dict[str, Dict[str, int]]): Dictionary mapping sequence type names to their variables and dimensions.
        output_dim (int): Number of output features per timestep.
        d_model (int): Dimension of model embeddings.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of layers in each TransformerEncoder.
        dim_feedforward (int): Dimension of the feedforward network inside the encoder.
        dropout (float): Dropout rate.
        output_seq_type (str): Key from sequence_dims to determine output sequence length.
        norm_first (bool): If True, apply layer normalization before other operations.
        max_sequence_length (int): Maximum supported sequence length.

    Expected input: A dictionary with keys corresponding to sequence types (max 2) and optionally 'global'.
                    Each sequence tensor should have shape [batch_size, seq_length, feature_dim].
                    If global features are provided, they should have shape [batch_size, global_dim].

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, seq_length, output_dim].
    """
    def __init__(
        self,
        global_variables: List[str],
        sequence_dims: Dict[str, Dict[str, int]],
        output_dim: int = 1,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_seq_type: Optional[str] = None,
        norm_first: bool = False,
        max_sequence_length: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        
        # Filter out empty sequence types
        self.sequence_dims = {
            seq_type: var_dict for seq_type, var_dict in sequence_dims.items() 
            if var_dict  # Only keep non-empty sequence types
        }
        self.sequence_types = list(self.sequence_dims.keys())
        self.global_variables = global_variables
        
        # Raise error if more than 2 sequence types are provided.
        if len(self.sequence_types) > 2:
            raise ValueError(f"Only up to 2 sequence types are supported, but received {len(self.sequence_types)}: {self.sequence_types}")
        
        self.has_global = len(global_variables) > 0
        
        # Validate output sequence type.
        if output_seq_type is None:
            raise ValueError("output_seq_type must be specified to match target length")
        if output_seq_type not in self.sequence_dims:
            raise ValueError(f"output_seq_type '{output_seq_type}' not found in non-empty sequence_types: {list(self.sequence_dims.keys())}")
        self.output_seq_type = output_seq_type
        
        # Create sequence encoders for each non-empty sequence type.
        self.sequence_encoders = nn.ModuleDict()
        for seq_type, var_dict in self.sequence_dims.items():
            input_dim = len(var_dict)
            self.sequence_encoders[seq_type] = SequenceEncoder(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=norm_first
            )
        
        # Global encoder and FiLM conditioning.
        if self.has_global:
            self.global_encoder = GlobalEncoder(input_dim=len(global_variables), d_model=d_model, dropout=dropout)
            self.film_layers = nn.ModuleDict({
                seq_type: FiLMLayer(d_model) for seq_type in self.sequence_types
            })
        
        # Cross-attention between sequence types (if two exist).
        self.use_cross_attention = len(self.sequence_types) == 2
        if self.use_cross_attention:
            self.cross_attention = nn.ModuleDict({
                f"{self.sequence_types[0]}_to_{self.sequence_types[1]}": CrossAttentionModule(d_model=d_model, nhead=nhead, dropout=dropout),
                f"{self.sequence_types[1]}_to_{self.sequence_types[0]}": CrossAttentionModule(d_model=d_model, nhead=nhead, dropout=dropout)
            })
        
        # Output projection layer.
        self.output_proj = nn.Linear(d_model, output_dim)
        
        self._init_parameters()
        
        logger.info(f"Created MultiEncoderTransformer with sequence types: {self.sequence_types}")
        if self.has_global:
            logger.info(f"Global variables: {self.global_variables}")
        logger.info(f"Using '{self.output_seq_type}' as output sequence type")
        if self.use_cross_attention:
            logger.info("Using cross-attention between sequence types")
        else:
            logger.info("Using self-attention only (no cross-attention)")
    
    def _init_parameters(self):
        """
        Initialize model parameters using Glorot/Xavier initialization.
        Uses targeted initialization for weight matrices and biases.
        """
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif 'weight' in name and '.norm' not in name:
                nn.init.uniform_(p, -0.1, 0.1)
        if hasattr(self, 'output_proj'):
            nn.init.xavier_uniform_(self.output_proj.weight, gain=0.01)
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)
    
    def _validate_inputs(self, inputs: dict) -> Dict[str, int]:
        """
        Validate input dictionary and tensor shapes.
        
        Args:
            inputs (dict): Dictionary containing input sequences and optional global features.
            
        Returns:
            Dict[str, int]: A dictionary mapping sequence types to their sequence lengths.
            
        Raises:
            ValueError: If required keys are missing or tensor shapes do not match expectations.
        """
        if not isinstance(inputs, dict):
            raise ValueError(f"Expected inputs to be a dictionary, got {type(inputs)}")
        
        sequence_lengths = {}
        for seq_type in self.sequence_types:
            if seq_type not in inputs:
                raise ValueError(f"Missing required sequence type: {seq_type}")
            
            seq = inputs[seq_type]
            if not isinstance(seq, torch.Tensor):
                raise ValueError(f"Expected tensor for {seq_type}, got {type(seq)}")
            
            expected_features = len(self.sequence_dims[seq_type])
            if seq.size(-1) != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features for {seq_type}, got {seq.size(-1)}"
                )
            sequence_lengths[seq_type] = seq.size(1)
        
        if self.has_global and 'global' not in inputs:
            raise ValueError("Missing required global features")
            
        logger.debug(f"Input sequence lengths: {sequence_lengths}")
        return sequence_lengths
    
    def forward(self, inputs: dict, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            inputs (dict): Dictionary containing input sequences and optional global features.
                - Each sequence should have shape [batch_size, seq_length, feature_dim].
                - Global features (if provided) should have shape [batch_size, global_dim].
            targets (torch.Tensor, optional): Target tensor for validation, shape [batch_size, seq_length, output_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_length, output_dim].

        Raises:
            ValueError: If target sequence length does not match the output sequence length.
        """
        try:
            sequence_lengths = self._validate_inputs(inputs)
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            raise
        
        # Validate target length if provided.
        if targets is not None:
            target_length = targets.size(1)
            output_seq_length = sequence_lengths[self.output_seq_type]
            if target_length != output_seq_length:
                raise ValueError(
                    f"Target length ({target_length}) does not match output sequence length "
                    f"({output_seq_length}) from '{self.output_seq_type}'"
                )
        
        global_features = None
        if self.has_global and 'global' in inputs:
            global_features = self.global_encoder(inputs['global'])
        
        encoded_sequences = {}
        for seq_type, encoder in self.sequence_encoders.items():
            if seq_type in inputs and inputs[seq_type].numel() > 0:
                encoded = encoder(inputs[seq_type])
                if self.has_global and global_features is not None and seq_type in self.film_layers:
                    encoded = self.film_layers[seq_type](encoded, global_features)
                encoded_sequences[seq_type] = encoded
        
        # If both sequence types are present and cross-attention is enabled, apply cross-attention between them.
        if self.use_cross_attention and len(encoded_sequences) == 2:
            try:
                # Use the actual sequence types present in encoded_sequences
                available_seq_types = list(encoded_sequences.keys())
                
                # Apply cross-attention only if we have the right modules for these sequences
                if len(available_seq_types) == 2:
                    seq0, seq1 = available_seq_types
                    
                    # First direction: seq0 attends to seq1
                    key = f"{seq0}_to_{seq1}"
                    if key in self.cross_attention:
                        encoded_sequences[seq0] = self.cross_attention[key](
                            encoded_sequences[seq0], encoded_sequences[seq1]
                        )
                    
                    # Second direction: seq1 attends to seq0
                    key = f"{seq1}_to_{seq0}"
                    if key in self.cross_attention:
                        encoded_sequences[seq1] = self.cross_attention[key](
                            encoded_sequences[seq1], encoded_sequences[seq0]
                        )
            except Exception as e:
                logger.error(f"Error during cross-attention: {e}")
                logger.error(f"Available sequence types: {list(encoded_sequences.keys())}")
                raise
        
        if self.output_seq_type not in encoded_sequences:
            raise ValueError(f"Output sequence type '{self.output_seq_type}' not found in encoded sequences")
            
        output_sequence = encoded_sequences[self.output_seq_type]
        
        # Final output projection.
        output = self.output_proj(output_sequence)
        return output

# ------------------------------------------------------------------------------
# Model Creation Function
# ------------------------------------------------------------------------------
def create_prediction_model(config: dict, device: Optional[torch.device] = None) -> nn.Module:
    """
    Create a prediction model from a configuration dictionary.

    The configuration should include:
      - "input_variables": A list of input variable names.
      - "target_variables": A list of target variable names.
      - "sequence_types": A dictionary mapping sequence type names to lists of variable names.
      - "global_variables": A list of global variable names.
      - Additional hyperparameters: "d_model", "nhead", "num_encoder_layers", "dim_feedforward",
         "dropout", "norm_first", and "max_sequence_length".

    Args:
        config (dict): Configuration dictionary.
        device (torch.device, optional): Device to which the model will be moved.

    Returns:
        nn.Module: The constructed MultiEncoderTransformer model.

    Raises:
        ValueError: If required configuration keys are missing or inconsistent.
    """
    try:
        input_variables = config.get("input_variables", [])
        target_variables = config.get("target_variables", [])
        
        if not input_variables:
            raise ValueError("No input variables specified in the configuration.")
        if not target_variables:
            raise ValueError("No target variables specified in the configuration.")
        
        # Get sequence types configuration
        sequence_types_config = config.get("sequence_types", {})
        if not sequence_types_config:
            raise ValueError("No sequence types specified in the configuration.")
        
        # Get global variables
        global_variables = config.get("global_variables", [])
        
        # Validate that all input variables are assigned to a sequence type or are global
        all_assigned_vars = set()
        for seq_type, var_list in sequence_types_config.items():
            all_assigned_vars.update(var_list)
        
        all_assigned_vars.update(global_variables)
        
        unassigned_inputs = set(input_variables) - all_assigned_vars
        if unassigned_inputs:
            raise ValueError(f"The following input variables are not assigned to any sequence type or global: {unassigned_inputs}")
        
        # Validate that all target variables are assigned to a sequence type
        unassigned_targets = set(target_variables) - all_assigned_vars
        if unassigned_targets:
            raise ValueError(f"The following target variables are not assigned to any sequence type: {unassigned_targets}")
        
        # Convert sequence_types_config to sequence_dims with variable counts
        sequence_dims = {}
        for seq_type, var_list in sequence_types_config.items():
            # Create a dictionary mapping variable name to its position in the sequence
            sequence_dims[seq_type] = {var: i for i, var in enumerate(var_list)}
        
        # Determine the output sequence type based on the first target variable
        output_target = target_variables[0]
        output_seq_type = None
        
        # Find which sequence type contains the first target variable
        for seq_type, var_dict in sequence_dims.items():
            if output_target in var_dict:
                output_seq_type = seq_type
                break
        
        if output_seq_type is None:
            raise ValueError(
                f"Target variable '{output_target}' is not assigned to any sequence type. "
                f"Available sequence types: {list(sequence_dims.keys())}"
            )
        
        # Filter out empty sequence types
        sequence_dims = {
            seq_type: var_dict for seq_type, var_dict in sequence_dims.items() 
            if var_dict  # Only keep non-empty sequence types
        }
        
        if not sequence_dims:
            raise ValueError("No valid sequence types with variables defined in configuration")
            
        if output_seq_type not in sequence_dims:
            raise ValueError(f"Output sequence type '{output_seq_type}' has no variables assigned")
        
        d_model = config.get("d_model", 256)
        nhead = config.get("nhead", 8)
        
        # Adjust nhead to divide d_model if necessary.
        if d_model % nhead != 0:
            for div in range(nhead, 0, -1):
                if d_model % div == 0:
                    nhead = div
                    logger.info(f"Adjusted nhead to {nhead} to divide d_model={d_model}")
                    break
        
        model = MultiEncoderTransformer(
            global_variables=global_variables,
            sequence_dims=sequence_dims,
            output_dim=len(target_variables),
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=config.get("num_encoder_layers", 6),
            dim_feedforward=config.get("dim_feedforward", 1024),
            dropout=config.get("dropout", 0.1),
            output_seq_type=output_seq_type,
            norm_first=config.get("norm_first", False),
            max_sequence_length=config.get("max_sequence_length", 512)
        )
        
        model.input_vars = input_variables
        model.target_vars = target_variables
        
        if device is not None:
            model = model.to(device)
        
        return model
    
    except Exception as e:
        logger.error(f"Error creating prediction model: {e}")
        raise