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
import torch.nn.functional as F # F is not used, consider removing if cleaning up further

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Positional Encoding (Using OLDER version - from second upload)
# ------------------------------------------------------------------------------
class SinePositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional encoding information into the sequence. (OLDER VERSION)

    Uses fixed sine and cosine functions based on position and feature index.
    The maximum sequence length is determined by the `max_len` parameter.
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even for sine positional encoding")
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Use register_buffer with name 'pe_table' and persistent=False as in the older version
        self.register_buffer("pe_table", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor."""
        seq_len = x.size(1)
        if seq_len > self.pe_table.size(1):
             raise ValueError(
                 f"Sequence length {seq_len} exceeds SinePositionalEncoding max_len {self.pe_table.size(1)}"
             )
        # Use pe_table and match dtype/device as in the older version
        return x + self.pe_table[:, :seq_len].to(dtype=x.dtype, device=x.device)

# ------------------------------------------------------------------------------
# Sequence Encoder (Using CURRENT version structure - without ConvBlock1D)
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
        # Use the OLDER SinePositionalEncoding defined above
        self.pos_encoder = SinePositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation="gelu" # Match activation from old version's encoder layer
        )
        # Use conditional norm based on norm_first, similar to old version
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if norm_first else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the sequence encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, input_dim].

        Returns:
            torch.Tensor: Encoded sequence tensor of shape [batch_size, seq_length, d_model].
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x) # Add positional encoding
        return self.encoder(x)

# ------------------------------------------------------------------------------
# Global Features Encoder (Using CURRENT version)
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
# Feature Integration with FiLM (Using OLDER version)
# ------------------------------------------------------------------------------
class FiLMLayer(nn.Module):
    """
    Applies Feature-wise Linear Modulation (FiLM) to condition a sequence. (OLDER VERSION)

    Uses a conditioning vector (typically from global features) to generate
    per-feature scaling (gamma) and shifting (beta) parameters, which are
    then applied element-wise to the input sequence.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Use 'fc' name from old version
        self.fc = nn.Linear(d_model, 2 * d_model)

    # Use 'seq' and 'cond' argument names from old version
    def forward(self, seq: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Applies FiLM conditioning."""
        gamma, beta = self.fc(cond).chunk(2, dim=-1)
        # Unsqueeze and apply as in old version
        return gamma.unsqueeze(1) * seq + beta.unsqueeze(1)

# ------------------------------------------------------------------------------
# Cross-Attention Module (Using CURRENT version)
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
# Multi-Encoder Transformer (Using CURRENT version's structure)
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
        max_sequence_length (int): Maximum supported sequence length (Used by Positional Encoding).
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

        # Keep current sequence_dims filtering/validation logic
        self.sequence_dims = {
            seq_type: var_dict for seq_type, var_dict in sequence_dims.items()
            if var_dict
        }
        self.sequence_types = list(self.sequence_dims.keys())
        self.global_variables = global_variables

        if len(self.sequence_types) > 2:
            raise ValueError(f"Only up to 2 sequence types are supported, but received {len(self.sequence_types)}: {self.sequence_types}")

        self.has_global = len(global_variables) > 0

        if output_seq_type is None:
            raise ValueError("output_seq_type must be specified to match target length")
        if output_seq_type not in self.sequence_dims:
            raise ValueError(f"output_seq_type '{output_seq_type}' not found in non-empty sequence_types: {list(self.sequence_dims.keys())}")
        self.output_seq_type = output_seq_type

        # Instantiate SequenceEncoder (uses OLDER PE)
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
            # Instantiate current GlobalEncoder
            self.global_encoder = GlobalEncoder(input_dim=len(global_variables), d_model=d_model, dropout=dropout)
            # Instantiate OLDER FiLMLayer
            self.film_layers = nn.ModuleDict({
                seq_type: FiLMLayer(d_model) for seq_type in self.sequence_types
            })
        else:
             self.global_encoder = None
             self.film_layers = None

        # Cross-attention between sequence types.
        self.use_cross_attention = len(self.sequence_types) == 2
        if self.use_cross_attention:
            # Instantiate current CrossAttentionModule
            self.cross_attention = nn.ModuleDict({
                f"{self.sequence_types[0]}_to_{self.sequence_types[1]}": CrossAttentionModule(d_model=d_model, nhead=nhead, dropout=dropout),
                f"{self.sequence_types[1]}_to_{self.sequence_types[0]}": CrossAttentionModule(d_model=d_model, nhead=nhead, dropout=dropout)
            })
        else:
             self.cross_attention = None

        # Output projection layer.
        self.output_proj = nn.Linear(d_model, output_dim)

        self._init_parameters() # Use current _init_parameters

        logger.info(f"Created MultiEncoderTransformer with sequence types: {self.sequence_types}")
        if self.has_global:
            logger.info(f"Global variables: {self.global_variables}")
        logger.info(f"Using '{self.output_seq_type}' as output sequence type")
        if self.use_cross_attention:
            logger.info("Using cross-attention between sequence types")
        else:
            logger.info("Using self-attention only (no cross-attention)")

    def _init_parameters(self): # Keep CURRENT _init_parameters
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

    def _validate_inputs(self, inputs: dict) -> Dict[str, int]: # Keep CURRENT validation
        """
        Validate input dictionary and tensor shapes.
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

    def forward(self, inputs: dict, targets: Optional[torch.Tensor] = None) -> torch.Tensor: # Keep CURRENT forward pass structure
        """
        Forward pass through the transformer.
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
                encoded = encoder(inputs[seq_type]) # Uses SequenceEncoder (with OLDER PE)
                if self.has_global and global_features is not None and self.film_layers is not None and seq_type in self.film_layers:
                    # Use OLDER FiLMLayer class, applied in the current position (before cross-attn)
                    encoded = self.film_layers[seq_type](encoded, global_features)
                encoded_sequences[seq_type] = encoded

        # Apply cross-attention using current module and logic
        if self.use_cross_attention and len(encoded_sequences) == 2:
            try:
                available_seq_types = list(encoded_sequences.keys())
                if len(available_seq_types) == 2:
                    seq0, seq1 = available_seq_types
                    key01 = f"{seq0}_to_{seq1}"
                    key10 = f"{seq1}_to_{seq0}"
                    if key01 in self.cross_attention and key10 in self.cross_attention:
                        enc0_attn = self.cross_attention[key01](encoded_sequences[seq0], encoded_sequences[seq1])
                        enc1_attn = self.cross_attention[key10](encoded_sequences[seq1], encoded_sequences[seq0])
                        encoded_sequences[seq0] = enc0_attn
                        encoded_sequences[seq1] = enc1_attn
                    else:
                         logger.warning(f"Cross attention modules missing for pair {seq0}/{seq1}, skipping.")
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
# Model Creation Function (Using CURRENT version's logic)
# ------------------------------------------------------------------------------
def create_prediction_model(config: dict, device: Optional[torch.device] = None) -> nn.Module:
    """
    Create a prediction model from a configuration dictionary.
    """
    try:
        # --- Keep all validation and setup logic from the CURRENT version ---
        input_variables = config.get("input_variables", [])
        target_variables = config.get("target_variables", [])
        if not input_variables: raise ValueError("No input variables specified.")
        if not target_variables: raise ValueError("No target variables specified.")

        sequence_types_config = config.get("sequence_types", {})
        if not sequence_types_config: raise ValueError("No sequence types specified.")

        global_variables = config.get("global_variables", [])

        all_assigned_vars = set(global_variables)
        for var_list in sequence_types_config.values(): all_assigned_vars.update(var_list)
        if unassigned_inputs := set(input_variables) - all_assigned_vars:
            raise ValueError(f"Input variables not assigned: {unassigned_inputs}")

        # Check if first target is in any sequence type.
        first_target = target_variables[0]
        target_found = False
        for var_list in sequence_types_config.values():
             if first_target in var_list:
                  target_found = True
                  break
        if not target_found:
             raise ValueError(f"First target variable '{first_target}' not assigned to any sequence type.")

        sequence_dims = {st: {var: i for i, var in enumerate(vl)} for st, vl in sequence_types_config.items() if vl}
        if not sequence_dims: raise ValueError("No valid sequence types with variables defined.")

        output_seq_type = None
        output_target = target_variables[0]
        for st, var_dict in sequence_dims.items():
            if output_target in var_dict:
                output_seq_type = st
                break
        if output_seq_type is None:
            raise ValueError(f"Target variable '{output_target}' not assigned to any sequence type.") # Should be caught above
        if output_seq_type not in sequence_dims:
            raise ValueError(f"Output sequence type '{output_seq_type}' has no variables assigned.") # Should be caught above


        d_model = config.get("d_model", 256)
        nhead = config.get("nhead", 8)

        # Adjust nhead if necessary (current logic)
        if d_model % nhead != 0:
            original_nhead = nhead
            adjusted = False
            for div in range(nhead -1, 0, -1):
                 if d_model % div == 0:
                      nhead = div
                      logger.warning(f"Adjusted nhead from {original_nhead} to {nhead} to divide d_model={d_model}")
                      config['nhead'] = nhead # Update config
                      adjusted = True
                      break
            if not adjusted:
                 raise ValueError(f"d_model ({d_model}) is not divisible by any nhead value <= {original_nhead}")

        # Instantiate the modified MultiEncoderTransformer
        model = MultiEncoderTransformer(
            global_variables=global_variables,
            sequence_dims=sequence_dims,
            output_dim=len(target_variables),
            d_model=d_model,
            nhead=nhead, # Use potentially adjusted nhead
            num_encoder_layers=config.get("num_encoder_layers", 6),
            dim_feedforward=config.get("dim_feedforward", 1024),
            dropout=config.get("dropout", 0.1),
            output_seq_type=output_seq_type,
            norm_first=config.get("norm_first", False),
            max_sequence_length=config.get("max_sequence_length", 512)
        )

        model.input_vars = input_variables
        model.target_vars = target_variables
        model.global_vars = global_variables

        if device is not None:
            model = model.to(device)

        return model

    except Exception as e:
        logger.error(f"Error creating prediction model: {e}")
        raise

__all__ = ["create_prediction_model", "MultiEncoderTransformer"]