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
from typing import Dict, Optional, List, Set, Tuple
import sys # Added for sys.exit

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------

class SinePositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding module.

    Generates fixed sinusoidal positional encoding and adds it to the input.
    Expected input shape: [batch_size, seq_length, d_model]
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor."""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length ({seq_len}) > max_len ({self.pe.size(1)})"
            )
        return x + self.pe[:, :seq_len, :]


class SequenceEncoder(nn.Module):
    """
    Bidirectional encoder for sequence data.

    Projects inputs, adds positional encodings, processes via TransformerEncoder.
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
        norm_first: bool = False,
        positional_encoding_type: Optional[str] = "sinusoidal",
        max_len: int = 512
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        # Determine the type of positional encoding
        if positional_encoding_type is None:
            self.pos_encoder = None
            logger.warning("No positional encoding will be used (type was None).")
        elif isinstance(positional_encoding_type, str):
            pe_type_lower = positional_encoding_type.lower()
            if pe_type_lower == 'none':
                self.pos_encoder = None
                logger.warning("No positional encoding will be used (type was 'none').")
            elif pe_type_lower in ["sinusoidal", "sine", "sin"]:
                self.pos_encoder = SinePositionalEncoding(d_model, max_len=max_len)
                logger.info(f"Using sinusoidal positional encoding (type was '{positional_encoding_type}').")
            else:
                logger.error(f"Unsupported positional_encoding_type string: '{positional_encoding_type}'. Exiting.")
                sys.exit(1)
        else:
            logger.error(f"Invalid positional_encoding_type: {positional_encoding_type} (type: {type(positional_encoding_type)}). Exiting.")
            sys.exit(1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=norm_first,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sequence encoder."""
        x = self.input_proj(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        return self.encoder(x)


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
        """Forward pass through the global encoder."""
        return self.encoder(x)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Conditions a sequence with global features via learned scale and shift.
    Args:
        d_model: The number of features in the sequence.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.film_proj = nn.Linear(d_model, d_model * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Applies FiLM conditioning."""
        film_params = self.film_proj(condition)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.unsqueeze(1).expand(-1, x.size(1), -1)
        beta = beta.unsqueeze(1).expand(-1, x.size(1), -1)
        return gamma * x + beta


class CrossAttentionModule(nn.Module):
    """
    Cross-attention module allowing one sequence to attend to another.

    Args:
        d_model: Dimensionality of input/output features.
        nhead: Number of attention heads.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention and Add & Norm."""
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_output))


# ------------------------------------------------------------------------------
# Main Transformer Model
# ------------------------------------------------------------------------------

class MultiEncoderTransformer(nn.Module):
    """
    Encoder-only transformer for multi-sequence data with cross-attention.

    Processes up to 2 sequence types, integrates global features via FiLM,
    and applies cross-attention if both sequence types are provided. Output
    sequence length determined by `output_seq_type`.

    Args:
        global_variables: List of global variable names.
        sequence_dims: Maps sequence type name to {variable_name: index}.
        output_dim: Number of output features per timestep.
        d_model: Dimension of model embeddings.
        nhead: Number of attention heads.
        num_encoder_layers: Number of layers in each TransformerEncoder.
        dim_feedforward: Dimension of the feedforward network.
        dropout: Dropout rate.
        output_seq_type: Key from sequence_dims defining output length.
        norm_first: If True, apply layer norm before other ops.
        max_sequence_length: Maximum supported sequence length.
        positional_encoding_type: Type of positional encoding for sequences.
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
        max_sequence_length: int = 512,
        positional_encoding_type: Optional[str] = "sinusoidal"
    ):
        super().__init__()
        self.d_model = d_model

        self.sequence_dims = {
            k: v for k, v in sequence_dims.items() if isinstance(v, dict) and v
        }
        self.sequence_types = list(self.sequence_dims.keys())
        self.global_variables = global_variables if global_variables is not None else []

        if len(self.sequence_types) > 2:
            raise ValueError(
                f"Supports up to 2 sequence types, got {len(self.sequence_types)}"
            )
        if not self.sequence_types:
            raise ValueError("At least one non-empty sequence type is required.")

        self.has_global = bool(self.global_variables)

        if output_seq_type is None:
            if len(self.sequence_types) == 1:
                output_seq_type = self.sequence_types[0]
            else:
                raise ValueError(
                    "output_seq_type must be specified for multiple sequence types."
                )
        elif output_seq_type not in self.sequence_dims:
            raise ValueError(
                f"output_seq_type '{output_seq_type}' not in sequence_dims."
            )
        self.output_seq_type = output_seq_type

        self.sequence_encoders = nn.ModuleDict()
        for seq_type, var_dict in self.sequence_dims.items():
            self.sequence_encoders[seq_type] = SequenceEncoder(
                input_dim=len(var_dict), d_model=d_model, nhead=nhead,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward, dropout=dropout,
                norm_first=norm_first,
                positional_encoding_type=positional_encoding_type,
                max_len=max_sequence_length
            )

        self.global_encoder = None
        self.film_layers = None
        if self.has_global:
            self.global_encoder = GlobalEncoder(
                input_dim=len(self.global_variables), d_model=d_model,
                dropout=dropout
            )
            self.film_layers = nn.ModuleDict({
                st: FiLMLayer(d_model) for st in self.sequence_types
            })

        self.cross_attention = None
        self.use_cross_attention = len(self.sequence_types) == 2
        if self.use_cross_attention:
            st0, st1 = self.sequence_types
            if st0 in self.sequence_encoders and st1 in self.sequence_encoders:
                self.cross_attention = nn.ModuleDict({
                    f"{st0}_to_{st1}": CrossAttentionModule(d_model, nhead, dropout),
                    f"{st1}_to_{st0}": CrossAttentionModule(d_model, nhead, dropout)
                })
            else:
                self.use_cross_attention = False

        self.output_proj = nn.Linear(d_model, output_dim)
        self._init_parameters()
        logger.info("MultiEncoderTransformer initialized.")

    def _init_parameters(self):
        """Initializes model parameters using Xavier uniform and zeros."""
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
        """Validates the structure and shapes of the input dictionary."""
        if not isinstance(inputs, dict):
            raise TypeError("Inputs must be a dictionary.")

        sequence_lengths = {}
        batch_size = -1

        for seq_type in self.sequence_types:
            if seq_type not in self.sequence_encoders: continue
            if seq_type not in inputs:
                raise ValueError(f"Missing input sequence: '{seq_type}'")

            seq_tensor = inputs[seq_type]
            if not isinstance(seq_tensor, torch.Tensor) or seq_tensor.ndim != 3:
                raise ValueError(f"Input '{seq_type}' must be 3D Tensor.")

            current_bs = seq_tensor.size(0)
            if batch_size == -1: batch_size = current_bs
            elif batch_size != current_bs:
                raise ValueError("Inconsistent batch sizes.")

            expected_features = len(self.sequence_dims[seq_type])
            if seq_tensor.size(2) != expected_features:
                raise ValueError(
                    f"Feature mismatch for '{seq_type}'. "
                    f"Expected {expected_features}, got {seq_tensor.size(2)}."
                )
            sequence_lengths[seq_type] = seq_tensor.size(1)

        if self.has_global:
            if 'global' not in inputs:
                raise ValueError("Missing 'global' input features.")
            global_tensor = inputs['global']
            if not isinstance(global_tensor, torch.Tensor) or global_tensor.ndim != 2:
                raise ValueError("'global' input must be 2D Tensor.")
            if batch_size != -1 and global_tensor.size(0) != batch_size:
                raise ValueError("Batch size mismatch: sequences vs global.")
            expected_global_features = len(self.global_variables)
            if global_tensor.size(1) != expected_global_features:
                raise ValueError(
                    f"Global feature mismatch. Expected {expected_global_features}, "
                    f"got {global_tensor.size(1)}."
                )

        if self.output_seq_type not in sequence_lengths:
            raise ValueError(f"Output sequence '{self.output_seq_type}' missing.")

        return sequence_lengths

    def forward(
        self, inputs: dict, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            inputs: Contains input sequences keyed by type name, and
                    optionally 'global' features.
            targets: Optional target tensor for validation checks.

        Returns:
            Output tensor shape [batch_size, seq_length, output_dim].
        """
        sequence_lengths = self._validate_inputs(inputs)
        output_seq_length = sequence_lengths[self.output_seq_type]

        if targets is not None and targets.size(1) != output_seq_length:
            # Keep essential target length check
            raise ValueError(
                f"Target length ({targets.size(1)}) != output length ({output_seq_length})."
            )

        global_features_encoded = None
        if self.has_global and self.global_encoder is not None:
            global_features_encoded = self.global_encoder(inputs['global'])

        encoded_sequences = {}
        for seq_type, encoder in self.sequence_encoders.items():
            if seq_type in inputs and inputs[seq_type].numel() > 0:
                encoded = encoder(inputs[seq_type])
                if self.has_global and global_features_encoded is not None and self.film_layers is not None and seq_type in self.film_layers:
                    encoded = self.film_layers[seq_type](
                        encoded, global_features_encoded
                    )
                encoded_sequences[seq_type] = encoded

        if not encoded_sequences:
            raise RuntimeError("No sequences were successfully encoded.")

        if self.use_cross_attention and self.cross_attention is not None:
            st0, st1 = self.sequence_types
            if st0 in encoded_sequences and st1 in encoded_sequences:
                enc0, enc1 = encoded_sequences[st0], encoded_sequences[st1]
                # Removed try/except around cross-attention for conciseness
                attn0 = self.cross_attention[f"{st0}_to_{st1}"](enc0, enc1)
                attn1 = self.cross_attention[f"{st1}_to_{st0}"](enc1, enc0)
                encoded_sequences[st0], encoded_sequences[st1] = attn0, attn1

        if self.output_seq_type not in encoded_sequences:
            raise RuntimeError(
                f"Output sequence type '{self.output_seq_type}' not found in "
                f"successfully encoded sequences."
            )

        output_sequence = encoded_sequences[self.output_seq_type]
        output = self.output_proj(output_sequence)
        return output


# ------------------------------------------------------------------------------
# Model Creation Function
# ------------------------------------------------------------------------------

def create_prediction_model(
    config: dict, device: Optional[torch.device] = None
) -> MultiEncoderTransformer:
    """
    Create a MultiEncoderTransformer model from a configuration dictionary.

    Matches the architecture of the original model_old.py.

    Args:
        config: Configuration dictionary holding hyperparameters and
                variable definitions.
        device: Device to move the model to.

    Returns:
        The instantiated MultiEncoderTransformer model.
    """
    # Removed outer try/except for conciseness
    input_variables = config.get("input_variables")
    target_variables = config.get("target_variables")
    sequence_types_config = config.get("sequence_types")
    global_variables = config.get("global_variables", [])
    positional_encoding_type = config.get("positional_encoding_type", "sinusoidal")


    if not input_variables: raise ValueError("'input_variables' missing/empty.")
    if not target_variables: raise ValueError("'target_variables' missing/empty.")
    if not sequence_types_config: raise ValueError("'sequence_types' missing/empty.")
    if not isinstance(global_variables, list): raise TypeError("'global_variables' must be list.")

    all_assigned_vars: Set[str] = set(global_variables)
    sequence_dims: Dict[str, Dict[str, int]] = {}

    for seq_type, var_list in sequence_types_config.items():
        if not isinstance(var_list, list):
            raise TypeError(f"Var list for '{seq_type}' must be a list.")
        if not var_list: continue # Skip empty

        current_vars = set(var_list)
        if len(current_vars) != len(var_list):
            raise ValueError(f"Duplicates in var list for '{seq_type}'.")
        if overlap := current_vars.intersection(all_assigned_vars):
            raise ValueError(f"Variable assignment overlap for '{seq_type}': {overlap}")

        all_assigned_vars.update(current_vars)
        sequence_dims[seq_type] = {var: i for i, var in enumerate(var_list)}

    if unassigned := set(input_variables) - all_assigned_vars:
        raise ValueError(f"Input variables not assigned: {unassigned}")
    if unassigned_targets := set(target_variables) - all_assigned_vars:
        raise ValueError(f"Target variables not assigned: {unassigned_targets}")
    if not sequence_dims:
        raise ValueError("No valid sequence types defined.")

    first_target = target_variables[0]
    output_seq_type = None
    for seq_type, var_dict in sequence_dims.items():
        if first_target in var_dict:
            output_seq_type = seq_type
            break
    if output_seq_type is None:
        raise ValueError(f"Target '{first_target}' not in any sequence type.")

    d_model = config.get("d_model", 256)
    nhead = config.get("nhead", 8)

    if not isinstance(d_model, int) or d_model <= 0: raise ValueError("d_model > 0 required.")
    if not isinstance(nhead, int) or nhead <= 0: raise ValueError("nhead > 0 required.")

    if d_model % nhead != 0:
        original_nhead = nhead
        adjusted = False
        for potential_nhead in range(nhead - 1, 0, -1):
            if d_model % potential_nhead == 0:
                nhead = potential_nhead
                adjusted = True
                break
        if not adjusted:
            raise ValueError(f"d_model ({d_model}) not divisible by nhead ({original_nhead}).")

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
        max_sequence_length=config.get("max_sequence_length", 512),
        positional_encoding_type=positional_encoding_type
    )

    model.input_vars = list(input_variables)
    model.target_vars = list(target_variables)

    logger.info("Model created successfully.")

    if device is not None:
        if isinstance(device, str): device = torch.device(device)
        if not isinstance(device, torch.device):
            raise TypeError(f"device must be torch.device or str, got {type(device)}")
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")

    return model


__all__ = ["MultiEncoderTransformer", "create_prediction_model"]