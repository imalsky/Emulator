#!/usr/bin/env python3
"""
model.py - Bidirectional encoder-only transformer for atmospheric data prediction

Features:
- Supports up to 2 sequence types for input.
- Global feature integration with FiLM conditioning (applied after every encoder layer if global features are provided).
- Cross-attention for inter-sequence information exchange (when multiple sequence types exist).
- Clean, robust error handling and clear documentation.

The model is designed for regression tasks on 1D atmospheric profiles. The output sequence length is
determined by the target variable specification.
"""

import math
import logging
from typing import Dict, Optional, List

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
        return x + self.pe[:, :x.size(1), :]

# ------------------------------------------------------------------------------
# Feature Integration with FiLM
# ------------------------------------------------------------------------------
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Conditions a sequence of features with a global conditioning vector.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.film_proj = nn.Linear(d_model, d_model * 2)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        film_params = self.film_proj(condition)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.unsqueeze(1).expand(-1, x.size(1), -1)
        beta = beta.unsqueeze(1).expand(-1, x.size(1), -1)
        return gamma * x + beta

# ------------------------------------------------------------------------------
# Sequence Encoder
# ------------------------------------------------------------------------------
class SequenceEncoder(nn.Module):
    """
    Bidirectional encoder for sequence data with optional FiLM conditioning applied after each layer.
    
    Args:
        use_film (bool): If True, apply FiLM conditioning after each encoder layer.
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
        use_film: bool = False
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = SinePositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first
            ) for _ in range(num_layers)
        ])
        if use_film:
            self.film_layers = nn.ModuleList([FiLMLayer(d_model) for _ in range(num_layers)])
        else:
            self.film_layers = None
        
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.film_layers is not None and condition is not None:
                x = self.film_layers[i](x, condition)
        return x

# ------------------------------------------------------------------------------
# Global Features Encoder
# ------------------------------------------------------------------------------
class GlobalEncoder(nn.Module):
    """
    Encodes global features using a multi-layer perceptron (MLP).
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
        return self.encoder(x)

# ------------------------------------------------------------------------------
# Cross-Attention Module
# ------------------------------------------------------------------------------
class CrossAttentionModule(nn.Module):
    """
    Cross-attention module that allows one sequence to attend to another.
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
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_output))

# ------------------------------------------------------------------------------
# Multi-Encoder Transformer
# ------------------------------------------------------------------------------
class MultiEncoderTransformer(nn.Module):
    """
    Encoder-only transformer for multi-sequence data with cross-attention.
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
        
        # Filter out empty sequence types.
        self.sequence_dims = {seq_type: var_dict for seq_type, var_dict in sequence_dims.items() if var_dict}
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
        
        # Create sequence encoders for each non-empty sequence type.
        self.sequence_encoders = nn.ModuleDict({
            seq_type: SequenceEncoder(
                input_dim=len(var_dict),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=norm_first,
                use_film=self.has_global
            ) for seq_type, var_dict in self.sequence_dims.items()
        })
        
        # Global encoder (only if global variables are present).
        if self.has_global:
            self.global_encoder = GlobalEncoder(input_dim=len(global_variables), d_model=d_model, dropout=dropout)
        
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
                raise ValueError(f"Expected {expected_features} features for {seq_type}, got {seq.size(-1)}")
            sequence_lengths[seq_type] = seq.size(1)
        
        if self.has_global and 'global' not in inputs:
            raise ValueError("Missing required global features")
            
        logger.debug(f"Input sequence lengths: {sequence_lengths}")
        return sequence_lengths
    
    def forward(self, inputs: dict, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.
        """
        sequence_lengths = self._validate_inputs(inputs)
        
        if targets is not None:
            target_length = targets.size(1)
            output_seq_length = sequence_lengths[self.output_seq_type]
            if target_length != output_seq_length:
                raise ValueError(
                    f"Target length ({target_length}) does not match output sequence length "
                    f"({output_seq_length}) from '{self.output_seq_type}'"
                )
        
        global_features = None
        if self.has_global:
            global_features = self.global_encoder(inputs['global'])
        
        encoded_sequences = {}
        for seq_type, encoder in self.sequence_encoders.items():
            if seq_type in inputs and inputs[seq_type].numel() > 0:
                if self.has_global:
                    encoded = encoder(inputs[seq_type], global_features)
                else:
                    encoded = encoder(inputs[seq_type])
                encoded_sequences[seq_type] = encoded
        
        # Apply cross-attention if two sequence types are present.
        if self.use_cross_attention and len(encoded_sequences) == 2:
            available_seq_types = list(encoded_sequences.keys())
            if len(available_seq_types) == 2:
                seq0, seq1 = available_seq_types
                key = f"{seq0}_to_{seq1}"
                if key in self.cross_attention:
                    encoded_sequences[seq0] = self.cross_attention[key](encoded_sequences[seq0], encoded_sequences[seq1])
                key = f"{seq1}_to_{seq0}"
                if key in self.cross_attention:
                    encoded_sequences[seq1] = self.cross_attention[key](encoded_sequences[seq1], encoded_sequences[seq0])
        
        if self.output_seq_type not in encoded_sequences:
            raise ValueError(f"Output sequence type '{self.output_seq_type}' not found in encoded sequences")
            
        output_sequence = encoded_sequences[self.output_seq_type]
        output = self.output_proj(output_sequence)
        return output

# ------------------------------------------------------------------------------
# Model Creation Function
# ------------------------------------------------------------------------------
def create_prediction_model(config: dict, device: Optional[torch.device] = None) -> nn.Module:
    """
    Create a prediction model from a configuration dictionary.
    """
    input_variables = config.get("input_variables", [])
    target_variables = config.get("target_variables", [])
    
    if not input_variables:
        raise ValueError("No input variables specified in the configuration.")
    if not target_variables:
        raise ValueError("No target variables specified in the configuration.")
    
    sequence_types_config = config.get("sequence_types", {})
    if not sequence_types_config:
        raise ValueError("No sequence types specified in the configuration.")
    
    global_variables = config.get("global_variables", [])
    
    all_assigned_vars = set()
    for seq_type, var_list in sequence_types_config.items():
        all_assigned_vars.update(var_list)
    all_assigned_vars.update(global_variables)
    
    unassigned_inputs = set(input_variables) - all_assigned_vars
    if unassigned_inputs:
        raise ValueError(f"The following input variables are not assigned to any sequence type or global: {unassigned_inputs}")
    
    unassigned_targets = set(target_variables) - all_assigned_vars
    if unassigned_targets:
        raise ValueError(f"The following target variables are not assigned to any sequence type: {unassigned_targets}")
    
    sequence_dims = {
        seq_type: {var: i for i, var in enumerate(var_list)}
        for seq_type, var_list in sequence_types_config.items() if var_list
    }
    
    output_target = target_variables[0]
    output_seq_type = None
    for seq_type, var_dict in sequence_dims.items():
        if output_target in var_dict:
            output_seq_type = seq_type
            break
    
    if output_seq_type is None:
        raise ValueError(f"Target variable '{output_target}' is not assigned to any sequence type. Available: {list(sequence_dims.keys())}")
    
    if output_seq_type not in sequence_dims:
        raise ValueError(f"Output sequence type '{output_seq_type}' has no variables assigned")
    
    d_model = config.get("d_model", 256)
    nhead = config.get("nhead", 8)
    
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
