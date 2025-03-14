#!/usr/bin/env python3
"""
model.py - Simplified transformer for atmospheric data prediction that combines
global and sequence features in a streamlined architecture.
"""

import math
import logging
from typing import Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# RMSNorm Implementation
# ------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        rms = norm / (x.size(-1) ** 0.5)
        return self.weight * (x / (rms + self.eps))

# ------------------------------------------------------------------------------
# Positional Encodings
# ------------------------------------------------------------------------------
class SinePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)
        
    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]

# ------------------------------------------------------------------------------
# Attention Module
# ------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)

# ------------------------------------------------------------------------------
# Feed-Forward Network
# ------------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation in ["silu", "swish"]:
            act_fn = nn.SiLU()
        else:
            act_fn = nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
    
    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------
# Encoder Layer
# ------------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first
    
    def forward(self, src, src_mask=None):
        if self.norm_first:
            src2 = self.norm1(src)
            src2 = self.self_attn(src2, src_mask)
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.feed_forward(src2)
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, src_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.feed_forward(src)
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src

# ------------------------------------------------------------------------------
# Output Head
# ------------------------------------------------------------------------------
class OutputHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_features: int,
        hidden_dim: Optional[int] = None,
        mlp_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation in ["silu", "swish"]:
            act_fn = nn.SiLU()
        else:
            act_fn = nn.GELU()
        
        if mlp_layers <= 1:
            self.net = nn.Linear(d_model, out_features)
        else:
            layers = []
            in_dim = d_model
            for _ in range(mlp_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(act_fn)
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_features))
            self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------
# Simplified Transformer Model
# ------------------------------------------------------------------------------
class MultiSourceTransformer(nn.Module):
    def __init__(
        self,
        global_dim: int,
        sequence_dims=None,  # For backward compatibility
        sequence_dim=None,   # New parameter
        output_dim: int = None,
        d_model: int = 256,
        nhead: int = 8,
        encoder_layers=None,  # For backward compatibility
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        mlp_layers: int = 3,
        mlp_hidden_dim: Optional[int] = None,
        max_seq_length: int = 512,
        output_proj: bool = True,
        batch_first: bool = True,
        layer_scale: float = 0.1,  # Kept for backward compatibility but not used
        positional_encoding: str = "sine"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.output_dim = output_dim
        self.batch_first = batch_first
        self.positional_encoding = positional_encoding.lower()
        
        # Handle backward compatibility for sequence_dims
        if sequence_dim is not None:
            self.sequence_dim = sequence_dim
        elif sequence_dims is not None:
            if len(sequence_dims) > 1:
                logger.warning(f"Multiple sequence types found: {list(sequence_dims.keys())}. "
                            f"Using only the first type. Other types will be ignored.")
            
            seq_type = list(sequence_dims.keys())[0]
            self.sequence_dim = sequence_dims[seq_type]
            self.seq_type = seq_type
        else:
            raise ValueError("Either sequence_dim or sequence_dims must be provided")
        
        # Handle backward compatibility for encoder_layers
        if encoder_layers is not None:
            if isinstance(encoder_layers, dict):
                if len(encoder_layers) > 0:
                    seq_type = list(encoder_layers.keys())[0]
                    self.num_encoder_layers = encoder_layers[seq_type]
                else:
                    self.num_encoder_layers = num_encoder_layers
            else:
                self.num_encoder_layers = encoder_layers
        else:
            self.num_encoder_layers = num_encoder_layers
        
        logger.info(f"Using simplified transformer architecture")
        logger.info(f"Global features: {global_dim} dimension(s)")
        logger.info(f"Sequence features: {self.sequence_dim} dimension(s)")
        
        # Validate that the sequence has at least 1 feature
        if self.sequence_dim < 1:
            error_msg = f"Sequence must have at least 1 feature, but has {self.sequence_dim}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Store global feature flag
        self.has_global = global_dim > 0
        
        # Compute input dimensions for projection
        self.global_dim = global_dim
        
        # Input projection - will handle both sequence and global features
        if self.has_global:
            # For sequences with global features, we'll expand global features 
            # as a separate input and concatenate with sequence data
            self.global_proj = nn.Linear(global_dim, d_model)
        
        # Sequence projection
        self.seq_proj = nn.Linear(self.sequence_dim, d_model)
        
        # Positional encoding
        if self.positional_encoding == "sine":
            self.pos_encoder = SinePositionalEncoding(d_model, max_seq_length)
        elif self.positional_encoding == "learned":
            self.pos_encoder = LearnedPositionalEncoding(d_model, max_seq_length)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first
            )
            for _ in range(self.num_encoder_layers)
        ])
        
        # Final norm layer
        self.norm = RMSNorm(d_model) if norm_first else None
        
        # Output head
        self.output_head = OutputHead(
            d_model=d_model,
            out_features=output_dim,
            hidden_dim=mlp_hidden_dim,
            mlp_layers=mlp_layers if output_proj else 1,
            dropout=dropout,
            activation=activation
        )
        
        self.initialize_parameters()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model created with {trainable_params:,} trainable parameters")
    
    def initialize_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
        
        # Special initialization for output layer
        for m in self.output_head.modules():
            if isinstance(m, nn.Linear) and hasattr(m, 'out_features') and m.out_features == self.output_dim:
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _validate_input(self, inputs):
        """Validate inputs and convert to tensor format."""
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary with sequence type and optional 'global' keys")
        
        result = {}
        
        # For backward compatibility, find the correct sequence key
        if hasattr(self, 'seq_type') and self.seq_type in inputs:
            seq_key = self.seq_type
        else:
            sequence_keys = [k for k in inputs.keys() if k != "global"]
            if not sequence_keys:
                error_msg = "Input must contain at least one sequence type key"
                logger.error(error_msg)
                raise ValueError(error_msg)
            seq_key = sequence_keys[0]
        
        sequence = inputs[seq_key]
        if not isinstance(sequence, torch.Tensor) or sequence.numel() == 0:
            error_msg = f"Sequence input '{seq_key}' must be a non-empty tensor"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate sequence dimensions
        if sequence.dim() == 3:  # [batch, seq_len, features]
            if sequence.shape[2] != self.sequence_dim:
                error_msg = f"Expected sequence with {self.sequence_dim} features, but got {sequence.shape[2]}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = f"Expected 3D tensor for sequence input, got {sequence.dim()}D"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Clean data (replace NaN/Inf values)
        sequence = torch.nan_to_num(sequence, nan=0.0, posinf=1e6, neginf=-1e6)
        result[seq_key] = sequence
        
        # Handle global input if present
        if self.has_global:
            if 'global' not in inputs:
                error_msg = "Global features expected but not provided in inputs"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            global_input = inputs['global']
            if not isinstance(global_input, torch.Tensor) or global_input.numel() == 0:
                error_msg = "Global input must be a non-empty tensor"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            global_input = torch.nan_to_num(global_input, nan=0.0, posinf=1e6, neginf=-1e6)
            result['global'] = global_input
        
        return result, seq_key

    def forward(self, inputs):
        """Forward pass for the transformer model."""
        # Validate inputs
        inputs, seq_key = self._validate_input(inputs)
        
        batch_size, seq_len, _ = inputs[seq_key].shape
        
        # Apply sequence projection 
        sequence_features = self.seq_proj(inputs[seq_key])
        
        # Incorporate global features if present
        if self.has_global and 'global' in inputs:
            # Project global features
            global_features = self.global_proj(inputs['global'])
            
            # Prepare global features for addition (expand to match sequence length)
            # This is a simpler alternative to feature integration
            global_expanded = global_features.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Combine with sequence features by simple addition
            sequence_features = sequence_features + global_expanded
        
        # Apply positional encoding
        if hasattr(self, 'pos_encoder'):
            sequence_features = self.pos_encoder(sequence_features)
        
        # Process through encoder layers
        for layer in self.encoder_layers:
            sequence_features = layer(sequence_features)
            
        # Apply final normalization if needed
        if self.norm is not None:
            sequence_features = self.norm(sequence_features)
        
        # Generate output through the output head
        output = self.output_head(sequence_features)
        
        return output
    
    def generate(self, inputs, **kwargs):
        """Alias for forward method for compatibility."""
        return self.forward(inputs)