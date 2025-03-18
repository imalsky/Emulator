#!/usr/bin/env python3
"""
model.py - Efficient transformer for atmospheric data prediction with:
- LayerNorm normalization
- Configurable positional encodings (rotary, sine, learned)
- Enhanced feature integration with gating
- Layer scaling for gradient flow
"""

import math
import logging
from typing import Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ------------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.dim_half = dim // 2
        
        positions = torch.arange(0, max_seq_len, dtype=torch.float)
        freqs = 1.0 / (10000 ** (torch.arange(0, self.dim_half, dtype=torch.float) / self.dim_half))
        angles = torch.outer(positions, freqs)
        self.register_buffer("sin", angles.sin())
        self.register_buffer("cos", angles.cos())
    
    def forward(self, q, k):
        batch_size, n_heads, seq_len, _ = q.shape
        seq_len = min(seq_len, self.max_seq_len)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads, -1, -1)
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads, -1, -1)
        q1, q2 = q[..., :self.dim_half], q[..., self.dim_half:]
        k1, k2 = k[..., :self.dim_half], k[..., self.dim_half:]
        q_out = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_out = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        return q_out, k_out

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
# Attention Modules
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
        self.rope = RotaryEmbedding(self.head_dim)
    
    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)

class StandardMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
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
# Encoder Layer (using LayerNorm and Layer Scaling)
# ------------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        layer_scale: float = 0.1,
        attention_layer = None
    ):
        super().__init__()
        self.self_attn = attention_layer if attention_layer is not None else MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.use_layer_scale = layer_scale > 0
        if self.use_layer_scale:
            self.layer_scale1 = nn.Parameter(torch.ones(1, 1, d_model) * layer_scale)
            self.layer_scale2 = nn.Parameter(torch.ones(1, 1, d_model) * layer_scale)
    
    def forward(self, src, src_mask=None):
        if self.norm_first:
            src2 = self.norm1(src)
            src2 = self.self_attn(src2, src_mask)
            src = src + (self.dropout1(self.layer_scale1 * src2) if self.use_layer_scale else self.dropout1(src2))
            src2 = self.norm2(src)
            src2 = self.feed_forward(src2)
            src = src + (self.dropout2(self.layer_scale2 * src2) if self.use_layer_scale else self.dropout2(src2))
        else:
            src2 = self.self_attn(src, src_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.feed_forward(src)
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src

# ------------------------------------------------------------------------------
# Sequence Encoder (using LayerNorm)
# ------------------------------------------------------------------------------
class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        max_seq_length: int = 512,
        layer_scale: float = 0.1,
        positional_encoding: str = "rotary"
    ):
        super().__init__()
        self.positional_encoding = positional_encoding.lower()
        self.input_proj = nn.Linear(input_dim, d_model)
        if self.positional_encoding == "sine":
            self.pos_encoder = SinePositionalEncoding(d_model, max_seq_length)
        elif self.positional_encoding == "learned":
            self.pos_encoder = LearnedPositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if self.positional_encoding == "rotary":
                attention = MultiHeadAttention(d_model, nhead, dropout)
            else:
                attention = StandardMultiHeadAttention(d_model, nhead, dropout)
            self.layers.append(
                EncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation, norm_first, layer_scale,
                    attention_layer=attention
                )
            )
        self.norm = nn.LayerNorm(d_model) if norm_first else None
    
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        if hasattr(self, 'pos_encoder'):
            x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

# ------------------------------------------------------------------------------
# Global Encoder (using LayerNorm)
# ------------------------------------------------------------------------------
class GlobalEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        # Simplified MLP with LayerNorm for global features
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation in ["silu", "swish"]:
            act_fn = nn.SiLU()
        else:
            act_fn = nn.GELU()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x):
        return self.encoder(x)

# ------------------------------------------------------------------------------
# Feature Integration
# ------------------------------------------------------------------------------
class EnhancedFeatureIntegration(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gamma_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Parameter(torch.ones(1, 1, 1) * 0.1)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.bias)
    
    def forward(self, seq_features, global_features):
        gamma = self.gamma_proj(global_features).unsqueeze(1)
        beta = self.beta_proj(global_features).unsqueeze(1)
        conditioned = seq_features * gamma + beta
        return seq_features + self.gate * (conditioned - seq_features)

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
# Multi-Source Transformer Model (Encoder-Only)
# ------------------------------------------------------------------------------
class MultiSourceTransformer(nn.Module):
    def __init__(
        self,
        global_dim: int,
        sequence_dims: Dict[str, int],
        output_dim: int = None,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: Union[int, Dict[str, int]] = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        mlp_layers: int = 3,
        mlp_hidden_dim: Optional[int] = None,
        max_seq_length: int = 512,
        output_proj: bool = True,
        batch_first: bool = True,
        layer_scale: float = 0.1,
        positional_encoding: str = "rotary"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.output_dim = output_dim
        self.batch_first = batch_first
        self.positional_encoding = positional_encoding.lower()
        
        if not sequence_dims:
            raise ValueError("sequence_dims must be provided and contain at least one sequence type")
            
        # Enforce only one sequence type
        if len(sequence_dims) > 1:
            raise ValueError(f"Only one sequence type is supported. Found {len(sequence_dims)}: {list(sequence_dims.keys())}")
            
        self.sequence_dims = sequence_dims
        
        # Handle encoder layers - either a single value for all sequences or a dict
        if isinstance(num_encoder_layers, int):
            self.num_encoder_layers = {seq_type: num_encoder_layers for seq_type in sequence_dims.keys()}
        else:
            self.num_encoder_layers = num_encoder_layers
            # Ensure all sequence types have an entry in num_encoder_layers
            for seq_type in sequence_dims.keys():
                if seq_type not in self.num_encoder_layers:
                    self.num_encoder_layers[seq_type] = 4  # Default value
                    logger.warning(f"No encoder layers specified for {seq_type}, using default of 4")
        
        logger.info(f"Using ENCODER-ONLY architecture with {positional_encoding} positional encoding")
        logger.info(f"Global features: {global_dim} dimension(s)")
        
        # Log sequence information
        for seq_type, dim in sequence_dims.items():
            logger.info(f"Sequence {seq_type}: {dim} dimension(s) with {self.num_encoder_layers[seq_type]} encoder layers")
            
            # Validate that each sequence has at least 1 feature
            if dim < 1:
                error_msg = f"Sequence {seq_type} must have at least 1 feature, but has {dim}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Global features encoder
        self.has_global = global_dim > 0
        if self.has_global:
            self.global_encoder = GlobalEncoder(global_dim, d_model, dropout, activation)
        
        # Sequence encoders - one for each sequence type
        self.sequence_encoders = nn.ModuleDict()
        for seq_type, dim in sequence_dims.items():
            self.sequence_encoders[seq_type] = SequenceEncoder(
                input_dim=dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=self.num_encoder_layers[seq_type],
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                max_seq_length=max_seq_length,
                layer_scale=layer_scale,
                positional_encoding=positional_encoding
            )
        
        # Feature integration (if global features are present)
        if self.has_global:
            self.feature_integration = nn.ModuleDict()
            for seq_type in sequence_dims.keys():
                self.feature_integration[seq_type] = EnhancedFeatureIntegration(d_model)
        
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
        
        # Initialize integration layers if present
        if self.has_global and hasattr(self, 'feature_integration'):
            for seq_type, integration in self.feature_integration.items():
                nn.init.normal_(integration.gamma_proj.weight, mean=0.0, std=0.02)
                nn.init.ones_(integration.gamma_proj.bias)
                nn.init.normal_(integration.beta_proj.weight, mean=0.0, std=0.02)
                nn.init.zeros_(integration.beta_proj.bias)
        
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
        valid_sequence_types = set(self.sequence_dims.keys())
        
        # Validate sequence inputs
        for seq_type in valid_sequence_types:
            if seq_type not in inputs:
                error_msg = f"Input missing required sequence type '{seq_type}'"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            sequence = inputs[seq_type]
            if not isinstance(sequence, torch.Tensor) or sequence.numel() == 0:
                error_msg = f"Sequence input '{seq_type}' must be a non-empty tensor"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Validate sequence dimensions
            if sequence.dim() == 3:  # [batch, seq_len, features]
                if sequence.shape[2] != self.sequence_dims[seq_type]:
                    error_msg = (f"Expected sequence '{seq_type}' with {self.sequence_dims[seq_type]} " 
                                f"features, but got {sequence.shape[2]}")
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                error_msg = f"Expected 3D tensor for sequence input '{seq_type}', got {sequence.dim()}D"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Clean data (replace NaN/Inf values)
            sequence = torch.nan_to_num(sequence, nan=0.0, posinf=1e6, neginf=-1e6)
            result[seq_type] = sequence
        
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
        
        return result

    def forward(self, inputs):
        """Forward pass for the transformer model."""
        # Validate inputs
        inputs = self._validate_input(inputs)
        
        # Process global features if present
        global_features = None
        if self.has_global and 'global' in inputs:
            global_features = self.global_encoder(inputs['global'])
        
        # Process the single sequence type and apply feature integration if needed
        # With the single sequence type constraint, we can simplify this logic
        seq_type = list(self.sequence_encoders.keys())[0]  # There's only one
        sequence_features = self.sequence_encoders[seq_type](inputs[seq_type])
        
        # Apply feature integration if global features are present
        if global_features is not None:
            sequence_features = self.feature_integration[seq_type](sequence_features, global_features)
        
        primary_output = sequence_features
        
        # Generate output through the output head
        output = self.output_head(primary_output)
        
        return output
    
    def generate(self, inputs, **kwargs):
        """Alias for forward method for compatibility."""
        return self.forward(inputs)
