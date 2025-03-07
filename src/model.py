#!/usr/bin/env python3
"""
model.py

Robust Encoder-Decoder Transformer model for atmospheric variable prediction.
Includes flexible torch.compile, autoregressive generation, and dynamic masking.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # You can adjust the logging level if needed

# ------------------------------------------------------------------------------
# Positional Encoding
# ------------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequence models."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1, encoding_type: str = "sine"):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoding_type = encoding_type.lower()
        
        if self.encoding_type == "sine":
            # Sinusoidal encoding from "Attention Is All You Need"
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0), persistent=True)
            self.trainable = False
            
        elif self.encoding_type == "learned":
            # Learned positional embeddings
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pe, std=0.02)
            self.trainable = True
            
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}. Use 'sine' or 'learned'.")

    def forward(self, x):
        """Add positional encoding to input."""
        return self.dropout(x + self.pe[:, :x.size(1)])


# ------------------------------------------------------------------------------
# Multi-Head Attention with optional Flash Attention
# ------------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Efficient multi-head attention with optional Flash Attention support."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        
        # Combined Q, K, V projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Check if Flash Attention is available
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
        
    def _reshape_for_multi_head(self, x, batch_size):
        """Reshape input for multi-head attention."""
        # x shape: [batch_size, seq_len, 3 * d_model]
        # Reshape to [3, batch_size, nhead, seq_len, head_dim]
        x = x.view(batch_size, -1, 3, self.nhead, self.head_dim)
        x = x.permute(2, 0, 3, 1, 4)
        return x[0], x[1], x[2]  # query, key, value
        
    def forward(self, x, attn_mask=None):
        """Forward pass for multi-head attention."""
        batch_size, seq_len = x.shape[:2]
        qkv = self.qkv_proj(x)
        q, k, v = self._reshape_for_multi_head(qkv, batch_size)
        
        # Use Flash Attention if available and on CUDA
        if self.use_flash and torch.cuda.is_available():
            try:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0
                )
            except Exception as e:
                logger.warning(f"Flash attention failed ({e}); falling back to standard attention")
                attn_output = self._standard_attention(q, k, v, attn_mask)
        else:
            attn_output = self._standard_attention(q, k, v, attn_mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)
    
    def _standard_attention(self, q, k, v, attn_mask):
        """Standard scaled dot-product attention."""
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)


# ------------------------------------------------------------------------------
# Cross-Attention for Decoder
# ------------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Cross-attention module for decoder to attend to encoder outputs."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        
        # Separate projections for query, key, and value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Flash Attention flag
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
    
    def _reshape_for_multi_head(self, x, batch_size):
        """Reshape input for multi-head attention."""
        return x.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
    
    def forward(self, query, key_value, attn_mask=None):
        """
        Forward pass for cross-attention.
        
        Args:
            query: Decoder queries [batch_size, tgt_len, d_model]
            key_value: Encoder outputs [batch_size, src_len, d_model]
            attn_mask: Optional mask [batch_size, nhead, tgt_len, src_len]
            
        Returns:
            Attention output [batch_size, tgt_len, d_model]
        """
        batch_size, tgt_len = query.shape[:2]
        k_len = key_value.shape[1]
        
        q = self._reshape_for_multi_head(self.q_proj(query), batch_size)
        k = self._reshape_for_multi_head(self.k_proj(key_value), batch_size)
        v = self._reshape_for_multi_head(self.v_proj(key_value), batch_size)
        
        if self.use_flash and torch.cuda.is_available():
            try:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0
                )
            except Exception as e:
                logger.warning(f"Flash attention failed ({e}); falling back to standard cross-attention")
                attn_output = self._standard_cross_attention(q, k, v, attn_mask)
        else:
            attn_output = self._standard_cross_attention(q, k, v, attn_mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        return self.out_proj(attn_output)
    
    def _standard_cross_attention(self, q, k, v, attn_mask):
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)


# ------------------------------------------------------------------------------
# Feed-Forward Network and Drop Path
# ------------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Feed-forward network with residual connection."""
    
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation in ["silu", "swish"]:
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class DropPath(nn.Module):
    """Drops paths (residual connections) per sample with probability drop_prob."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ------------------------------------------------------------------------------
# Transformer Encoder and Decoder Layers
# ------------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm or post-norm architecture."""
    
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
        dropout: float = 0.1, activation: str = "gelu", norm_first: bool = True,
        drop_path_rate: float = 0.0, layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.norm_first = norm_first
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
    def forward(self, src, src_mask=None):
        if self.norm_first:
            src2 = self.norm1(src)
            src2 = self.self_attn(src2, src_mask)
            src = src + self.drop_path(self.dropout1(src2))
            
            src2 = self.norm2(src)
            src2 = self.ff(src2)
            src = src + self.drop_path(self.dropout2(src2))
        else:
            src2 = self.self_attn(src, src_mask)
            src = src + self.drop_path(self.dropout1(src2))
            src = self.norm1(src)
            
            src2 = self.ff(src)
            src = src + self.drop_path(self.dropout2(src2))
            src = self.norm2(src)
            
        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with pre-norm or post-norm architecture."""
    
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 2048,
        dropout: float = 0.1, activation: str = "gelu", norm_first: bool = True,
        drop_path_rate: float = 0.0, layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.norm_first = norm_first
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        
        self.cross_attn = CrossAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ff = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout3 = nn.Dropout(dropout)
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        if self.norm_first:
            tgt2 = self.norm1(tgt)
            tgt2 = self.self_attn(tgt2, tgt_mask)
            tgt = tgt + self.drop_path(self.dropout1(tgt2))
            
            tgt2 = self.norm2(tgt)
            tgt2 = self.cross_attn(tgt2, memory, memory_mask)
            tgt = tgt + self.drop_path(self.dropout2(tgt2))
            
            tgt2 = self.norm3(tgt)
            tgt2 = self.ff(tgt2)
            tgt = tgt + self.drop_path(self.dropout3(tgt2))
        else:
            tgt2 = self.self_attn(tgt, tgt_mask)
            tgt = tgt + self.drop_path(self.dropout1(tgt2))
            tgt = self.norm1(tgt)
            
            tgt2 = self.cross_attn(tgt, memory, memory_mask)
            tgt = tgt + self.drop_path(self.dropout2(tgt2))
            tgt = self.norm2(tgt)
            
            tgt2 = self.ff(tgt)
            tgt = tgt + self.drop_path(self.dropout3(tgt2))
            tgt = self.norm3(tgt)
            
        return tgt


# ------------------------------------------------------------------------------
# Transformer Encoder and Decoder Stacks
# ------------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(
        self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 2048,
        dropout: float = 0.1, activation: str = "gelu", norm_first: bool = True,
        layer_norm_eps: float = 1e-5, stochastic_depth_rate: float = 0.0
    ):
        super().__init__()
        
        drop_path_rates = torch.linspace(0, stochastic_depth_rate, num_layers).tolist() if stochastic_depth_rate > 0 else [0.0] * num_layers
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                layer_norm_eps=layer_norm_eps,
                drop_path_rate=drop_path_rates[i]
            )
            for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if norm_first else None
    
    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers."""
    
    def __init__(
        self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 2048,
        dropout: float = 0.1, activation: str = "gelu", norm_first: bool = True,
        layer_norm_eps: float = 1e-5, stochastic_depth_rate: float = 0.0
    ):
        super().__init__()
        
        drop_path_rates = torch.linspace(0, stochastic_depth_rate, num_layers).tolist() if stochastic_depth_rate > 0 else [0.0] * num_layers
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                layer_norm_eps=layer_norm_eps,
                drop_path_rate=drop_path_rates[i]
            )
            for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if norm_first else None
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


# ------------------------------------------------------------------------------
# Output Head
# ------------------------------------------------------------------------------

class OutputHead(nn.Module):
    """Output projection with optional MLP."""
    
    def __init__(self, d_model: int, out_features: int, mlp_layers: int = 2, 
                 hidden_dim: Optional[int] = None, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = d_model
            
        if mlp_layers <= 1:
            self.mlp = nn.Linear(d_model, out_features)
        else:
            layers = []
            in_dim = d_model
            for _ in range(mlp_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation in ["silu", "swish"]:
                    layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_features))
            self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


# ------------------------------------------------------------------------------
# AtmosphericModel: Encoder-Decoder with Global-Local Feature Integration
# ------------------------------------------------------------------------------

class AtmosphericModel(nn.Module):
    """
    Transformer encoder-decoder model for atmospheric variable prediction.
    
    Sequential features are processed via a projection and positional encoding.
    Global features (if provided) are integrated using FiLM, concatenation, or addition.
    """
    
    def __init__(
        self, nx_seq: int, nx_global: int, ny: int,
        seq_indices: List[int], global_indices: List[int],
        d_model: int = 256, nhead: int = 8, 
        num_encoder_layers: int = 6, num_decoder_layers: int = 6,
        dim_feedforward: int = 1024, dropout: float = 0.1, activation: str = "gelu",
        norm_first: bool = True, max_seq_length: int = 512, pos_encoding_type: str = "sine",
        mlp_layers: int = 3, mlp_hidden_dim: Optional[int] = None,
        stochastic_depth_rate: float = 0.0, layer_norm_eps: float = 1e-5,
        integration_method: str = "film", batch_first: bool = True,
        default_input_seq_length: Optional[int] = None,
        default_output_seq_length: Optional[int] = None,
        use_torch_compile: bool = False
    ):
        super().__init__()
        
        self.nx_seq = nx_seq
        self.nx_global = nx_global
        self.ny = ny
        self.d_model = d_model
        self.integration_method = integration_method.lower()
        self.batch_first = batch_first

        self.input_seq_length = default_input_seq_length
        self.output_seq_length = default_output_seq_length
        
        self.seq_indices = seq_indices
        self.global_indices = global_indices
        
        if mlp_hidden_dim is None:
            mlp_hidden_dim = d_model
        
        # Projection for sequential features
        self.seq_proj = nn.Linear(nx_seq, d_model)
        
        # Positional encodings for encoder and decoder
        self.encoder_pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout, pos_encoding_type)
        self.decoder_pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout, pos_encoding_type)
        
        # Global feature processing if provided
        if nx_global > 0:
            self.global_proj = nn.Sequential(
                nn.Linear(nx_global, d_model),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )
            if integration_method == "film":
                self.gamma_proj = nn.Linear(d_model, d_model)  # scaling
                self.beta_proj = nn.Linear(d_model, d_model)   # shift
            elif integration_method == "concat":
                self.integration_proj = nn.Linear(d_model * 2, d_model)
            elif integration_method != "add":
                raise ValueError(f"Unknown integration method: {integration_method}")
        
        # Transformer encoder and decoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            layer_norm_eps=layer_norm_eps,
            stochastic_depth_rate=stochastic_depth_rate
        )
        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            layer_norm_eps=layer_norm_eps,
            stochastic_depth_rate=stochastic_depth_rate
        )
        
        # Instead of a decoder input projection from 1 -> d_model,
        # we use a learned start token and a projection to embed previous outputs.
        self.decoder_start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.decoder_start_token, std=0.02)
        self.output_to_decoder_input = nn.Linear(ny, d_model)
        
        # Output head
        self.output_head = OutputHead(
            d_model=d_model,
            out_features=ny,
            mlp_layers=mlp_layers,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            activation=activation
        )
        
        # Parameter initialization
        self._initialize_parameters()
        
        # Causal mask cache: keys are (tgt_len, device_type, device_index)
        self._causal_mask_cache = {}
        
        # Torch compile flag
        self.use_torch_compile = use_torch_compile
        
        # Try torch.compile if enabled
        self._try_compile()
    
    def _initialize_parameters(self):
        """Initialize model parameters with appropriate scaling."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
        
        # Special initialization for FiLM conditioning
        if self.nx_global > 0 and self.integration_method == "film":
            nn.init.normal_(self.gamma_proj.weight, mean=0.0, std=0.02)
            nn.init.ones_(self.gamma_proj.bias)
            nn.init.normal_(self.beta_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.beta_proj.bias)
        
        # Initialize output layer with smaller weights
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.out_features == self.ny:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _try_compile(self):
        """Try to compile the model using torch.compile if enabled."""
        if self.use_torch_compile and hasattr(torch, 'compile'):
            try:
                compiled_model = torch.compile(self)
                logger.info("Torch compilation succeeded.")
                return compiled_model
            except Exception as e:
                logger.warning(f"Torch compilation failed: {e}")
        return self
    
    def _generate_square_subsequent_mask(self, sz, device):
        """Generate a square mask to hide future positions."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _create_causal_mask(self, tgt_len, device):
        """Create (and cache) a causal mask for the decoder."""
        device_key = (device.type, device.index if device.index is not None else 0)
        key = (tgt_len, device_key)
        if key not in self._causal_mask_cache:
            mask = self._generate_square_subsequent_mask(tgt_len, device)
            self._causal_mask_cache[key] = mask
            logger.debug(f"Created new causal mask for tgt_len={tgt_len} on device={device}.")
        return self._causal_mask_cache[key]
        
    def forward(self, src, target_seq_length=None, tgt_mask=None, memory_mask=None):
        """
        Forward pass integrating sequential and global features.
        
        Args:
            src: Input tensor [batch_size, seq_length, features]
            target_seq_length: Optional output sequence length
            
        Returns:
            outputs: Predicted outputs [batch_size, target_seq_length, output_features]
        """
        src = src.to(torch.float32)
        batch_size, seq_length = src.shape[0], src.shape[1]
        
        if self.input_seq_length != seq_length:
            self.input_seq_length = seq_length
            logger.debug(f"Input sequence length updated to {self.input_seq_length}")
        
        if target_seq_length is not None:
            if self.output_seq_length != target_seq_length:
                self.output_seq_length = target_seq_length
                logger.debug(f"Output sequence length updated to {self.output_seq_length}")
        elif self.output_seq_length is None:
            self.output_seq_length = self.input_seq_length
            logger.debug(f"No target sequence length provided; using input sequence length: {self.output_seq_length}")
        
        target_seq_length = self.output_seq_length
        device = src.device
        
        # Process global and sequential features
        if self.nx_global > 0:
            x_seq = src[:, :, self.seq_indices]
            x_global = src[:, 0, self.global_indices]
            
            seq_embedding = self.seq_proj(x_seq)
            seq_embedding = self.encoder_pos_encoder(seq_embedding)
            
            global_embedding = self.global_proj(x_global)
            
            if self.integration_method == "film":
                gamma = self.gamma_proj(global_embedding).unsqueeze(1)
                beta = self.beta_proj(global_embedding).unsqueeze(1)
                encoder_input = gamma * seq_embedding + beta
            elif self.integration_method == "concat":
                global_expanded = global_embedding.unsqueeze(1).expand(-1, seq_length, -1)
                x_combined = torch.cat([seq_embedding, global_expanded], dim=2)
                encoder_input = self.integration_proj(x_combined)
            elif self.integration_method == "add":
                encoder_input = seq_embedding + global_embedding.unsqueeze(1)
            else:
                raise ValueError(f"Unknown integration method: {self.integration_method}")
        else:
            x_seq = src[:, :, self.seq_indices]
            encoder_input = self.seq_proj(x_seq)
            encoder_input = self.encoder_pos_encoder(encoder_input)
        
        memory = self.encoder(encoder_input)
        
        # Use a learned start token repeated for the target sequence length
        decoder_input = self.decoder_start_token.expand(batch_size, target_seq_length, self.d_model)
        decoder_input = self.decoder_pos_encoder(decoder_input)
        
        decoder_output = self.decoder(decoder_input, memory, tgt_mask, memory_mask)
        outputs = self.output_head(decoder_output)
        return outputs
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def generate(self, src, max_length=None, temperature=1.0, autoregressive=True):
        """
        Generate outputs with optional autoregressive decoding.
        
        Args:
            src: Input tensor [batch_size, seq_length, features]
            max_length: Maximum sequence length to generate
            temperature: Temperature for sampling
            autoregressive: Use autoregressive decoding if True
            
        Returns:
            outputs: Generated outputs [batch_size, max_length, output_features]
        """
        if not autoregressive:
            logger.info("Using standard parallel generation.")
            return self.forward(src, target_seq_length=max_length)
        
        logger.info("Starting autoregressive generation.")
        batch_size = src.shape[0]
        device = src.device
        
        if max_length is None:
            max_length = self.output_seq_length or self.input_seq_length
        
        # Process encoder
        with torch.no_grad():
            if self.nx_global > 0:
                x_seq = src[:, :, self.seq_indices]
                x_global = src[:, 0, self.global_indices]
                seq_embedding = self.seq_proj(x_seq)
                seq_embedding = self.encoder_pos_encoder(seq_embedding)
                global_embedding = self.global_proj(x_global)
                
                if self.integration_method == "film":
                    gamma = self.gamma_proj(global_embedding).unsqueeze(1)
                    beta = self.beta_proj(global_embedding).unsqueeze(1)
                    encoder_input = gamma * seq_embedding + beta
                elif self.integration_method == "concat":
                    global_expanded = global_embedding.unsqueeze(1).expand(-1, x_seq.shape[1], -1)
                    x_combined = torch.cat([seq_embedding, global_expanded], dim=2)
                    encoder_input = self.integration_proj(x_combined)
                elif self.integration_method == "add":
                    encoder_input = seq_embedding + global_embedding.unsqueeze(1)
                else:
                    raise ValueError(f"Unknown integration method: {self.integration_method}")
            else:
                x_seq = src[:, :, self.seq_indices]
                encoder_input = self.seq_proj(x_seq)
                encoder_input = self.encoder_pos_encoder(encoder_input)
            
            memory = self.encoder(encoder_input)
            
            all_outputs = torch.zeros(batch_size, max_length, self.ny, device=device)
            # Start with the learned decoder start token (embedded)
            decoder_input = self.decoder_start_token.expand(batch_size, 1, self.d_model)
            
            for i in range(max_length):
                tgt_mask = self._create_causal_mask(decoder_input.size(1), device)
                decoder_output = self.decoder(decoder_input, memory, tgt_mask)
                next_output = self.output_head(decoder_output[:, -1:])
                all_outputs[:, i:i+1] = next_output
                
                if i < max_length - 1:
                    # Use the predicted output to generate the next decoder input
                    next_decoder_input = self.output_to_decoder_input(next_output)  # shape: (batch, 1, d_model)
                    decoder_input = torch.cat([decoder_input, next_decoder_input], dim=1)
            
            return all_outputs


# ------------------------------------------------------------------------------
# Utility: Detect Global vs Sequential Variables
# ------------------------------------------------------------------------------

def detect_variable_types(sample_data, input_var_names, rtol=1e-5, atol=1e-8):
    """
    Detect global vs sequential variables from sample data.
    
    Returns a dictionary with:
      - 'global_vars': List of global variable names
      - 'seq_vars': List of sequential variable names
      - 'global_indices': List of indices for global variables
      - 'seq_indices': List of indices for sequential variables
    """
    if sample_data is None:
        logger.warning("No sample data provided; treating all features as sequential.")
        return {
            'global_vars': [],
            'seq_vars': list(input_var_names),
            'global_indices': [],
            'seq_indices': list(range(len(input_var_names)))
        }
    
    if len(sample_data.shape) == 2:
        logger.info("Sample data shape [sequence_length, features]; adding batch dimension.")
        sample_data = sample_data.unsqueeze(0)
        batch_size = 1
        seq_length, num_features = sample_data.shape[1:]
    elif len(sample_data.shape) == 3:
        batch_size, seq_length, num_features = sample_data.shape
    else:
        logger.warning(f"Unexpected sample data shape: {sample_data.shape}; treating all as sequential.")
        return {
            'global_vars': [],
            'seq_vars': list(input_var_names),
            'global_indices': [],
            'seq_indices': list(range(len(input_var_names)))
        }
    
    if isinstance(input_var_names[0], int):
        input_var_names = [str(idx) for idx in input_var_names]
    
    if len(input_var_names) != num_features:
        logger.warning("Mismatch between number of input variables and data features; treating all as sequential.")
        return {
            'global_vars': [],
            'seq_vars': list(input_var_names),
            'global_indices': [],
            'seq_indices': list(range(len(input_var_names)))
        }
    
    global_vars, seq_vars, global_indices, seq_indices = [], [], [], []
    
    for i, var_name in enumerate(input_var_names):
        feature_data = sample_data[:, :, i]
        is_global = True
        for sample_idx in range(batch_size):
            sample = feature_data[sample_idx]
            first_val = sample[0]
            if not torch.all(torch.isclose(sample, first_val, rtol=rtol, atol=atol)):
                is_global = False
                break
        if is_global:
            global_vars.append(var_name)
            global_indices.append(i)
            logger.info(f"Detected global variable: '{var_name}'")
        else:
            seq_vars.append(var_name)
            seq_indices.append(i)
            logger.info(f"Detected sequential variable: '{var_name}'")
    
    return {
        'global_vars': global_vars,
        'seq_vars': seq_vars,
        'global_indices': global_indices,
        'seq_indices': seq_indices
    }


# ------------------------------------------------------------------------------
# Model Creation Function
# ------------------------------------------------------------------------------

def create_prediction_model(config, sample_data=None):
    """
    Create a prediction model based on configuration.
    """
    input_variables = config.get("input_variables", [])
    target_variables = config.get("target_variables", [])
    
    nx = len(input_variables)
    ny = len(target_variables)
    
    default_input_seq_length = config.get("input_seq_length")
    default_output_seq_length = config.get("output_seq_length")
    
    d_model = config.get("d_model", 256)
    dropout = config.get("dropout", 0.1)
    
    nhead = config.get("nhead", min(8, d_model // 32))
    if d_model % nhead != 0:
        old_nhead = nhead
        for divisor in range(nhead, 0, -1):
            if d_model % divisor == 0:
                nhead = divisor
                logger.warning(f"Adjusted nhead from {old_nhead} to {nhead} to ensure divisibility with d_model={d_model}")
                break
    
    num_layers = config.get("num_layers", 6)
    dim_feedforward = config.get("dim_feedforward", d_model * 4)
    activation = config.get("activation", "gelu")
    
    # Auto-detect variable types if sample data is provided
    if sample_data is not None:
        sample_data = sample_data.float()
        feature_types = detect_variable_types(sample_data, input_variables)
        global_indices = feature_types['global_indices']
        seq_indices = feature_types['seq_indices']
    else:
        global_indices = config.get('global_feature_indices', [])
        seq_indices = config.get('seq_feature_indices', list(range(nx)))
        logger.info(f"Using global indices from config: {global_indices}")
        logger.info(f"Using sequential indices from config: {seq_indices}")

    nx_seq = len(seq_indices)
    nx_global = len(global_indices)
    
    # Handle max sequence length config key flexibility
    max_seq_length = config.get('max_seq_length', config.get('max_sequence_length', 512))
    
    model = AtmosphericModel(
        nx_seq=nx_seq,
        nx_global=nx_global,
        ny=ny,
        seq_indices=seq_indices,
        global_indices=global_indices,
        default_input_seq_length=default_input_seq_length,
        default_output_seq_length=default_output_seq_length,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        integration_method=config.get('integration_method', 'film'),
        stochastic_depth_rate=config.get('stochastic_depth_rate', 0.0),
        norm_first=config.get("norm_first", True),
        max_seq_length=max_seq_length,
        pos_encoding_type=config.get('pos_encoding_type', 'sine'),
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        mlp_layers=config.get('mlp_layers', 3),
        mlp_hidden_dim=config.get('mlp_hidden_dim'),
        batch_first=True,
        use_torch_compile=config.get("use_torch_compile", True)
    )
    
    model.nx_names = input_variables
    model.ny_names = target_variables
    
    logger.info(f"Created model with {model.count_parameters():,} trainable parameters.")
    return model
