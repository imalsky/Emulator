#!/usr/bin/env python3
"""
model.py

Neural network models for atmospheric and spectral prediction:
A transformer-based architecture optimized for atmospheric profiles and spectral data.
This version adds an explicit decoder for sequence-to-sequence prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer sequence models.
    Supports sinusoidal and learned encodings.
    """
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1, encoding_type: str = "sine"):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoding_type = encoding_type.lower()
        if self.encoding_type == "sine":
            # Sinusoidal encoding from "Attention Is All You Need"
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0), persistent=True)
        elif self.encoding_type == "learned":
            # Learned positional embeddings
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pe, std=0.02)
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}. Use 'sine' or 'learned'.")
    
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
    
    def forward_with_offset(self, x, offset=0):
        if offset > 0 and offset + x.size(1) <= self.pe.size(1):
            return self.dropout(x + self.pe[:, offset:offset + x.size(1)])
        else:
            return self.forward(x)


class SequenceLengthAdapter(nn.Module):
    """
    Module for adapting between different sequence lengths.
    (This module is retained from the original design but is not used in the explicit encoder–decoder model.)
    """
    def __init__(self, d_model: int, in_seq_len: int, out_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.dropout = nn.Dropout(p=dropout)
        if in_seq_len != out_seq_len:
            self.adaptation_method = "attn"
            self.query_proj = nn.Linear(d_model, d_model)
            self.key_proj = nn.Linear(d_model, d_model)
            self.value_proj = nn.Linear(d_model, d_model)
            self.target_pos_embeddings = nn.Parameter(torch.zeros(1, out_seq_len, d_model))
            nn.init.trunc_normal_(self.target_pos_embeddings, std=0.02)
            self.out_proj = nn.Linear(d_model, d_model)
        else:
            self.adaptation_method = "identity"
    
    def forward(self, x):
        if self.adaptation_method == "identity":
            return x
        batch_size = x.shape[0]
        queries = self.query_proj(self.target_pos_embeddings.expand(batch_size, -1, -1))
        keys = self.key_proj(x)
        values = self.value_proj(x)
        scale = float(self.d_model) ** -0.5
        attn = torch.matmul(queries, keys.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, values)
        out = self.out_proj(out)
        return out


class MultiHeadAttention(nn.Module):
    """
    Efficient multi-head attention with Flash Attention support.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
    
    def _reshape_for_multi_head(self, x, batch_size):
        x = x.view(batch_size, -1, 3, self.nhead, self.head_dim)
        x = x.permute(2, 0, 3, 1, 4)
        return x[0], x[1], x[2]
    
    def forward(self, x, attn_mask=None):
        batch_size, seq_len = x.shape[:2]
        qkv = self.qkv_proj(x)
        q, k, v = self._reshape_for_multi_head(qkv, batch_size)
        if self.use_flash and torch.cuda.is_available():
            try:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0
                )
            except Exception as e:
                logger.warning(f"Flash attention failed ({e}); falling back to standard attention")
                attn_output = self._standard_attention(q, k, v, attn_mask)
        else:
            attn_output = self._standard_attention(q, k, v, attn_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)
    
    def _standard_attention(self, q, k, v, attn_mask):
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)


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
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm architecture."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu", norm_first: bool = True,
                 drop_path_rate: float = 0.0, layer_norm_eps: float = 1e-5):
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
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src_mask)
        src = src + self.drop_path(self.dropout1(src2))
        src2 = self.norm2(src)
        src2 = self.ff(src2)
        src = src + self.drop_path(self.dropout2(src2))
        return src


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu", norm_first: bool = True,
                 layer_norm_eps: float = 1e-5, stochastic_depth_rate: float = 0.0):
        super().__init__()
        if stochastic_depth_rate > 0:
            drop_path_rates = torch.linspace(0, stochastic_depth_rate, num_layers).tolist()
        else:
            drop_path_rates = [0.0] * num_layers
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
            ) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if norm_first else None
    
    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with pre-norm architecture.
    Performs masked self-attention followed by cross-attention with encoder outputs and a feed-forward network.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu", norm_first: bool = True,
                 drop_path_rate: float = 0.0, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout3 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Masked self-attention with causal mask
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt_mask)
        tgt = tgt + self.drop_path(self.dropout1(tgt2))
        # Cross-attention: decoder queries attend to encoder outputs (memory)
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt2, attn_mask=memory_mask)  # Here, cross-attention uses memory as keys/values internally.
        # To implement cross-attention properly, one common strategy is to concatenate decoder queries with memory;
        # for simplicity, we assume our MultiHeadAttention can be modified accordingly. (Alternatively, modify the module.)
        tgt = tgt + self.drop_path(self.dropout2(tgt2))
        # Feed-forward network
        tgt2 = self.norm3(tgt)
        tgt2 = self.ff(tgt2)
        tgt = tgt + self.drop_path(self.dropout3(tgt2))
        return tgt


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers."""
    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = norm
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


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


class TransformerModel(nn.Module):
    """
    Transformer-based model for prediction with flexibility for varying sequence lengths.
    Extended to include an explicit decoder for sequence-to-sequence prediction.
    Expects both a source (src) and a target (tgt) input.
    """
    def __init__(
        self,
        nx: int,                      # Number of input features
        ny: int,                      # Number of output features
        default_output_seq_length: Optional[int] = None,  # (Not used in decoder mode)
        default_input_seq_length: Optional[int] = None,   # (Not used in decoder mode)
        d_model: int = 256,           # Model dimension
        nhead: int = 8,               # Number of attention heads
        num_layers: int = 6,          # Number of encoder layers
        num_decoder_layers: int = 6,  # Number of decoder layers
        dim_feedforward: int = 1024,  # Feedforward dimension
        dropout: float = 0.1,         # Dropout probability
        activation: str = "gelu",     # Activation function
        chunk_size: int = 1000,       # (Retained but not used in decoder mode)
        global_feature_indices: Optional[List[int]] = None,  # Indices of global features
        seq_feature_indices: Optional[List[int]] = None,     # Indices of sequential features
        integration_method: str = "film", # Method to integrate global and profile features
        stochastic_depth_rate: float = 0.0, # Probability of dropping paths
        norm_first: bool = True,      # Whether to use pre-norm architecture
        pos_encoding_type: str = "sine", # Positional encoding type
        use_torch_compile: bool = False,  # Whether to use torch.compile
        layer_norm_eps: float = 1e-5, # Epsilon for LayerNorm
        mlp_layers: int = 3,          # Number of layers in output MLP
        mlp_hidden_dim: Optional[int] = None, # Hidden dimension in output MLP
        sample_data: Optional[torch.Tensor] = None # Sample data for auto-detecting global/seq features
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.nx = nx
        self.ny = ny
        self.d_model = d_model
        self.integration_method = integration_method.lower()
        
        # Auto-detect global vs sequential features if sample data is provided.
        if global_feature_indices is None and seq_feature_indices is None and sample_data is not None:
            sample_data = sample_data.float()
            var_indices = list(range(nx))
            feature_types = detect_variable_types(sample_data, var_indices, rtol=1e-5, atol=1e-8)
            self.global_feature_indices = feature_types['global_indices']
            self.seq_feature_indices = feature_types['seq_indices']
            self.has_global_features = len(self.global_feature_indices) > 0
        else:
            self.global_feature_indices = global_feature_indices if global_feature_indices is not None else []
            self.seq_feature_indices = seq_feature_indices if seq_feature_indices is not None else list(range(nx))
            self.has_global_features = len(self.global_feature_indices) > 0
        
        logger.info(f"Initializing TransformerModel with {len(self.global_feature_indices)} global and {len(self.seq_feature_indices)} sequential features")
        
        # Encoder: project sequential features from input space to model dimension.
        self.encoder_input_proj = nn.Linear(len(self.seq_feature_indices), d_model)
        
        # Global feature processing (if any)
        if self.has_global_features:
            self.global_proj = nn.Sequential(
                nn.Linear(len(self.global_feature_indices), d_model),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )
        
        # Positional encodings for encoder and decoder.
        self.encoder_pos_encoder = PositionalEncoding(d_model, max_len=2048, dropout=dropout, encoding_type=pos_encoding_type)
        self.decoder_pos_encoder = PositionalEncoding(d_model, max_len=2048, dropout=dropout, encoding_type=pos_encoding_type)
        
        # Encoder: transformer encoder stack.
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            layer_norm_eps=layer_norm_eps,
            stochastic_depth_rate=stochastic_depth_rate
        )
        
        # Decoder: explicit transformer decoder stack.
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            drop_path_rate=0.0,
            layer_norm_eps=layer_norm_eps
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, norm=nn.LayerNorm(d_model, eps=layer_norm_eps))
        
        # Output projection for final prediction.
        self.output_head = nn.Linear(d_model, ny)
        
        # Retain original positional encoder and sequence adapter for backward compatibility,
        # though they will not be used in the decoder branch.
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(self.default_input_seq_length * 2 if default_input_seq_length else 200, 2048), dropout=dropout, encoding_type=pos_encoding_type)
        self.seq_adapter = None  # Not used in decoder mode.
        
        # Initialize parameters.
        self._initialize_parameters()
        
        if use_torch_compile and hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                logger.info("Using torch.compile for model acceleration")
                self.forward = torch.compile(self.forward)
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        logger.info(f"TransformerModel initialized with {self.count_parameters():,} parameters")
    
    def _initialize_parameters(self):
        """Initialize parameters with appropriate scaling."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
        if self.has_global_features and self.integration_method == "film":
            nn.init.normal_(self.gamma_proj.weight, mean=0.0, std=0.02)
            nn.init.ones_(self.gamma_proj.bias)
            nn.init.normal_(self.beta_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.beta_proj.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.out_features == self.ny:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def create_sequence_adapter(self, in_seq_len, out_seq_len, device):
        """Create adapter for transforming between sequence lengths (unused in decoder mode)."""
        adapter = SequenceLengthAdapter(
            d_model=self.d_model,
            in_seq_len=in_seq_len,
            out_seq_len=out_seq_len,
            dropout=0.1
        ).to(device)
        return adapter
    
    def forward(self, src, tgt):
        """
        Forward pass through the encoder-decoder model.
        
        Args:
            src: Input tensor of shape [batch, src_seq_len, nx].
            tgt: Decoder input tensor of shape [batch, tgt_seq_len, ny] (e.g. shifted target sequence).
        
        Returns:
            Output tensor of shape [batch, tgt_seq_len, ny].
        """
        src = src.to(torch.float32)
        # Process encoder input.
        try:
            seq_features = src[:, :, self.seq_feature_indices]
        except IndexError as e:
            logger.error(f"Sequential feature extraction error: {e}. Input shape: {src.shape}, seq indices: {self.seq_feature_indices}")
            raise ValueError(f"Sequential feature extraction failed. Check that feature indices match input dimensions: {src.shape[2]} features available")
        encoder_emb = self.encoder_input_proj(seq_features)
        encoder_emb = self.encoder_pos_encoder(encoder_emb)
        if self.has_global_features:
            try:
                global_features = src[:, 0, self.global_feature_indices]
            except IndexError as e:
                logger.error(f"Global feature extraction error: {e}. Input shape: {src.shape}, global indices: {self.global_feature_indices}")
                raise ValueError(f"Global feature extraction failed. Check that feature indices match input dimensions: {src.shape[2]} features available")
            global_emb = self.global_proj(global_features)
            encoder_emb = encoder_emb + global_emb.unsqueeze(1)
        # Encoder forward pass.
        memory = self.transformer(encoder_emb)
        
        # Process decoder input.
        tgt = tgt.to(torch.float32)
        decoder_emb = self.decoder_pos_encoder(tgt)
        tgt_seq_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device) * float('-inf'), diagonal=1)
        decoder_output = self.decoder(decoder_emb, memory, tgt_mask=causal_mask)
        output = self.output_head(decoder_output)
        return output
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def detect_variable_types(sample_data, input_var_names, rtol=1e-5, atol=1e-8):
    """
    Automatically detect global vs sequential variables from sample data.
    """
    sample_data = sample_data.float()
    batch_size, seq_length, num_features = sample_data.shape
    if isinstance(input_var_names[0], int):
        input_var_names = [str(idx) for idx in input_var_names]
    if len(input_var_names) != num_features:
        raise ValueError(f"Number of input variables ({len(input_var_names)}) doesn't match data features ({num_features})")
    global_vars = []
    seq_vars = []
    global_indices = []
    seq_indices = []
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
            logger.info(f"Detected global variable: '{var_name}' (constant across sequence)")
        else:
            seq_vars.append(var_name)
            seq_indices.append(i)
            logger.info(f"Detected sequential variable: '{var_name}' (varies across sequence)")
    return {
        'global_vars': global_vars,
        'seq_vars': seq_vars,
        'global_indices': global_indices,
        'seq_indices': seq_indices
    }


def validate_target_sequence_lengths(targets, feature_names=None):
    """
    Validate that all target features have the same sequence length.
    
    Parameters
    ----------
    targets : torch.Tensor
        Target tensor of shape [batch, seq_len, num_features]
    feature_names : List[str], optional
        Names of target features for more informative warnings
        
    Returns
    -------
    int
        The common sequence length if consistent, otherwise the maximum length
    """
    if targets.dim() < 3:
        return targets.shape[1]
    batch_size, _, num_features = targets.shape
    if num_features > 1:
        seq_lengths = []
        for i in range(num_features):
            feature = targets[:, :, i]
            is_sequential = False
            for sample_idx in range(min(batch_size, 10)):
                sample = feature[sample_idx]
                first_val = sample[0]
                if not torch.all(torch.isclose(sample, first_val, rtol=1e-5, atol=1e-8)):
                    is_sequential = True
                    break
            if is_sequential:
                feature_name = feature_names[i] if feature_names else f"Feature {i}"
                seq_lengths.append((i, feature_name, feature.shape[1]))
        if seq_lengths and len(set([length for _, _, length in seq_lengths])) > 1:
            feature_details = ', '.join([f"{name}: {length}" for _, name, length in seq_lengths])
            logger.warning(f"Target features have inconsistent sequence lengths: {feature_details}")
            max_length = max([length for _, _, length in seq_lengths])
            logger.warning(f"Using maximum sequence length ({max_length}) for output")
            return max_length
        elif seq_lengths:
            return seq_lengths[0][2]
    return targets.shape[1]


def create_prediction_model(config, sample_data=None):
    """
    Create a prediction model based on configuration.
    """
    nx = len(config.get("input_variables", []))
    ny = len(config.get("target_variables", []))
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
                logger.warning(f"Adjusted nhead from {old_nhead} to {nhead} to ensure it divides d_model={d_model}")
                break
    num_layers = config.get("num_layers", 3)
    dim_feedforward = config.get("dim_feedforward", d_model * 4)
    activation = config.get("activation", "gelu")
    global_feature_indices = config.get("global_feature_indices", None)
    seq_feature_indices = config.get("seq_feature_indices", None)
    integration_method = config.get("integration_method", "add")
    stochastic_depth_rate = config.get("stochastic_depth_rate", 0.0)
    pos_encoding_type = config.get("pos_encoding_type", "sine")
    use_torch_compile = config.get("use_torch_compile", False)
    chunk_size = config.get("chunk_size", 1000)
    if use_torch_compile:
        use_torch_compile = hasattr(torch, 'compile') and torch.cuda.is_available()
        if not use_torch_compile:
            logger.warning("torch.compile disabled - requires CUDA and PyTorch 2.0+")
    mlp_layers = config.get("mlp_layers", 3)
    mlp_hidden_dim = config.get("mlp_hidden_dim", None)
    
    logger.info("Creating TransformerModel (with explicit decoder)")
    
    model = TransformerModel(
        nx=nx,
        ny=ny,
        default_output_seq_length=default_output_seq_length,
        default_input_seq_length=default_input_seq_length,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_decoder_layers=config.get("num_decoder_layers", 3),
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        chunk_size=chunk_size,
        global_feature_indices=global_feature_indices,
        seq_feature_indices=seq_feature_indices,
        integration_method=integration_method,
        stochastic_depth_rate=stochastic_depth_rate,
        norm_first=config.get("norm_first", True),
        pos_encoding_type=pos_encoding_type,
        use_torch_compile=use_torch_compile,
        layer_norm_eps=config.get("layer_norm_eps", 1e-5),
        mlp_layers=mlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        sample_data=sample_data
    )
    
    logger.info(f"Created model with {model.count_parameters():,} parameters")
    return model
