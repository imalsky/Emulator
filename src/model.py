#!/usr/bin/env python3
"""
model.py

Neural network models for prediction:
1. TransformerModel - A transformer-based architecture optimized for spectral data
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SimplePositionalEncoding(nn.Module):
    """
    Simple sinusoidal encoding for coordinate values.
    Maps a 1D coordinate to a higher dimensional space.
    """
    def __init__(self, dim=64, max_freq=10.0):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        
    def forward(self, x):
        """
        Encode coordinates using sine/cosine functions at different frequencies.
        """
        # Ensure input has correct shape
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)
            
        # Create frequency bands
        freqs = torch.linspace(1.0, self.max_freq, self.dim // 2, device=x.device)
        
        # Calculate encodings
        args = x * freqs.unsqueeze(0).unsqueeze(0) * math.pi
        encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return encoding


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequence models."""
    
    def __init__(self, d_model, max_len=2048, dropout=0.1, encoding_type="sine"):
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
            
        elif self.encoding_type == "learned":
            # Learned positional embeddings
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pe, std=0.02)
            
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}. Use 'sine' or 'learned'.")

    def forward(self, x):
        """Add positional encoding to input."""
        return self.dropout(x + self.pe[:, :x.size(1)])
        
    def forward_with_offset(self, x, offset=0):
        """Add positional encoding to input with position offset."""
        if offset > 0 and offset + x.size(1) <= self.pe.size(1):
            return self.dropout(x + self.pe[:, offset:offset + x.size(1)])
        else:
            # Fallback to regular forward if offset is invalid
            return self.forward(x)


class MultiHeadAttention(nn.Module):
    """Efficient multi-head attention with Flash Attention support."""
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        
        # Projection layers
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
        # Return query, key, value each with shape [batch_size, nhead, seq_len, head_dim]
        return x[0], x[1], x[2]
        
    def forward(self, x, attn_mask=None):
        """Forward pass for multi-head attention."""
        batch_size, seq_len = x.shape[:2]
        
        # Combined projection for efficiency
        qkv = self.qkv_proj(x)
        q, k, v = self._reshape_for_multi_head(qkv, batch_size)
        
        # Use Flash Attention if available (for newer GPUs)
        if self.use_flash and torch.cuda.is_available():
            try:
                # Expects shape: [batch_size, nhead, seq_len, head_dim]
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0
                )
            except Exception as e:
                logger.warning(f"Flash attention failed ({e}); falling back to standard attention")
                attn_output = self._standard_attention(q, k, v, attn_mask)
        else:
            attn_output = self._standard_attention(q, k, v, attn_mask)
        
        # Reshape output and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)
    
    def _standard_attention(self, q, k, v, attn_mask):
        """Standard scaled dot-product attention."""
        # Scale dot product
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
            
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        return torch.matmul(attn_weights, v)


class FeedForward(nn.Module):
    """Feed-forward network with residual connection."""
    
    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation="gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.SiLU()  # Swish activation
    
    def forward(self, x):
        """Forward pass for feed-forward network."""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class DropPath(nn.Module):
    """Drops paths (residual connections) per sample with probability drop_prob."""
    
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm architecture and stochastic depth."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="gelu", drop_path_rate=0.0):
        super().__init__()
        
        # Attention block
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feedforward block
        self.ff = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
    def forward(self, src, src_mask=None):
        """Forward pass with pre-norm architecture."""
        # Self-attention block with pre-norm
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src_mask)
        src = src + self.drop_path(self.dropout1(src2))
        
        # Feedforward block with pre-norm
        src2 = self.norm2(src)
        src2 = self.ff(src2)
        src = src + self.drop_path(self.dropout2(src2))
        
        return src


class TransformerModel(nn.Module):
    """
    Transformer-based model for spectral prediction.
    """
    def __init__(
        self,
        nx,                      # Number of input features
        ny,                      # Number of output features
        output_seq_length,       # Length of output sequence
        input_seq_length=100,    # Length of input sequence
        hidden_dim=256,          # Hidden dimension size
        nhead=8,                 # Number of attention heads
        num_layers=3,            # Number of transformer layers
        dim_feedforward=1024,    # Feedforward dimension
        dropout=0.1,             # Dropout probability
        activation="gelu",       # Activation function
        pos_encoding_dim=64,     # Dimension for coordinate encoding
        max_freq=10.0,           # Maximum frequency for coordinate encoding
        use_coordinate=True,     # Whether to use coordinate information
        chunk_size=1000,         # Chunk size for processing long sequences
        global_feature_indices=None,  # Indices of global features (auto-detect if None)
        seq_feature_indices=None,     # Indices of sequential features (auto-detect if None)
        integration_method="add", # Method to integrate global and profile features: "add", "film", or "concat"
        stochastic_depth_rate=0.0, # Probability of dropping paths for regularization
        pos_encoding_type="sine", # Positional encoding type: "sine" or "learned"
        use_torch_compile=False  # Whether to use torch.compile for acceleration
    ):
        super().__init__()
        
        self.nx = nx
        self.ny = ny
        self.output_seq_length = output_seq_length
        self.input_seq_length = input_seq_length
        self.use_coordinate = use_coordinate
        self.chunk_size = min(chunk_size, 1000)  # Limit chunk size for memory efficiency
        self.integration_method = integration_method.lower()
        
        # Determine global and sequential feature indices
        if global_feature_indices is not None and seq_feature_indices is not None:
            self.global_feature_indices = global_feature_indices
            self.seq_feature_indices = seq_feature_indices
            self.has_global_features = len(global_feature_indices) > 0
        else:
            # Default: assume all features are sequential if not specified
            self.global_feature_indices = []
            self.seq_feature_indices = list(range(nx))
            self.has_global_features = False
        
        logger.info(f"Initializing TransformerModel for {output_seq_length} output points")
        
        # Global feature encoder (if using global features)
        if self.has_global_features:
            self.global_encoder = nn.Sequential(
                nn.Linear(len(self.global_feature_indices), hidden_dim),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            )
            
            # Feature integration mechanism for global features
            if self.integration_method == "film":
                # Feature-wise Linear Modulation
                self.gamma_proj = nn.Linear(hidden_dim, hidden_dim)  # Scaling
                self.beta_proj = nn.Linear(hidden_dim, hidden_dim)   # Shift
            elif self.integration_method == "concat":
                # Concatenation followed by projection
                self.integration_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Profile encoder for sequential features
        if len(self.seq_feature_indices) > 0:
            self.profile_encoder = nn.Sequential(
                nn.Linear(len(self.seq_feature_indices) * input_seq_length, hidden_dim * 2),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        else:
            # Handle edge case of no sequential features
            logger.warning("No sequential features detected, using identity mapping")
            self.profile_encoder = nn.Identity()
        
        # Coordinate encoder if using coordinates
        if use_coordinate:
            self.coord_encoder = SimplePositionalEncoding(pos_encoding_dim, max_freq)
            # Project combined features to hidden_dim
            self.combined_proj = nn.Linear(hidden_dim + pos_encoding_dim, hidden_dim)
        
        # Position encoding for transformer sequence
        self.pos_encoder = PositionalEncoding(
            hidden_dim, 
            max_len=max(output_seq_length, 2048), 
            dropout=dropout, 
            encoding_type=pos_encoding_type
        )
        
        # Generate stochastic depth rates if using it
        drop_path_rates = None
        if stochastic_depth_rate > 0:
            drop_path_rates = torch.linspace(0, stochastic_depth_rate, num_layers).tolist()
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                drop_path_rate=drop_path_rates[i] if drop_path_rates else 0.0
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm for pre-norm architecture
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, ny)
        
        # Initialize parameters
        self._init_parameters()
        
        # Apply torch.compile if requested
        if use_torch_compile and hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                logger.info("Using torch.compile for model acceleration")
                self.forward = torch.compile(self.forward)
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        logger.info(f"TransformerModel initialized with {self.count_parameters():,} parameters")
    
    def _init_parameters(self):
        """Initialize parameters with appropriate scaling."""
        # Xavier initialization for most layers
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Special initialization for FiLM conditioning
        if self.has_global_features and self.integration_method == "film":
            # Initialize gamma close to 1 (identity scaling)
            nn.init.normal_(self.gamma_proj.weight, mean=0.0, std=0.02)
            nn.init.ones_(self.gamma_proj.bias)
            
            # Initialize beta close to 0 (no shift)
            nn.init.normal_(self.beta_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.beta_proj.bias)
        
        # Initialize output layer with smaller weights
        nn.init.normal_(self.output_proj.weight, mean=0, std=0.01)
        nn.init.zeros_(self.output_proj.bias)
    
    def generate_coordinates(self, batch_size, device):
        """Generate default coordinates from 0 to 1."""
        coords = torch.linspace(0, 1, self.output_seq_length, device=device)
        if batch_size > 1:
            coords = coords.unsqueeze(0).expand(batch_size, -1)
        return coords
    
    def forward(self, src, coordinates=None):
        """
        Forward pass through transformer with coordinate integration.
        
        Args:
            src: Input profiles of shape [batch_size, input_seq_length, nx]
            coordinates: Optional coordinate values of shape [batch_size, output_seq_length]
                      If None, uses default coordinates from 0 to 1
        
        Returns:
            Predictions of shape [batch_size, output_seq_length, ny]
        """
        batch_size = src.shape[0]
        
        # Process global and sequential features
        if self.has_global_features:
            # Extract global features (assumed constant across sequence)
            global_features = src[:, 0, self.global_feature_indices]
            global_encoded = self.global_encoder(global_features)
            
            if len(self.seq_feature_indices) > 0:
                # Extract sequential features
                seq_features = src[:, :, self.seq_feature_indices]
                seq_flat = seq_features.reshape(batch_size, -1)
                seq_encoded = self.profile_encoder(seq_flat)
                
                # Combine global and sequential features based on integration method
                if self.integration_method == "add":
                    profile_features = global_encoded + seq_encoded
                elif self.integration_method == "film":
                    # FiLM conditioning (Feature-wise Linear Modulation)
                    gamma = self.gamma_proj(global_encoded)  # Scaling
                    beta = self.beta_proj(global_encoded)    # Shift
                    profile_features = gamma * seq_encoded + beta
                elif self.integration_method == "concat":
                    # Concatenate and project
                    profile_features = torch.cat([seq_encoded, global_encoded], dim=1)
                    profile_features = self.integration_proj(profile_features)
                else:
                    raise ValueError(f"Unknown integration method: {self.integration_method}")
            else:
                # No sequential features, use only global
                profile_features = global_encoded
        else:
            # Only process sequential features
            seq_flat = src.reshape(batch_size, -1)
            profile_features = self.profile_encoder(seq_flat)  # [batch_size, hidden_dim]
        
        # Generate default coordinates if not provided
        if coordinates is None and self.use_coordinate:
            coordinates = self.generate_coordinates(batch_size, src.device)
        
        # For very long sequences, process in chunks to save memory
        if self.output_seq_length > 2000:
            return self._process_long_sequence(profile_features, coordinates)
            
        # Ensure coordinates have correct shape if using them
        if self.use_coordinate:
            if coordinates.dim() == 1:
                coordinates = coordinates.unsqueeze(0)
            if coordinates.size(0) == 1 and batch_size > 1:
                coordinates = coordinates.expand(batch_size, -1)
            
            # Prepare coordinate features for each spectral point
            coords = coordinates.unsqueeze(-1)  # [batch_size, output_seq_length, 1]
            coord_features = self.coord_encoder(coords)  # [batch_size, output_seq_length, pos_encoding_dim]
            
            # Expand profile features to match output sequence length
            profile_expanded = profile_features.unsqueeze(1).expand(-1, self.output_seq_length, -1)
            
            # Combine profile and coordinate features
            combined = torch.cat([profile_expanded, coord_features], dim=-1)
            x = self.combined_proj(combined)  # Project to hidden_dim
        else:
            # Use only profile features expanded across sequence
            x = profile_features.unsqueeze(1).expand(-1, self.output_seq_length, -1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x)
            
        # Final normalization
        x = self.norm(x)
        
        # Project to output dimension
        output = self.output_proj(x)
        
        return output
    
    def _process_long_sequence(self, profile_features, coordinates=None):
        """Process long sequences in chunks to save memory."""
        batch_size = profile_features.shape[0]
        device = profile_features.device
        outputs = []
        
        # Process in chunks
        for chunk_start in range(0, self.output_seq_length, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.output_seq_length)
            chunk_size = chunk_end - chunk_start
            
            # Get coordinates for this chunk
            if self.use_coordinate and coordinates is not None:
                chunk_coords = coordinates[:, chunk_start:chunk_end]
            else:
                chunk_coords = torch.linspace(
                    chunk_start / self.output_seq_length,
                    chunk_end / self.output_seq_length,
                    chunk_size, device=device
                ).unsqueeze(0).expand(batch_size, -1)
            
            # Process this chunk
            if self.use_coordinate:
                # Prepare coordinate features
                coords = chunk_coords.unsqueeze(-1)
                coord_features = self.coord_encoder(coords)
                
                # Expand profile features to match chunk size
                profile_expanded = profile_features.unsqueeze(1).expand(-1, chunk_size, -1)
                
                # Combine profile and coordinate features
                combined = torch.cat([profile_expanded, coord_features], dim=-1)
                x = self.combined_proj(combined)
            else:
                # Use only profile features
                x = profile_features.unsqueeze(1).expand(-1, chunk_size, -1)
            
            # Add positional encoding for this chunk with position offset
            if hasattr(self.pos_encoder, 'forward_with_offset'):
                x = self.pos_encoder.forward_with_offset(x, offset=chunk_start)
            else:
                # Fall back to regular encoding if offset not supported
                x = self.pos_encoder(x)
            
            # Process through transformer layers
            for layer in self.layers:
                x = layer(x)
                
            # Final normalization and output projection
            x = self.norm(x)
            chunk_output = self.output_proj(x)
            outputs.append(chunk_output)
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Combine all chunks
        return torch.cat(outputs, dim=1)
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_prediction_model(config):
    """
    Create a prediction model based on configuration.
    """
    # Extract required parameters
    nx = len(config.get("input_variables", []))
    ny = len(config.get("target_variables", []))
    output_seq_length = config.get("output_seq_length", 2000)
    input_seq_length = config.get("input_seq_length", 100)
    
    # Basic model configuration
    hidden_dim = config.get("hidden_dim", config.get("d_model", 256))
    dropout = config.get("dropout", 0.1)
    use_coordinate = len(config.get("coordinate_variable", [])) > 0
    
    # Advanced model parameters
    nhead = config.get("nhead", min(8, hidden_dim // 32))  # Ensure nhead divides hidden_dim
    # Adjust nhead if it doesn't divide hidden_dim evenly
    if hidden_dim % nhead != 0:
        old_nhead = nhead
        for divisor in range(nhead, 0, -1):
            if hidden_dim % divisor == 0:
                nhead = divisor
                logger.warning(f"Adjusted nhead from {old_nhead} to {nhead} to ensure it divides hidden_dim={hidden_dim}")
                break
    
    num_layers = config.get("num_layers", 3)
    dim_feedforward = config.get("dim_feedforward", hidden_dim * 4)
    activation = config.get("activation", "gelu")
    pos_encoding_dim = config.get("pos_encoding_dim", 64)
    max_freq = config.get("max_freq", 10.0)
    
    # Chunk size for processing long sequences
    chunk_size = config.get("chunk_size", 1000)
    
    # Advanced features
    integration_method = config.get("integration_method", "add")
    stochastic_depth_rate = config.get("stochastic_depth_rate", 0.0)
    pos_encoding_type = config.get("pos_encoding_type", "sine")
    use_torch_compile = config.get("use_torch_compile", False)
    
    # Check for hardware support for compilation
    if use_torch_compile:
        use_torch_compile = hasattr(torch, 'compile') and torch.cuda.is_available()
        if not use_torch_compile:
            logger.warning("torch.compile disabled - requires CUDA and PyTorch 2.0+")
    
    # Create the transformer model
    logger.info(f"Creating TransformerModel with {output_seq_length} output points")
    model = TransformerModel(
        nx=nx,
        ny=ny,
        output_seq_length=output_seq_length,
        input_seq_length=input_seq_length,
        hidden_dim=hidden_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        pos_encoding_dim=pos_encoding_dim,
        max_freq=max_freq,
        use_coordinate=use_coordinate,
        chunk_size=chunk_size,
        integration_method=integration_method,
        stochastic_depth_rate=stochastic_depth_rate,
        pos_encoding_type=pos_encoding_type,
        use_torch_compile=use_torch_compile
    )
    
    logger.info(f"Created model with {model.count_parameters():,} parameters")
    
    return model