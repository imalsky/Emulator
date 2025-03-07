#!/usr/bin/env python3
"""
enhanced_transformer_model.py

Transformer models for atmospheric variable prediction, with model creation functions
that maintain compatibility with train.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Union, List, Dict, Tuple, Any

logger = logging.getLogger(__name__)


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


class MultiHeadAttention(nn.Module):
    """Efficient multi-head attention with Flash Attention support."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        
        # Projection layers
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)  # Combined Q, K, V projections
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
        """Forward pass for feed-forward network."""
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
        output = x.div(keep_prob) * random_tensor
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm or post-norm architecture."""
    
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
        dropout: float = 0.1, activation: str = "gelu", norm_first: bool = True,
        drop_path_rate: float = 0.0, layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.norm_first = norm_first
        
        # Attention block
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feedforward block
        self.ff = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)
        
        # Optional stochastic depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
    def forward(self, src, src_mask=None):
        """Forward pass with pre-norm or post-norm architecture."""
        # Self-attention block
        if self.norm_first:
            # Pre-norm
            src2 = self.norm1(src)
            src2 = self.self_attn(src2, src_mask)
            src = src + self.drop_path(self.dropout1(src2))
            
            # Feedforward block
            src2 = self.norm2(src)
            src2 = self.ff(src2)
            src = src + self.drop_path(self.dropout2(src2))
        else:
            # Post-norm
            src2 = self.self_attn(src, src_mask)
            src = src + self.drop_path(self.dropout1(src2))
            src = self.norm1(src)
            
            src2 = self.ff(src)
            src = src + self.drop_path(self.dropout2(src2))
            src = self.norm2(src)
            
        return src


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(
        self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 2048,
        dropout: float = 0.1, activation: str = "gelu", norm_first: bool = True,
        layer_norm_eps: float = 1e-5, stochastic_depth_rate: float = 0.0
    ):
        super().__init__()
        
        # Generate stochastic depth rates if needed
        drop_path_rates = None
        if stochastic_depth_rate > 0:
            drop_path_rates = torch.linspace(0, stochastic_depth_rate, num_layers).tolist()
        
        # Create layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                layer_norm_eps=layer_norm_eps,
                drop_path_rate=drop_path_rates[i] if drop_path_rates else 0.0
            )
            for i in range(num_layers)
        ])
        
        # Final normalization for pre-norm architecture
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if norm_first else None
    
    def forward(self, src, mask=None):
        """Process input through all encoder layers."""
        output = src
        
        for layer in self.layers:
            output = layer(output, mask)
            
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
            
            # Hidden layers
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
            
            # Output layer
            layers.append(nn.Linear(in_dim, out_features))
            self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """Project input to output dimension."""
        return self.mlp(x)


class AtmosphericModel(nn.Module):
    """
    Transformer model that handles both sequential and global features.
    
    Sequential features: Variables that change by position in sequence (e.g., pressure profiles)
    Global features: Variables constant across all positions (e.g., orbital parameters)
    """
    
    def __init__(
        self, nx_seq: int, nx_global: int, ny: int,
        seq_indices: List[int], global_indices: List[int],
        d_model: int = 256, nhead: int = 8, num_encoder_layers: int = 6,
        dim_feedforward: int = 1024, dropout: float = 0.1, activation: str = "gelu",
        norm_first: bool = True, max_seq_length: int = 512, pos_encoding_type: str = "sine",
        mlp_layers: int = 3, mlp_hidden_dim: Optional[int] = None,
        stochastic_depth_rate: float = 0.0, layer_norm_eps: float = 1e-5,
        integration_method: str = "film", batch_first: bool = True,
        default_input_seq_length: Optional[int] = None,
        default_output_seq_length: Optional[int] = None
    ):
        super().__init__()
        
        self.nx_seq = nx_seq
        self.nx_global = nx_global
        self.ny = ny
        self.d_model = d_model
        self.integration_method = integration_method.lower()
        self.batch_first = batch_first
        
        # Sequence length tracking
        self.input_seq_length = None
        self.output_seq_length = default_output_seq_length
        self.default_input_seq_length = default_input_seq_length or max_seq_length
        
        # Store feature indices for forward pass
        self.seq_indices = seq_indices
        self.global_indices = global_indices
        
        if mlp_hidden_dim is None:
            mlp_hidden_dim = d_model
            
        # Sequential feature projection
        self.seq_proj = nn.Linear(nx_seq, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model, max_seq_length, dropout, pos_encoding_type
        )
        
        # Global feature processing (if any)
        if nx_global > 0:
            # Project global features
            self.global_proj = nn.Sequential(
                nn.Linear(nx_global, d_model),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )
            
            # Integration mechanism
            if integration_method == "film":
                # Feature-wise Linear Modulation
                self.gamma_proj = nn.Linear(d_model, d_model)  # Scaling
                self.beta_proj = nn.Linear(d_model, d_model)   # Shift
            elif integration_method == "concat":
                # Concatenation followed by projection
                self.integration_proj = nn.Linear(d_model * 2, d_model)
            elif integration_method != "add":
                raise ValueError(f"Unknown integration method: {integration_method}")
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
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
        
        # Output projection
        self.output_head = OutputHead(
            d_model=d_model,
            out_features=ny,
            mlp_layers=mlp_layers,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            activation=activation
        )
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Try to compile model if supported
        self._try_compile()
    
    def _initialize_parameters(self):
        """Initialize model parameters with appropriate scaling."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
        
        # Special initialization for FiLM conditioning
        if self.nx_global > 0 and self.integration_method == "film":
            # Initialize gamma close to 1 (identity scaling)
            nn.init.normal_(self.gamma_proj.weight, mean=0.0, std=0.02)
            nn.init.ones_(self.gamma_proj.bias)
            
            # Initialize beta close to 0 (no shift)
            nn.init.normal_(self.beta_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.beta_proj.bias)
        
        # Initialize output layer with smaller weights
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.out_features == self.ny:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _try_compile(self):
        """Try to compile the model using torch.compile if supported."""
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                if any(gpu in device_name for gpu in ["A100", "H100"]):
                    torch._dynamo.config.suppress_errors = True
                    return torch.compile(self)
            except Exception as e:
                logger.warning(f"Compilation failed: {e}")
        return self
    
    def forward(self, src, target_seq_length=None):
        """
        Forward pass integrating sequential and global features.
        
        Args:
            src: Input tensor [batch_size, seq_length, features]
            target_seq_length: Optional output sequence length
            
        Returns:
            outputs: Model outputs [batch_size, seq_length, output_features]
        """
        # Convert to float32 for better numerical stability
        src = src.to(torch.float32)
        batch_size, seq_length = src.shape[0], src.shape[1]
        
        # Update sequence lengths if needed
        current_input_seq_length = seq_length
        if self.input_seq_length != current_input_seq_length:
            self.input_seq_length = current_input_seq_length
            logger.debug(f"Input sequence length set to {self.input_seq_length}")
        
        if target_seq_length is not None:
            if self.output_seq_length != target_seq_length:
                self.output_seq_length = target_seq_length
                logger.debug(f"Output sequence length set to {self.output_seq_length}")
        elif self.output_seq_length is None:
            self.output_seq_length = self.input_seq_length
            logger.debug(f"No target sequence length specified, using input length: {self.output_seq_length}")
        
        # Process according to whether we have global features
        if self.nx_global > 0:
            # Extract features using the stored indices
            x_seq = src[:, :, self.seq_indices]          # Sequential features using indices
            x_global = src[:, 0, self.global_indices]    # Global features from first position
            
            # Process sequential features
            seq_embedding = self.seq_proj(x_seq)
            seq_embedding = self.pos_encoder(seq_embedding)
            
            # Process global features
            global_embedding = self.global_proj(x_global)
            
            # Integrate global and sequential features
            if self.integration_method == "film":
                # FiLM conditioning: Generate scaling (gamma) and shift (beta)
                gamma = self.gamma_proj(global_embedding).unsqueeze(1)  # [batch, 1, d_model]
                beta = self.beta_proj(global_embedding).unsqueeze(1)    # [batch, 1, d_model]
                x = gamma * seq_embedding + beta
            elif self.integration_method == "concat":
                # Expand global embedding to match sequence length
                global_expanded = global_embedding.unsqueeze(1).expand(-1, seq_length, -1)
                # Concatenate and project
                x_combined = torch.cat([seq_embedding, global_expanded], dim=2)
                x = self.integration_proj(x_combined)
            elif self.integration_method == "add":
                # Simple addition of global features to each position
                x = seq_embedding + global_embedding.unsqueeze(1)
            else:
                raise ValueError(f"Unknown integration method: {self.integration_method}")
        else:
            # No global features - just process sequential data
            x_seq = src[:, :, self.seq_indices]
            x = self.seq_proj(x_seq)
            x = self.pos_encoder(x)
        
        # Process through transformer
        transformer_output = self.transformer(x)
        
        # Generate output
        outputs = self.output_head(transformer_output)
        
        return outputs
    
    def count_parameters(self):
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def detect_variable_types(sample_data, input_var_names, rtol=1e-5, atol=1e-8):
    """
    Automatically detect global vs sequential variables from sample data.
    
    A variable is considered 'global' if it has the same value across 
    the entire sequence dimension for ALL samples in the batch.
    
    Parameters
    ----------
    sample_data : torch.Tensor
        A batch of data with shape [batch_size, sequence_length, features]
        or just [sequence_length, features]
    input_var_names : List[str]
        Names of input variables in order
    rtol : float, optional
        Relative tolerance for floating point comparison
    atol : float, optional
        Absolute tolerance for floating point comparison
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'global_vars': List of global variable names
        - 'seq_vars': List of sequential variable names
        - 'global_indices': List of indices for global variables
        - 'seq_indices': List of indices for sequential variables
    """
    # Handle different input shapes
    if sample_data is None:
        logger.warning("No sample data provided, treating all features as sequential")
        global_indices = []
        seq_indices = list(range(len(input_var_names)))
        return {
            'global_vars': [],
            'seq_vars': list(input_var_names),
            'global_indices': global_indices,
            'seq_indices': seq_indices
        }
    
    # Check the dimensionality of the sample data
    if len(sample_data.shape) == 2:
        # Shape is [sequence_length, features], add batch dimension
        logger.info("Sample data has shape [sequence_length, features], adding batch dimension")
        sample_data = sample_data.unsqueeze(0)  # Convert to [1, sequence_length, features]
        batch_size = 1
        seq_length, num_features = sample_data.shape[1:]
    elif len(sample_data.shape) == 3:
        # Shape is already [batch_size, sequence_length, features]
        batch_size, seq_length, num_features = sample_data.shape
    else:
        # Unexpected shape
        logger.warning(f"Unexpected sample data shape: {sample_data.shape}, treating all features as sequential")
        global_indices = []
        seq_indices = list(range(len(input_var_names)))
        return {
            'global_vars': [],
            'seq_vars': list(input_var_names),
            'global_indices': global_indices,
            'seq_indices': seq_indices
        }
    
    # Handle numeric variable names by converting to strings
    if isinstance(input_var_names[0], int):
        input_var_names = [str(idx) for idx in input_var_names]
    
    if len(input_var_names) != num_features:
        logger.warning(f"Number of input variables ({len(input_var_names)}) doesn't match data features ({num_features})")
        # Use a safe fallback
        global_indices = []
        seq_indices = list(range(len(input_var_names)))
        return {
            'global_vars': [],
            'seq_vars': list(input_var_names),
            'global_indices': global_indices,
            'seq_indices': seq_indices
        }
    
    global_vars = []
    seq_vars = []
    global_indices = []
    seq_indices = []
    
    # Examine each feature to see if it's constant across the sequence
    for i, var_name in enumerate(input_var_names):
        # Check if this feature is constant across the sequence dimension for all samples
        feature_data = sample_data[:, :, i]  # [batch_size, seq_length]
        
        # A feature is global if ALL samples have constant values across sequence
        is_global = True
        
        for sample_idx in range(batch_size):
            sample = feature_data[sample_idx]  # [seq_length]
            first_val = sample[0]
            
            # If any value differs from the first, this isn't a global feature
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


def create_prediction_model(config, sample_data=None):
    """Create a prediction model based on configuration."""
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
                logger.warning(f"Adjusted nhead from {old_nhead} to {nhead} to ensure it divides d_model={d_model}")
                break
    
    num_layers = config.get("num_layers", 3)
    dim_feedforward = config.get("dim_feedforward", d_model * 4)
    activation = config.get("activation", "gelu")
    
    # Auto-detect global vs sequence variables if sample data is provided
    if sample_data is not None:
        sample_data = sample_data.float()
        feature_types = detect_variable_types(sample_data, input_variables)
        global_indices = feature_types['global_indices']
        seq_indices = feature_types['seq_indices']
    else:
        # Without sample data, use config or default to all sequential
        global_indices = config.get('global_feature_indices', [])
        seq_indices = config.get('seq_feature_indices', list(range(nx)))
        logger.info(f"Using global indices from config: {global_indices}")
        logger.info(f"Using sequential indices from config: {seq_indices}")

    # Count number of sequential and global features
    nx_seq = len(seq_indices)
    nx_global = len(global_indices)
    
    # Create the model with enhanced features
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
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        integration_method=config.get('integration_method', 'film'),
        stochastic_depth_rate=config.get('stochastic_depth_rate', 0.0),
        norm_first=config.get("norm_first", True),
        max_seq_length=config.get('max_seq_length', 512),
        pos_encoding_type=config.get('pos_encoding_type', 'sine'),
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        mlp_layers=config.get('mlp_layers', 3),
        mlp_hidden_dim=config.get('mlp_hidden_dim'),
        batch_first=True
    )
    
    # Store variable names for reference
    model.nx_names = input_variables
    model.ny_names = target_variables
    
    logger.info(f"Created model with {model.count_parameters():,} parameters")
    
    return model