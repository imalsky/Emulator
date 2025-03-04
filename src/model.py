#!/usr/bin/env python3
"""
sequence_model.py

Neural network models for predicting sequence data from input profiles.
Optimized for ultra-large output sequences (up to 40,000 points)
with memory-efficient architecture and processing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def is_mps_device(device):
    """Check if device is Apple Silicon MPS."""
    return device.type == 'mps' or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence models."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1, encoding_type="sine"):
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
            self.register_buffer('pe', pe.unsqueeze(0))
            
        elif self.encoding_type == "learned":
            # Learned positional embeddings
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pe, std=0.02)
            
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}. Use 'sine' or 'learned'.")

    def forward(self, x):
        """Add positional encoding to input."""
        return self.dropout(x + self.pe[:, :x.size(1)])


class OptimizedLargeSequenceModel(nn.Module):
    """
    Highly optimized model for ultra-large output sequences (>30,000 points).
    Uses a minimalist architecture with progressive dimension reduction and
    aggressive memory management for maximum efficiency.
    """
    
    def __init__(
        self,
        nx,                      # Number of input features
        ny,                      # Number of output features (typically 1)
        output_seq_length,       # Length of output sequence
        input_seq_length=100,    # Length of input sequence
        hidden_dim=128,          # Hidden dimension size (reduced for memory efficiency)
        num_layers=2,            # Number of hidden layers
        dropout=0.1,             # Dropout probability
        activation="relu",       # Activation function
        chunk_size=100,          # Process this many output points at once (very small chunks)
        use_coordinate=True,     # Whether to use coordinate as input to the MLP
        bottleneck_factor=4      # Additional compression factor for progressive reduction
    ):
        super().__init__()
        
        self.nx = nx
        self.ny = ny
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.chunk_size = min(chunk_size, 100)  # Force very small chunks for ultra-large sequences
        self.use_coordinate = use_coordinate
        
        logger.info(f"Initializing OptimizedLargeSequenceModel for {output_seq_length} output points")
        logger.info(f"Using extremely small chunk size: {self.chunk_size} for maximum memory efficiency")
        
        # Efficient profile encoder with layer normalization for better training stability
        # Modified to handle the flattened input size correctly
        self.profile_encoder = nn.Sequential(
            # No need for nn.Flatten() - we'll do reshape in forward method
            nn.Linear(nx * input_seq_length, hidden_dim * 2),  # Fix: input dimension is nx * input_seq_length
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Helps with training stability
            self._get_activation(activation)
        )
        
        # Memory-efficient decoder with progressive dimension reduction
        sequence_input_dim = hidden_dim + 1 if use_coordinate else hidden_dim
        
        decoder_layers = []
        current_dim = sequence_input_dim
        
        for i in range(num_layers):
            # Progressive dimension reduction for memory efficiency
            output_dim = hidden_dim // (bottleneck_factor if i > 0 else 1)
            decoder_layers.extend([
                nn.Linear(current_dim, output_dim),
                nn.LayerNorm(output_dim) if i < num_layers-1 else nn.Identity(),
                self._get_activation(activation) if i < num_layers-1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_layers-1 else nn.Identity()
            ])
            current_dim = output_dim
        
        # Final output layer
        decoder_layers.append(nn.Linear(current_dim, ny))
        
        self.sequence_decoder = nn.Sequential(*decoder_layers)
        
        # Initialize parameters with careful scaling
        self._initialize_parameters()
        
        logger.info(f"Model initialized with {self.count_parameters():,} parameters")
        
    def _get_activation(self, activation):
        """Get activation function by name."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU()
        else:
            return nn.ReLU()
    
    def _initialize_parameters(self):
        """Initialize model parameters with careful scaling for better training dynamics."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # Initialize output layer with smaller weights
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.out_features == self.ny:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def generate_default_coordinates(self, batch_size, device):
        """Generate default coordinates when none are provided."""
        # Create normalized coordinates from 0 to 1
        coords = torch.linspace(0, 1, self.output_seq_length, device=device)
        # Expand to batch dimension if needed
        if batch_size > 1:
            coords = coords.unsqueeze(0).expand(batch_size, -1)
        return coords
    
    def forward(self, src, coordinates=None):
        """
        Ultra-memory-efficient forward pass for very large sequences.
        Uses mixed precision and aggressive chunking to handle 40k+ sequences.
        
        Parameters
        ----------
        src : torch.Tensor
            Input tensor of profile [batch_size, input_seq_length, nx]
        coordinates : torch.Tensor, optional
            Coordinate values for output points [batch_size, output_seq_length] or 
            [output_seq_length] (will be broadcasted)
            
        Returns
        -------
        torch.Tensor
            Output sequence tensor [batch_size, output_seq_length, ny]
        """
        batch_size = src.shape[0]
        
        # Manually reshape input to [batch_size, input_seq_length * nx]
        src_flat = src.reshape(batch_size, -1)
        
        # Process profile - disable mixed precision on MPS
        if hasattr(src, 'device') and is_mps_device(src.device):
            # MPS doesn't fully support autocast - run without it
            profile_repr = self.profile_encoder(src_flat)
        else:
            # Use autocast for CUDA or CPU
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
                profile_repr = self.profile_encoder(src_flat)
        
        # Generate default coordinates if needed and use_coordinate is True
        if self.use_coordinate and coordinates is None:
            logger.info("No coordinates provided, generating default normalized coordinates (0-1)")
            coordinates = self.generate_default_coordinates(batch_size, src.device)
        
        # Check coordinates dimensions and broadcast if needed
        if self.use_coordinate and coordinates is not None:
            if coordinates.dim() == 1:
                # Convert [output_seq_length] to [1, output_seq_length] for broadcasting
                coordinates = coordinates.unsqueeze(0)
            
            # Ensure coordinates match batch size
            if coordinates.size(0) == 1 and batch_size > 1:
                # Broadcast to match batch size
                coordinates = coordinates.expand(batch_size, -1)
        
        # Process in extremely small chunks to minimize memory usage
        output_chunks = []
        
        for chunk_start in range(0, self.output_seq_length, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.output_seq_length)
            chunk_size = chunk_end - chunk_start
            
            # Create embeddings for this chunk
            chunk_input = profile_repr.unsqueeze(1).expand(-1, chunk_size, -1)  # [batch_size, chunk_size, hidden_dim]
            
            # Add coordinates if using them and they're available
            if self.use_coordinate and coordinates is not None:
                coord_chunk = coordinates[:, chunk_start:chunk_end].unsqueeze(-1)
                chunk_input = torch.cat([chunk_input, coord_chunk], dim=-1)
            
            # Process chunk - disable mixed precision on MPS
            if hasattr(src, 'device') and is_mps_device(src.device):
                # MPS doesn't support autocast - run without it
                flat_input = chunk_input.reshape(-1, chunk_input.size(-1))
                flat_output = self.sequence_decoder(flat_input)
                chunk_output = flat_output.reshape(batch_size, chunk_size, self.ny)
            else:
                # Use autocast for CUDA or CPU
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
                    flat_input = chunk_input.reshape(-1, chunk_input.size(-1))
                    flat_output = self.sequence_decoder(flat_input)
                    chunk_output = flat_output.reshape(batch_size, chunk_size, self.ny)
            
            output_chunks.append(chunk_output)
            
            # Explicitly free memory
            del chunk_input, flat_input, flat_output
            if self.use_coordinate and coordinates is not None:
                del coord_chunk
            
            # Force garbage collection on CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate chunks to get full output sequence
        return torch.cat(output_chunks, dim=1)
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LargeSequenceModel(nn.Module):
    """
    Specialized model for very large output sequences (>10k points).
    Uses a simpler architecture with minimal memory footprint.
    """
    
    def __init__(
        self,
        nx,                      # Number of input features
        ny,                      # Number of output features (typically 1)
        output_seq_length,       # Length of output sequence
        input_seq_length=100,    # Length of input sequence 
        hidden_dim=128,          # Hidden dimension size
        num_layers=2,            # Number of layers
        dropout=0.1,             # Dropout probability
        activation="relu",       # Activation function
        chunk_size=250,          # Process this many output points at once (smaller default)
        use_coordinate=True      # Whether to use coordinate as input to the MLP
    ):
        super().__init__()
        
        self.nx = nx
        self.ny = ny
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.chunk_size = min(chunk_size, 250)  # Enforce smaller chunks for large sequences
        self.use_coordinate = use_coordinate
        
        logger.info(f"Initializing LargeSequenceModel for {output_seq_length} output points")
        
        # Simple embedding network for the input profile
        self.profile_encoder = nn.Sequential(
            # No need for nn.Flatten() - we'll do reshape in forward method
            nn.Linear(nx * input_seq_length, hidden_dim * 2),  # Fix: input dimension is nx * input_seq_length
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Added layer normalization for better training stability
            self._get_activation(activation)
        )
        
        # Simple decoder for the output sequence
        sequence_input_dim = hidden_dim + 1 if use_coordinate else hidden_dim
        
        # Smaller decoder for memory efficiency
        self.sequence_decoder = nn.Sequential(
            nn.Linear(sequence_input_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ny)
        )
        
        # Initialize parameters with careful scaling
        self._initialize_parameters()
        
        logger.info(f"Model initialized with {self.count_parameters():,} parameters")
    
    def _get_activation(self, activation):
        """Get activation function by name."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU()
        else:
            return nn.ReLU()
    
    def _initialize_parameters(self):
        """Initialize model parameters with careful scaling for better training dynamics."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # Initialize output layer with smaller weights
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.out_features == self.ny:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def generate_default_coordinates(self, batch_size, device):
        """Generate default coordinates when none are provided."""
        # Create normalized coordinates from 0 to 1
        coords = torch.linspace(0, 1, self.output_seq_length, device=device)
        # Expand to batch dimension if needed
        if batch_size > 1:
            coords = coords.unsqueeze(0).expand(batch_size, -1)
        return coords
    
    def forward(self, src, coordinates=None):
        """
        Forward pass optimized for very large output sequences.
        
        Parameters
        ----------
        src : torch.Tensor
            Input tensor of profile [batch_size, input_seq_length, nx]
        coordinates : torch.Tensor, optional
            Coordinate values for output points [batch_size, output_seq_length] or
            [output_seq_length] (will be broadcasted)
            
        Returns
        -------
        torch.Tensor
            Output sequence tensor [batch_size, output_seq_length, ny]
        """
        batch_size = src.shape[0]
        
        # Manually reshape input to [batch_size, input_seq_length * nx]
        src_flat = src.reshape(batch_size, -1)
        
        # Get profile embedding
        profile_repr = self.profile_encoder(src_flat)  # [batch_size, hidden_dim]
        
        # Generate default coordinates if needed and use_coordinate is True
        if self.use_coordinate and coordinates is None:
            logger.info("No coordinates provided, generating default normalized coordinates (0-1)")
            coordinates = self.generate_default_coordinates(batch_size, src.device)
        
        # Check coordinates dimensions and broadcast if needed
        if self.use_coordinate and coordinates is not None:
            if coordinates.dim() == 1:
                # Convert [output_seq_length] to [1, output_seq_length] for broadcasting
                coordinates = coordinates.unsqueeze(0)
            
            # Ensure coordinates match batch size
            if coordinates.size(0) == 1 and batch_size > 1:
                # Broadcast to match batch size
                coordinates = coordinates.expand(batch_size, -1)
        
        # Process in very small chunks to minimize memory usage
        output_chunks = []
        
        for chunk_start in range(0, self.output_seq_length, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.output_seq_length)
            chunk_size = chunk_end - chunk_start
            
            # Create embeddings for this chunk
            chunk_input = profile_repr.unsqueeze(1).expand(-1, chunk_size, -1)  # [batch_size, chunk_size, hidden_dim]
            
            # Add coordinates if available
            if self.use_coordinate and coordinates is not None:
                coord_chunk = coordinates[:, chunk_start:chunk_end].unsqueeze(-1)
                chunk_input = torch.cat([chunk_input, coord_chunk], dim=-1)
            
            # Process chunk efficiently
            flat_input = chunk_input.reshape(-1, chunk_input.size(-1))
            flat_output = self.sequence_decoder(flat_input)
            chunk_output = flat_output.reshape(batch_size, chunk_size, self.ny)
            
            output_chunks.append(chunk_output)
            
            # Explicitly free memory
            del chunk_input, flat_input, flat_output, chunk_output
            if self.use_coordinate and coordinates is not None:
                del coord_chunk
            
            # Force garbage collection on CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate chunks to get full output sequence
        return torch.cat(output_chunks, dim=1)
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SequenceTransformerModel(nn.Module):
    """
    Memory-efficient model for predicting very large output sequences from input profiles.
    Uses a transformer encoder for input profiles and a memory-efficient MLP for output sequence.
    """
    
    def __init__(
        self,
        nx,                      # Number of input features
        ny,                      # Number of output features (typically 1)
        output_seq_length,       # Length of output sequence
        d_model=64,              # Hidden dimension size (reduced default for large sequences)
        nhead=4,                 # Number of attention heads
        num_encoder_layers=2,    # Number of encoder layers (reduced default)
        dim_feedforward=256,     # Feedforward dimension (reduced default)
        dropout=0.1,             # Dropout probability
        activation="gelu",       # Activation function
        mlp_layers=2,            # Output MLP layers
        chunk_size=500,          # Process this many output points at once (reduced default)
        use_coordinate=False,    # Whether to use coordinate as input to the MLP
        use_mixed_precision=False # Whether to use mixed precision for prediction
    ):
        super().__init__()
        
        self.nx = nx
        self.ny = ny
        self.d_model = d_model
        self.output_seq_length = output_seq_length
        # Hard limit on chunk size for memory efficiency
        self.chunk_size = min(chunk_size, 500)  
        self.use_coordinate = use_coordinate
        self.use_mixed_precision = use_mixed_precision
        
        logger.info(f"Initializing SequenceTransformerModel with output sequence length: {output_seq_length}")
        logger.info(f"Using chunk size: {self.chunk_size} for memory efficiency")
        
        # Input projection and positional encoding
        self.input_proj = nn.Linear(nx, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, encoding_type="sine")
        
        # Transformer encoder (smaller than original)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # The sequence MLP takes the encoded profile + coordinate and predicts the value
        mlp_input_dim = d_model
        if use_coordinate:
            mlp_input_dim += 1  # Add dimension for coordinate
        
        # Build MLP layers (more efficient architecture for large sequences)
        mlp_layers_list = []
        current_dim = mlp_input_dim
        
        for i in range(mlp_layers):
            if i == 0:
                mlp_layers_list.append(nn.Linear(current_dim, d_model))
                current_dim = d_model
            else:
                output_dim = d_model // 2
                mlp_layers_list.append(nn.Linear(current_dim, output_dim))
                current_dim = output_dim
                
            # Add activation, normalization and dropout for all but last layer
            if i < mlp_layers - 1:
                if activation == "relu":
                    mlp_layers_list.append(nn.ReLU())
                elif activation == "gelu":
                    mlp_layers_list.append(nn.GELU())
                else:
                    mlp_layers_list.append(nn.SiLU())
                mlp_layers_list.append(nn.LayerNorm(current_dim))  # Added normalization
                mlp_layers_list.append(nn.Dropout(dropout))
        
        # Final output layer
        mlp_layers_list.append(nn.Linear(current_dim, ny))
        
        self.sequence_mlp = nn.Sequential(*mlp_layers_list)
        
        # Initialize parameters
        self._initialize_parameters()
        
        logger.info(f"Model initialized with {self.count_parameters():,} parameters")
        
    def _initialize_parameters(self):
        """Initialize model parameters for better training dynamics."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # Initialize output layer with smaller weights
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.out_features == self.ny:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def generate_default_coordinates(self, batch_size, device):
        """Generate default coordinates when none are provided."""
        # Create normalized coordinates from 0 to 1
        coords = torch.linspace(0, 1, self.output_seq_length, device=device)
        # Expand to batch dimension if needed
        if batch_size > 1:
            coords = coords.unsqueeze(0).expand(batch_size, -1)
        return coords
                
    def forward(self, src, coordinates=None):
        """
        Forward pass that efficiently handles extremely long output sequences.
        
        Parameters
        ----------
        src : torch.Tensor
            Input tensor of profile [batch_size, input_seq_length, nx]
        coordinates : torch.Tensor, optional
            Coordinate values for output points [batch_size, output_seq_length] or
            [output_seq_length] (will be broadcasted)
            
        Returns
        -------
        torch.Tensor
            Output sequence tensor [batch_size, output_seq_length, ny]
        """
        batch_size = src.shape[0]
        
        # Generate default coordinates if needed and use_coordinate is True
        if self.use_coordinate and coordinates is None:
            logger.info("No coordinates provided, generating default normalized coordinates (0-1)")
            coordinates = self.generate_default_coordinates(batch_size, src.device)
            
        # Use mixed precision if requested - but disable on MPS
        if self.use_mixed_precision and src.device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', enabled=True):
                # Project and encode the input profile
                src = self.input_proj(src)
                src = self.pos_encoder(src)
                
                # Pass through the transformer encoder
                encoded_profile = self.transformer_encoder(src)
                
                # Calculate profile representation (mean across sequence dimension)
                profile_repr = encoded_profile.mean(dim=1)  # [batch_size, d_model]
        else:
            # Standard precision (always use this for MPS)
            src = self.input_proj(src)
            src = self.pos_encoder(src)
            encoded_profile = self.transformer_encoder(src)
            profile_repr = encoded_profile.mean(dim=1)
        
        # Check coordinates dimensions and broadcast if needed
        if self.use_coordinate and coordinates is not None:
            if coordinates.dim() == 1:
                # Convert [output_seq_length] to [1, output_seq_length] for broadcasting
                coordinates = coordinates.unsqueeze(0)
            
            # Ensure coordinates match batch size
            if coordinates.size(0) == 1 and batch_size > 1:
                # Broadcast to match batch size
                coordinates = coordinates.expand(batch_size, -1)
        
        # Generate output sequence in small chunks to be memory efficient
        output_chunks = []
        
        # Process in chunks with explicit memory management
        for chunk_start in range(0, self.output_seq_length, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.output_seq_length)
            chunk_size = chunk_end - chunk_start
            
            # Create input for MLP by repeating profile representation
            chunk_input = profile_repr.unsqueeze(1).expand(-1, chunk_size, -1)  # [batch_size, chunk_size, d_model]
            
            # If using coordinates, concatenate them to the input
            if self.use_coordinate and coordinates is not None:
                coord_chunk = coordinates[:, chunk_start:chunk_end].unsqueeze(-1)  # [batch_size, chunk_size, 1]
                chunk_input = torch.cat([chunk_input, coord_chunk], dim=-1)
            
            # Reshape for the MLP: [batch_size * chunk_size, d_model(+1)]
            flat_input = chunk_input.reshape(-1, chunk_input.size(-1))
            
            # Use mixed precision if requested - but disable on MPS
            if self.use_mixed_precision and src.device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    flat_output = self.sequence_mlp(flat_input)
            else:
                flat_output = self.sequence_mlp(flat_input)
            
            # Reshape back to [batch_size, chunk_size, ny]
            chunk_output = flat_output.reshape(batch_size, chunk_size, self.ny)
            output_chunks.append(chunk_output)
            
            # Clear memory if needed
            del chunk_input, flat_input, flat_output, chunk_output
            if self.use_coordinate and coordinates is not None:
                del coord_chunk
            
            # Force garbage collection on CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate chunks to get full output sequence
        return torch.cat(output_chunks, dim=1)
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Renamed from SpectralMLPModel
class MLPModel(nn.Module):
    """
    A simpler MLP-based model for sequence prediction.
    More efficient for cases where the relationship between profiles and output is direct.
    """
    
    def __init__(
        self,
        nx,                      # Number of input features
        ny,                      # Number of output features (typically 1)
        output_seq_length,       # Length of output sequence
        input_seq_length=100,    # Length of input sequence
        hidden_dim=256,          # Hidden dimension size (reduced for large sequences)
        num_layers=2,            # Number of hidden layers (reduced)
        dropout=0.1,             # Dropout probability
        activation="relu",       # Activation function
        chunk_size=500,          # Process this many output points at once (reduced)
        use_coordinate=True      # Whether to use coordinate as input to the MLP
    ):
        super().__init__()
        
        self.nx = nx
        self.ny = ny
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.chunk_size = min(chunk_size, 500)  # Enforce smaller chunk size
        self.use_coordinate = use_coordinate
        
        # Simplified architecture with layer normalization for better stability
        self.profile_encoder = nn.Sequential(
            # No need for nn.Flatten() - we'll do reshape in forward method
            nn.Linear(nx * input_seq_length, hidden_dim * 2),  # Fix: input dimension is nx * input_seq_length
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Added layer normalization
            self._get_activation(activation),
            nn.Dropout(dropout)
        )
        
        # MLP for sequence prediction
        mlp_input_dim = hidden_dim + 1 if use_coordinate else hidden_dim
        
        decoder_layers = []
        current_dim = mlp_input_dim
        
        for i in range(num_layers):
            if i == 0:
                decoder_layers.append(nn.Linear(current_dim, hidden_dim))
                current_dim = hidden_dim
            else:
                decoder_layers.append(nn.Linear(current_dim, hidden_dim))
                
            decoder_layers.append(self._get_activation(activation))
            
            # Add layer normalization for stability
            if i < num_layers - 1:
                decoder_layers.append(nn.LayerNorm(hidden_dim))
                
            decoder_layers.append(nn.Dropout(dropout))
        
        # Final output layer
        decoder_layers.append(nn.Linear(current_dim, ny))
        
        self.sequence_decoder = nn.Sequential(*decoder_layers)
        
        # Initialize parameters
        self._initialize_parameters()
        
        logger.info(f"Model initialized with {self.count_parameters():,} parameters")
    
    def _get_activation(self, activation):
        """Get activation function by name."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU()
        else:
            return nn.ReLU()
        
    def _initialize_parameters(self):
        """Initialize model parameters for better training dynamics."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # Initialize output layer with smaller weights
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.out_features == self.ny:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def generate_default_coordinates(self, batch_size, device):
        """Generate default coordinates when none are provided."""
        # Create normalized coordinates from 0 to 1
        coords = torch.linspace(0, 1, self.output_seq_length, device=device)
        # Expand to batch dimension if needed
        if batch_size > 1:
            coords = coords.unsqueeze(0).expand(batch_size, -1)
        return coords
    
    def forward(self, src, coordinates=None):
        """Memory-efficient forward pass."""
        batch_size = src.shape[0]
        
        # Manually reshape input to [batch_size, input_seq_length * nx]
        src_flat = src.reshape(batch_size, -1)
        
        profile_repr = self.profile_encoder(src_flat)  # [batch_size, hidden_dim]
        
        # Generate default coordinates if needed and use_coordinate is True
        if self.use_coordinate and coordinates is None:
            logger.info("No coordinates provided, generating default normalized coordinates (0-1)")
            coordinates = self.generate_default_coordinates(batch_size, src.device)
        
        # Check coordinates dimensions and broadcast if needed
        if self.use_coordinate and coordinates is not None:
            if coordinates.dim() == 1:
                # Convert [output_seq_length] to [1, output_seq_length] for broadcasting
                coordinates = coordinates.unsqueeze(0)
            
            # Ensure coordinates match batch size
            if coordinates.size(0) == 1 and batch_size > 1:
                # Broadcast to match batch size
                coordinates = coordinates.expand(batch_size, -1)
        
        output_chunks = []
        
        for chunk_start in range(0, self.output_seq_length, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.output_seq_length)
            chunk_size = chunk_end - chunk_start
            
            # Create embeddings for this chunk
            chunk_input = profile_repr.unsqueeze(1).expand(-1, chunk_size, -1)
            
            # Add coordinates if available
            if self.use_coordinate and coordinates is not None:
                coord_chunk = coordinates[:, chunk_start:chunk_end].unsqueeze(-1)
                chunk_input = torch.cat([chunk_input, coord_chunk], dim=-1)
            
            # Process chunk
            flat_input = chunk_input.reshape(-1, chunk_input.size(-1))
            flat_output = self.sequence_decoder(flat_input)
            chunk_output = flat_output.reshape(batch_size, chunk_size, self.ny)
            
            output_chunks.append(chunk_output)
            
            # Clean up memory
            del chunk_input, flat_input, flat_output, chunk_output
            if self.use_coordinate and coordinates is not None:
                del coord_chunk
            
            # Force garbage collection on CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return torch.cat(output_chunks, dim=1)
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_prediction_model(config):
    """
    Create a sequence prediction model based on configuration.
    For extremely large output sequences (>30k), selects the ultra-optimized architecture.
    
    Parameters
    ----------
    config : dict
        Model configuration dictionary
    
    Returns
    -------
    nn.Module
        Initialized sequence prediction model
    """
    logger.info("Creating prediction model")
    
    # Extract required parameters
    nx = len(config.get("input_variables", []))
    ny = len(config.get("target_variables", []))
    output_seq_length = config.get("output_seq_length", 2000)
    
    # Extract or infer input sequence length
    input_seq_length = config.get("input_seq_length", 100)  # Default to 100 based on logs
    
    # For extremely large output sequences, adjust the model type automatically
    if output_seq_length > 30000:
        logger.info(f"Ultra-large output sequence detected ({output_seq_length} points)")
        logger.info("Switching to OptimizedLargeSequenceModel for maximum memory efficiency")
        model_type = "optimized_large_sequence"
    elif output_seq_length > 10000:
        logger.info(f"Very large output sequence detected ({output_seq_length} points)")
        if config.get("model_type", "").lower() != "large_sequence":
            logger.info("Switching to LargeSequenceModel for better memory efficiency")
            model_type = "large_sequence"
        else:
            model_type = "large_sequence"
    else:
        model_type = config.get("model_type", "transformer").lower()
    
    # Check if any coordinate variables are defined
    use_coordinate = len(config.get("coordinate_variable", [])) > 0
    
    # Choose appropriate model based on sequence length and model type
    if model_type == "optimized_large_sequence":
        # Ultra-optimized model for extremely large sequences
        model = OptimizedLargeSequenceModel(
            nx=nx,
            ny=ny,
            output_seq_length=output_seq_length,
            input_seq_length=input_seq_length,
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "relu"),
            chunk_size=config.get("chunk_size", 100),
            use_coordinate=use_coordinate,
            bottleneck_factor=config.get("bottleneck_factor", 4)
        )
    elif model_type == "large_sequence":
        # Specialized model for very large sequences
        model = LargeSequenceModel(
            nx=nx,
            ny=ny,
            output_seq_length=output_seq_length,
            input_seq_length=input_seq_length,
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "relu"),
            chunk_size=config.get("chunk_size", 250),
            use_coordinate=use_coordinate
        )
    elif model_type == "mlp":
        # Standard MLP model (adjusted for large sequences)
        model = MLPModel(
            nx=nx,
            ny=ny,
            output_seq_length=output_seq_length,
            input_seq_length=input_seq_length,
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "relu"),
            chunk_size=config.get("chunk_size", 500),
            use_coordinate=use_coordinate
        )
    else:
        # Default transformer model with memory optimizations
        model = SequenceTransformerModel(
            nx=nx,
            ny=ny,
            output_seq_length=output_seq_length,
            d_model=config.get("d_model", 64),
            nhead=config.get("nhead", 4),
            num_encoder_layers=config.get("num_encoder_layers", 2),
            dim_feedforward=config.get("dim_feedforward", 256),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "gelu"),
            mlp_layers=config.get("mlp_layers", 2),
            chunk_size=config.get("chunk_size", 500),
            use_coordinate=use_coordinate,
            use_mixed_precision=config.get("use_mixed_precision", False)
        )
    
    logger.info(f"Created {model_type} model with {model.count_parameters():,} parameters")
    if use_coordinate:
        logger.info(f"Model will use coordinate variable(s) from: {config.get('coordinate_variable', [])}")
    
    return model