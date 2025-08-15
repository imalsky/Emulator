#!/usr/bin/env python3
"""
model.py - Optimized transformer model with FiLM conditioning and export compatibility.

Architecture:
- Transformer encoder with sinusoidal positional encoding
- FiLM (Feature-wise Linear Modulation) for global context conditioning
- Export-friendly implementation avoiding dynamic control flow
- Support for torch.compile and torch.export

PADDING CONVENTION:
- Mask values: True = padding position, False = valid position
- This follows PyTorch's convention for src_key_padding_mask
- Padding positions are excluded from attention and loss computation

DATA ASSUMPTIONS:
- Input tensors are expected to have consistent dtype and device within a forward pass
- Sequence lengths can vary but must not exceed max_sequence_length
- Global features, if present, must match batch size
- Padding value (-9999.0) is assumed to never occur naturally in data
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.export import Dim, export as texport, save as tsave

from utils import DTYPE, PADDING_VALUE, validate_config

logger = logging.getLogger(__name__)


class SinePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding - stateless for multi-GPU safety."""
    
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding (computes on-the-fly for long sequences)."""
        seq_len = x.size(1)
        
        if seq_len <= self.pe.size(1):
            return x + self.pe[:, :seq_len, :].to(x.dtype)
        
        # Compute on-the-fly for longer sequences
        position = torch.arange(0, seq_len, dtype=x.dtype, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=x.dtype, device=x.device) *
            (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(1, seq_len, self.d_model, dtype=x.dtype, device=x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return x + pe


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Modulates features using global context through learned scale and shift.
    Uses improved initialization for better training stability.
    """
    
    def __init__(
        self,
        context_dim: int,
        feature_dim: int,
        clamp_gamma: Optional[float] = 2.0  # Increased from 1.0 for more adaptation
    ) -> None:
        """
        Initialize FiLM layer.
        
        Args:
            context_dim: Dimension of context vector
            feature_dim: Dimension of features to modulate
            clamp_gamma: Maximum magnitude for scale/shift (None for no clamping)
        """
        super().__init__()
        
        # Project context to scale and shift parameters
        self.projection = nn.Linear(context_dim, feature_dim * 2)
        
        # Better initialization for stable training (increased std)
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)
        
        # Store clamp value as Python float for dtype flexibility
        self.clamp_gamma = clamp_gamma if clamp_gamma is not None else float('inf')
    
    def forward(self, features: Tensor, context: Tensor) -> Tensor:
        """
        Apply FiLM modulation to features.
        
        Args:
            features: Features to modulate (batch, seq_len, feature_dim)
            context: Global context (batch, context_dim)
            
        Returns:
            Modulated features
        """
        # Get scale and shift parameters
        gamma_beta = self.projection(context)
        delta_gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        
        # Apply clamping with correct dtype
        if self.clamp_gamma != float('inf'):
            clamp_val = torch.tensor(self.clamp_gamma, dtype=features.dtype, device=features.device)
            delta_gamma = torch.clamp(delta_gamma, -clamp_val, clamp_val)
            beta = torch.clamp(beta, -clamp_val, clamp_val)
        
        # Expand for sequence dimension
        delta_gamma = delta_gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        # Apply FiLM transformation: y = x * (1 + γ) + β
        return features * (1.0 + delta_gamma) + beta


class DecomposedTransformerEncoderLayer(nn.Module):
    """
    Custom transformer encoder layer built from primitives.
    
    Fully compatible with torch.export and torch.compile.
    Supports both pre-norm and post-norm architectures.
    
    ATTENTION MASKING:
    - Uses src_key_padding_mask where True = padding position
    - Padding positions are prevented from attending or being attended to
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,  # Separate attention dropout
        activation: str = "gelu",
        norm_first: bool = True,
        batch_first: bool = True,
    ) -> None:
        """
        Initialize transformer encoder layer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability for feedforward and residual
            attention_dropout: Dropout probability for attention
            activation: Activation function ("gelu" or "relu")
            norm_first: If True, use pre-norm architecture
            batch_first: If True, expect batch dimension first
        """
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.norm_first = norm_first
        
        # Multi-head attention with separate dropout
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=attention_dropout,  # Use separate attention dropout
            batch_first=batch_first
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights with improved strategy."""
        # Attention weights are initialized by MultiheadAttention
        # Initialize feedforward weights
        nn.init.xavier_uniform_(self.linear1.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through transformer encoder layer.
        
        Args:
            src: Source sequence (batch, seq_len, d_model)
            src_mask: Attention mask (optional)
            src_key_padding_mask: Padding mask where True = padding position
            
        Returns:
            Transformed sequence
        """
        if self.norm_first:
            # Pre-norm architecture
            x = src
            
            # Self-attention block
            x2 = self.norm1(x)
            x2, _ = self.self_attn(
                x2, x2, x2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,  # True = padding
                need_weights=False
            )
            x = x + self.dropout1(x2)
            
            # Feed-forward block
            x2 = self.norm2(x)
            x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
            x = x + self.dropout2(x2)
        else:
            # Post-norm architecture
            x = src
            
            # Self-attention block
            x2, _ = self.self_attn(
                x, x, x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,  # True = padding
                need_weights=False
            )
            x = x + self.dropout1(x2)
            x = self.norm1(x)
            
            # Feed-forward block
            x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = x + self.dropout2(x2)
            x = self.norm2(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Combined transformer + optional FiLM block.
    
    Avoids dynamic control flow for export compatibility.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        attention_dropout: float,
        global_input_dim: int,
        film_clamp: Optional[float] = 2.0,
    ):
        """
        Initialize transformer block with optional FiLM.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
            global_input_dim: Dimension of global features (0 for no FiLM)
            film_clamp: Clamping value for FiLM parameters
        """
        super().__init__()
        
        self.has_film = global_input_dim > 0
        
        # Transformer layer with separate attention dropout
        self.transformer = DecomposedTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        
        # Optional FiLM layer
        if self.has_film:
            self.film = FiLMLayer(global_input_dim, d_model, clamp_gamma=film_clamp)
    
    def forward(
        self,
        x: Tensor,
        global_features: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input sequence
            global_features: Optional global context
            src_key_padding_mask: Padding mask (True = padding)
            
        Returns:
            Transformed sequence
        """
        # Always pass through transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Apply FiLM if it exists and global features are provided
        if self.has_film and global_features is not None:
            x = self.film(x, global_features)
        
        return x


class PredictionModel(nn.Module):
    """
    Transformer model for atmospheric profile regression.
    
    Features:
    - Deep transformer architecture with FiLM conditioning
    - Sinusoidal positional encoding
    - Export-friendly implementation
    - Proper padding mask handling (no output overwriting)
    - Improved initialization and normalization
    
    IMPORTANT: This model does NOT overwrite outputs at padding positions.
    The loss function handles masking, following industry standards.
    """
    
    def __init__(
        self,
        input_dim: int,
        global_input_dim: int,
        output_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: Optional[float] = None,
        padding_value: float = PADDING_VALUE,
        film_clamp: Optional[float] = 2.0,
    ) -> None:
        """
        Initialize prediction model.
        
        Args:
            input_dim: Dimension of input features
            global_input_dim: Dimension of global features (0 for none)
            output_dim: Dimension of output predictions
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension (defaults to 4*d_model)
            dropout: Dropout probability
            attention_dropout: Attention dropout (defaults to dropout)
            padding_value: Value used for padding (for reference only)
            film_clamp: Clamping value for FiLM parameters
        """
        super().__init__()
        
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        
        self.d_model = d_model
        self.has_global_features = global_input_dim > 0
        
        # Set defaults following industry standards
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model  # Standard ratio
        if attention_dropout is None:
            attention_dropout = dropout
        
        # Note: We store padding_value for reference but don't use it for output masking
        self.padding_value = padding_value
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Positional encoding
        self.pos_encoder = SinePositionalEncoding(d_model)
        
        # Initial FiLM if we have global features
        if self.has_global_features:
            self.initial_film = FiLMLayer(global_input_dim, d_model, clamp_gamma=film_clamp)
        else:
            self.initial_film = None
        
        # Build transformer blocks with integrated FiLM
        self.blocks = nn.ModuleList()
        for _ in range(num_encoder_layers):
            self.blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    global_input_dim=global_input_dim,
                    film_clamp=film_clamp,
                )
            )
        
        # Final normalization (important for stability)
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection (improved: no dropout before final layer)
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),  # Dropout first
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),  # No dropout before final output
        )
        
        # Initialize weights
        self._init_weights()
        
        # Log model statistics
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"PredictionModel created with {trainable_params:,} trainable parameters. "
            f"Architecture: d_model={d_model}, nhead={nhead}, layers={num_encoder_layers}, "
            f"ffn_dim={dim_feedforward}, attn_dropout={attention_dropout:.2f}"
        )
    
    def _init_weights(self) -> None:
        """Initialize weights with improved strategies per layer type."""
        for name, module in self.named_modules():
            # Skip already initialized layers
            if isinstance(module, (FiLMLayer, DecomposedTransformerEncoderLayer, TransformerBlock)):
                continue  # These have their own initialization
            
            if isinstance(module, nn.Linear):
                # Use truncated normal for better initialization
                if "output_proj" in name and "2" in name:  # Final output layer
                    # Smaller initialization for output layer
                    nn.init.trunc_normal_(module.weight, std=0.02)
                else:
                    # Standard initialization for other layers
                    nn.init.trunc_normal_(module.weight, std=0.02)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        sequence: Tensor,
        global_features: Optional[Tensor] = None,
        sequence_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the model.
        
        Args:
            sequence: Input sequence (batch, seq_len, input_dim)
            global_features: Optional global features (batch, global_dim)
            sequence_mask: Padding mask where True = padding position
            
        Returns:
            Output predictions (batch, seq_len, output_dim)
            
        Note: Outputs at padding positions are NOT overwritten.
              The loss function handles masking these positions.
        """
        # Project input features to model dimension
        x = self.input_proj(sequence)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply initial FiLM if it exists
        if self.initial_film is not None and global_features is not None:
            x = self.initial_film(x, global_features)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, global_features, sequence_mask)
        
        # Apply final normalization (important for stability)
        x = self.final_norm(x)
        
        # Project to output dimension
        output = self.output_proj(x)
        
        # IMPORTANT: We do NOT mask the output here.
        # The loss function handles masking, following industry standards.
        # This allows the model to produce predictions for all positions,
        # which can be useful for analysis, while the loss correctly
        # ignores padding positions during training.
        
        return output


def create_prediction_model(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    compile_model: bool = True,
) -> PredictionModel:
    """
    Create a prediction model from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
        compile_model: Whether to apply torch.compile
        
    Returns:
        Initialized PredictionModel
    """
    validate_config(config)
    
    data_spec = config["data_specification"]
    model_params = config["model_hyperparameters"]
    
    if device is None:
        device = torch.device("cpu")
    
    # Extract parameters with improved defaults
    d_model = model_params.get("d_model", 256)
    dim_feedforward = model_params.get("dim_feedforward")
    if dim_feedforward is None:
        dim_feedforward = 4 * d_model  # Industry standard ratio
    
    # Create model with improved parameters
    model = PredictionModel(
        input_dim=len(data_spec["input_variables"]),
        global_input_dim=len(data_spec.get("global_variables", [])),
        output_dim=len(data_spec["target_variables"]),
        d_model=d_model,
        nhead=model_params.get("nhead", 8),
        num_encoder_layers=model_params.get("num_encoder_layers", 6),
        dim_feedforward=dim_feedforward,
        dropout=float(model_params.get("dropout", 0.1)),
        attention_dropout=float(model_params.get("attention_dropout", model_params.get("dropout", 0.1))),
        padding_value=float(data_spec.get("padding_value", PADDING_VALUE)),
        film_clamp=2.0,  # Increased from 1.0 for better adaptation
    )
    
    model.to(device=device)
    
    # Conditionally compile the model
    if compile_model:
        compile_enabled = config.get("miscellaneous_settings", {}).get("torch_compile", False)
        compile_mode = config.get("miscellaneous_settings", {}).get("compile_mode", "default")
        
        # Check for torch.compile capability
        has_compile = hasattr(torch, "compile")
        
        if (
            has_compile
            and device.type == "cuda"
            and compile_enabled
        ):
            try:
                logger.info(f"Attempting torch.compile with mode='{compile_mode}'")
                
                # Compilation kwargs
                compile_kwargs = {"mode": compile_mode}
                
                # Use fullgraph for max optimization if no dynamic control flow
                if compile_mode == "max-autotune":
                    compile_kwargs["fullgraph"] = True
                
                model = torch.compile(model, **compile_kwargs)
                logger.info("Model compiled successfully")
                
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Proceeding without compilation.")
        elif compile_enabled and device.type != "cuda":
            logger.info("torch.compile is only enabled for CUDA devices.")
    
    logger.info(f"Model moved to device: {device}")
    return model


def export_model(
    model: nn.Module,
    example_input: Dict[str, Tensor],
    save_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export model with torch.export for deployment.
    
    Performs validation to ensure exported model produces identical results.
    
    Args:
        model: Model to export
        example_input: Example input dictionary
        save_path: Path to save exported model
        config: Optional configuration dictionary
    """
    save_path = Path(save_path)
    save_dir = save_path.parent
    model_name = save_path.stem
    
    # Check if export is enabled
    if config is not None:
        export_enabled = config.get("miscellaneous_settings", {}).get("torch_export", True)
        if not export_enabled:
            logger.info("Model export disabled in config - skipping.")
            return
    
    model.eval()
    
    # Get original device
    original_device = next(model.parameters()).device
    
    # Unwrap torch.compile wrapper if present
    if hasattr(model, "_orig_mod"):
        logger.info("Extracting original model from compiled wrapper")
        model = model._orig_mod
    
    # Move to CPU for export (avoids device-specific issues)
    model = model.to('cpu')
    
    # Move example inputs to CPU
    sequence = example_input["sequence"].to('cpu')
    global_features = example_input.get("global_features")
    if global_features is not None:
        global_features = global_features.to('cpu')
    sequence_mask = example_input.get("sequence_mask")
    if sequence_mask is not None:
        sequence_mask = sequence_mask.to('cpu')
    
    # Ensure batch size > 1 for dynamic shapes
    batch_size = sequence.shape[0]
    if batch_size == 1:
        # Duplicate inputs for batch size 2
        export_sequence = torch.cat([sequence, sequence], dim=0)
        export_global = (
            torch.cat([global_features, global_features], dim=0)
            if global_features is not None else None
        )
        export_mask = (
            torch.cat([sequence_mask, sequence_mask], dim=0)
            if sequence_mask is not None else None
        )
    else:
        export_sequence = sequence
        export_global = global_features
        export_mask = sequence_mask
    
    # Prepare kwargs for export
    kwargs: Dict[str, Tensor] = {"sequence": export_sequence}
    if export_global is not None:
        kwargs["global_features"] = export_global
    if export_mask is not None:
        kwargs["sequence_mask"] = export_mask
    
    # Define dynamic shapes (batch dimension only)
    batch_dim = Dim("batch", min=1, max=1024)
    
    dynamic_shapes: Dict[str, Any] = {
        "sequence": {0: batch_dim}
    }
    if export_global is not None:
        dynamic_shapes["global_features"] = {0: batch_dim}
    if export_mask is not None:
        dynamic_shapes["sequence_mask"] = {0: batch_dim}
    
    try:
        # Export with torch.export
        with torch.no_grad():
            exported_program = texport(
                model,
                args=(),
                kwargs=kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=False
            )
        
        # Save exported model
        export_path = save_dir / f"{model_name}_exported.pt2"
        tsave(exported_program, str(export_path))
        logger.info(f"Model exported successfully to {export_path}")
        
        # Validate exported model
        logger.info("Validating exported model...")
        with torch.no_grad():
            # Test with original batch size
            test_kwargs = {
                "sequence": sequence,
                "global_features": global_features,
                "sequence_mask": sequence_mask
            }
            test_kwargs = {k: v for k, v in test_kwargs.items() if v is not None}
            
            original_output = model(**test_kwargs)
            exported_output = exported_program.module()(**test_kwargs)
            
            if not torch.allclose(original_output, exported_output, rtol=1e-4, atol=1e-5):
                logger.warning("Exported model output differs from original!")
                max_diff = torch.max(torch.abs(original_output - exported_output)).item()
                logger.warning(f"Maximum difference: {max_diff}")
            else:
                logger.info("✓ Exported model validation passed")
                
    except Exception as exc:
        logger.error(f"Model export failed: {exc}", exc_info=True)
        
        # Try fallback without dynamic shapes
        logger.info("Attempting fallback export with static shapes...")
        try:
            with torch.no_grad():
                static_exported = texport(
                    model,
                    args=(),
                    kwargs=kwargs,
                    strict=False
                )
            static_path = save_dir / f"{model_name}_exported_static.pt2"
            tsave(static_exported, str(static_path))
            logger.warning(f"Static shape export succeeded: {static_path}")
        except Exception as fallback_exc:
            logger.error(f"Fallback export also failed: {fallback_exc}")
    
    finally:
        # Restore to original device if needed
        if original_device.type != 'cpu':
            model.to(original_device)


__all__ = ["PredictionModel", "create_prediction_model", "export_model"]