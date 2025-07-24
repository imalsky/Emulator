#!/usr/bin/env python3
"""
model.py - Optimized transformer model with FiLM conditioning and full export compatibility.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
import math
from packaging import version
from utils import PADDING_VALUE, DTYPE, validate_config
from torch.export import Dim, export as texport, save as tsave
import torch.onnx
from torch.nn.attention import sdpa_kernel, SDPBackend

logger = logging.getLogger(__name__)


class SinePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequential data."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        # Pre-register buffer for better export compatibility
        self.register_buffer('_cached_pe', None, persistent=False)
        self._cached_seq_len = -1

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Cache positional encoding for efficiency and export compatibility
        if self._cached_pe is None or self._cached_seq_len != seq_len:
            position = torch.arange(seq_len, device=x.device, dtype=DTYPE).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, device=x.device, dtype=DTYPE) * 
                (-math.log(10000.0) / self.d_model)
            )
            pe = torch.zeros(1, seq_len, self.d_model, device=x.device, dtype=DTYPE)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('_cached_pe', pe, persistent=False)
            self._cached_seq_len = seq_len
        
        return x + self._cached_pe


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer with near-identity initialization.
    Export-compatible implementation without dynamic branching.
    """

    def __init__(self, context_dim: int, feature_dim: int, clamp_gamma: Optional[float] = 1.0) -> None:
        super().__init__()
        self.projection = nn.Linear(context_dim, feature_dim * 2)
        
        # Near-identity initialization
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.projection.bias)

        # Store clamp value as buffer for export compatibility
        if clamp_gamma is not None:
            self.register_buffer('clamp_value', torch.tensor(clamp_gamma))
        else:
            self.register_buffer('clamp_value', torch.tensor(float('inf')))

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Get scale and shift parameters
        gamma_beta = self.projection(context)
        delta_gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        # Always apply clamping (with inf it becomes no-op)
        delta_gamma = torch.clamp(delta_gamma, -self.clamp_value, self.clamp_value)
        beta = torch.clamp(beta, -self.clamp_value, self.clamp_value)

        # Expand for sequence dimension
        delta_gamma = delta_gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # Apply FiLM transformation
        return features * (1.0 + delta_gamma) + beta


class DecomposedTransformerEncoderLayer(nn.Module):
    """
    Custom decomposed transformer encoder layer built from primitive operations.
    Fully compatible with torch.export and torch.compile.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.norm_first = norm_first
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, 
            nhead, 
            dropout=dropout, 
            batch_first=batch_first
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Activation
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
        """Initialize weights following PyTorch's TransformerEncoderLayer."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
    
    def forward(
        self, 
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Use static branching based on norm_first
        if self.norm_first:
            # Pre-norm architecture
            x = src
            # Self-attention block
            x2 = self.norm1(x)
            x2, _ = self.self_attn(x2, x2, x2, attn_mask=src_mask, 
                                  key_padding_mask=src_key_padding_mask, need_weights=False)
            x = x + self.dropout1(x2)
            
            # Feed-forward block  
            x2 = self.norm2(x)
            x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
            x = x + self.dropout2(x2)
        else:
            # Post-norm architecture
            x = src
            # Self-attention block
            x2, _ = self.self_attn(x, x, x, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask, need_weights=False)
            x = x + self.dropout1(x2)
            x = self.norm1(x)
            
            # Feed-forward block
            x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = x + self.dropout2(x2)
            x = self.norm2(x)
        
        return x


class TransformerBlock(nn.Module):
    """Combined transformer + optional FiLM block to avoid dynamic control flow."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        global_input_dim: int,
        film_clamp: Optional[float] = 1.0,
    ):
        super().__init__()
        self.has_film = global_input_dim > 0
        
        self.transformer = DecomposedTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        
        if self.has_film:
            self.film = FiLMLayer(global_input_dim, d_model, clamp_gamma=film_clamp)
    
    def forward(
        self,
        x: Tensor,
        global_features: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Always pass through transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Apply FiLM if it exists and global features are provided
        # This avoids dynamic branching - if no FiLM layer exists, this is skipped at module level
        if self.has_film and global_features is not None:
            x = self.film(x, global_features)
        
        return x


class PredictionModel(nn.Module):
    """Transformer for atmospheric profile regression with deep FiLM conditioning."""

    def __init__(
        self,
        input_dim: int,
        global_input_dim: int,
        output_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        padding_value: float = PADDING_VALUE,
        film_clamp: Optional[float] = 1.0,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.has_global_features = global_input_dim > 0
        # Store padding value as buffer for export compatibility
        self.register_buffer('padding_value', torch.tensor(padding_value))

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), 
            nn.LayerNorm(d_model), 
            nn.GELU()
        )

        # Positional encoding
        self.pos_encoder = SinePositionalEncoding(d_model)

        # Build transformer blocks with integrated FiLM
        self.blocks = nn.ModuleList()
        
        # Initial FiLM if we have global features
        if self.has_global_features:
            self.initial_film = FiLMLayer(global_input_dim, d_model, clamp_gamma=film_clamp)
        else:
            self.initial_film = None
            
        # Transformer blocks
        for _ in range(num_encoder_layers):
            self.blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    global_input_dim=global_input_dim,
                    film_clamp=film_clamp,
                )
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

        self._init_weights()

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"PredictionModel created with {trainable_params:,} trainable parameters. "
            f"Architecture: d_model={d_model}, nhead={nhead}, layers={num_encoder_layers}"
        )

    def _init_weights(self) -> None:
        """Initialize weights for all layers except FiLM and Transformer blocks."""
        for module in self.modules():
            if isinstance(module, (FiLMLayer, DecomposedTransformerEncoderLayer, TransformerBlock)):
                continue  # These have their own initialization
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        sequence: Tensor,
        global_features: Optional[Tensor] = None,
        sequence_mask: Optional[Tensor] = None,
    ) -> Tensor:
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

        # Project to output dimension
        output = self.output_proj(x)

        # Mask padding positions
        if sequence_mask is not None:
            # Create mask for output dimension
            output_mask = sequence_mask.unsqueeze(-1).expand_as(output)
            output = torch.where(output_mask, self.padding_value, output)

        return output


def create_prediction_model(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    compile_model: bool = True,
) -> PredictionModel:
    """Create a prediction model with optional compilation."""
    validate_config(config)

    data_spec = config["data_specification"]
    model_params = config["model_hyperparameters"]

    if device is None:
        device = torch.device("cpu")

    model = PredictionModel(
        input_dim=len(data_spec["input_variables"]),
        global_input_dim=len(data_spec.get("global_variables", [])),
        output_dim=len(data_spec["target_variables"]),
        d_model=model_params.get("d_model", 256),
        nhead=model_params.get("nhead", 8),
        num_encoder_layers=model_params.get("num_encoder_layers", 6),
        dim_feedforward=model_params.get("dim_feedforward", 1024),
        dropout=float(model_params.get("dropout", 0.1)),
        padding_value=float(data_spec.get("padding_value", PADDING_VALUE)),
        film_clamp=1.0,
    )

    model.to(device=device)

    # Conditionally compile the model
    if compile_model:
        compile_enabled = config.get("miscellaneous_settings", {}).get("torch_compile", False)
        compile_mode = config.get("miscellaneous_settings", {}).get("compile_mode", "default")

        if (
            version.parse(torch.__version__) >= version.parse("2.0.0")
            and device.type == "cuda"
            and compile_enabled
        ):
            try:
                logger.info(f"Attempting torch.compile with mode='{compile_mode}'")
                # Add fullgraph=True for better optimization if no dynamic control flow
                compile_kwargs = {"mode": compile_mode}
                # Only use fullgraph if we're confident there are no graph breaks
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
    Export model with torch.export, ensuring compatibility and correctness.
    Performs validation to ensure exported model produces identical results.
    """
    save_path = Path(save_path)
    save_dir = save_path.parent
    model_name = save_path.stem

    if config is not None and not config.get("miscellaneous_settings", {}).get("torch_export", True):
        logger.info("Model export disabled in config – skipping.")
        return

    model.eval()

    # Always export from CPU to avoid device-specific issues
    original_device = next(model.parameters()).device
    model = model.to('cpu')

    # Unwrap torch.compile wrapper if present
    if hasattr(model, "_orig_mod"):
        logger.info("Extracting original model from compiled wrapper")
        model = model._orig_mod

    # Move example inputs to CPU
    sequence = example_input["sequence"].to('cpu')
    global_features = example_input.get("global_features")
    if global_features is not None:
        global_features = global_features.to('cpu')
    sequence_mask = example_input.get("sequence_mask")
    if sequence_mask is not None:
        sequence_mask = sequence_mask.to('cpu')

    # Prepare inputs for export (ensure batch size > 1 for dynamic shapes)
    batch_size = sequence.shape[0]
    if batch_size == 1:
        # Duplicate inputs to create batch size 2
        export_sequence = torch.cat([sequence, sequence], dim=0)
        export_global = torch.cat([global_features, global_features], dim=0) if global_features is not None else None
        export_mask = torch.cat([sequence_mask, sequence_mask], dim=0) if sequence_mask is not None else None
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

    # Define dynamic shapes
    batch_dim = Dim("batch", min=1, max=1024)  # Reasonable max batch size
    
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
            # Use strict=False to allow some flexibility in tracing
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
        # Restore model to original device
        model.to(original_device)


__all__ = ["PredictionModel", "create_prediction_model", "export_model"]