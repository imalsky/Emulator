#!/usr/bin/env python3
"""
model.py - Optimized transformer model with FiLM conditioning.
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

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device, dtype=DTYPE).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device, dtype=DTYPE) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(1, seq_len, self.d_model, device=x.device, dtype=DTYPE)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return x + pe


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer with near-identity initialization and optional clamping for stability.
    Based on best practices from the original FiLM paper (Perez et al., 2018): Initialize to produce values near identity,
    allow gamma to take negative and large values, but add optional soft clamping to prevent explosions during early training.
    """

    def __init__(self, context_dim: int, feature_dim: int, clamp_gamma: Optional[float] = 1.0) -> None:
        super().__init__()
        # Single projection that outputs both scale and shift
        self.projection = nn.Linear(context_dim, feature_dim * 2)
        
        # Safer near-identity initialization: Small normal for weights (std=0.01 for slight randomness without explosion),
        # zeros for bias. This ensures delta_gamma ~0, beta~0 at start.
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.projection.bias)

        self.clamp_gamma = clamp_gamma  # If None, no clamping; otherwise clamp delta_gamma to [-clamp, clamp]

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Get scale and shift parameters (delta_gamma and beta)
        gamma_beta = self.projection(context)
        delta_gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        # Optional clamping for stability (soft bound to prevent extreme scaling early on)
        if self.clamp_gamma is not None:
            delta_gamma = torch.clamp(delta_gamma, -self.clamp_gamma, self.clamp_gamma)
            beta = torch.clamp(beta, -self.clamp_gamma, self.clamp_gamma)

        # Expand for sequence dimension (broadcastable)
        delta_gamma = delta_gamma.unsqueeze(1)  # [batch, 1, feature_dim]
        beta = beta.unsqueeze(1)                # [batch, 1, feature_dim]

        # Apply with residual formulation: Equivalent to gamma * features + beta where gamma = 1 + delta_gamma
        # This allows identity at init and preserves ability for negative/large gamma as per FiLM paper.
        return features * (1 + delta_gamma) + beta


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
        self.padding_value = padding_value
        self.has_global_features = global_input_dim > 0

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), 
            nn.LayerNorm(d_model), 
            nn.GELU()
        )

        # Positional encoding
        self.pos_encoder = SinePositionalEncoding(d_model)

        # Build transformer with interleaved FiLM conditioning
        self.layers = nn.ModuleList()
        
        # Add initial FiLM if we have global features
        if self.has_global_features:
            self.layers.append(FiLMLayer(global_input_dim, d_model, clamp_gamma=film_clamp))
        
        # Interleave transformer and FiLM layers
        for _ in range(num_encoder_layers):
            # Add transformer encoder layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                norm_first=True,
                batch_first=True,
            )
            self.layers.append(encoder_layer)
            
            # Add FiLM after each transformer layer
            if self.has_global_features:
                self.layers.append(FiLMLayer(global_input_dim, d_model, clamp_gamma=film_clamp))

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
        """Initialize weights for all layers except FiLM (which self-initializes)."""
        for module in self.modules():
            if isinstance(module, FiLMLayer):
                continue 
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
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

        # Pass through interleaved transformer and FiLM layers
        for layer in self.layers:
            if isinstance(layer, nn.TransformerEncoderLayer):
                # Transformer layer
                x = layer(src=x, src_key_padding_mask=sequence_mask)
            elif isinstance(layer, FiLMLayer) and global_features is not None:
                # FiLM conditioning layer
                x = layer(x, global_features)

        # Project to output dimension
        output = self.output_proj(x)

        # Mask padding
        if sequence_mask is not None:
            output = output.masked_fill(sequence_mask.unsqueeze(-1), self.padding_value)

        return output
    

def create_prediction_model(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    compile_model: bool = True,
) -> PredictionModel:
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

    # Conditionally compile the model based on the compile_model flag
    if compile_model:
        compile_enabled = config.get("miscellaneous_settings", {}).get(
            "torch_compile", False
        )
        compile_mode = config.get("miscellaneous_settings", {}).get(
            "compile_mode", "default"
        )

        if (
            version.parse(torch.__version__) >= version.parse("2.0.0")
            and device.type == "cuda"
            and compile_enabled
        ):
            try:
                logger.info(f"Attempting torch.compile with mode='{compile_mode}'")
                model = torch.compile(model, mode=compile_mode)
                logger.info("Model compiled successfully")
            except RuntimeError as e:
                logger.warning(
                    f"torch.compile failed: {e}. Proceeding without compilation."
                )
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
    1. Exports a **torch.export** artefact with a *dynamic batch* dimension.
    2. Exports an **ONNX** model with the same dynamic batch axis.
    3. Optionally simplifies the ONNX graph if onnx‑sim is available.
    """
    save_path = Path(save_path)
    save_dir = save_path.parent
    model_name = save_path.stem

    if config is not None and not config.get("miscellaneous_settings", {}).get("torch_export", True):
        logger.info("Model export disabled in config – skipping.")
        return

    # Set environment variables to prevent ONNX runtime thread affinity issues
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['ORT_DISABLE_THREAD_AFFINITY'] = '1'
    
    model.eval()

    # Move model to CPU for export to avoid MPS-specific issues with fake tensors
    original_device = next(model.parameters()).device
    model = model.to('cpu')

    # Unwrap torch.compile wrapper if present ─ it may freeze shapes
    if hasattr(model, "_orig_mod"):
        logger.info("Extracting original model from compiled wrapper")
        model = model._orig_mod

    # Move example inputs to CPU to avoid device mismatch in fake tensors
    sequence = example_input["sequence"].to('cpu')
    global_features = example_input.get("global_features")
    if global_features is not None:
        global_features = global_features.to('cpu')
    sequence_mask = example_input.get("sequence_mask")
    if sequence_mask is not None:
        sequence_mask = sequence_mask.to('cpu')

    # Conditionally create dummy with batch=2 only if original batch=1
    batch_size = sequence.shape[0]
    if batch_size == 1:
        dummy_sequence = torch.cat([sequence, sequence], dim=0)
        dummy_global_features = torch.cat([global_features, global_features], dim=0) if global_features is not None else None
        dummy_sequence_mask = torch.cat([sequence_mask, sequence_mask], dim=0) if sequence_mask is not None else None
    else:
        dummy_sequence = sequence
        dummy_global_features = global_features
        dummy_sequence_mask = sequence_mask

    # Use only kwargs to avoid mixed key types in dynamic_shapes
    args = ()
    kwargs: Dict[str, Tensor] = {"sequence": dummy_sequence}
    if dummy_global_features is not None:
        kwargs["global_features"] = dummy_global_features
    if dummy_sequence_mask is not None:
        kwargs["sequence_mask"] = dummy_sequence_mask

    # Adjust max based on your use case (e.g., hardware limits)
    batch = Dim("batch", min=1, max=128)  
    
    dynamic_shapes: Dict[str, Any] = {
        "sequence": {0: batch}
    }
    if dummy_global_features is not None:
        dynamic_shapes["global_features"] = {0: batch}
    if dummy_sequence_mask is not None:
        dynamic_shapes["sequence_mask"] = {0: batch}

    try:
        with torch.no_grad():
            if original_device.type == 'mps':
                with sdpa_kernel(SDPBackend.MATH):
                    prog = texport(model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
            else:
                prog = texport(model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
        
        tsave(prog, str(save_dir / f"{model_name}_torch.pt"))
        logger.info("torch.export artefact written with dynamic batch size")
    except Exception as exc:
        logger.error("torch.export failed: %s", exc)
        logger.info("Attempting fallback export without dynamic shapes...")
        
        # Fallback: Try exporting without dynamic shapes
        try:
            with torch.no_grad():
                if original_device.type == 'mps':
                    with sdpa_kernel(SDPBackend.MATH):
                        prog = texport(model, args, kwargs=kwargs)
                else:
                    prog = texport(model, args, kwargs=kwargs)
            tsave(prog, str(save_dir / f"{model_name}_torch_static.pt"))
            logger.warning("Exported with static shapes as fallback")
        except Exception as fallback_exc:
            logger.error("Fallback export also failed: %s", fallback_exc)
            return

    # Quick numerical sanity‑check
    with torch.no_grad():
        ref = model(**kwargs)
        out = prog.module()(**kwargs)
        if not torch.allclose(ref, out, rtol=1e-4, atol=1e-5):
            logger.warning("Exported (torch.export) output differs from original")
    
    return
    """
    # --------------------------------------------------------------------- #
    #                         ONNX dynamic‑batch export                     #
    # --------------------------------------------------------------------- #
    onnx_path = save_dir / f"{model_name}.onnx"
    try:
        # Build tuple for ONNX export (ordered inputs)
        onnx_args = (sequence,)
        if global_features is not None:
            onnx_args += (global_features,)
        if sequence_mask is not None:
            onnx_args += (sequence_mask,)

        input_names = ["sequence"]
        dynamic_axes = {"sequence": {0: "batch"}, "output": {0: "batch"}}
        if global_features is not None:
            input_names.append("global_features")
            dynamic_axes["global_features"] = {0: "batch"}
        if sequence_mask is not None:
            input_names.append("sequence_mask")
            dynamic_axes["sequence_mask"] = {0: "batch"}

        torch.onnx.export(
            model,
            onnx_args,  # Use tuple
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        logger.info("ONNX model saved with dynamic batch size: %s", onnx_path)

        # ---- Simplify / optimise ONNX graph if onnx‑sim is available ---- #
        try:
            import onnx
            import onnxsim

            onnx_model = onnx.load(str(onnx_path))
            # Remove deprecated dynamic_input_shape parameter
            onnx_model_opt, ok = onnxsim.simplify(onnx_model)
            if ok:
                opt_path = save_dir / f"{model_name}_optimized.onnx"
                onnx.save(onnx_model_opt, str(opt_path))
                logger.info("Simplified ONNX graph saved to: %s", opt_path)
            else:
                logger.warning("onnx‑sim simplification check failed – keeping raw graph")
        except ImportError:
            logger.info("onnxsim not installed – skipping ONNX optimisation")
        except Exception as exc:  # pragma: no cover
            logger.warning("ONNX optimisation failed: %s", exc)

    except Exception as exc:  # pragma: no cover
        logger.error("ONNX export failed: %s", exc)
    finally:
        # Reset environment variables to defaults
        if 'OMP_NUM_THREADS' in os.environ:
            del os.environ['OMP_NUM_THREADS']
        if 'MKL_NUM_THREADS' in os.environ:
            del os.environ['MKL_NUM_THREADS']
        if 'ORT_DISABLE_THREAD_AFFINITY' in os.environ:
            del os.environ['ORT_DISABLE_THREAD_AFFINITY']

    """

__all__ = ["PredictionModel", "create_prediction_model", "export_model"]