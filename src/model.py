#!/usr/bin/env python3
"""
model.py – Encoder‑only multi‑source transformer for atmospheric profile prediction.

This module defines a transformer architecture suitable for regression tasks on
atmospheric profile data. It supports multiple input sequence types, optional
global conditioning features, and an optional 1D convolutional block within
the sequence encoder.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

# =========================================================================== #
# Core Building Blocks                                                        #
# =========================================================================== #

class SinePositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional encoding information into the sequence.

    Uses fixed sine and cosine functions based on position and feature index.
    The maximum sequence length is determined by the `max_len` parameter.
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even for sine positional encoding")
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe_table", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Adds positional encoding to the input tensor."""
        seq_len = x.size(1)
        if seq_len > self.pe_table.size(1):
             raise ValueError(
                 f"Sequence length {seq_len} exceeds SinePositionalEncoding max_len {self.pe_table.size(1)}"
             )
        return x + self.pe_table[:, :seq_len].to(dtype=x.dtype, device=x.device)


class ConvBlock1D(nn.Module):
    """
    A stack of 1D dilated convolutional layers with residual connections.

    Applies a series of convolutions with increasing dilation factors,
    interspersed with Group Normalization, GELU activation, and Dropout.
    Residual connections are added after each block.
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers):
            dilation = 2 ** i
            pad = dilation * (kernel_size - 1) // 2
            layers.append(
                nn.Sequential(
                    nn.GroupNorm(1, d_model),
                    nn.Conv1d(
                        d_model,
                        d_model,
                        kernel_size,
                        padding=pad,
                        dilation=dilation,
                        bias=True,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the 1D convolutional stack."""
        x_perm = x.permute(0, 2, 1)
        for layer in self.layers:
            res = x_perm
            x_perm = layer(x_perm) + res
        return x_perm.permute(0, 2, 1)


class SequenceEncoder(nn.Module):
    """
    Encodes a single sequence type using optional CNN and Transformer blocks.

    Processes an input sequence through:
    1. Linear projection to `d_model`.
    2. Optional 1D Convolutional block (`ConvBlock1D`).
    3. Addition of sinusoidal positional encodings.
    4. A standard Transformer Encoder stack.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        norm_first: bool,
        *,
        use_cnn: bool,
        cnn_kernel: int,
        cnn_layers: int,
        cnn_dropout: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.cnn = (ConvBlock1D(d_model, cnn_layers, cnn_kernel, cnn_dropout) if use_cnn else None)
        self.pe = SinePositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if norm_first else None,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input sequence through the encoder."""
        x = self.proj(x)
        if self.cnn is not None:
            x = x + self.cnn(x)
        x = self.pe(x)
        return self.encoder(x)


class GlobalEncoder(nn.Module):
    """
    Encodes global (non-sequential) features using an MLP.

    Projects global features to `d_model` dimensions using a small feed-forward
    network with GELU activation, Dropout, and Layer Normalization.
    """
    def __init__(self, inp: int, d_model: int, dropout: float) -> None:
        super().__init__()
        hidden = max(d_model, inp * 2)
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encodes the global features."""
        return self.net(x)


class FiLMLayer(nn.Module):
    """
    Applies Feature-wise Linear Modulation (FiLM) to condition a sequence.

    Uses a conditioning vector (typically from global features) to generate
    per-feature scaling (gamma) and shifting (beta) parameters, which are
    then applied element-wise to the input sequence.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 2 * d_model)

    def forward(self, seq: Tensor, cond: Tensor) -> Tensor:
        """Applies FiLM conditioning."""
        gamma, beta = self.fc(cond).chunk(2, dim=-1)
        return gamma.unsqueeze(1) * seq + beta.unsqueeze(1)


class CrossAttention(nn.Module):
    """
    Performs cross-attention from a query sequence to a context sequence.

    Allows the query sequence to attend to the key/value derived from the
    context sequence. Includes Layer Normalization and dropout.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float, norm_first: bool):
        super().__init__()
        self.norm_first = norm_first
        self.nq = nn.LayerNorm(d_model)
        self.nkv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.nout = nn.LayerNorm(d_model) if not norm_first else nn.Identity()

    def forward(self, q: Tensor, ctx: Tensor) -> Tensor:
        """Applies cross-attention."""
        res = q
        q_normed = self.nq(q)
        ctx_normed = self.nkv(ctx)

        if self.norm_first:
           attn_in_q = q_normed
           attn_in_kv = ctx_normed
        else:
           attn_in_q = q
           attn_in_kv = ctx_normed

        out, _ = self.attn(attn_in_q, attn_in_kv, attn_in_kv, need_weights=False)
        out = self.drop(out) + res
        return self.nout(out)

# =========================================================================== #
# Main Model Architecture                                                     #
# =========================================================================== #

class MultiEncoderTransformer(nn.Module):
    """
    Encoder-only transformer handling multiple sequence types and global features.

    This model independently encodes multiple input sequences (e.g., different
    types of atmospheric profiles). If global features are provided, they are
    encoded and used to condition the sequences via FiLM layers before cross-attention.
    If multiple sequence types are present, cross-attention is applied between them.
    A final linear layer projects the representation of a designated output
    sequence type to the target dimension.

    Attributes:
        output_seq_type: The key (string name) of the sequence type whose final
                         representation is used for output prediction.
        sequence_types: List of keys for the sequence types being processed.
        use_global: Boolean indicating if global feature conditioning is used.
        use_cross_attention: Boolean indicating if cross-attention is applied.
    """
    def __init__(
        self,
        *,
        global_variables: List[str],
        sequence_dims: Dict[str, List[str]],
        output_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        norm_first: bool,
        output_seq_type: str,
        use_sequence_cnn: bool = False,
        cnn_kernel_size: int = 3,
        cnn_num_layers: int = 4,
        cnn_dropout: float = 0.0,
        max_sequence_length: int = 512,
    ) -> None:
        super().__init__()

        self.output_seq_type = output_seq_type
        self.sequence_dims_internal = {k: v for k, v in sequence_dims.items() if v}
        self.sequence_types = list(self.sequence_dims_internal.keys())
        if not self.sequence_types:
             raise ValueError("No valid sequence types provided in sequence_dims.")
        if output_seq_type not in self.sequence_types:
            raise ValueError(
                f"output_seq_type '{output_seq_type}' not found in valid sequence types: {self.sequence_types}"
            )

        self.encoders = nn.ModuleDict(
            {
                st: SequenceEncoder(
                    input_dim=len(self.sequence_dims_internal[st]),
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    norm_first=norm_first,
                    use_cnn=use_sequence_cnn,
                    cnn_kernel=cnn_kernel_size,
                    cnn_layers=cnn_num_layers,
                    cnn_dropout=cnn_dropout,
                )
                for st in self.sequence_types
            }
        )

        self.use_global = bool(global_variables)
        if self.use_global:
            self.glob_enc = GlobalEncoder(len(global_variables), d_model, dropout)
            self.film_layers = nn.ModuleDict({st: FiLMLayer(d_model) for st in self.sequence_types})
        else:
            self.glob_enc = None
            self.film_layers = None

        self.use_cross_attention = len(self.sequence_types) > 1
        if self.use_cross_attention:
             if len(self.sequence_types) != 2:
                  logger.warning(f"Cross-attention enabled but {len(self.sequence_types)} sequence types found (expected 2). Check configuration.")
             self.xattn = nn.ModuleDict({
                st: CrossAttention(d_model, nhead, dropout, norm_first) for st in self.sequence_types
             })
        else:
             self.xattn = None

        self.head = nn.Linear(d_model, output_dim)

        self._init_parameters()
        logger.info(f"MultiEncoderTransformer initialised with sequence types: {self.sequence_types}")
        if self.use_global:
            logger.info(f"Global conditioning enabled for variables: {global_variables}")
        if self.use_cross_attention:
             logger.info("Cross-attention enabled between sequence types.")

    def _init_parameters(self):
        """Initializes model parameters using Xavier uniform and other methods."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif 'weight' in name and '.norm' not in name and 'ln_' not in name:
                 nn.init.uniform_(p, -0.1, 0.1)
        if hasattr(self, 'head') and isinstance(self.head, nn.Linear):
            nn.init.xavier_uniform_(self.head.weight, gain=0.01)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward(self, inp: Dict[str, Tensor]) -> Tensor:
        """
        Defines the forward pass of the model.

        Args:
            inp: A dictionary where keys are sequence type names (matching those
                 in `sequence_dims` during initialization) or 'global'. Values
                 are tensors:
                 - Sequence tensors: Shape [batch_size, seq_length, feature_dim]
                 - Global tensor: Shape [batch_size, global_feature_dim]

        Returns:
            The final output tensor of shape [batch_size, output_seq_length, output_dim].
        """
        cond: Optional[Tensor] = None
        if self.use_global:
            g = inp.get("global")
            if g is None:
                raise ValueError("Global input 'global' missing but expected.")
            if g.dim() == 1:
                g = g.unsqueeze(0)
            cond = self.glob_enc(g)

        enc: Dict[str, Tensor] = {}
        for st in self.sequence_types:
            x = inp.get(st)
            if x is None:
                raise ValueError(f"Input for sequence type '{st}' missing.")
            seq = self.encoders[st](x)

            if cond is not None and self.film_layers is not None:
                 if st not in self.film_layers:
                      logger.warning(f"FiLM layer not found for sequence type '{st}', skipping conditioning.")
                 else:
                      seq = self.film_layers[st](seq, cond)
            enc[st] = seq

        if self.xattn is not None and self.use_cross_attention:
            if len(self.sequence_types) == 2: # Only apply if exactly 2 types
                 st1, st2 = self.sequence_types[0], self.sequence_types[1]
                 ctx1 = enc[st2] # Context for st1 is st2
                 ctx2 = enc[st1] # Context for st2 is st1
                 enc[st1] = self.xattn[st1](enc[st1], ctx1)
                 enc[st2] = self.xattn[st2](enc[st2], ctx2)
            else:
                 logger.warning("Cross-attention module exists but not applied (requires exactly 2 sequence types).")


        if self.output_seq_type not in enc:
             raise RuntimeError( # Use RuntimeError for unexpected internal state
                 f"Designated output sequence type '{self.output_seq_type}' not found in encoded representations."
             )
        return self.head(enc[self.output_seq_type])

# =========================================================================== #
# Factory Function                                                            #
# =========================================================================== #

def create_prediction_model(cfg: Dict[str, any]) -> nn.Module:
    """
    Builds and returns a `MultiEncoderTransformer` instance from a configuration dict.

    Args:
        cfg: A dictionary containing model configuration parameters, including
             `input_variables`, `target_variables`, `sequence_types`,
             `global_variables`, `output_seq_type`, `d_model`, `nhead`, etc.

    Returns:
        An initialized `MultiEncoderTransformer` model.

    Raises:
        ValueError: If essential configuration keys are missing or inconsistent.
    """
    required = [
        "input_variables", "target_variables", "sequence_types", "output_seq_type",
        "d_model", "nhead", "num_encoder_layers", "dim_feedforward", "dropout",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    sequence_dims_lists = {k: list(v) for k, v in cfg["sequence_types"].items()}

    model = MultiEncoderTransformer(
        global_variables=cfg.get("global_variables", []),
        sequence_dims=sequence_dims_lists,
        output_dim=len(cfg["target_variables"]),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["num_encoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        norm_first=cfg.get("norm_first", True),
        output_seq_type=cfg["output_seq_type"],
        use_sequence_cnn=cfg.get("use_sequence_cnn", False),
        cnn_kernel_size=cfg.get("cnn_kernel_size", 3),
        cnn_num_layers=cfg.get("cnn_num_layers", 4),
        cnn_dropout=cfg.get("cnn_dropout", cfg["dropout"]),
        max_sequence_length=cfg.get("max_sequence_length", 512)
    )

    # Store variable names for reference
    model.input_vars = cfg["input_variables"]
    model.target_vars = cfg["target_variables"]
    model.global_vars = cfg.get("global_variables", [])

    return model


__all__ = ["create_prediction_model", "MultiEncoderTransformer"]