#!/usr/bin/env python3
"""
model.py – encoder-only multi-source transformer for atmospheric-profile prediction.

Updates
-------
* SinePositionalEncoding now expands its lookup table on-the-fly if a sequence
  exceeds the initial max length.
* CrossAttention always layer-normalises the context (K,V), even when
  `norm_first=False`, improving stability.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Positional encoding                                                          #
# --------------------------------------------------------------------------- #


class SinePositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding that grows if needed."""

    def __init__(self, d_model: int, max_len: int = 5_000) -> None:
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even for sine positional encoding")
        self.d_model = d_model
        self._init_table(max_len)

    def _init_table(self, length: int) -> None:
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(1, length, self.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe_table", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:  # [B,L,D]
        seq_len = x.size(1)
        if seq_len > self.pe_table.size(1):
            logger.debug("PositionalEncoding: expanding table to %d", seq_len)
            self._init_table(seq_len)
            self.pe_table = self.pe_table.to(x.device, dtype=x.dtype)
        return x + self.pe_table[:, :seq_len]


# --------------------------------------------------------------------------- #
# Dilated Conv1d block                                                         #
# --------------------------------------------------------------------------- #


class ConvBlock1D(nn.Module):
    """Dilated Conv1d residual stack."""

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

    def forward(self, x: Tensor) -> Tensor:  # [B,L,D]
        x = x.permute(0, 2, 1)  # -> [B,D,L]
        for layer in self.layers:
            res = x
            x = layer(x) + res
        return x.permute(0, 2, 1)


# --------------------------------------------------------------------------- #
# Sequence encoder                                                             #
# --------------------------------------------------------------------------- #


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ff: int,
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
        self.cnn = (
            ConvBlock1D(d_model, cnn_layers, cnn_kernel, cnn_dropout) if use_cnn else None
        )
        self.pe = SinePositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_ff,
            dropout,
            batch_first=True,
            norm_first=norm_first,
            activation=nn.GELU(),
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, None if norm_first else nn.LayerNorm(d_model)
        )

    def forward(self, x: Tensor) -> Tensor:  # [B,L,F]
        x = self.proj(x)
        if self.cnn is not None:
            x = x + self.cnn(x)
        x = self.pe(x)
        return self.encoder(x)


# --------------------------------------------------------------------------- #
# Global conditioning & FiLM                                                   #
# --------------------------------------------------------------------------- #


class GlobalEncoder(nn.Module):
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
        return self.net(x)


class FiLMLayer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 2 * d_model)

    def forward(self, seq: Tensor, cond: Tensor) -> Tensor:  # [B,L,D], [B,D]
        gamma, beta = self.fc(cond).chunk(2, dim=-1)
        return gamma.unsqueeze(1) * seq + beta.unsqueeze(1)


# --------------------------------------------------------------------------- #
# Cross-sequence attention                                                     #
# --------------------------------------------------------------------------- #


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, norm_first: bool):
        super().__init__()
        self.norm_first = norm_first
        self.nq = nn.LayerNorm(d_model)
        self.nkv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.nout = nn.LayerNorm(d_model) if not norm_first else nn.Identity()

    def forward(self, q: Tensor, ctx: Tensor) -> Tensor:  # [B,L,D] each
        res = q
        if self.norm_first:
            q = self.nq(q)
            ctx = self.nkv(ctx)
        else:
            ctx = self.nkv(ctx)  # still normalise context
        out, _ = self.attn(q, ctx, ctx, need_weights=False)
        out = self.drop(out) + res
        return self.nout(out)


# --------------------------------------------------------------------------- #
# Multi-encoder transformer                                                    #
# --------------------------------------------------------------------------- #


class MultiEncoderTransformer(nn.Module):
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
        head_activation: str = "none",
    ) -> None:
        super().__init__()

        self.output_seq_type = output_seq_type
        self.sequence_types = [k for k, v in sequence_dims.items() if v]
        if output_seq_type not in self.sequence_types:
            raise ValueError(f"output_seq_type '{output_seq_type}' not valid")

        # sequence encoders
        self.encoders = nn.ModuleDict(
            {
                st: SequenceEncoder(
                    input_dim=len(sequence_dims[st]),
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    dim_ff=dim_feedforward,
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

        # global conditioning
        self.use_global = bool(global_variables)
        if self.use_global:
            self.glob_enc = GlobalEncoder(len(global_variables), d_model, dropout)
            self.film_pre = nn.ModuleDict({st: FiLMLayer(d_model) for st in self.sequence_types})
            self.film_post = nn.ModuleDict({st: FiLMLayer(d_model) for st in self.sequence_types})
        else:
            self.glob_enc = None
            self.film_pre = self.film_post = None

        # cross-attention
        self.xattn = (
            nn.ModuleDict(
                {st: CrossAttention(d_model, nhead, dropout, norm_first) for st in self.sequence_types}
            )
            if len(self.sequence_types) > 1
            else None
        )

        # head
        act_map = {"none": nn.Identity(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}
        if head_activation not in act_map:
            raise ValueError(f"head_activation '{head_activation}' not in {list(act_map)}")
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim),
            act_map[head_activation],
        )

        self._init_weights()
        logger.info("MultiEncoderTransformer initialised")

    # ------------------------------------------------------------------ #
    # weight init                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _init_linear(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_weights(self) -> None:
        self.apply(self._init_linear)

    # ------------------------------------------------------------------ #
    # forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, inp: Dict[str, Tensor]) -> Tensor:
        # --- global conditioning --------------------------------------
        cond: Optional[Tensor] = None
        if self.use_global:
            g = inp.get("global")
            if g is None:
                raise ValueError("missing 'global' input")
            if g.dim() == 1:  # batch=1 scalar squeeze
                g = g.unsqueeze(0)
            cond = self.glob_enc(g)

        # --- encode each sequence ------------------------------------
        enc: Dict[str, Tensor] = {}
        for st in self.sequence_types:
            x = inp.get(st)
            if x is None:
                raise ValueError(f"missing input for sequence '{st}'")
            seq = self.encoders[st](x)
            if cond is not None:
                seq = self.film_pre[st](seq, cond)
            enc[st] = seq

        # --- optional cross-attention --------------------------------
        if self.xattn is not None:
            enc = {
                st: self.xattn[st](enc[st], torch.cat([enc[o] for o in self.sequence_types if o != st], dim=1))
                for st in self.sequence_types
            }

        # --- second FiLM ---------------------------------------------
        if cond is not None:
            enc = {st: self.film_post[st](enc[st], cond) for st in self.sequence_types}

        # --- output head ---------------------------------------------
        return self.head(enc[self.output_seq_type])


# --------------------------------------------------------------------------- #
# Factory helper                                                               #
# --------------------------------------------------------------------------- #


def create_prediction_model(cfg: Dict[str, any]) -> nn.Module:
    """Build a MultiEncoderTransformer from *cfg* dict."""
    required = [
        "input_variables",
        "target_variables",
        "sequence_types",
        "output_seq_type",
        "d_model",
        "nhead",
        "num_encoder_layers",
        "dim_feedforward",
        "dropout",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    model = MultiEncoderTransformer(
        global_variables=cfg.get("global_variables", []),
        sequence_dims=cfg["sequence_types"],
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
        head_activation=cfg.get("head_activation", "none"),
    )

    # attach some helpful attributes for downstream code
    model.input_vars = cfg["input_variables"]
    model.target_vars = cfg["target_variables"]
    model.global_vars = cfg.get("global_variables", [])

    return model


__all__ = ["create_prediction_model", "MultiEncoderTransformer"]
