#!/usr/bin/env python3
"""
model.py – Encoder-only transformer for multi-source atmospheric data.

Features:
- Sinusoidal positional encoding only
- Separate SequenceEncoders per sequence type
- Global feature FiLM conditioning
- General cross-attention: each sequence attends to the concatenation of all other sequences
- MLP head with LayerNorm, GELU, and dropout
- Xavier parameter initialization
"""
import math
import logging
from typing import Any, Dict, List, Optional

import torch
from torch import nn, Tensor

logger = logging.getLogger(__name__)


class SinePositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        if d_model % 2 != 0:
             raise ValueError(f"d_model must be even for SinePositionalEncoding, got {d_model}")
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [d_model/2]
        pe = torch.zeros(1, max_len, d_model) # [1, max_len, d_model]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, D]
        L = x.size(1)
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} > max_len {self.max_len}")
        # Add positional encoding [1, L, D] to input [B, L, D]
        return x + self.pe[:, :L, :]


class SequenceEncoder(nn.Module):
    """TransformerEncoder for one sequence type."""
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        norm_first: bool = False,
        max_len: int = 512, # Pass max_len for SinePositionalEncoding
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinePositionalEncoding(d_model, max_len) # Use Sine PE
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        # Norm to apply after the stack if norm_first=False
        encoder_norm = nn.LayerNorm(d_model) if not norm_first else None
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=encoder_norm)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, input_dim]
        x = self.input_proj(x)            # [B, L, d_model]
        x = self.pos_enc(x)               # add positional encodings
        return self.encoder(x)            # [B, L, d_model]


class GlobalEncoder(nn.Module):
    """MLP that encodes global features into d_model dimension."""
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Adjusted hidden layer size based on common practice (e.g., 2x d_model)
        hidden_dim = 2 * d_model
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model), # LayerNorm often applied at the end
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, global_dim]
        return self.net(x)  # [B, d_model]


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM)."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Linear layer projects conditioning vector to produce scale (gamma) and shift (beta)
        self.scale_shift_proj = nn.Linear(d_model, 2 * d_model)

    def forward(self, seq: Tensor, cond: Tensor) -> Tensor:
        # seq: [B, L, d_model], cond: [B, d_model]
        # Project condition to get FiLM parameters
        params = self.scale_shift_proj(cond)       # [B, 2*d_model]
        # Split into scale (gamma) and shift (beta)
        gamma, beta = params.chunk(2, dim=-1) # each [B, d_model]
        # Unsqueeze to allow broadcasting over sequence length L: [B, 1, d_model]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        # Apply FiLM: gamma * sequence + beta
        return gamma * seq + beta


class CrossAttentionModule(nn.Module):
    """Cross-attention from one sequence (query) to an aggregated context (key/value)."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True, # Expects [B, L, D]
        )
        # LayerNorm and Dropout for residual connection
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        # query:   [B, Lq, d_model]
        # context: [B, Lc, d_model] (Lc = total length of other sequences)
        # Attend: query attends to context (keys=context, values=context)
        attn_out, _ = self.attn(query=query, key=context, value=context, need_weights=False)
        # Residual connection: query + dropout(attention_output) -> LayerNorm
        return self.norm(query + self.dropout(attn_out))


class MultiEncoderTransformer(nn.Module):
    """
    Multi-source, encoder-only transformer matching the README description.

    Expects inputs dict with keys matching sequence_types and optionally 'global'.
    Each sequence tensor: [B, L, feature_dim].
    Global tensor:    [B, global_dim].
    """
    def __init__(
        self,
        global_variables: List[str],
        sequence_dims: Dict[str, List[str]], # Changed from Dict[str, Dict[str, int]] for simplicity
        output_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        norm_first: bool,
        output_seq_type: str,
        max_sequence_length: int,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        # Filter out empty sequence definitions from config
        self.sequence_types = [k for k, v in sequence_dims.items() if v]
        if not self.sequence_types:
             raise ValueError("sequence_dims must contain at least one non-empty sequence type.")
        if output_seq_type not in self.sequence_types:
            raise ValueError(f"output_seq_type '{output_seq_type}' not in sequence_types {self.sequence_types}")
        self.output_seq_type = output_seq_type

        # Sequence encoders
        self.encoders = nn.ModuleDict({
            st: SequenceEncoder(
                input_dim=len(sequence_dims[st]), # Input dim is number of vars in the list
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=norm_first,
                max_len=max_sequence_length, # Pass max_len here
            ) for st in self.sequence_types
        })

        # Global encoder and FiLM layers
        self.use_global = bool(global_variables)
        if self.use_global:
            if not global_variables:
                 raise ValueError("global_variables list cannot be empty if use_global is intended.")
            self.global_enc = GlobalEncoder(len(global_variables), d_model, dropout)
            # FiLM layers applied BEFORE and AFTER cross-attention
            self.film1 = nn.ModuleDict({st: FiLMLayer(d_model) for st in self.sequence_types})
            self.film2 = nn.ModuleDict({st: FiLMLayer(d_model) for st in self.sequence_types})

        # Cross-attention module for each sequence type (if >1 type exists)
        self.cross_attn = nn.ModuleDict()
        if len(self.sequence_types) > 1:
            for st in self.sequence_types:
                self.cross_attn[st] = CrossAttentionModule(d_model, nhead, dropout)

        # MLP head for final output projection
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), # Intermediate layer
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim), # Final projection
        )

        # Store max sequence length for input validation
        self.max_seq_len = max_sequence_length

        self._init_weights()
        logger.info(
            f"Initialized MultiEncoderTransformer (Simple): seq_types={self.sequence_types}, "
            f"output_seq_type='{self.output_seq_type}', d_model={d_model}"
        )

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform for matrices > 1D."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name: # Initialize biases to zero
                nn.init.zeros_(p)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        # 1. Encode global features (if used)
        global_cond: Optional[Tensor] = None
        if self.use_global:
            if 'global' not in inputs:
                raise ValueError("Missing 'global' features in inputs when expected.")
            global_features_input = inputs['global']
            if global_features_input.ndim == 1: # Ensure batch dimension
                 global_features_input = global_features_input.unsqueeze(0)
            global_cond = self.global_enc(global_features_input)  # [B, d_model]

        # 2. Encode each sequence type
        encoded_sequences: Dict[str, Tensor] = {}
        for seq_type in self.sequence_types:
            if seq_type not in inputs:
                raise ValueError(f"Missing required sequence input '{seq_type}'")
            x = inputs[seq_type] # [B, L, feat_dim]
            if x.size(1) > self.max_seq_len:
                raise ValueError(
                    f"Input sequence '{seq_type}' length {x.size(1)} exceeds model max_len {self.max_seq_len}"
                )
            # Encode sequence
            encoded = self.encoders[seq_type](x) # [B, L, d_model]
            # Apply FiLM Layer 1 (if global features exist)
            if global_cond is not None:
                encoded = self.film1[seq_type](encoded, global_cond)
            encoded_sequences[seq_type] = encoded

        # 3. Apply Cross-Attention (if more than one sequence type)
        if len(self.sequence_types) > 1:
            cross_attended_sequences = {}
            for current_seq_type in self.sequence_types:
                # Build context from concatenation of *other* encoded sequences
                context_list = [
                    encoded_sequences[other_type]
                    for other_type in self.sequence_types
                    if other_type != current_seq_type
                ]
                if not context_list: # Should not happen if len > 1, but safeguard
                     cross_attended_sequences[current_seq_type] = encoded_sequences[current_seq_type]
                     continue

                context = torch.cat(context_list, dim=1) # [B, L_total_others, d_model]
                # Apply cross-attention: current sequence attends to context
                attended_output = self.cross_attn[current_seq_type](
                    query=encoded_sequences[current_seq_type],
                    context=context
                )
                cross_attended_sequences[current_seq_type] = attended_output
            encoded_sequences = cross_attended_sequences # Update with attended versions


        # 4. Apply FiLM Layer 2 (if global features exist)
        if global_cond is not None:
            for seq_type in self.sequence_types:
                 # Ensure key exists before attempting FiLM
                 if seq_type in encoded_sequences and seq_type in self.film2:
                    encoded_sequences[seq_type] = self.film2[seq_type](
                        encoded_sequences[seq_type], global_cond
                    )


        # 5. Select the designated output sequence
        if self.output_seq_type not in encoded_sequences:
             raise ValueError(f"Output sequence type '{self.output_seq_type}' not found after processing steps.")
        output_sequence = encoded_sequences[self.output_seq_type]  # [B, L_out, d_model]

        # 6. Project through the final MLP head
        output = self.head(output_sequence)  # [B, L_out, output_dim]
        return output


def create_prediction_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to build the MultiEncoderTransformer from a configuration dictionary.
    Expects config to include necessary keys like 'sequence_types', 'output_seq_type', etc.
    """
    # Validate presence of required keys
    required_keys = [
        'input_variables', 'target_variables', 'global_variables',
        'sequence_types', 'output_seq_type',
        'd_model', 'nhead', 'num_encoder_layers',
        'dim_feedforward', 'dropout', 'norm_first',
        'max_sequence_length'
    ]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys for model creation: {missing_keys}")

    # Check sequence_dims format expected by model (list of vars, not dict)
    sequence_dims_model = {k: list(v) for k, v in config['sequence_types'].items() if isinstance(v, list)}
    if not sequence_dims_model:
         raise ValueError("Config 'sequence_types' must be a dict mapping type name to a list of variable names.")


    model = MultiEncoderTransformer(
        global_variables=config.get('global_variables', []), # Allow empty list
        sequence_dims=sequence_dims_model,
        output_dim=len(config['target_variables']),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        norm_first=config.get('norm_first', False), # Default to False if not specified
        output_seq_type=config['output_seq_type'],
        max_sequence_length=config['max_sequence_length'],
    )

    # Optionally store variable names on model instance (useful for debugging/inspection)
    model.input_vars = config['input_variables']
    model.target_vars = config['target_variables']

    return model


__all__ = ["create_prediction_model", "MultiEncoderTransformer"]