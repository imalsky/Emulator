#!/usr/bin/env python3
"""
model.py – Encoder-only transformer for multi-source atmospheric data.

Implements the simpler model architecture featuring:
- Sinusoidal positional encoding only.
- Separate SequenceEncoders for each defined sequence type.
- Optional global feature conditioning using FiLM layers applied before
  and after cross-attention.
- Standard cross-attention mechanism where each sequence attends to the
  concatenation of all other encoded sequences.
- An MLP head with Layer Normalization for final prediction.
- Includes a factory function `create_prediction_model` for easy instantiation
  from a configuration dictionary.
"""

import math
import logging
from typing import Any, Dict, List, Optional

import torch
from torch import nn, Tensor

logger = logging.getLogger(__name__)

# =============================================================================
# Building Blocks
# =============================================================================


class SinePositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding module.

    Generates positional embeddings based on sine and cosine functions
    of different frequencies and adds them to the input tensor.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        """
        Initializes the SinePositionalEncoding layer.

        Args:
            d_model: The embedding dimension (must be even).
            max_len: The maximum sequence length anticipated. The positional
                     encoding table will be pre-computed up to this length.

        Raises:
            ValueError: If d_model is not an even number.
        """
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even for SinePositionalEncoding, got {d_model}"
            )
        self.max_len = max_len

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe_table", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model].

        Returns:
            The input tensor with positional encodings added.

        Raises:
            ValueError: If the input sequence length exceeds `max_len`.
        """
        batch_size, seq_length, _ = x.shape
        if seq_length > self.max_len:
            raise ValueError(
                f"Input sequence length ({seq_length}) exceeds the maximum "
                f"pre-computed length ({self.max_len}) for positional encoding. "
                f"Increase 'max_sequence_length' in the config."
            )
        return x + self.pe_table[:, :seq_length, :]


class SequenceEncoder(nn.Module):
    """
    TransformerEncoder stack specifically designed for one type of sequence input.

    Applies an initial linear projection, adds positional encoding, and then
    processes the sequence through TransformerEncoder layers.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_sequence_length: int,
        dropout: float = 0.1,
        norm_first: bool = False,
    ) -> None:
        """
        Initializes the SequenceEncoder.

        Args:
            input_dim: Number of features in the raw input sequence.
            d_model: The internal embedding dimension of the transformer.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of the feedforward network in each layer.
            max_sequence_length: Maximum sequence length for positional encoding table.
            dropout: Dropout probability.
            norm_first: If True, applies layer norm before attention/feedforward layers (Pre-LN).
                        If False, applies layer norm after (Post-LN).
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinePositionalEncoding(
            d_model, max_len=max_sequence_length
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )

        encoder_norm = nn.LayerNorm(d_model) if not norm_first else None

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
        )
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Processes the input sequence through the encoder stack.

        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim].

        Returns:
            Encoded sequence tensor of shape [batch_size, seq_length, d_model].
        """
        x = self.input_proj(x)
        x = self.pos_enc(x)
        # x = self.dropout(x)
        x = self.encoder(x)
        return x


class GlobalEncoder(nn.Module):
    """
    Encodes scalar global features into a fixed-size embedding vector (d_model).

    Uses a simple Multi-Layer Perceptron (MLP) structure.
    """

    def __init__(
        self, input_dim: int, d_model: int, dropout: float = 0.1
    ) -> None:
        """
        Initializes the GlobalEncoder MLP.

        Args:
            input_dim: Number of global input features.
            d_model: The target output embedding dimension.
            dropout: Dropout probability applied within the MLP.
        """
        super().__init__()
        hidden_dim = 2 * d_model
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Encodes the global input features.

        Args:
            x: Input tensor of global features, shape [batch_size, input_dim].

        Returns:
            Encoded global features tensor, shape [batch_size, d_model].
        """
        return self.net(x)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Conditionally modulates a sequence tensor based on a conditioning vector
    (e.g., encoded global features). It learns to predict a scale (gamma)
    and shift (beta) from the condition, applying them element-wise to the sequence.
    Formula: `output = gamma * sequence + beta`
    """

    def __init__(self, d_model: int) -> None:
        """
        Initializes the FiLMLayer.

        Args:
            d_model: The feature dimension of both the sequence and the condition.
        """
        super().__init__()
        self.scale_shift_projection = nn.Linear(d_model, 2 * d_model)

    def forward(self, sequence: Tensor, condition: Tensor) -> Tensor:
        """
        Applies FiLM modulation.

        Args:
            sequence: The sequence tensor to modulate, shape [batch_size, seq_length, d_model].
            condition: The conditioning tensor, shape [batch_size, d_model].

        Returns:
            The modulated sequence tensor, shape [batch_size, seq_length, d_model].
        """
        film_params = self.scale_shift_projection(condition)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return gamma * sequence + beta


class CrossAttentionModule(nn.Module):
    """
    Performs cross-attention from a query sequence to a context sequence.

    Uses PyTorch's standard MultiheadAttention module, followed by dropout
    and layer normalization with a residual connection.
    """

    def __init__(
        self, d_model: int, nhead: int, dropout: float = 0.1
    ) -> None:
        """
        Initializes the CrossAttentionModule.

        Args:
            d_model: The embedding dimension.
            nhead: The number of attention heads.
            dropout: Dropout probability for the attention mechanism and residual connection.
        """
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        """
        Applies cross-attention.

        Args:
            query: The query sequence tensor, shape [batch_size, query_length, d_model].
            context: The context sequence tensor (used for keys and values),
                     shape [batch_size, context_length, d_model].

        Returns:
            The output tensor after cross-attention and residual connection,
            shape [batch_size, query_length, d_model].
        """
        attn_output, _attn_weights = self.multihead_attn(
            query=query, key=context, value=context, need_weights=False
        )
        output = self.norm(query + self.dropout(attn_output))
        return output


# =============================================================================
# Main Model Class
# =============================================================================


class MultiEncoderTransformer(nn.Module):
    """
    Multi-source, encoder-only transformer for atmospheric profile prediction.

    Processes multiple input sequence types and optional global features. Uses
    separate encoders for each sequence type, FiLM for global feature conditioning,
    and cross-attention for inter-sequence communication. Predicts target sequences.
    """

    def __init__(
        self,
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
        max_sequence_length: int,
    ) -> None:
        """
        Initializes the MultiEncoderTransformer model.

        Args:
            global_variables: List of names for scalar global input features.
            sequence_dims: Dictionary mapping sequence type names (e.g., "atmosphere")
                           to lists of variable names within that sequence.
            output_dim: The number of features in the target output sequence.
            d_model: Internal embedding dimension of the model.
            nhead: Number of attention heads.
            num_encoder_layers: Number of layers in each sequence encoder stack.
            dim_feedforward: Dimension of the feedforward network in encoder layers.
            dropout: Dropout probability used throughout the model.
            norm_first: Whether to use Pre-LN normalization in transformer layers.
            output_seq_type: The key from `sequence_dims` that defines the length
                             of the output prediction sequence.
            max_sequence_length: Maximum sequence length for positional encoding table.
        """
        super().__init__()

        self.d_model = d_model
        self.sequence_types = [k for k, v in sequence_dims.items() if v]
        if not self.sequence_types:
            raise ValueError(
                "Configuration 'sequence_dims' must contain at least one sequence type with variables."
            )
        if output_seq_type not in self.sequence_types:
            raise ValueError(
                f"Configuration 'output_seq_type' ('{output_seq_type}') "
                f"must be one of the defined sequence types: {self.sequence_types}"
            )
        self.output_seq_type = output_seq_type

        self.encoders = nn.ModuleDict()
        for seq_type in self.sequence_types:
            input_feature_dim = len(sequence_dims[seq_type])
            if input_feature_dim == 0:
                logger.warning(
                    "Sequence type '%s' has no variables assigned, skipping encoder creation.",
                    seq_type,
                )
                continue
            self.encoders[seq_type] = SequenceEncoder(
                input_dim=input_feature_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                max_sequence_length=max_sequence_length,
                dropout=dropout,
                norm_first=norm_first,
            )

        self.use_global = bool(global_variables)
        if self.use_global:
            num_global_features = len(global_variables)
            if num_global_features == 0:
                raise ValueError(
                    "Configuration lists `global_variables` but the list is empty."
                )
            self.global_encoder = GlobalEncoder(
                num_global_features, d_model, dropout
            )
            self.film1 = nn.ModuleDict(
                {st: FiLMLayer(d_model) for st in self.sequence_types}
            )
            self.film2 = nn.ModuleDict(
                {st: FiLMLayer(d_model) for st in self.sequence_types}
            )
            logger.info("Global feature conditioning enabled using FiLM.")
        else:
            self.global_encoder = None
            self.film1 = None
            self.film2 = None
            logger.info(
                "No global variables specified; FiLM conditioning disabled."
            )

        self.cross_attn = nn.ModuleDict()
        if len(self.sequence_types) > 1:
            logger.info(
                "Initializing cross-attention modules between sequence types."
            )
            for seq_type in self.sequence_types:
                self.cross_attn[seq_type] = CrossAttentionModule(
                    d_model, nhead, dropout
                )
        else:
            logger.info("Only one sequence type; cross-attention disabled.")

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

        self.max_seq_len = max_sequence_length

        self._initialize_weights()
        logger.info(
            f"Initialized MultiEncoderTransformer (Simple Arch): "
            f"Sequences={self.sequence_types}, Output='{self.output_seq_type}', d_model={d_model}"
        )

    def _initialize_weights(self) -> None:
        """Initializes model weights using Xavier uniform for multi-dimensional tensors."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Defines the forward pass of the model.

        Args:
            inputs: A dictionary containing input tensors. Expected keys match
                    `sequence_types` and optionally 'global'.
                    - Sequence tensors shape: [batch_size, seq_length, feature_dim]
                    - Global tensor shape: [batch_size, global_dim]

        Returns:
            The prediction tensor, shape [batch_size, output_seq_length, output_dim].

        Raises:
            ValueError: If required inputs are missing or sequence lengths exceed maximum.
        """
        global_condition: Optional[Tensor] = None
        if self.use_global:
            if "global" not in inputs:
                raise ValueError(
                    "Model configured to use global features, but 'global' key missing from input dictionary."
                )
            global_features_input = inputs["global"]
            if global_features_input.ndim == 1:
                global_features_input = global_features_input.unsqueeze(0)
            if self.global_encoder is None:
                raise RuntimeError(
                    "Global encoder not initialized despite use_global being True."
                )
            global_condition = self.global_encoder(global_features_input)

        encoded_sequences: Dict[str, Tensor] = {}
        for seq_type in self.sequence_types:
            if seq_type not in inputs:
                raise ValueError(
                    f"Missing required sequence input for type '{seq_type}'."
                )
            x = inputs[seq_type]

            if x.size(1) > self.max_seq_len:
                raise ValueError(
                    f"Input sequence '{seq_type}' length ({x.size(1)}) exceeds model "
                    f"max_sequence_length ({self.max_seq_len}). "
                    f"Check data or increase 'max_sequence_length' in config."
                )

            encoded = self.encoders[seq_type](x)

            if global_condition is not None and self.film1 is not None:
                encoded = self.film1[seq_type](encoded, global_condition)

            encoded_sequences[seq_type] = encoded

        if len(self.sequence_types) > 1:
            cross_attended_sequences: Dict[str, Tensor] = {}
            for current_seq_type in self.sequence_types:
                context_list = [
                    encoded_sequences[other_type]
                    for other_type in self.sequence_types
                    if other_type != current_seq_type
                ]

                if not context_list:
                    logger.warning(
                        "Cross-attention context list empty for sequence type '%s'. Skipping.",
                        current_seq_type,
                    )
                    cross_attended_sequences[current_seq_type] = (
                        encoded_sequences[current_seq_type]
                    )
                    continue

                context = torch.cat(context_list, dim=1)

                attended_output = self.cross_attn[current_seq_type](
                    query=encoded_sequences[current_seq_type], context=context
                )
                cross_attended_sequences[current_seq_type] = attended_output

            encoded_sequences = cross_attended_sequences

        if global_condition is not None and self.film2 is not None:
            for seq_type in self.sequence_types:
                if seq_type in encoded_sequences and seq_type in self.film2:
                    encoded_sequences[seq_type] = self.film2[seq_type](
                        encoded_sequences[seq_type], global_condition
                    )

        if self.output_seq_type not in encoded_sequences:
            raise ValueError(
                f"Designated output sequence type '{self.output_seq_type}' "
                f"not found in processed sequences: {list(encoded_sequences.keys())}"
            )
        output_sequence = encoded_sequences[self.output_seq_type]

        output_prediction = self.head(output_sequence)

        return output_prediction


# =============================================================================
# Factory Function
# =============================================================================


def create_prediction_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to build the MultiEncoderTransformer from a configuration dictionary.

    Validates required keys before instantiating the model.

    Args:
        config: The configuration dictionary containing all necessary parameters.

    Returns:
        An initialized MultiEncoderTransformer model instance.

    Raises:
        ValueError: If required configuration keys are missing or invalid.
    """
    required_keys = [
        "input_variables",
        "target_variables",
        "sequence_types",
        "output_seq_type",
        "d_model",
        "nhead",
        "num_encoder_layers",
        "dim_feedforward",
        "dropout",
        "norm_first",
        "max_sequence_length",
    ]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required configuration keys for model creation: {missing_keys}"
        )

    sequence_dims_model = config["sequence_types"]
    if not isinstance(sequence_dims_model, dict) or not all(
        isinstance(v, list) for v in sequence_dims_model.values()
    ):
        raise ValueError(
            "Config 'sequence_types' must be a dict mapping type name to a list of variable names."
        )

    model = MultiEncoderTransformer(
        global_variables=config.get("global_variables", []),
        sequence_dims=sequence_dims_model,
        output_dim=len(config["target_variables"]),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        norm_first=config.get("norm_first", True),
        output_seq_type=config["output_seq_type"],
        max_sequence_length=config["max_sequence_length"],
    )

    model.input_vars = config["input_variables"]
    model.target_vars = config["target_variables"]
    model.global_vars = config.get("global_variables", [])

    return model


# =============================================================================
# Module Exports
# =============================================================================

__all__ = ["create_prediction_model", "MultiEncoderTransformer"]
