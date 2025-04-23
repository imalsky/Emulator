#!/usr/bin/env python3
"""
model.py – Encoder-only transformer for multi-source atmospheric data.

Implements the model architecture featuring:
- Sinusoidal positional encoding (dynamically sized table based on input).
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
    of different frequencies and adds them to the input tensor. The table
    is pre-computed up to a fixed maximum length.
    """
    # Define a maximum sequence length for the pre-computed table.
    # Adjust if your sequences can be longer.
    _MAX_LEN = 5000

    def __init__(self, d_model: int) -> None:
        """
        Initializes the SinePositionalEncoding layer.

        Args:
            d_model: The embedding dimension (must be even).

        Raises:
            ValueError: If d_model is not an even number.
        """
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even for SinePositionalEncoding, got {d_model}"
            )

        position = torch.arange(self._MAX_LEN, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, self._MAX_LEN, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it's part of the model state but not trained
        self.register_buffer("pe_table", pe)
        logger.debug(f"Initialized SinePositionalEncoding table up to length {self._MAX_LEN}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model].

        Returns:
            The input tensor with positional encodings added.

        Raises:
            ValueError: If the input sequence length exceeds the pre-computed `_MAX_LEN`.
        """
        batch_size, seq_length, _ = x.shape
        if seq_length > self._MAX_LEN:
            raise ValueError(
                f"Input sequence length ({seq_length}) exceeds the maximum "
                f"pre-computed length ({self._MAX_LEN}) for positional encoding. "
                f"Increase SinePositionalEncoding._MAX_LEN in model.py if needed."
            )
        # Add positional encoding from the pre-computed table
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
            dropout: Dropout probability.
            norm_first: If True, applies layer norm before attention/feedforward layers (Pre-LN).
                        If False, applies layer norm after (Post-LN).
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # Use the SinePositionalEncoding with its fixed internal max length
        self.pos_enc = SinePositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation=nn.GELU(), # Use GELU activation
        )

        # LayerNorm applied after the stack if norm_first is False
        encoder_norm = nn.LayerNorm(d_model) if not norm_first else None

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
        )
        # Dropout is typically applied within TransformerEncoderLayer and potentially after PE
        # self.dropout = nn.Dropout(dropout) # Often not needed here if applied inside layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Processes the input sequence through the encoder stack.

        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim].

        Returns:
            Encoded sequence tensor of shape [batch_size, seq_length, d_model].
        """
        x = self.input_proj(x)
        x = self.pos_enc(x) # Applies PE, raises error if seq_len > _MAX_LEN
        # x = self.dropout(x) # Optional dropout after PE
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
        # Use a slightly wider hidden layer
        hidden_dim = max(d_model, input_dim * 2) # Ensure hidden is at least d_model
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model), # Apply LayerNorm at the end
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
        # Projection to generate scale (gamma) and shift (beta) parameters
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
        # Generate gamma and beta from the condition vector
        # film_params shape: [batch_size, 2 * d_model]
        film_params = self.scale_shift_projection(condition)

        # Split into gamma and beta, each shape: [batch_size, d_model]
        gamma, beta = film_params.chunk(2, dim=-1)

        # Unsqueeze gamma and beta to allow broadcasting across the sequence length dimension
        # gamma shape: [batch_size, 1, d_model]
        # beta shape: [batch_size, 1, d_model]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # Apply FiLM: output = gamma * sequence + beta
        return gamma * sequence + beta


class CrossAttentionModule(nn.Module):
    """
    Performs cross-attention from a query sequence to a context sequence.

    Uses PyTorch's standard MultiheadAttention module, followed by dropout
    and layer normalization with a residual connection. Assumes Pre-LN structure.
    """

    def __init__(
        self, d_model: int, nhead: int, dropout: float = 0.1, norm_first: bool = True
    ) -> None:
        """
        Initializes the CrossAttentionModule.

        Args:
            d_model: The embedding dimension.
            nhead: The number of attention heads.
            dropout: Dropout probability for the attention mechanism and residual connection.
            norm_first: Whether the surrounding architecture uses Pre-LN (True) or Post-LN (False).
        """
        super().__init__()
        self.norm_first = norm_first
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model) # Normalize context separately
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        # LayerNorm after residual connection only needed for Post-LN
        self.norm_out = nn.LayerNorm(d_model) if not norm_first else nn.Identity()


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
        query_orig = query # For residual connection

        # Pre-Normalization (if norm_first)
        if self.norm_first:
            query = self.norm_q(query)
            context = self.norm_kv(context)

        # Multi-Head Attention
        # K and V are derived from the (potentially normalized) context
        attn_output, _attn_weights = self.multihead_attn(
            query=query, key=context, value=context, need_weights=False
        )

        # Residual connection and Dropout
        output = query_orig + self.dropout(attn_output)

        # Post-Normalization (if not norm_first)
        output = self.norm_out(output)

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
    Sequence length is determined dynamically from input, positional encoding table
    has a fixed maximum size internal to SinePositionalEncoding.
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
        # max_sequence_length is removed
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
            output_seq_type: The key from `sequence_dims` that defines the sequence
                             whose final representation is used for prediction.
        """
        super().__init__()

        self.d_model = d_model
        # Filter out empty sequence types from the input dict
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

        # --- Sequence Encoders ---
        self.encoders = nn.ModuleDict()
        for seq_type in self.sequence_types:
            input_feature_dim = len(sequence_dims[seq_type])
            if input_feature_dim == 0:
                # This case should ideally be filtered out earlier or raise an error in config validation
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
                # max_sequence_length removed here
                dropout=dropout,
                norm_first=norm_first,
            )

        # --- Global Feature Handling (FiLM) ---
        self.use_global = bool(global_variables)
        if self.use_global:
            num_global_features = len(global_variables)
            if num_global_features == 0:
                # Config validation should catch this, but safeguard
                raise ValueError(
                    "Configuration lists `global_variables` but the list is empty."
                )
            self.global_encoder = GlobalEncoder(
                num_global_features, d_model, dropout
            )
            # FiLM layers applied *after* initial encoding and *after* cross-attention
            self.film1 = nn.ModuleDict(
                {st: FiLMLayer(d_model) for st in self.sequence_types if st in self.encoders}
            )
            self.film2 = nn.ModuleDict(
                {st: FiLMLayer(d_model) for st in self.sequence_types if st in self.encoders}
            )
            logger.info("Global feature conditioning enabled using FiLM.")
        else:
            self.global_encoder = None
            self.film1 = None
            self.film2 = None
            logger.info(
                "No global variables specified; FiLM conditioning disabled."
            )

        # --- Cross-Attention ---
        self.cross_attn = nn.ModuleDict()
        # Only add cross-attention if there's more than one sequence type *with an encoder*
        if len(self.encoders) > 1:
            logger.info(
                "Initializing cross-attention modules between sequence types."
            )
            for seq_type in self.encoders.keys(): # Iterate over keys with actual encoders
                self.cross_attn[seq_type] = CrossAttentionModule(
                    d_model, nhead, dropout, norm_first
                )
        else:
            logger.info("Only one sequence type; cross-attention disabled.")

        # --- Prediction Head ---
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2), # Slightly wider hidden layer
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim),
        )

        # No max_seq_len attribute needed here anymore
        # self.max_seq_len = max_sequence_length

        self._initialize_weights()
        logger.info(
            f"Initialized MultiEncoderTransformer: "
            f"Sequences={list(self.encoders.keys())}, Output Source='{self.output_seq_type}', d_model={d_model}"
        )

    def _initialize_weights(self) -> None:
        """Initializes model weights using Xavier uniform for multi-dimensional tensors."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                # Use Xavier initialization for linear layers' weights
                if "weight" in name and isinstance(self.get_submodule(name.rsplit('.', 1)[0] if '.' in name else ''), nn.Linear):
                     nn.init.xavier_uniform_(param)
                # Consider specific initialization for embedding layers if added later
            elif "bias" in name:
                # Initialize biases to zero
                nn.init.zeros_(param)
            # LayerNorm weights initialized to 1, bias to 0 by default, which is usually fine

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Defines the forward pass of the model.

        Args:
            inputs: A dictionary containing input tensors. Expected keys match
                    active `sequence_types` and optionally 'global'.
                    - Sequence tensors shape: [batch_size, seq_length, feature_dim]
                    - Global tensor shape: [batch_size, global_dim]

        Returns:
            The prediction tensor, shape [batch_size, output_seq_length, output_dim].
            The output_seq_length is determined by the length of the sequence
            specified by `self.output_seq_type`.

        Raises:
            ValueError: If required inputs are missing or sequence lengths exceed
                        the internal maximum of SinePositionalEncoding.
            RuntimeError: If internal processing fails.
        """
        global_condition: Optional[Tensor] = None
        if self.use_global:
            if "global" not in inputs:
                raise ValueError(
                    "Model configured to use global features, but 'global' key missing from input dictionary."
                )
            global_features_input = inputs["global"]
            # Ensure global features are 2D [batch, features]
            if global_features_input.ndim == 1:
                global_features_input = global_features_input.unsqueeze(0) # Add batch dim if needed
            if global_features_input.ndim != 2:
                 raise ValueError(f"Expected global input to be 2D [batch, features], but got shape {global_features_input.shape}")

            if self.global_encoder is None: # Should not happen if use_global is True
                raise RuntimeError("Global encoder not initialized despite use_global being True.")
            global_condition = self.global_encoder(global_features_input)

        # --- Initial Encoding and First FiLM ---
        encoded_sequences: Dict[str, Tensor] = {}
        active_encoders = self.encoders.keys() # Use only keys with actual encoders
        for seq_type in active_encoders:
            if seq_type not in inputs:
                raise ValueError(
                    f"Missing required sequence input for type '{seq_type}'."
                )
            x = inputs[seq_type]
            # Length check now happens inside SinePositionalEncoding.forward

            encoded = self.encoders[seq_type](x)

            # Apply first FiLM layer if global conditioning is enabled
            if global_condition is not None and self.film1 is not None and seq_type in self.film1:
                encoded = self.film1[seq_type](encoded, global_condition)

            encoded_sequences[seq_type] = encoded

        # --- Cross-Attention ---
        if len(active_encoders) > 1:
            cross_attended_sequences: Dict[str, Tensor] = {}
            all_encoded_values = list(encoded_sequences.values()) # Get all currently encoded sequences

            for current_seq_type in active_encoders:
                # Prepare context: concatenate all *other* sequences
                context_list = [
                    encoded_sequences[other_type]
                    for other_type in active_encoders
                    if other_type != current_seq_type
                ]

                if not context_list:
                     # This case should not happen if len(active_encoders) > 1
                     logger.warning(f"Context list unexpectedly empty for {current_seq_type}")
                     cross_attended_sequences[current_seq_type] = encoded_sequences[current_seq_type]
                     continue

                # Concatenate along the sequence length dimension (dim=1)
                context = torch.cat(context_list, dim=1)

                # Perform cross-attention
                attended_output = self.cross_attn[current_seq_type](
                    query=encoded_sequences[current_seq_type], context=context
                )
                cross_attended_sequences[current_seq_type] = attended_output

            encoded_sequences = cross_attended_sequences # Update with cross-attended results

        # --- Second FiLM ---
        if global_condition is not None and self.film2 is not None:
            for seq_type in active_encoders:
                if seq_type in encoded_sequences and seq_type in self.film2:
                    encoded_sequences[seq_type] = self.film2[seq_type](
                        encoded_sequences[seq_type], global_condition
                    )

        # --- Prediction Head ---
        if self.output_seq_type not in encoded_sequences:
            # This error indicates a logic problem or config mismatch
            raise ValueError(
                f"Designated output sequence type '{self.output_seq_type}' "
                f"not found among processed sequences: {list(encoded_sequences.keys())}. "
                f"Ensure it's included in 'sequence_types' in the config and has variables."
            )
        # Select the final representation of the designated output sequence type
        output_sequence_representation = encoded_sequences[self.output_seq_type]

        # Apply the prediction head
        output_prediction = self.head(output_sequence_representation)

        return output_prediction


# =============================================================================
# Factory Function
# =============================================================================


def create_prediction_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to build the MultiEncoderTransformer from a configuration dictionary.

    Validates required keys before instantiating the model. Handles dynamic sequence length.

    Args:
        config: The configuration dictionary containing all necessary parameters.
                `max_sequence_length` is no longer required.

    Returns:
        An initialized MultiEncoderTransformer model instance.

    Raises:
        ValueError: If required configuration keys are missing or invalid.
    """
    # Define required keys for model instantiation
    required_keys = [
        "input_variables",      # Needed to determine input features implicitly
        "target_variables",     # Needed for output_dim
        "sequence_types",       # Defines structure and input features per sequence
        "output_seq_type",      # Specifies which sequence representation feeds the head
        "d_model",              # Core model dimension
        "nhead",                # Number of attention heads
        "num_encoder_layers",   # Depth of sequence encoders
        "dim_feedforward",      # Width of FFN in encoders
        "dropout",              # Dropout rate
        "norm_first",           # Pre-LN or Post-LN
    ]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required configuration keys for model creation: {missing_keys}"
        )

    # Validate sequence_types structure
    sequence_dims_model = config["sequence_types"]
    if not isinstance(sequence_dims_model, dict) or not all(
        isinstance(v, list) for v in sequence_dims_model.values()
    ):
        raise ValueError(
            "Config 'sequence_types' must be a dict mapping type name to a list of variable names."
        )

    # Instantiate the model
    model = MultiEncoderTransformer(
        global_variables=config.get("global_variables", []), # Optional global vars
        sequence_dims=sequence_dims_model,                  # Sequence structure
        output_dim=len(config["target_variables"]),         # Number of target features
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        norm_first=config.get("norm_first", True),          # Default to Pre-LN
        output_seq_type=config["output_seq_type"],
        # max_sequence_length argument is removed
    )

    # Store variable lists on the model instance for potential reference (optional)
    model.input_vars = config["input_variables"]
    model.target_vars = config["target_variables"]
    model.global_vars = config.get("global_variables", [])

    return model


# =============================================================================
# Module Exports
# =============================================================================

__all__ = ["create_prediction_model", "MultiEncoderTransformer"]
