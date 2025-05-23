#!/usr/bin/env python3
import math
import sys # Used for critical error exits in the model factory function.
from typing import Dict, Optional, List, Any, Tuple, Set 

import torch
import torch.nn as nn
from torch import Tensor

import logging
logger = logging.getLogger(__name__)


class SinePositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding, adding time-step information to input sequences.
    This technique helps the transformer model understand the order of elements in a sequence,
    as the self-attention mechanism itself is permutation-invariant.
    The encodings are pre-computed up to `max_len` and added to the input embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        # `d_model` must be even for sinusoidal encoding, as pairs of dimensions are used
        # for sine and cosine components. This validation is handled by `create_prediction_model`.
        
        # Pre-allocate a tensor for positional encodings.
        pe = torch.zeros(max_len, d_model)
        # Create a tensor representing positions (0, 1, ..., max_len-1).
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # Calculate the division term for the wavelengths of the sine/cosine functions.
        # The wavelengths form a geometric progression from 2*pi to 10000*2*pi.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices in the `d_model` dimension.
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the `d_model` dimension.
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register `pe` as a buffer. Buffers are part of the model's state but not trained.
        # `unsqueeze(0)` adds a batch dimension for broadcasting.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Adds positional encoding to the input tensor.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model).
        Returns:
            Tensor with added positional encodings, of the same shape as input.
        """
        seq_len = x.size(1)
        # Ensure the input sequence length does not exceed the pre-computed max_len.
        # This check is important for JIT compilation and runtime safety.
        torch._assert(seq_len <= self.pe.size(1), 
                      f"Input sequence length {seq_len} exceeds PE table max_len {self.pe.size(1)}")
        # Add the corresponding slice of positional encodings to the input.
        return x + self.pe[:, :seq_len, :]


class SequenceEncoder(nn.Module):
    """
    A standard transformer encoder module for processing sequential data.
    It consists of an input projection, optional positional encoding, and a stack
    of transformer encoder layers.
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
        positional_encoding_module: Optional[SinePositionalEncoding] = None
    ):
        super().__init__()
        # Project the input features to the model's dimension `d_model`.
        self.input_proj = nn.Linear(input_dim, d_model)
        # Store the positional encoding module (e.g., SinePositionalEncoding).
        self.pos_encoder = positional_encoding_module 
        # Define a single transformer encoder layer. `norm_first=True` applies layer norm
        # before self-attention and feedforward sub-layers, which can improve stability.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=norm_first,
            activation='gelu' # GELU activation is common in transformers.
        )
        # Stack multiple encoder layers to form the complete encoder.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Processes the input sequence through the encoder.
        Args:
            x: Input sequence tensor (batch_size, seq_len, input_dim).
            src_key_padding_mask: Optional mask indicating padded elements in `x`
                                  (batch_size, seq_len), where True means PAD.
        Returns:
            Encoded sequence tensor (batch_size, seq_len, d_model).
        """
        x = self.input_proj(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class GlobalEncoder(nn.Module):
    """
    Encodes global (non-sequential) features into a fixed-size representation.
    This typically involves a few linear layers with activations and normalization.
    """
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model * 2), # Expand dimensionality.
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), # Project back to `d_model`.
            nn.LayerNorm(d_model) # Normalize the output.
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Processes global features.
        Args:
            x: Global features tensor (batch_size, input_dim).
        Returns:
            Encoded global features tensor (batch_size, d_model).
        """
        return self.encoder(x)


class FiLMLayer(nn.Module):
    """
    Implements Feature-wise Linear Modulation (FiLM).
    FiLM layers adapt sequence representations based on a conditioning vector (e.g.,
    encoded global features) by applying an affine transformation (scale and shift)
    feature-wise.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Linear layer to generate FiLM parameters (gamma and beta) from the condition.
        # It outputs 2 * d_model, as gamma and beta each have d_model dimensions.
        self.film_proj = nn.Linear(d_model, d_model * 2)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """
        Applies FiLM to the input sequence `x` using `condition`.
        Args:
            x: Input sequence tensor (batch_size, seq_len, d_model).
            condition: Conditioning tensor (batch_size, d_model).
        Returns:
            Modulated sequence tensor (batch_size, seq_len, d_model).
        """
        # Generate gamma (scale) and beta (shift) parameters from the condition.
        film_params = self.film_proj(condition)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        # Apply the affine transformation: gamma * x + beta.
        # Unsqueeze gamma and beta to allow broadcasting across the sequence length.
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


class CrossAttentionModule(nn.Module):
    """
    A module for performing cross-attention between two sequences, or between
    a query sequence and a key-value sequence. It's a standard attention block
    with residual connection and layer normalization.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        # Multi-head attention layer. `batch_first=True` means input/output tensors
        # have batch dimension first.
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout) # Dropout for the residual connection.

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_value_padding_mask: Optional[Tensor] = None, 
        query_padding_mask: Optional[Tensor] = None      
    ) -> Tensor:
        """
        Performs cross-attention.
        Args:
            query: Query sequence (batch_size, query_len, d_model).
            key_value: Key and Value sequence (batch_size, kv_len, d_model).
            key_value_padding_mask: Mask for `key_value` (batch_size, kv_len), True means PAD.
            query_padding_mask: Mask for `query` (batch_size, query_len), True means PAD.
                                Used to zero out outputs at padded query positions.
        Returns:
            Output tensor after attention, residual connection, and normalization
            (batch_size, query_len, d_model).
        """
        # Compute attention output. `key_value` is used for both keys and values.
        attn_out, _ = self.attn(query, key_value, key_value, key_padding_mask=key_value_padding_mask)
        # Residual connection: add attention output to the original query.
        res_query = query + self.dropout_layer(attn_out)
        # Apply layer normalization.
        out_norm = self.norm(res_query)
        # If a query padding mask is provided, zero out the outputs at padded query positions.
        # This ensures that padded parts of the query sequence do not carry meaningful information.
        if query_padding_mask is not None:
            out_norm = out_norm.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)
        return out_norm


class MultiEncoderTransformer(nn.Module):
    """
    A transformer model that can process multiple input sequences (up to 2) and
    optional global features. It uses separate encoders for each sequence type,
    can fuse information using FiLM layers if global features are present,
    and can perform cross-attention between sequences if two are provided.
    The final output is projected to the target dimension.
    """
    def __init__(
        self,
        sequence_input_dims_ref: Dict[str, int], 
        global_input_dim: int,
        output_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        ordered_active_seq_names: List[str], 
        output_seq_processing_idx: int, 
        pe_module: Optional[SinePositionalEncoding],
        padding_value: float = 0.0,
        dropout: float = 0.1,
        norm_first: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.padding_value = padding_value # Value used for padding in input data.
        self.output_dim = output_dim # Dimension of the final output prediction.
        # Index indicating which processed sequence (0 or 1) is used for the final output.
        self.output_processing_idx = output_seq_processing_idx
        
        # Stores the input feature dimension for each active sequence type, used for validation.
        self.sequence_input_dims_ref = sequence_input_dims_ref 
        
        self.num_active_sequences = len(ordered_active_seq_names)
        # The factory `create_prediction_model` ensures 0 < num_active_sequences <= 2.

        # Initialize components for the first sequence encoder (if active).
        self.seq_name_0: Optional[str] = None
        self.encoder_0: Optional[SequenceEncoder] = None
        self.film_0: Optional[FiLMLayer] = None
        
        # Initialize components for the second sequence encoder (if active).
        self.seq_name_1: Optional[str] = None
        self.encoder_1: Optional[SequenceEncoder] = None
        self.film_1: Optional[FiLMLayer] = None

        if self.num_active_sequences >= 1:
            self.seq_name_0 = ordered_active_seq_names[0]
            self.encoder_0 = SequenceEncoder(
                input_dim=self.sequence_input_dims_ref[self.seq_name_0],
                d_model=d_model, nhead=nhead, num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward, dropout=dropout, norm_first=norm_first,
                positional_encoding_module=pe_module # Share PE module if applicable.
            )
        
        if self.num_active_sequences > 1:
            self.seq_name_1 = ordered_active_seq_names[1]
            # Internal consistency check, should be guaranteed by factory.
            torch._assert(self.seq_name_1 is not None and self.seq_name_1 in self.sequence_input_dims_ref, 
                          "Internal config error: seq_name_1 not in sequence_input_dims_ref.")
            if self.seq_name_1 is not None : # Check for type checker.
                 self.encoder_1 = SequenceEncoder(
                    input_dim=self.sequence_input_dims_ref[self.seq_name_1],
                    d_model=d_model, nhead=nhead, num_layers=num_encoder_layers,
                    dim_feedforward=dim_feedforward, dropout=dropout, norm_first=norm_first,
                    positional_encoding_module=pe_module # Share PE module.
                )
        
        # Global feature encoder and FiLM layers are only created if global_input_dim > 0.
        self.global_encoder: Optional[GlobalEncoder] = None
        if global_input_dim > 0:
            self.global_encoder = GlobalEncoder(global_input_dim, d_model, dropout)
            # FiLM layers to modulate sequence encodings with global features.
            if self.encoder_0 is not None: self.film_0 = FiLMLayer(d_model)
            if self.encoder_1 is not None: self.film_1 = FiLMLayer(d_model)
        
        # Cross-attention modules are only created if there are two active sequences.
        self.cross_attention_0_to_1: Optional[CrossAttentionModule] = None
        self.cross_attention_1_to_0: Optional[CrossAttentionModule] = None
        if self.num_active_sequences == 2:
            self.cross_attention_0_to_1 = CrossAttentionModule(d_model, nhead, dropout)
            self.cross_attention_1_to_0 = CrossAttentionModule(d_model, nhead, dropout)

        # Final linear projection from `d_model` to the `output_dim`.
        self.output_proj = nn.Linear(d_model, output_dim)
        self._init_parameters() # Apply custom weight initialization.

    def _init_parameters(self):
        """Applies custom weight initialization to model parameters."""
        for n, p in self.named_parameters():
            if p.dim() > 1: # For weight matrices (2D or higher).
                nn.init.xavier_uniform_(p)
            elif 'bias' in n: # For bias terms.
                nn.init.zeros_(p)
            elif 'weight' in n: # For 1D weight vectors (e.g., LayerNorm weights).
                if 'norm' in n.lower() and p.ndim == 1: # Specifically for LayerNorm weights.
                    nn.init.ones_(p)
                else: # Other 1D weights.
                    nn.init.uniform_(p, -0.1, 0.1)
        # Initialize output projection layer with a smaller gain, as it's often beneficial
        # for the final layer before a loss function.
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.01)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def _generate_pytorch_padding_mask(self, x: Tensor) -> Tensor:
        """
        Generates a padding mask where True indicates a padded element.
        This is used if input_masks are not directly provided to the forward method.
        A position is considered padded if all features at that position match self.padding_value.
        Args:
            x: Input tensor (batch_size, seq_len, features).
        Returns:
            Boolean mask tensor (batch_size, seq_len), True for PAD.
        """
        return torch.all(x == self.padding_value, dim=-1)

    def forward(
        self,
        inputs: Dict[str, Tensor],
        input_masks: Optional[Dict[str, Tensor]] = None 
    ) -> Tensor:
        """
        Forward pass of the MultiEncoderTransformer.
        `input_masks` provided here should have True for VALID data points.
        These are converted internally to PyTorch's convention (True for PADDED elements)
        before being passed to attention mechanisms.
        """
        # Determine which input sequence's shape will define the output sequence length.
        # This is based on `output_seq_processing_idx`.
        output_source_seq_name_for_shape: str = "" 
        if self.output_processing_idx == 0:
            torch._assert(self.seq_name_0 is not None, 
                          "Output index 0 selected, but seq_name_0 is not configured.")
            if self.seq_name_0 is not None : output_source_seq_name_for_shape = self.seq_name_0
        elif self.output_processing_idx == 1:
            torch._assert(self.seq_name_1 is not None, 
                          "Output index 1 selected, but seq_name_1 is not configured.")
            if self.seq_name_1 is not None : output_source_seq_name_for_shape = self.seq_name_1
        else: # Should be caught by factory config validation.
            torch._assert(False, f"Invalid output_processing_idx: {self.output_processing_idx}.") 
        
        torch._assert(output_source_seq_name_for_shape in inputs, 
                      f"Key '{output_source_seq_name_for_shape}' (for output shape) not in inputs dict.")
        
        # Get batch size, padded length, and device from the reference input tensor.
        ref_input_for_shape = inputs[output_source_seq_name_for_shape]
        batch_size = ref_input_for_shape.size(0)
        output_padded_length = ref_input_for_shape.size(1) 
        device = ref_input_for_shape.device

        # Prepare PyTorch-style padding mask for sequence 0 (True means PAD).
        # If `input_masks` (True=VALID) are provided, invert them. Otherwise, generate from padding value.
        mask_0_pytorch: Optional[Tensor] = None
        if self.seq_name_0 is not None:
            torch._assert(self.seq_name_0 in inputs, f"Input for sequence '{self.seq_name_0}' missing.")
            input_seq_0 = inputs[self.seq_name_0]
            # Validate feature dimension of input sequence 0.
            expected_dim0 = self.sequence_input_dims_ref[self.seq_name_0]
            torch._assert(input_seq_0.size(2) == expected_dim0, 
                           f"Feature dim for {self.seq_name_0}. Expected {expected_dim0}, got {input_seq_0.size(2)}")
            if input_masks is not None and self.seq_name_0 in input_masks:
                mask_0_pytorch = ~input_masks[self.seq_name_0] # Invert: True=VALID to True=PAD.
            else:
                mask_0_pytorch = self._generate_pytorch_padding_mask(input_seq_0)
        
        # Prepare PyTorch-style padding mask for sequence 1.
        mask_1_pytorch: Optional[Tensor] = None
        if self.seq_name_1 is not None:
            torch._assert(self.seq_name_1 in inputs, f"Input for sequence '{self.seq_name_1}' missing.")
            input_seq_1 = inputs[self.seq_name_1]
            expected_dim1 = self.sequence_input_dims_ref[self.seq_name_1]
            torch._assert(input_seq_1.size(2) == expected_dim1,
                           f"Feature dim for {self.seq_name_1}. Expected {expected_dim1}, got {input_seq_1.size(2)}")
            if input_masks is not None and self.seq_name_1 in input_masks:
                mask_1_pytorch = ~input_masks[self.seq_name_1] # Invert.
            else:
                mask_1_pytorch = self._generate_pytorch_padding_mask(input_seq_1)

        # Encode global features if the global encoder exists.
        global_enc: Optional[Tensor] = None
        if self.global_encoder is not None:
            torch._assert('global' in inputs, "Input 'global' features missing when global_encoder is active.")
            global_enc = self.global_encoder(inputs['global'])

        # Process sequence 0 through its encoder and optional FiLM layer.
        encoded_0: Optional[Tensor] = None
        if self.encoder_0 is not None and self.seq_name_0 is not None:
            encoded_0 = self.encoder_0(inputs[self.seq_name_0], mask_0_pytorch)
            if self.film_0 is not None and global_enc is not None:
                encoded_0 = self.film_0(encoded_0, global_enc) # Modulate with global features.
            if mask_0_pytorch is not None and encoded_0 is not None:
                # Zero out padded positions after all processing for this sequence.
                # The mask `mask_0_pytorch` should already be boolean. `unsqueeze` makes it broadcastable.
                encoded_0 = encoded_0.masked_fill(mask_0_pytorch.unsqueeze(-1), 0.0)
        
        # Process sequence 1 similarly.
        encoded_1: Optional[Tensor] = None
        if self.encoder_1 is not None and self.seq_name_1 is not None:
            encoded_1 = self.encoder_1(inputs[self.seq_name_1], mask_1_pytorch)
            if self.film_1 is not None and global_enc is not None:
                encoded_1 = self.film_1(encoded_1, global_enc)
            if mask_1_pytorch is not None and encoded_1 is not None:
                encoded_1 = encoded_1.masked_fill(mask_1_pytorch.unsqueeze(-1), 0.0)

        # Perform cross-attention if two sequences are active and encoders are defined.
        if self.cross_attention_0_to_1 is not None and self.cross_attention_1_to_0 is not None:
            torch._assert(encoded_0 is not None and encoded_1 is not None, 
                          "Cross-attention is active, but required encoded sequences (0 and 1) are not available. Check inputs.")
            if encoded_0 is not None and encoded_1 is not None : 
                original_encoded_0 = encoded_0 
                original_encoded_1 = encoded_1
                # Sequence 0 attends to sequence 1.
                encoded_0 = self.cross_attention_0_to_1(original_encoded_0, original_encoded_1, mask_1_pytorch, mask_0_pytorch)
                # Sequence 1 attends to sequence 0.
                encoded_1 = self.cross_attention_1_to_0(original_encoded_1, original_encoded_0, mask_0_pytorch, mask_1_pytorch)
        
        # Select the final sequence representation based on `output_processing_idx`.
        final_processed_sequence: Optional[Tensor] = None
        final_output_padding_mask: Optional[Tensor] = None # This mask is True=PAD.

        if self.output_processing_idx == 0:
            final_processed_sequence = encoded_0
            final_output_padding_mask = mask_0_pytorch
        elif self.output_processing_idx == 1: 
            final_processed_sequence = encoded_1
            final_output_padding_mask = mask_1_pytorch
        
        torch._assert(final_processed_sequence is not None, 
                      "Selected output sequence (final_processed_sequence) is None. This indicates missing input for the designated output stream.")
        
        # Fallback for JIT if `final_processed_sequence` somehow ends up as None despite asserts.
        if final_processed_sequence is None: 
            return torch.zeros(batch_size, output_padded_length, self.output_dim, device=device, dtype=torch.float32)

        # Project the chosen sequence representation to the final output dimension.
        output = self.output_proj(final_processed_sequence)

        # Zero out padded positions in the final output.
        if final_output_padding_mask is not None:
            # `final_output_padding_mask` is already boolean (True=PAD).
            output = output.masked_fill(final_output_padding_mask.unsqueeze(-1), 0.0)
            
        return output


def create_prediction_model(
    config: Dict[str, Any], device: Optional[torch.device] = None
) -> MultiEncoderTransformer:
    """
    Factory function to create and configure the MultiEncoderTransformer model.
    It performs extensive validation of the configuration related to model
    architecture, input/output variables, and sequence types.
    Exits with sys.exit(1) on critical configuration errors.
    """
    # Validate presence of essential keys in the configuration.
    required_keys = [
        "input_variables", "target_variables", "sequence_types", "output_seq_type",
        "d_model", "nhead", "num_encoder_layers", "dim_feedforward"
    ]
    for key in required_keys:
        if key not in config:
            logger.critical(f"Config missing required key: '{key}'. Exiting.")
            sys.exit(1) 

    # Extract and validate core configuration values.
    input_vars_cfg = config["input_variables"]
    target_vars_cfg = config["target_variables"]
    seq_types_from_config: Dict[str, List[str]] = config["sequence_types"] 
    output_seq_type_from_config = config["output_seq_type"]
    d_model_cfg = config["d_model"]
    nhead_from_config = config["nhead"]
    pos_enc_type_str = config.get("positional_encoding_type", "sinusoidal")
    max_seq_len_for_pe = config.get("max_sequence_length", 512)
    padding_val_cfg = float(config.get("padding_value", 0.0))
    dropout_val = config.get("dropout", 0.1)
    norm_first_val = config.get("norm_first", False)
    num_encoder_layers_val = config["num_encoder_layers"]
    dim_feedforward_val = config["dim_feedforward"]

    # Basic type and content validation for critical config items.
    if not (isinstance(input_vars_cfg, list) and input_vars_cfg): logger.critical("'input_variables' must be non-empty list. Exiting."); sys.exit(1)
    if not (isinstance(target_vars_cfg, list) and target_vars_cfg): logger.critical("'target_variables' must be non-empty list. Exiting."); sys.exit(1)
    if not (isinstance(seq_types_from_config, dict) and seq_types_from_config): logger.critical("'sequence_types' must be non-empty dict. Exiting."); sys.exit(1)
    if not isinstance(output_seq_type_from_config, str) or not output_seq_type_from_config:
        logger.critical("'output_seq_type' must be non-empty string. Exiting."); sys.exit(1)

    global_vars_cfg = config.get("global_variables", [])
    if not isinstance(global_vars_cfg, list): logger.critical("'global_variables' must be a list. Exiting."); sys.exit(1)

    # Process `sequence_types` to determine active sequences, their input dimensions,
    # and maintain a fixed order for consistency.
    active_sequence_input_dims: Dict[str, int] = {} # Stores input_dim for each active sequence.
    ordered_active_type_names: List[str] = [] # Maintains order of active sequences.
    all_vars_in_active_sequences: Set[str] = set() # Tracks all variables used in sequences.

    # Iterate through sequence types in the order they appear in the config.
    for seq_name_key in seq_types_from_config.keys():
        var_name_list = seq_types_from_config[seq_name_key]
        if not isinstance(var_name_list, list): logger.critical(f"Var list for seq type '{seq_name_key}' must be a list. Exiting."); sys.exit(1)
        # Skip sequence types with no variables defined, as they cannot be processed.
        if not var_name_list: 
            logger.info(f"Sequence type '{seq_name_key}' has an empty variable list in config; it will be ignored for model construction.")
            continue 

        # Ensure no duplicate variables within a single sequence type.
        if len(set(var_name_list)) != len(var_name_list): logger.critical(f"Duplicate vars in seq type '{seq_name_key}'. Exiting."); sys.exit(1)
        
        current_type_vars = set(var_name_list)
        # Ensure variables are not shared across different active sequence types.
        if not current_type_vars.isdisjoint(all_vars_in_active_sequences):
            overlap = current_type_vars.intersection(all_vars_in_active_sequences)
            logger.critical(f"Variable overlap ({overlap}) with other active sequence types for '{seq_name_key}'. Exiting."); sys.exit(1)
        # Ensure sequence variables do not overlap with global variables.
        if not current_type_vars.isdisjoint(set(global_vars_cfg)):
            overlap = current_type_vars.intersection(set(global_vars_cfg))
            logger.critical(f"Variable overlap ({overlap}) between seq type '{seq_name_key}' and global variables. Exiting."); sys.exit(1)
        
        all_vars_in_active_sequences.update(current_type_vars)
        ordered_active_type_names.append(seq_name_key)
        active_sequence_input_dims[seq_name_key] = len(var_name_list)

    # The model requires at least one active sequence type.
    if not ordered_active_type_names: logger.critical("No sequence types with actual variables defined. Model requires at least one. Exiting."); sys.exit(1)
    # This specific implementation supports a maximum of two active sequence types.
    if len(ordered_active_type_names) > 2: logger.critical(f"Model configured for {len(ordered_active_type_names)} active sequence types. Supports max 2. Exiting."); sys.exit(1)

    # Ensure the specified `output_seq_type` is one of the active sequence types.
    if output_seq_type_from_config not in active_sequence_input_dims:
        logger.critical(f"'output_seq_type' ('{output_seq_type_from_config}') is not among the active sequence types with variables: {ordered_active_type_names}. Exiting."); sys.exit(1)

    # Determine the index of the output sequence type in the ordered list.
    output_processing_idx_val = ordered_active_type_names.index(output_seq_type_from_config)

    # Validate that all declared `input_variables` and `target_variables` are assigned
    # to either global features or one of the active sequence types.
    all_vars_defined_in_model = set(global_vars_cfg).union(all_vars_in_active_sequences)
    if unassigned_in := set(input_vars_cfg) - all_vars_defined_in_model:
        logger.critical(f"Input variables {unassigned_in} not defined in global or any active sequence type. Exiting."); sys.exit(1)
    if unassigned_tgt := set(target_vars_cfg) - all_vars_defined_in_model:
        logger.critical(f"Target variables {unassigned_tgt} not defined in global or any active sequence type. Exiting."); sys.exit(1)

    # Validate transformer core dimensions.
    if not (isinstance(d_model_cfg, int) and d_model_cfg > 0): logger.critical("d_model must be > 0. Exiting."); sys.exit(1)
    if not (isinstance(nhead_from_config, int) and nhead_from_config > 0): logger.critical("nhead must be > 0. Exiting."); sys.exit(1)
    
    # Adjust `nhead` if `d_model` is not divisible by the configured `nhead`.
    # It finds the largest divisor of `d_model` that is less than or equal to the original `nhead`.
    final_nhead_val = nhead_from_config
    if d_model_cfg % nhead_from_config != 0:
        original_nhead = nhead_from_config
        for potential_nhead in range(original_nhead -1, 0, -1): 
            if d_model_cfg % potential_nhead == 0: final_nhead_val = potential_nhead; break
        else: 
             if d_model_cfg % original_nhead != 0: 
                logger.critical(f"d_model ({d_model_cfg}) is not divisible by any nhead <= {original_nhead}. Adjust config. Exiting."); sys.exit(1)
        logger.warning(f"d_model ({d_model_cfg}) not divisible by nhead ({original_nhead}). Adjusted nhead to {final_nhead_val}.")
        config["nhead"] = final_nhead_val # Update config in-place for consistency if used later.

    # Instantiate the positional encoding module if specified.
    pe_module_instance: Optional[SinePositionalEncoding] = None
    _pe_type_lower = pos_enc_type_str.lower() if pos_enc_type_str else "none"
    if _pe_type_lower in ["sinusoidal", "sine", "sin"]:
        # Sinusoidal PE requires `d_model` to be even.
        if d_model_cfg % 2 != 0: logger.critical(f"SinePE requires even d_model, got {d_model_cfg}. Exiting."); sys.exit(1)
        if max_seq_len_for_pe <=0 : logger.critical(f"max_sequence_length for PE table must be > 0, got {max_seq_len_for_pe}. Exiting."); sys.exit(1)
        pe_module_instance = SinePositionalEncoding(d_model_cfg, max_len=max_seq_len_for_pe)
    elif _pe_type_lower != "none": # Only 'sinusoidal' or 'none' are supported.
        logger.critical(f"Unsupported positional_encoding_type: '{pos_enc_type_str}'. Valid: 'sinusoidal' or 'none'. Exiting."); sys.exit(1)

    # Create the MultiEncoderTransformer instance with validated and processed parameters.
    model = MultiEncoderTransformer(
        sequence_input_dims_ref=active_sequence_input_dims,
        global_input_dim=len(global_vars_cfg),
        output_dim=len(target_vars_cfg),
        d_model=d_model_cfg,
        nhead=final_nhead_val,
        num_encoder_layers=num_encoder_layers_val,
        dim_feedforward=dim_feedforward_val,
        ordered_active_seq_names=ordered_active_type_names,
        output_seq_processing_idx=output_processing_idx_val,
        pe_module=pe_module_instance,
        padding_value=padding_val_cfg,
        dropout=dropout_val,
        norm_first=norm_first_val
    )
    
    logger.info("MultiEncoderTransformer instance created successfully by factory.")

    # Move the model to the specified device if provided.
    if device is not None:
        actual_device = device
        if isinstance(device, str): # Convert string device name to torch.device object.
            try: actual_device = torch.device(device)
            except RuntimeError as e: logger.critical(f"Invalid device string '{device}': {e}. Exiting."); sys.exit(1)
        
        if not isinstance(actual_device, torch.device): 
             logger.critical("Device must be torch.device instance or valid string. Exiting."); sys.exit(1)
        try:
            model = model.to(actual_device)
            logger.info(f"Model moved to device: {actual_device}")
        except Exception as e: # Catch any errors during model transfer.
            logger.critical(f"Failed to move model to device '{actual_device}': {e}. Exiting.", exc_info=True); sys.exit(1)
            
    return model

__all__ = ["MultiEncoderTransformer", "create_prediction_model"]