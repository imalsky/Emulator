#!/usr/bin/env python3
import math
import logging
import sys
from typing import Dict, Optional, List, Set, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class SinePositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding module.

    Generates fixed sinusoidal positional encoding and adds it to the input.
    Expected input shape: [batch_size, seq_length, d_model].
    The d_model must be an even number.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        if d_model % 2 != 0:
            msg = (f"SinePositionalEncoding requires an even d_model, "f"got {d_model}.")
            logger.critical(msg)
            raise ValueError(msg)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()* (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor."""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            logger.error(
                "Input sequence length (%d) exceeds PE table max_len (%d). "
                "Adjust 'max_sequence_length' in config.",
                seq_len, self.pe.size(1)
            )
            raise ValueError(
                f"Input sequence length ({seq_len}) > PE max_len ({self.pe.size(1)})"
            )
        return x + self.pe[:, :seq_len, :]


class SequenceEncoder(nn.Module):
    """
    Bidirectional encoder for sequence data.

    Projects inputs, optionally adds positional encodings, and processes
    via TransformerEncoder.
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
        positional_encoding_type: Optional[str] = "sinusoidal",
        max_len: int = 512
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder: Optional[nn.Module] = None
        self.d_model = d_model

        pe_type_lower = (
            positional_encoding_type.lower()
            if positional_encoding_type
            else "none"
        )

        if pe_type_lower == "none":
            self.pos_encoder = None
        elif pe_type_lower in ["sinusoidal", "sine", "sin"]:
            try:
                self.pos_encoder = SinePositionalEncoding(d_model, max_len=max_len)
            except ValueError as e:
                logger.critical(
                    "Failed to init SinePositionalEncoding for SequenceEncoder "
                    "(d_model likely odd): %s. Exiting.", e
                )
                sys.exit(1)
        else:
            logger.critical(
                "Unsupported positional_encoding_type: '%s'. Valid: "
                "'sinusoidal' or 'none'. Exiting.", positional_encoding_type
            )
            sys.exit(1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=norm_first,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the sequence encoder.

        Args:
            x: Input tensor [batch_size, seq_length, input_dim].
            src_key_padding_mask: Mask for padded tokens in x.
                                  Shape [batch_size, seq_length],
                                  True indicates a padded position.
        Returns:
            Encoded tensor [batch_size, seq_length, d_model].
        """
        x = self.input_proj(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class GlobalEncoder(nn.Module):
    """
    Encodes global features using an MLP.
    Input: [batch_size, global_dim], Output: [batch_size, d_model].
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Conditions a sequence with global features.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.film_proj = nn.Linear(d_model, d_model * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sequence tensor [batch_size, seq_length, d_model].
            condition: Global condition tensor [batch_size, d_model].
        Returns:
            Conditioned sequence tensor [batch_size, seq_length, d_model].
        """
        film_params = self.film_proj(condition)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


class CrossAttentionModule(nn.Module):
    """Cross-attention module with Add & Norm."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_value_padding_mask: Optional[torch.Tensor] = None,
        query_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Query sequence [B, query_len, D].
            key_value: Key/Value sequence [B, kv_len, D].
            key_value_padding_mask: Mask for key_value [B, kv_len], True=PAD.
            query_padding_mask: Mask for query [B, query_len], True=PAD.
        Returns:
            Output tensor [B, query_len, D].
        """
        attn_output, _ = self.multihead_attn(
            query, key_value, key_value,
            key_padding_mask=key_value_padding_mask
        )
        res_query = query + self.dropout(attn_output)
        norm_output = self.norm(res_query)

        if query_padding_mask is not None:
            norm_output = norm_output.masked_fill(
                query_padding_mask.unsqueeze(-1), 0.0
            )
        return norm_output


class MultiEncoderTransformer(nn.Module):
    """
    Encoder-only transformer for multi-sequence data with optional global
    features and cross-attention.
    """

    def __init__(
        self,
        global_variables: List[str],
        sequence_dims: Dict[str, Dict[str, int]],
        output_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        output_seq_type: str, # Creator (create_prediction_model) ensures this is valid
        dropout: float = 0.1,
        norm_first: bool = False,
        max_sequence_length: int = 512,
        positional_encoding_type: Optional[str] = "sinusoidal",
        padding_value: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.padding_value = padding_value
        self.output_dim = output_dim
        self.nhead = nhead

        self.sequence_dims = sequence_dims
        self.sequence_types = list(self.sequence_dims.keys())
        self.global_variables = global_variables
        self.has_global = bool(self.global_variables)
        self.output_seq_type = output_seq_type

        self.sequence_encoders = nn.ModuleDict()
        for seq_type, var_dict in self.sequence_dims.items():
            if not var_dict:
                 logger.warning(
                    "Empty var_dict for seq_type '%s' in MultiEncoderTransformer __init__; "
                    "encoder will not be created. This should be caught by create_prediction_model.",
                    seq_type
                 )
                 continue
            self.sequence_encoders[seq_type] = SequenceEncoder(
                input_dim=len(var_dict), d_model=d_model, nhead=self.nhead,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward, dropout=dropout,
                norm_first=norm_first,
                positional_encoding_type=positional_encoding_type,
                max_len=max_sequence_length
            )

        self.global_encoder: Optional[GlobalEncoder] = None
        self.film_layers: Optional[nn.ModuleDict] = None
        if self.has_global:
            self.global_encoder = GlobalEncoder(
                input_dim=len(self.global_variables), d_model=d_model,
                dropout=dropout
            )
            self.film_layers = nn.ModuleDict({
                st: FiLMLayer(d_model) for st in self.sequence_types
                if st in self.sequence_encoders
            })

        self.cross_attention: Optional[nn.ModuleDict] = None
        self.use_cross_attention = (
            len(self.sequence_types) == 2 and
            self.sequence_types[0] in self.sequence_encoders and
            self.sequence_types[1] in self.sequence_encoders
        )
        if self.use_cross_attention:
            st0, st1 = self.sequence_types[0], self.sequence_types[1]
            self.cross_attention = nn.ModuleDict({
                f"{st0}_to_{st1}": CrossAttentionModule(d_model, self.nhead, dropout),
                f"{st1}_to_{st0}": CrossAttentionModule(d_model, self.nhead, dropout)
            })
        elif len(self.sequence_types) == 2:
             logger.warning(
                "Cross-attention intended (2 seq types) but one/both encoders "
                "not initialized. Disabling cross-attention."
            )

        self.output_proj = nn.Linear(d_model, self.output_dim)
        self._init_parameters()
        logger.info(
            "MultiEncoderTransformer initialized. Output dim: %d. "
            "Cross-attention: %s.",
            self.output_dim, self.use_cross_attention
        )

    def _init_parameters(self):
        """Initializes model parameters using common schemes."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif 'weight' in name:
                if 'norm' in name.lower() and p.ndim == 1:
                    nn.init.ones_(p)
                else:
                    nn.init.uniform_(p, -0.1, 0.1)

        if hasattr(self, 'output_proj'):
            nn.init.xavier_uniform_(self.output_proj.weight, gain=0.01)
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)

    def _generate_pytorch_padding_mask(
        self, seq_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Generates PyTorch-style padding mask (True means PAD)."""
        return torch.all(seq_tensor == self.padding_value, dim=-1)

    def _validate_inputs(
        self,
        inputs: Dict[str, Tensor],
        input_masks_true_is_valid: Optional[Dict[str, Tensor]] = None
    ) -> Dict[str, int]:
        """
        Validates input structure for the forward pass.
        Raises ValueError on critical inconsistencies.
        """
        if not isinstance(inputs, dict):
            raise TypeError("Inputs must be a dictionary.")

        sequence_padded_lengths: Dict[str, int] = {}
        batch_size = -1

        for seq_type in self.sequence_types:
            if seq_type not in self.sequence_encoders:
                # This indicates an internal setup error if this path is reached.
                logger.error("Internal: No encoder for configured seq_type '%s'.", seq_type)
                continue

            if seq_type not in inputs:
                raise ValueError(f"Missing input tensor for sequence type: '{seq_type}'.")

            seq_tensor = inputs[seq_type]
            if not isinstance(seq_tensor, torch.Tensor) or seq_tensor.ndim != 3:
                raise ValueError(
                    f"Input for '{seq_type}' must be a 3D Tensor (B, L, F)."
                )

            current_bs = seq_tensor.size(0)
            if batch_size == -1: batch_size = current_bs
            elif batch_size != current_bs:
                raise ValueError("Inconsistent batch sizes across sequence inputs.")

            # self.sequence_dims is guaranteed to have seq_type by __init__ if encoder exists
            expected_features = len(self.sequence_dims[seq_type])
            if seq_tensor.size(2) != expected_features:
                raise ValueError(
                    f"Feature dimension mismatch for '{seq_type}'. "
                    f"Expected {expected_features}, got {seq_tensor.size(2)}."
                )
            sequence_padded_lengths[seq_type] = seq_tensor.size(1)

            if input_masks_true_is_valid and seq_type in input_masks_true_is_valid:
                mask = input_masks_true_is_valid[seq_type]
                if mask.shape != seq_tensor.shape[:-1]:
                    raise ValueError(
                        f"Shape mismatch for input '{seq_type}': tensor is "
                        f"{seq_tensor.shape[:-1]} but mask is {mask.shape}."
                    )

        if self.has_global:
            if 'global' not in inputs:
                raise ValueError("Missing 'global' input features when expected.")
            global_tensor = inputs['global']
            if global_tensor.ndim != 2:
                raise ValueError("'global' input must be a 2D Tensor (B, F).")
            if batch_size != -1 and global_tensor.size(0) != batch_size:
                raise ValueError("Batch size mismatch: sequences vs global features.")
            expected_global_features = len(self.global_variables)
            if global_tensor.size(1) != expected_global_features:
                raise ValueError(
                    f"Global feature count mismatch. Expected {expected_global_features}, "
                    f"got {global_tensor.size(1)}."
                )
        return sequence_padded_lengths


    def forward(
        self,
        inputs: Dict[str, Tensor],
        input_masks: Optional[Dict[str, Tensor]] = None,
        targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        `input_masks` should have True for VALID data points (non-padded).
        It will be inverted internally for PyTorch's True=PAD convention.
        """
        sequence_padded_lengths = self._validate_inputs(
            inputs, input_masks_true_is_valid=input_masks
        )
        output_seq_padded_length = sequence_padded_lengths[self.output_seq_type]

        pytorch_padding_masks: Dict[str, Tensor] = {} # True means PAD
        if input_masks:
            for seq_type, mask_true_is_valid in input_masks.items():
                if seq_type in self.sequence_encoders:
                    pytorch_padding_masks[seq_type] = ~mask_true_is_valid
        else:
            logger.debug("No input_masks provided; generating PyTorch masks from padding_value.")
            for seq_type in self.sequence_types:
                if seq_type in inputs and inputs[seq_type].ndim == 3:
                    pytorch_padding_masks[seq_type] = \
                        self._generate_pytorch_padding_mask(inputs[seq_type])

        global_features_encoded = None
        if self.has_global and self.global_encoder is not None:
            global_tensor = inputs['global']
            global_features_encoded = self.global_encoder(global_tensor)

        encoded_sequences: Dict[str, Optional[torch.Tensor]] = {}
        for seq_type, encoder in self.sequence_encoders.items():
            seq_input_tensor = inputs.get(seq_type)
            if seq_input_tensor is None or seq_input_tensor.numel() == 0:
                encoded_sequences[seq_type] = None
                continue

            current_padding_mask = pytorch_padding_masks.get(seq_type)
            encoded = encoder(seq_input_tensor, current_padding_mask)

            if self.has_global and global_features_encoded is not None and \
               self.film_layers and seq_type in self.film_layers:
                encoded = self.film_layers[seq_type](encoded, global_features_encoded)

            if current_padding_mask is not None:
                encoded = encoded.masked_fill(current_padding_mask.unsqueeze(-1), 0.0)
            encoded_sequences[seq_type] = encoded

        active_encoded_sequences = {k:v for k,v in encoded_sequences.items() if v is not None}
        if not active_encoded_sequences:
            bs = next((t.size(0) for t in inputs.values() if isinstance(t, Tensor) and t.ndim > 0), 1)
            logger.warning("No sequences encoded; returning zeros. Shape: (%d, %d, %d)",
                           bs, output_seq_padded_length, self.output_dim)
            return torch.zeros(
                bs, output_seq_padded_length, self.output_dim,
                device=next(self.parameters()).device
            )

        if self.use_cross_attention and self.cross_attention:
            st0, st1 = self.sequence_types[0], self.sequence_types[1]
            enc0, enc1 = encoded_sequences.get(st0), encoded_sequences.get(st1)
            if enc0 is not None and enc1 is not None:
                mask0_pad = pytorch_padding_masks.get(st0)
                mask1_pad = pytorch_padding_masks.get(st1)
                encoded_sequences[st0] = self.cross_attention[f"{st0}_to_{st1}"](
                    enc0, enc1, mask1_pad, mask0_pad)
                encoded_sequences[st1] = self.cross_attention[f"{st1}_to_{st0}"](
                    enc1, enc0, mask0_pad, mask1_pad)
            else:
                logger.warning("Cross-attn skipped: one/both sequences ('%s', '%s') are None.", st0, st1)

        final_output_candidate = encoded_sequences.get(self.output_seq_type)
        if final_output_candidate is None:
            bs = next((t.size(0) for t in inputs.values() if isinstance(t, Tensor) and t.ndim > 0), 1)
            logger.error("Output sequence '%s' is None after encoding. Returning zeros.", self.output_seq_type)
            return torch.zeros(
                bs, output_seq_padded_length, self.output_dim,
                device=next(self.parameters()).device
            )

        output = self.output_proj(final_output_candidate)
        output_mask_pad = pytorch_padding_masks.get(self.output_seq_type)
        if output_mask_pad is not None:
            output = output.masked_fill(output_mask_pad.unsqueeze(-1), 0.0)
        return output


def create_prediction_model(
    config: Dict[str, Any], device: Optional[torch.device] = None
) -> MultiEncoderTransformer:
    """
    Factory function for MultiEncoderTransformer.
    Performs critical config validations and exits on failure.
    """
    required_keys = [
        "input_variables", "target_variables", "sequence_types", "output_seq_type",
        "d_model", "nhead", "num_encoder_layers", "dim_feedforward"
    ]
    for key in required_keys:
        if key not in config:
            logger.critical("Config missing required key: '%s'. Exiting.", key)
            sys.exit(1)

    input_vars = config["input_variables"]
    target_vars = config["target_variables"]
    seq_types_cfg = config["sequence_types"]
    output_seq_type_cfg = config["output_seq_type"]
    d_model = config["d_model"]
    nhead_cfg = config["nhead"]
    pos_enc_type = config.get("positional_encoding_type", "sinusoidal") # Default to sinusoidal

    # Validate types and non-emptiness for essential list/dict fields
    if not (isinstance(input_vars, list) and input_vars):
        logger.critical("'input_variables' must be a non-empty list. Exiting.")
        sys.exit(1)
    if not (isinstance(target_vars, list) and target_vars):
        logger.critical("'target_variables' must be a non-empty list. Exiting.")
        sys.exit(1)
    if not (isinstance(seq_types_cfg, dict) and seq_types_cfg):
        logger.critical("'sequence_types' must be a non-empty dict. Exiting.")
        sys.exit(1)
    if not isinstance(output_seq_type_cfg, str) or not output_seq_type_cfg:
        logger.critical("'output_seq_type' must be a non-empty string. Exiting.")
        sys.exit(1)

    global_vars = config.get("global_variables", [])
    if not isinstance(global_vars, list): # Allow empty list, but must be list
        logger.critical("'global_variables' must be a list if provided. Exiting.")
        sys.exit(1)

    # Process sequence_types to build sequence_dims and check for errors
    processed_sequence_dims: Dict[str, Dict[str, int]] = {}
    all_defined_vars: Set[str] = set(global_vars)

    for seq_type, var_list in seq_types_cfg.items():
        if not isinstance(var_list, list): # var_list must be a list
            logger.critical("Var list for seq_type '%s' must be a list. Exiting.", seq_type)
            sys.exit(1)
        if not var_list: # Allow empty var_list for a seq_type, but it won't be used
            logger.info("Seq_type '%s' has an empty variable list; it will be ignored.", seq_type)
            continue # Skip this sequence type if its variable list is empty

        if len(set(var_list)) != len(var_list): # Check for duplicates within a list
            logger.critical("Duplicate vars in seq_type '%s'. Exiting.", seq_type)
            sys.exit(1)
        if overlap := set(var_list).intersection(all_defined_vars): # Check for overlap with already defined
            logger.critical("Var '%s' in seq_type '%s' overlaps with globals or another seq_type. Exiting.",
                           next(iter(overlap)), seq_type) # Show one overlapping var
            sys.exit(1)
        all_defined_vars.update(var_list)
        processed_sequence_dims[seq_type] = {var: i for i, var in enumerate(var_list)}

    if not processed_sequence_dims: # Must have at least one valid sequence type
        logger.critical("No valid, non-empty sequence types defined in config. Exiting.")
        sys.exit(1)
    if output_seq_type_cfg not in processed_sequence_dims:
        logger.critical("Config 'output_seq_type' ('%s') not among valid sequence types %s. Exiting.",
                        output_seq_type_cfg, list(processed_sequence_dims.keys()))
        sys.exit(1)

    # Validate that all input and target variables are covered
    if unassigned_in := set(input_vars) - all_defined_vars:
        logger.critical("Input vars %s not in any seq_type or global_variables. Exiting.", unassigned_in)
        sys.exit(1)
    if unassigned_tgt := set(target_vars) - all_defined_vars:
        logger.critical("Target vars %s not in any seq_type or global_variables. Exiting.", unassigned_tgt)
        sys.exit(1)

    # Ensure all non-global target variables belong to the output_seq_type for consistent length handling
    output_seq_type_vars = processed_sequence_dims.get(output_seq_type_cfg, {})
    for tv in target_vars:
        if tv not in global_vars and tv not in output_seq_type_vars:
            logger.critical(
                "Target variable '%s' is a sequence variable but not part of the "
                "designated 'output_seq_type' ('%s'). This is required for consistent "
                "target length determination. Exiting.", tv, output_seq_type_cfg
            )
            sys.exit(1)


    # Validate model dimensions and PE compatibility
    if not (isinstance(d_model, int) and d_model > 0):
        logger.critical("'d_model' (%s) must be a positive integer. Exiting.", d_model)
        sys.exit(1)
    # Check d_model parity specifically if sinusoidal PE is chosen
    # SequenceEncoder will also check this, but good to catch early.
    pe_type_lower_check = pos_enc_type.lower() if pos_enc_type else "none"
    if pe_type_lower_check in ["sinusoidal", "sine", "sin"] and d_model % 2 != 0:
        logger.critical("Sinusoidal PE selected, but 'd_model' (%d) is odd. Must be even. Exiting.", d_model)
        sys.exit(1)

    if not (isinstance(nhead_cfg, int) and nhead_cfg > 0):
        logger.critical("'nhead' (%s) must be a positive integer. Exiting.", nhead_cfg)
        sys.exit(1)

    final_nhead = nhead_cfg
    if d_model % nhead_cfg != 0:
        original_nhead = nhead_cfg
        for potential_nhead in range(original_nhead, 0, -1): # Find largest divisor <= original
            if d_model % potential_nhead == 0:
                final_nhead = potential_nhead
                break
        logger.warning(
            "Config 'd_model' (%d) is not divisible by 'nhead' (%d). "
            "Adjusting nhead to %d for model compatibility.",
            d_model, original_nhead, final_nhead
        )
        config["nhead"] = final_nhead # Update config dict for checkpoint consistency

    # Instantiate the model
    try:
        model_instance = MultiEncoderTransformer(
            global_variables=global_vars,
            sequence_dims=processed_sequence_dims,
            output_dim=len(target_vars),
            d_model=d_model,
            nhead=final_nhead,
            num_encoder_layers=config["num_encoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            output_seq_type=output_seq_type_cfg,
            dropout=config.get("dropout", 0.1),
            norm_first=config.get("norm_first", False),
            max_sequence_length=config.get("max_sequence_length", 512),
            positional_encoding_type=pos_enc_type,
            padding_value=float(config.get("padding_value", 0.0))
        )
    except Exception as e:
        logger.critical("Failed to instantiate MultiEncoderTransformer: %s. Exiting.", e, exc_info=True)
        sys.exit(1)

    model_instance.input_vars = list(input_vars) # Store for reference
    model_instance.target_vars = list(target_vars) # Store for reference

    logger.info("Model created successfully with d_model=%d, nhead=%d.", d_model, final_nhead)

    if device is not None:
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except RuntimeError as e: # Catch invalid device string
                logger.critical("Invalid device string '%s': %s. Exiting.", device, e)
                sys.exit(1)
        # Ensure device is actually a torch.device object before model.to()
        if not isinstance(device, torch.device):
            logger.critical("Device argument must be a torch.device instance or valid string. Exiting.")
            sys.exit(1)
        try:
            model_instance = model_instance.to(device)
            logger.info("Model moved to device: %s.", device)
        except Exception as e: # Catch errors from model.to(device)
            logger.critical("Failed to move model to device '%s': %s. Exiting.", device, e, exc_info=True)
            sys.exit(1)
    return model_instance

__all__ = ["MultiEncoderTransformer", "create_prediction_model"]
