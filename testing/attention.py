#!/usr/bin/env python3
"""Visualize attention patterns and analyze which pressure levels get most attention."""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

sys.path.append("../src")

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple

from dataset import create_dataset, create_collate_fn
from model import create_prediction_model
from normalizer import DataNormalizer
from utils import load_config, PADDING_VALUE

try:
    plt.style.use('science.mplstyle')
except:
    pass

MODEL_DIR = Path("../models/trained_model")
PROCESSED_DIR = Path("../data/processed/test")
N_SAMPLES = 5  # Analyze a few samples in detail


class AttentionExtractor(nn.Module):
    """Wrapper to extract attention weights from model."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.attention_weights = []

        # Hook into attention layers
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        self.hooks = []

        # Find all MultiheadAttention modules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(self._attention_hook)
                self.hooks.append(hook)

    def _attention_hook(self, module, input, output):
        """Hook to capture attention weights."""
        # output is (attn_output, attn_weights)
        if len(output) == 2 and output[1] is not None:
            self.attention_weights.append(output[1].detach().cpu())

    def forward(self, *args, **kwargs):
        """Forward pass that captures attention."""
        self.attention_weights = []
        output = self.model(*args, **kwargs)
        return output, self.attention_weights

    def remove_hooks(self):
        """Clean up hooks."""
        for hook in self.hooks:
            hook.remove()


def extract_attention_patterns(model, dataset, collate_fn, config, device, n_samples=5):
    """Extract attention patterns from model for analysis."""
    # Modify model to return attention weights
    model_modified = modify_model_for_attention(model)

    attention_data = []
    pressure_values_list = []

    input_vars = config["data_specification"]["input_variables"]
    pressure_idx = input_vars.index("pressure_bar") if "pressure_bar" in input_vars else None

    for i in range(min(n_samples, len(dataset))):
        inputs, targets = dataset[i]
        batch_inputs, batch_masks, _, _ = collate_fn([(inputs, targets)])

        # Move to device
        for k in batch_inputs:
            batch_inputs[k] = batch_inputs[k].to(device)
        batch_masks["sequence"] = batch_masks["sequence"].to(device)

        # Get attention weights
        with torch.no_grad():
            _, attention_weights = model_modified(
                sequence=batch_inputs["sequence"],
                global_features=batch_inputs.get("global_features"),
                sequence_mask=batch_masks["sequence"],
                return_attention=True
            )

        # Store attention and pressure values
        if attention_weights:
            attention_data.append(attention_weights)

            if pressure_idx is not None:
                pressure = batch_inputs["sequence"][0, :, pressure_idx].cpu().numpy()
                valid_mask = ~batch_masks["sequence"][0].cpu().numpy()
                pressure_values_list.append(pressure[valid_mask])

    return attention_data, pressure_values_list


def modify_model_for_attention(model):
    """Modify model to return attention weights."""

    class AttentionModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.attention_weights = []

        def forward(self, sequence, global_features=None, sequence_mask=None, return_attention=False):
            if not return_attention:
                return self.base_model(sequence, global_features, sequence_mask)

            # Store attention weights from each layer
            attention_weights = []

            # Manual forward pass through model components
            x = self.base_model.input_proj(sequence)
            x = self.base_model.pos_encoder(x)

            if self.base_model.initial_film is not None and global_features is not None:
                x = self.base_model.initial_film(x, global_features)

            # Pass through transformer blocks and collect attention
            for block in self.base_model.blocks:
                # Get attention from transformer layer
                x_norm = block.transformer.norm1(x) if block.transformer.norm_first else x
                attn_out, attn_weights = block.transformer.self_attn(
                    x_norm, x_norm, x_norm,
                    key_padding_mask=sequence_mask,
                    need_weights=True,
                    average_attn_weights=True
                )

                if attn_weights is not None:
                    attention_weights.append(attn_weights.detach().cpu())

                # Complete the forward pass
                if block.transformer.norm_first:
                    x = x + block.transformer.dropout1(attn_out)
                    x2 = block.transformer.norm2(x)
                    x2 = block.transformer.linear2(
                        block.transformer.dropout(
                            block.transformer.activation(block.transformer.linear1(x2))
                        )
                    )
                    x = x + block.transformer.dropout2(x2)
                else:
                    x = x + block.transformer.dropout1(attn_out)
                    x = block.transformer.norm1(x)
                    x2 = block.transformer.linear2(
                        block.transformer.dropout(
                            block.transformer.activation(block.transformer.linear1(x))
                        )
                    )
                    x = x + block.transformer.dropout2(x2)
                    x = block.transformer.norm2(x)

                # Apply FiLM if exists
                if block.has_film and global_features is not None:
                    x = block.film(x, global_features)

            x = self.base_model.final_norm(x)
            output = self.base_model.output_proj(x)

            return output, attention_weights

    return AttentionModel(model)


def plot_attention_heatmaps(attention_data, pressure_values_list, norm_metadata):
    """Plot attention heatmaps for multiple layers and samples."""
    if not attention_data:
        print("No attention data to plot")
        return

    n_samples = len(attention_data)
    n_layers = len(attention_data[0])

    # Create figure with subplots for each layer
    fig, axes = plt.subplots(n_layers, n_samples, figsize=(4 * n_samples, 3 * n_layers))

    if n_layers == 1:
        axes = axes.reshape(1, -1)
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for sample_idx in range(n_samples):
        sample_attention = attention_data[sample_idx]

        # Denormalize pressure values if available
        if pressure_values_list and sample_idx < len(pressure_values_list):
            pressure = pressure_values_list[sample_idx]
            pressure_denorm = DataNormalizer.denormalize_tensor(
                torch.from_numpy(pressure).float(),
                norm_metadata["normalization_methods"].get("pressure_bar", "none"),
                norm_metadata["per_key_stats"].get("pressure_bar", {})
            ).numpy()
        else:
            pressure_denorm = None

        for layer_idx, attn in enumerate(sample_attention):
            ax = axes[layer_idx, sample_idx]

            # Average attention across heads
            attn_avg = attn.mean(dim=0).numpy()  # Shape: [seq_len, seq_len]

            # Plot heatmap
            im = ax.imshow(attn_avg, cmap='hot', aspect='auto', vmin=0, vmax=attn_avg.max())

            # Labels
            ax.set_title(f'Sample {sample_idx + 1}, Layer {layer_idx + 1}', fontsize=10)
            ax.set_xlabel('Key Position', fontsize=9)
            ax.set_ylabel('Query Position', fontsize=9)

            # Add pressure labels if available
            if pressure_denorm is not None and len(pressure_denorm) == attn_avg.shape[0]:
                # Show pressure values at certain positions
                n_ticks = min(10, len(pressure_denorm))
                tick_positions = np.linspace(0, len(pressure_denorm) - 1, n_ticks, dtype=int)
                ax.set_xticks(tick_positions)
                ax.set_yticks(tick_positions)
                ax.set_xticklabels([f'{pressure_denorm[i]:.1e}' for i in tick_positions],
                                   rotation=45, fontsize=7)
                ax.set_yticklabels([f'{pressure_denorm[i]:.1e}' for i in tick_positions],
                                   fontsize=7)

    plt.suptitle('Attention Patterns Across Layers and Samples\n(Averaged over heads)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    save_path = MODEL_DIR / "plots" / "attention_heatmaps.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")



def analyze_pressure_attention(attention_data, pressure_values_list, norm_metadata):
    """Analyze which pressure levels receive most attention."""
    if not attention_data or not pressure_values_list:
        print("Insufficient data for pressure attention analysis")
        return

    # Aggregate attention scores by pressure ranges
    pressure_attention = []

    for sample_idx, sample_attention in enumerate(attention_data):
        if sample_idx >= len(pressure_values_list):
            continue

        pressure = pressure_values_list[sample_idx]
        pressure_denorm = DataNormalizer.denormalize_tensor(
            torch.from_numpy(pressure).float(),
            norm_metadata["normalization_methods"].get("pressure_bar", "none"),
            norm_metadata["per_key_stats"].get("pressure_bar", {})
        ).numpy()

        # For each layer, calculate mean attention received by each position
        for layer_idx, attn in enumerate(sample_attention):
            # Average across heads and sum across queries (how much each position is attended to)
            attn_received = attn.mean(dim=0).sum(dim=0).numpy()  # Shape: [seq_len]

            for pos_idx, (p, a) in enumerate(zip(pressure_denorm, attn_received)):
                pressure_attention.append({
                    'pressure': p,
                    'attention': a,
                    'layer': layer_idx,
                    'position': pos_idx,
                    'sample': sample_idx
                })

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Attention vs Pressure (scatter plot)
    ax = axes[0, 0]
    for layer in range(min(3, len(attention_data[0]))):  # Show first 3 layers
        layer_data = [x for x in pressure_attention if x['layer'] == layer]
        if layer_data:
            pressures = [x['pressure'] for x in layer_data]
            attentions = [x['attention'] for x in layer_data]
            ax.scatter(pressures, attentions, alpha=0.5, label=f'Layer {layer + 1}', s=20)

    ax.set_xscale('log')
    ax.set_xlabel('Pressure (bar)')
    ax.set_ylabel('Total Attention Received')
    ax.set_title('Attention Distribution Across Pressure Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Binned pressure attention (bar plot)
    ax = axes[0, 1]

    # Create pressure bins (log-spaced)
    all_pressures = [x['pressure'] for x in pressure_attention]
    if all_pressures:
        min_p, max_p = min(all_pressures), max(all_pressures)
        pressure_bins = np.logspace(np.log10(min_p), np.log10(max_p), 10)

        # Bin attention scores
        binned_attention = {f'Layer {i + 1}': [] for i in range(len(attention_data[0]))}
        bin_centers = []

        for i in range(len(pressure_bins) - 1):
            bin_center = np.sqrt(pressure_bins[i] * pressure_bins[i + 1])
            bin_centers.append(bin_center)

            for layer in range(len(attention_data[0])):
                layer_data = [x['attention'] for x in pressure_attention
                              if x['layer'] == layer and
                              pressure_bins[i] <= x['pressure'] < pressure_bins[i + 1]]
                binned_attention[f'Layer {layer + 1}'].append(
                    np.mean(layer_data) if layer_data else 0
                )

        # Plot bars
        x = np.arange(len(bin_centers))
        width = 0.8 / len(attention_data[0])

        for i, (layer_name, values) in enumerate(binned_attention.items()):
            if i < 3:  # Show first 3 layers
                ax.bar(x + i * width, values, width, label=layer_name, alpha=0.7)

        ax.set_xlabel('Pressure Range (bar)')
        ax.set_ylabel('Mean Attention')
        ax.set_title('Binned Attention by Pressure Range')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{p:.1e}' for p in bin_centers], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # 3. Attention evolution across layers
    ax = axes[1, 0]

    # Calculate attention entropy for each layer (measure of attention spread)
    layer_entropies = []
    for layer_idx in range(len(attention_data[0])):
        layer_attentions = []
        for sample_attention in attention_data:
            if layer_idx < len(sample_attention):
                attn = sample_attention[layer_idx].mean(dim=0).numpy()
                # Calculate entropy of attention distribution
                attn_flat = attn.flatten()
                attn_flat = attn_flat[attn_flat > 0]
                if len(attn_flat) > 0:
                    attn_norm = attn_flat / attn_flat.sum()
                    entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-10))
                    layer_attentions.append(entropy)

        if layer_attentions:
            layer_entropies.append(np.mean(layer_attentions))

    ax.plot(range(1, len(layer_entropies) + 1), layer_entropies, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Attention Spread Evolution Across Layers\n(Higher = more distributed attention)')
    ax.grid(True, alpha=0.3)

    # 4. Head specialization analysis
    ax = axes[1, 1]

    # Analyze first layer's heads
    if attention_data and len(attention_data[0]) > 0:
        first_layer_attention = []
        for sample_attention in attention_data:
            attn = sample_attention[0]  # First layer
            first_layer_attention.append(attn.numpy())

        # Calculate variance across heads
        head_variances = []
        n_heads = first_layer_attention[0].shape[0]

        for head_idx in range(n_heads):
            head_attn = np.concatenate([a[head_idx:head_idx + 1] for a in first_layer_attention])
            variance = np.var(head_attn)
            head_variances.append(variance)

        ax.bar(range(1, n_heads + 1), head_variances, color='skyblue', edgecolor='navy')
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Attention Variance')
        ax.set_title('Head Specialization in First Layer\n(Higher variance = more specialized)')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Pressure-Level Attention Analysis', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save plot
    save_path = MODEL_DIR / "plots" / "pressure_attention_analysis.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def main():
    """Main analysis function."""
    print("=" * 60)
    print("ATTENTION MECHANISM ANALYSIS")
    print("=" * 60)

    # Load model and data
    print("\nLoading model and data...")

    config_paths = [
        MODEL_DIR / "train_config.json",
        MODEL_DIR / "best_config.json",
        MODEL_DIR / "normalize_config.json"
    ]

    config = None
    for config_path in config_paths:
        if config_path.exists():
            config = load_config(config_path)
            break

    if config is None:
        raise FileNotFoundError(f"No config file found in {MODEL_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_prediction_model(config, device, compile_model=False)
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Load normalization metadata
    metadata_path = Path("../data/processed/normalization_metadata.json")
    with open(metadata_path, 'r') as f:
        norm_metadata = json.load(f)

    # Create dataset
    test_dataset = create_dataset(PROCESSED_DIR, config, list(range(N_SAMPLES)))
    collate_fn = create_collate_fn(PADDING_VALUE)

    # Extract attention patterns
    print(f"\nExtracting attention patterns from {N_SAMPLES} samples...")
    attention_data, pressure_values = extract_attention_patterns(
        model, test_dataset, collate_fn, config, device, N_SAMPLES
    )

    if attention_data:
        print(f"Extracted attention from {len(attention_data)} samples")
        print(f"Model has {len(attention_data[0])} transformer layers")

        # Visualize attention heatmaps
        print("\nGenerating attention heatmaps...")
        plot_attention_heatmaps(attention_data, pressure_values, norm_metadata)

        # Analyze pressure-level attention
        print("Analyzing pressure-level attention patterns...")
        analyze_pressure_attention(attention_data, pressure_values, norm_metadata)

        # Print summary
        print("\n" + "=" * 60)
        print("ATTENTION SUMMARY")
        print("=" * 60)

        # Calculate average attention statistics
        total_entropy = []
        for sample_attention in attention_data:
            for layer_attn in sample_attention:
                attn_avg = layer_attn.mean(dim=0).numpy()
                attn_flat = attn_avg.flatten()
                attn_flat = attn_flat[attn_flat > 0]
                if len(attn_flat) > 0:
                    attn_norm = attn_flat / attn_flat.sum()
                    entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-10))
                    total_entropy.append(entropy)

        if total_entropy:
            print(f"\nAverage attention entropy: {np.mean(total_entropy):.3f}")
            print(f"Min entropy (most focused): {np.min(total_entropy):.3f}")
            print(f"Max entropy (most distributed): {np.max(total_entropy):.3f}")
    else:
        print("No attention data extracted. Check model architecture.")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()