#!/usr/bin/env python3
"""Visualize attention patterns and target-specific input importance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import load_config, PADDING_VALUE
from model import create_prediction_model
from dataset import create_dataset, create_collate_fn

plt.style.use('science.mplstyle')
def load_model(model_dir):
    """Load model from checkpoint."""
    config = load_config(model_dir / "train_config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_prediction_model(config, device, compile_model=False)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    return model, config, device


def extract_attention(model, sequence, mask, global_features=None):
    """Extract attention weights from each encoder layer."""
    attention_weights = []

    x = model.input_proj(sequence)
    x = model.pos_encoder(x)

    if model.initial_film is not None and global_features is not None:
        x = model.initial_film(x, global_features)

    for block in model.blocks:
        transformer = block.transformer

        if transformer.norm_first:
            x_norm = transformer.norm1(x)
        else:
            x_norm = x

        with torch.no_grad():
            attn_output, attn_weights = transformer.self_attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=mask,
                need_weights=True,
                average_attn_weights=False
            )

        if attn_weights is not None:
            attention_weights.append(attn_weights.cpu())

        if transformer.norm_first:
            x = x + transformer.dropout1(attn_output)
            x2 = transformer.norm2(x)
            x2 = transformer.linear2(transformer.dropout(transformer.activation(transformer.linear1(x2))))
            x = x + transformer.dropout2(x2)
        else:
            x = x + transformer.dropout1(attn_output)
            x = transformer.norm1(x)
            x2 = transformer.linear2(transformer.dropout(transformer.activation(transformer.linear1(x))))
            x = x + transformer.dropout2(x2)
            x = transformer.norm2(x)

        if block.has_film and global_features is not None:
            x = block.film(x, global_features)

    return attention_weights


def compute_gradient_importance(model, sequence, mask, global_features, target_idx):
    """Compute input importance for specific target via gradients."""
    sequence.requires_grad_(True)

    # Forward pass
    output = model(sequence, global_features, mask)

    # Get gradient for specific target
    valid_mask = ~mask[0]
    target_output = output[0, valid_mask, target_idx].sum()
    target_output.backward()

    # Get gradient magnitude
    grad_importance = sequence.grad.abs().mean(dim=-1)[0]  # Average over input features

    return grad_importance.detach().cpu()


def plot_combined(attention_weights, grad_importance, mask, target_names):
    """Plot attention and gradient importance together."""
    n_layers = len(attention_weights)
    valid_mask = ~mask[0].cpu().numpy()
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    # Create figure: top 2 rows for attention, bottom row for gradient importance
    fig = plt.figure(figsize=(16, 10))

    # Attention plots (2 rows)
    n_cols = (n_layers + 1) // 2
    for layer_idx, attn in enumerate(attention_weights):
        ax = plt.subplot(3, n_cols, layer_idx + 1)

        attn_full = attn[0].mean(dim=0).numpy()
        attn_valid = attn_full[valid_indices][:, valid_indices]

        im = ax.imshow(attn_valid, cmap='hot', aspect='auto')
        ax.set_title(f'Layer {layer_idx + 1}')
        ax.set_xlabel('Key Pos')
        ax.set_ylabel('Query Pos')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Gradient importance plots (bottom row)
    for target_idx, (target_name, grads) in enumerate(zip(target_names, grad_importance)):
        ax = plt.subplot(3, 2, 5 + target_idx)

        grads_valid = grads[valid_mask].numpy()
        positions = np.arange(n_valid)

        ax.bar(positions, grads_valid, color=f'C{target_idx}')
        ax.set_title(f'Input Importance for {target_name}')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Gradient Magnitude')
        ax.grid(True, alpha=0.3)

        # Mark top 5 positions
        top_5 = np.argsort(grads_valid)[-5:]
        for pos in top_5:
            ax.axvline(pos, color='red', alpha=0.3, linestyle='--')

    plt.suptitle('Attention Patterns and Target-Specific Input Importance')
    plt.tight_layout()

    save_path = Path("../models/trained_model/plots/attention_and_gradients.png")
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def main():
    model_dir = Path("../models/trained_model")

    # Load model
    model, config, device = load_model(model_dir)

    # Get target names
    target_vars = config["data_specification"]["target_variables"]
    target_names = target_vars[:2]  # Use first two targets

    # Load test sample
    test_dir = Path("../data/processed/test")
    dataset = create_dataset(test_dir, config, [0])
    collate_fn = create_collate_fn(PADDING_VALUE)

    inputs, targets = dataset[0]
    batch_inputs, batch_masks, _, _ = collate_fn([(inputs, targets)])

    # Move to device
    sequence = batch_inputs["sequence"].to(device)
    mask = batch_masks["sequence"].to(device)
    global_features = batch_inputs.get("global_features")
    if global_features is not None:
        global_features = global_features.to(device)

    # Extract attention
    print(f"Extracting attention from {len(model.blocks)} layers...")
    attention_weights = extract_attention(model, sequence, mask, global_features)

    # Compute gradient importance for each target
    print(f"Computing gradient importance for targets: {target_names}")
    grad_importance = []
    for idx in range(2):
        grads = compute_gradient_importance(
            model, sequence.clone(), mask, global_features, idx
        )
        grad_importance.append(grads)

    # Plot
    plot_combined(attention_weights, grad_importance, mask, target_names)


if __name__ == "__main__":
    main()