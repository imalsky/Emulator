#!/usr/bin/env python3
"""Test model predictions with denormalization and log scale."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.append("../src")

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

from dataset import create_dataset, create_collate_fn
from model import create_prediction_model
from normalizer import DataNormalizer
from utils import load_config, PADDING_VALUE

plt.style.use('science.mplstyle')

MODEL_DIR = Path("../models/trained_model_v1")
PROCESSED_DIR = Path("../data/processed/test")
N_SAMPLES = 1

def load_normalization_metadata():
    """Load normalization metadata for denormalization."""
    metadata_path = Path("../data/processed/normalization_metadata.json")
    with open(metadata_path, 'r') as f:
        return json.load(f)

def denormalize_variable(data, var_name, norm_metadata):
    """Denormalize a single variable."""
    data_tensor = torch.from_numpy(data).float()
    method = norm_metadata["normalization_methods"].get(var_name, "none")
    stats = norm_metadata["per_key_stats"].get(var_name, {})
    
    if method != "none" and stats:
        return DataNormalizer.denormalize_tensor(data_tensor, method, stats).numpy()
    return data

def load_model_and_data():
    """Load model and test data."""
    config = load_config(MODEL_DIR / "train_config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = create_prediction_model(config, device, compile_model=False)
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load test data
    test_dataset = create_dataset(PROCESSED_DIR, config, list(range(N_SAMPLES)))
    collate_fn = create_collate_fn(PADDING_VALUE)
    
    return model, test_dataset, collate_fn, config, device

def test_predictions():
    """Test model on samples and visualize."""
    model, dataset, collate_fn, config, device = load_model_and_data()
    norm_metadata = load_normalization_metadata()
    
    target_vars = config["data_specification"]["target_variables"]
    
    # Find indices for thermal and reflected fluxes
    thermal_idx = target_vars.index("net_thermal_flux") if "net_thermal_flux" in target_vars else None
    reflected_idx = target_vars.index("net_reflected_flux") if "net_reflected_flux" in target_vars else None
    
    if thermal_idx is None or reflected_idx is None:
        print("Error: thermal_flux or reflected_flux not found in target variables")
        print(f"Available targets: {target_vars}")
        return
    
    # Create figure with subplots for thermal and reflected fluxes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Process each sample
    for i in range(min(N_SAMPLES, len(dataset))):
        inputs, targets = dataset[i]
        
        # Create batch
        batch_inputs, batch_masks, batch_targets, target_masks = collate_fn([(inputs, targets)])
        
        # Move to device
        for k in batch_inputs:
            batch_inputs[k] = batch_inputs[k].to(device)
        batch_masks["sequence"] = batch_masks["sequence"].to(device)
        
        # Predict
        with torch.no_grad():
            preds = model(
                sequence=batch_inputs["sequence"],
                global_features=batch_inputs.get("global_features"),
                sequence_mask=batch_masks["sequence"]
            )
        
        # Get predictions and targets
        preds_np = preds.cpu().numpy()[0]
        targets_np = batch_targets.numpy()[0]
        mask_np = target_masks.numpy()[0]
        
        # Get valid positions (remove padding)
        valid_mask = ~mask_np
        valid_idx = np.where(valid_mask)[0]
        
        if len(valid_idx) == 0:
            continue
        
        # Denormalize thermal flux
        thermal_pred = denormalize_variable(
            preds_np[valid_idx, thermal_idx], 
            "net_thermal_flux", 
            norm_metadata
        )
        thermal_target = denormalize_variable(
            targets_np[valid_idx, thermal_idx], 
            "net_thermal_flux", 
            norm_metadata
        )
        
        # Denormalize reflected flux
        reflected_pred = denormalize_variable(
            preds_np[valid_idx, reflected_idx], 
            "net_reflected_flux", 
            norm_metadata
        )
        reflected_target = denormalize_variable(
            targets_np[valid_idx, reflected_idx], 
            "net_reflected_flux", 
            norm_metadata
        )
        
        # Plot thermal flux with symlog scale
        axes[0].plot(valid_idx, thermal_target, color='red',
                    label=f"Target (Sample {i+1})", alpha=0.7, linewidth=2)
        axes[0].plot(valid_idx, thermal_pred, '--', color='black',
                    label=f"Prediction (Sample {i+1})", alpha=0.7, linewidth=2)
        
        # Plot reflected flux with symlog scale
        axes[1].plot(valid_idx, reflected_target, color='red',
                    label=f"Target (Sample {i+1})", alpha=0.7, linewidth=2)
        axes[1].plot(valid_idx, reflected_pred, '--', color='black',
                    label=f"Prediction (Sample {i+1})", alpha=0.7, linewidth=2)
    
    # Format thermal flux subplot with symlog scale
    axes[0].set_yscale('symlog')
    #axes[0].set_ylim(-1e10, 1e10)
    axes[0].set_xlabel("Sequence Position")
    axes[0].set_ylabel("Net Thermal Flux (Ergs/cm²)")
    axes[0].set_title("Net Thermal Flux - Denormalized (Symlog Scale)")
    axes[0].legend(loc="best", fontsize=8)
    
    # Format reflected flux subplot with symlog scale
    axes[1].set_yscale('symlog')
    #axes[1].set_ylim(-1e10, 1e10)
    axes[1].set_xlabel("Sequence Position")
    axes[1].set_ylabel("Net Reflected Flux (Ergs/cm²)")
    axes[1].set_title("Net Reflected Flux - Denormalized (Symlog Scale)")
    axes[1].legend(loc="best", fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    save_path = MODEL_DIR / "plots" / "flux_predictions_denorm_log.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    
    # Calculate metrics for fluxes
    print("\n" + "="*50)
    print("DENORMALIZED METRICS (Physical Units)")
    print("="*50)
    
    for flux_name in ["net_thermal_flux", "net_reflected_flux"]:
        print(f"\n{flux_name}:")
        
        # Collect all predictions and targets for this flux
        all_preds = []
        all_targets = []
        
        flux_idx = target_vars.index(flux_name)
        
        for i in range(min(N_SAMPLES, len(dataset))):
            inputs, targets = dataset[i]
            batch_inputs, batch_masks, batch_targets, target_masks = collate_fn([(inputs, targets)])
            
            for k in batch_inputs:
                batch_inputs[k] = batch_inputs[k].to(device)
            batch_masks["sequence"] = batch_masks["sequence"].to(device)
            
            with torch.no_grad():
                preds = model(
                    sequence=batch_inputs["sequence"],
                    global_features=batch_inputs.get("global_features"),
                    sequence_mask=batch_masks["sequence"]
                )
            
            preds_np = preds.cpu().numpy()[0]
            targets_np = batch_targets.numpy()[0]
            mask_np = target_masks.numpy()[0]
            valid_mask = ~mask_np
            
            if np.any(valid_mask):
                flux_pred = denormalize_variable(
                    preds_np[valid_mask, flux_idx], 
                    flux_name, 
                    norm_metadata
                )
                flux_target = denormalize_variable(
                    targets_np[valid_mask, flux_idx], 
                    flux_name, 
                    norm_metadata
                )
                all_preds.append(flux_pred)
                all_targets.append(flux_target)
        
        if all_preds:
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            
            mse = np.mean((all_preds - all_targets)**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(all_preds - all_targets))
            
            print(f"  RMSE: {rmse:.3e} Ergs/cm²")
            print(f"  MAE:  {mae:.3e} Ergs/cm²")
            print(f"  Target Range: [{np.min(all_targets):.3e}, {np.max(all_targets):.3e}] Ergs/cm²")
            print(f"  Pred Range:   [{np.min(all_preds):.3e}, {np.max(all_preds):.3e}] Ergs/cm²")

if __name__ == "__main__":
    test_predictions()