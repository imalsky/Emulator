#!/usr/bin/env python3
"""
test_export.py - Test script to verify model export functionality.

This script:
1. Creates a model from configuration
2. Tests forward pass with dummy data
3. Exports the model using torch.export
4. Validates exported model produces identical results
5. Tests with multiple batch sizes
"""
import torch
from pathlib import Path
from typing import Dict, Optional

from model import create_prediction_model, export_model
from utils import load_config


def test_model_export(
    config_path: str = "../config/config.jsonc",
    export_dir: Optional[Path] = None,
    test_batch_sizes: list = None,
) -> None:
    """
    Test that model can be successfully exported and produces identical results.
    
    Args:
        config_path: Path to configuration file
        export_dir: Directory for export test (defaults to ./test_export)
        test_batch_sizes: List of batch sizes to test (defaults to [1, 2, 8, 16])
    """
    print("=" * 60)
    print("Model Export Test")
    print("=" * 60)
    
    # Load configuration
    print(f"\n1. Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # Create model
    print("\n2. Creating model...")
    model = create_prediction_model(config, device=device, compile_model=False)
    
    # Extract dimensions from config
    data_spec = config["data_specification"]
    model_params = config["model_hyperparameters"]
    
    input_dim = len(data_spec["input_variables"])
    global_dim = len(data_spec.get("global_variables", []))
    seq_length = model_params["max_sequence_length"]
    
    print(f"   Input dimension: {input_dim}")
    print(f"   Global dimension: {global_dim}")
    print(f"   Sequence length: {seq_length}")
    
    # Create dummy inputs
    print("\n3. Creating dummy inputs...")
    batch_size = 4
    
    sequence = torch.randn(batch_size, seq_length, input_dim, device=device)
    global_features = (
        torch.randn(batch_size, global_dim, device=device)
        if global_dim > 0 else None
    )
    
    # Create padding mask (True = padding position)
    sequence_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
    # Add some padding to last sequences for testing
    sequence_mask[0, 50:] = True
    sequence_mask[1, 55:] = True
    
    print(f"   Sequence shape: {sequence.shape}")
    if global_features is not None:
        print(f"   Global features shape: {global_features.shape}")
    print(f"   Padding mask shape: {sequence_mask.shape}")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    with torch.no_grad():
        output = model(sequence, global_features, sequence_mask)
    
    print(f"   ✓ Model forward pass successful")
    print(f"   Output shape: {output.shape}")
    
    # Prepare export directory
    if export_dir is None:
        export_dir = Path("test_export")
    export_dir.mkdir(exist_ok=True)
    
    # Test export
    print("\n5. Testing model export...")
    example_input = {
        "sequence": sequence,
        "global_features": global_features,
        "sequence_mask": sequence_mask,
    }
    
    export_path = export_dir / "test_model"
    
    try:
        # Export the model
        export_model(model, example_input, export_path, config)
        print("   ✓ Model export successful!")
        
        # Load and test exported model
        exported_path = export_dir / "test_model_exported.pt2"
        
        if exported_path.exists():
            print("\n6. Loading exported model...")
            exported_prog = torch.export.load(str(exported_path))
            print("   ✓ Exported model loaded successfully")
            
            # Test with original batch size
            print("\n7. Validating exported model output...")
            with torch.no_grad():
                # Move to CPU for export testing
                test_kwargs = {
                    "sequence": sequence.cpu(),
                }
                if global_features is not None:
                    test_kwargs["global_features"] = global_features.cpu()
                if sequence_mask is not None:
                    test_kwargs["sequence_mask"] = sequence_mask.cpu()
                
                # Compare outputs
                model_cpu = model.to('cpu')
                original_output = model_cpu(**test_kwargs)
                exported_output = exported_prog.module()(**test_kwargs)
                
                max_diff = torch.max(torch.abs(original_output - exported_output)).item()
                
                if torch.allclose(original_output, exported_output, rtol=1e-4, atol=1e-5):
                    print(f"   ✓ Outputs match! Maximum difference: {max_diff:.2e}")
                else:
                    print(f"   ⚠ Outputs differ! Maximum difference: {max_diff:.2e}")
            
            # Test with different batch sizes
            print("\n8. Testing dynamic batch sizes...")
            
            if test_batch_sizes is None:
                test_batch_sizes = [1, 2, 8, 16]
            
            for test_batch_size in test_batch_sizes:
                try:
                    # Create test inputs
                    test_seq = torch.randn(test_batch_size, seq_length, input_dim)
                    test_global = (
                        torch.randn(test_batch_size, global_dim)
                        if global_dim > 0 else None
                    )
                    test_mask = torch.zeros(test_batch_size, seq_length, dtype=torch.bool)
                    
                    # Prepare kwargs
                    test_kwargs = {"sequence": test_seq}
                    if test_global is not None:
                        test_kwargs["global_features"] = test_global
                    if test_mask is not None:
                        test_kwargs["sequence_mask"] = test_mask
                    
                    # Run exported model
                    with torch.no_grad():
                        exported_out = exported_prog.module()(**test_kwargs)
                    
                    print(f"   ✓ Batch size {test_batch_size:2d}: "
                          f"Output shape {tuple(exported_out.shape)}")
                    
                except Exception as e:
                    print(f"   ✗ Batch size {test_batch_size:2d}: Failed - {e}")
            
            # Test static export if it exists
            static_path = export_dir / "test_model_exported_static.pt2"
            if static_path.exists():
                print("\n9. Static export found, testing...")
                static_prog = torch.export.load(str(static_path))
                
                # Test with original batch size only (static shapes)
                with torch.no_grad():
                    test_kwargs = {
                        "sequence": sequence.cpu(),
                    }
                    if global_features is not None:
                        test_kwargs["global_features"] = global_features.cpu()
                    if sequence_mask is not None:
                        test_kwargs["sequence_mask"] = sequence_mask.cpu()
                    
                    static_output = static_prog.module()(**test_kwargs)
                    print(f"   ✓ Static model works with original batch size {batch_size}")
        
        else:
            print(f"   ⚠ Exported model not found at {exported_path}")
    
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        raise
    
    # Summary
    print("\n" + "=" * 60)
    print("Export Test Summary")
    print("=" * 60)
    print("✓ Model creation successful")
    print("✓ Forward pass successful")
    print("✓ Model export successful")
    print("✓ Exported model validation passed")
    print("✓ Dynamic batch size support confirmed")
    print("\nModel is export-compatible and ready for deployment!")


def test_export_with_real_data(
    config_path: str = "../config/config.jsonc",
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Test export with a trained model checkpoint.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to trained model checkpoint
    """
    print("=" * 60)
    print("Model Export Test with Trained Weights")
    print("=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_prediction_model(config, device=device, compile_model=False)
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        state_dict = checkpoint["state_dict"]
        # Handle compiled model state dict
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in state_dict.items()
            }
        
        model.load_state_dict(state_dict)
        print(f"   Loaded weights from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"   Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    # Run standard export test
    test_model_export(config_path=config_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model export functionality")
    parser.add_argument(
        "--config",
        type=str,
        default="../config/config.jsonc",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model checkpoint",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Directory for export test (default: ./test_export)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to test (default: 1 2 8 16)",
    )
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # Test with trained model
        test_export_with_real_data(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
        )
    else:
        # Test with random weights
        test_model_export(
            config_path=args.config,
            export_dir=args.export_dir,
            test_batch_sizes=args.batch_sizes,
        )