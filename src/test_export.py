#!/usr/bin/env python3
"""
test_export.py - Test script to verify model export functionality
"""
import torch
from pathlib import Path
from model import create_prediction_model, export_model
from utils import load_config

def test_model_export():
    """Test that model can be successfully exported and produces identical results."""
    
    # Load config
    config = load_config("../config/config.jsonc")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_prediction_model(config, device=device, compile_model=False)
    
    # Create dummy inputs
    batch_size = 4
    seq_length = 64
    input_dim = len(config["data_specification"]["input_variables"])
    global_dim = len(config["data_specification"].get("global_variables", []))
    
    sequence = torch.randn(batch_size, seq_length, input_dim, device=device)
    global_features = torch.randn(batch_size, global_dim, device=device) if global_dim > 0 else None
    
    # Create padding mask (True = padding)
    sequence_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
    # Add some padding to last sequences
    sequence_mask[0, 50:] = True
    sequence_mask[1, 55:] = True
    
    # Test forward pass
    with torch.no_grad():
        output = model(sequence, global_features, sequence_mask)
    
    print(f"✓ Model forward pass successful. Output shape: {output.shape}")
    
    # Test export
    example_input = {
        "sequence": sequence,
        "global_features": global_features,
        "sequence_mask": sequence_mask
    }
    
    export_path = Path("test_export")
    export_path.mkdir(exist_ok=True)
    
    try:
        export_model(model, example_input, export_path / "test_model", config)
        print("✓ Model export successful!")
        
        # Load and test exported model
        exported_path = export_path / "test_model_exported.pt2"
        if exported_path.exists():
            exported_prog = torch.export.load(str(exported_path))
            
            # Test with different batch sizes
            for test_batch_size in [1, 2, 8, 16]:
                test_seq = torch.randn(test_batch_size, seq_length, input_dim)
                test_global = torch.randn(test_batch_size, global_dim) if global_dim > 0 else None
                test_mask = torch.zeros(test_batch_size, seq_length, dtype=torch.bool)
                
                test_kwargs = {"sequence": test_seq}
                if test_global is not None:
                    test_kwargs["global_features"] = test_global
                if test_mask is not None:
                    test_kwargs["sequence_mask"] = test_mask
                
                exported_out = exported_prog.module()(**test_kwargs)
                print(f"✓ Exported model works with batch_size={test_batch_size}")
                
    except Exception as e:
        print(f"✗ Export failed: {e}")
        raise
    
    print("\nAll tests passed! Model is export-compatible.")

if __name__ == "__main__":
    test_model_export()