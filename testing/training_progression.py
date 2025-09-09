#!/usr/bin/env python3
"""Plot training progression from log."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('science.mplstyle')

MODEL_DIR = Path("../models/trained_model")

def plot_training():
    """Plot training curves."""
    log_path = MODEL_DIR / "training_log.csv"
    if not log_path.exists():
        print(f"No log found: {log_path}")
        return
    
    df = pd.read_csv(log_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.scatter(df["epoch"], df["train_loss"], color='black', label="Train", alpha=0.7)
    ax1.scatter(df["epoch"], df["val_loss"], color='red',label="Validation", alpha=0.7)

    
    # Mark best epoch
    best_idx = df["val_loss"].idxmin()
    #ax1.axvline(df.loc[best_idx, "epoch"], color="black", linestyle="--", alpha=0.5)
    #ax1.scatter(df.loc[best_idx, "epoch"], df.loc[best_idx, "val_loss"], 
    #            color="blue", s=50, label=f"Best (epoch {df.loc[best_idx, 'epoch']})")
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_yscale('log')
    ax1.legend()
    
    # Learning rate
    ax2.semilogy(df["epoch"], df["lr"], color="black")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")

    plt.tight_layout()
    save_path = MODEL_DIR / "plots" / "training_curves.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"Total epochs: {len(df)}")
    print(f"Best val loss: {df.loc[best_idx, 'val_loss']:.3e} (epoch {df.loc[best_idx, 'epoch']})")
    print(f"Final train loss: {df.iloc[-1]['train_loss']:.3e}")
    print(f"Final val loss: {df.iloc[-1]['val_loss']:.3e}")

if __name__ == "__main__":
    plot_training()