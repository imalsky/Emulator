#!/usr/bin/env python3
"""
plot_model_training.py - Plots the training and validation loss curves
                         from the training_log.csv file. Simple version.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Loads training log data and generates a loss curve plot.
    """
    try:
        # Assume the script is run from a directory where '../data' and './' are valid
        SCRIPT_DIR = Path(__file__).resolve().parent
        # Simplified path assumptions - adjust if needed
        LOG_FILE_PATH = SCRIPT_DIR / ".." / "data" / "model" / "training_log.csv"
        PLOTS_DIR = SCRIPT_DIR / ".." / "plots"
        PLOTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure plots directory exists

        STYLE_FILE = SCRIPT_DIR / "science.mplstyle" # Style file in the same directory as the script

        # Apply plotting style if found
        if STYLE_FILE.exists():
            try:
                plt.style.use(str(STYLE_FILE)) # Use str() for older matplotlib versions if needed
                logger.info(f"Using style file: {STYLE_FILE}")
            except Exception as e:
                logger.warning(f"Could not apply style file {STYLE_FILE}: {e}. Using default styles.")
        else:
            logger.warning(f"Style file not found: {STYLE_FILE}. Using default matplotlib styles.")

        # --- Load Data ---
        if not LOG_FILE_PATH.exists():
            logger.error(f"Training log file not found: {LOG_FILE_PATH}")
            sys.exit(1) # Exit if log file is missing

        logger.info(f"Loading training log from: {LOG_FILE_PATH}")
        try:
            data = pd.read_csv(LOG_FILE_PATH)
        except pd.errors.EmptyDataError:
            logger.error(f"Training log file is empty: {LOG_FILE_PATH}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading training log file: {e}")
            sys.exit(1)

        # Check for required columns
        required_columns = ['epoch', 'train_loss', 'val_loss']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Log file {LOG_FILE_PATH} missing one or more required columns: {required_columns}")
            sys.exit(1)

        # Extract data
        epochs = data['epoch']
        train_loss = data['train_loss']
        val_loss = data['val_loss']

        # --- Plotting ---
        logger.info("Generating loss curve plot...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 8)) # Use ax for consistency

        # Use scatter as in the original snippet, adjust 's' for point size
        ax.scatter(epochs, train_loss, label='Training Loss', color='blue', s=50)
        ax.scatter(epochs, val_loss, label='Validation Loss', color='orange', s=50)

        # Configure plot axes and labels
        ax.set_xlabel("Epoch", fontsize=14) # Slightly smaller font
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_title("Model Training Progress", fontsize=16)
        ax.legend(fontsize=12)
        ax.set_yscale('log') # Keep log scale for loss

        plt.tight_layout() # Apply tight layout

        # --- Save Figure ---
        plot_filename = PLOTS_DIR / "training_validation_loss.png"
        try:
            plt.savefig(plot_filename, dpi=250)
            logger.info(f"Plot saved successfully to: {plot_filename}")
        except Exception as e:
            logger.error(f"Failed to save plot to {plot_filename}: {e}")

        plt.close(fig) # Close the figure to free memory

    except FileNotFoundError as e:
        logger.error(f"A required directory or file was not found: {e}. Check paths.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
