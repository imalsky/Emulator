#!/usr/bin/env python3
"""
train.py – Model training loop for multi-source transformer models.

Implements a standard training workflow including data splitting, epoch loops,
validation, metric calculation (Loss, Pearson R, Explained Variance),
learning rate scheduling, early stopping, checkpointing, optional mixed precision
(AMP) on CUDA, and optional graph compilation (`torch.compile`) on CUDA.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset

# Assuming these modules are in the same directory or project structure
from hardware import configure_dataloader_settings
from model import create_prediction_model
from utils import save_json

logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================


def _split_dataset(
    dataset: Dataset, val_frac: float, test_frac: float, seed: int = 42
) -> Tuple[Subset, Subset, Subset, List[int]]: # Return test_indices again
    """
    Splits a dataset into training, validation, and test subsets.

    Uses a fixed random seed for reproducible splits.

    Args:
        dataset: The complete dataset instance to split.
        val_frac: The fraction of data to use for the validation set (e.g., 0.15).
        test_frac: The fraction of data to use for the test set (e.g., 0.15).
        seed: The random seed for shuffling before splitting.

    Returns:
        A tuple containing: (train_subset, validation_subset, test_subset, test_indices).

    Raises:
        ValueError: If fractions are invalid or the dataset is too small for the split.
    """
    n_samples = len(dataset)

    if not (0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1):
        raise ValueError(
            "Validation and test fractions must be between 0 and 1, "
            "and their sum must be less than 1."
        )

    n_val = int(n_samples * val_frac)
    n_test = int(n_samples * test_frac)
    n_train = n_samples - n_val - n_test

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Dataset size {n_samples} is too small for the specified fractions "
            f"(val={val_frac:.2f}, test={test_frac:.2f}). Resulting sizes: "
            f"Train={n_train}, Val={n_val}, Test={n_test}."
        )

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_samples, generator=generator).tolist()

    test_indices = perm[:n_test]
    val_indices = perm[n_test : n_test + n_val]
    train_indices = perm[n_test + n_val :]

    logger.info(
        "Dataset split: %d train, %d validation, %d test samples.",
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    # Return test indices for potential saving
    return train_subset, val_subset, test_subset, test_indices


# =============================================================================
# Model Trainer Class
# =============================================================================


class ModelTrainer:
    """
    Orchestrates the training, validation, and testing of multi-source models.
    Includes checkpointing.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        save_dir: Path,
        dataset: Dataset,
        collate_fn: Optional[Callable] = None,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
    ) -> None:
        """
        Initializes the ModelTrainer.

        Args:
            config: The configuration dictionary containing hyperparameters and settings.
            device: The torch device (e.g., 'cuda', 'mps', 'cpu') to use for training.
            save_dir: The directory where model checkpoints and logs will be saved.
            dataset: The complete, initialized dataset instance.
            collate_fn: The custom collate function for the DataLoaders.
            val_frac: Fraction of the dataset to use for validation.
            test_frac: Fraction of the dataset to use for testing.
        """
        self.cfg = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.original_dataset = dataset # Keep reference for saving test list

        (
            self.train_ds,
            self.val_ds,
            self.test_ds,
            self.test_indices, # Store test indices
        ) = _split_dataset(
            dataset,
            val_frac=self.cfg.get("val_frac", val_frac),
            test_frac=self.cfg.get("test_frac", test_frac),
            seed=self.cfg.get("random_seed", 42),
        )

        self._build_dataloaders(collate_fn)
        self._build_model()
        self._build_optimiser_criterion()
        self._build_scheduler()

        self.use_amp = (
            self.cfg.get("use_amp", False) and self.device.type == "cuda"
        )
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("Automatic Mixed Precision (AMP) enabled for CUDA.")
        elif self.cfg.get("use_amp", False) and self.device.type != "cuda":
            logger.info(
                f"AMP requested but device is '{self.device.type}'. AMP disabled."
            )

        self.log_path = self.save_dir / "training_log.csv"
        try:
            # Write header for the CSV log file
            header = "epoch,train_loss,val_loss,pearson_r,explained_variance,learning_rate,time_seconds\n"
            self.log_path.write_text(header)
        except OSError as e:
            logger.error(
                f"Failed to write training log header to {self.log_path}: {e}"
            )

        # Save test profile list for reference
        self._save_test_list()

        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def _build_dataloaders(self, collate_fn: Optional[Callable]) -> None:
        """Creates DataLoaders for train, validation, and test sets."""
        hw_settings = configure_dataloader_settings()
        batch_size = self.cfg.get("batch_size", 16)
        num_workers = self.cfg.get("num_workers", hw_settings["num_workers"])

        if num_workers < 0:
            logger.warning("num_workers cannot be negative, setting to 0.")
            num_workers = 0

        recommended_workers = hw_settings["num_workers"]
        if num_workers != recommended_workers:
            logger.info(
                f"Using num_workers={num_workers} (recommended: {recommended_workers} for device '{self.device.type}')."
            )

        # Determine if persistent workers can be used
        persistent = hw_settings["persistent_workers"] and num_workers > 0
        common_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=hw_settings["pin_memory"],
            persistent_workers=persistent,
            collate_fn=collate_fn,
        )

        # Adjust drop_last for training loader if dataset size is smaller than batch size
        drop_last_train = self.cfg.get("drop_last_train", False)
        if len(self.train_ds) < batch_size:
            if drop_last_train:
                logger.warning(
                    f"Training dataset size ({len(self.train_ds)}) < batch size ({batch_size}). "
                    f"Setting drop_last=False for train loader."
                )
            drop_last_train = False

        try:
            # Initialize DataLoaders
            self.train_loader = DataLoader(
                self.train_ds,
                shuffle=True,
                drop_last=drop_last_train,
                **common_kwargs,
            )
            self.val_loader = DataLoader(
                self.val_ds, shuffle=False, drop_last=False, **common_kwargs
            )
            self.test_loader = DataLoader(
                self.test_ds, shuffle=False, drop_last=False, **common_kwargs
            )
        except NotImplementedError as e:
            # Fallback if persistent_workers cause issues (e.g., on some platforms)
            logger.warning(
                "DataLoader raised NotImplementedError (potentially related to persistent_workers). "
                "Retrying with persistent_workers=False. Original error: %s",
                e,
            )
            common_kwargs["persistent_workers"] = False
            self.train_loader = DataLoader(
                self.train_ds,
                shuffle=True,
                drop_last=drop_last_train,
                **common_kwargs,
            )
            self.val_loader = DataLoader(
                self.val_ds, shuffle=False, drop_last=False, **common_kwargs
            )
            self.test_loader = DataLoader(
                self.test_ds, shuffle=False, drop_last=False, **common_kwargs
            )

        # Check if the training loader is empty
        if len(self.train_loader) == 0:
            raise ValueError(
                f"Training DataLoader is empty. Check training dataset size ({len(self.train_ds)}), "
                f"batch size ({batch_size}), and drop_last ({drop_last_train})."
            )
        logger.info(
            "DataLoaders created (Batch size: %d, Workers: %d)",
            batch_size,
            num_workers,
        )

    def _build_model(self) -> None:
        """Instantiates the model and optionally applies torch.compile."""
        logger.info("Building model from configuration...")
        # Create model using the factory function from model.py
        self.model: nn.Module = create_prediction_model(self.cfg).to(self.device)

        # Apply torch.compile if configured and on CUDA
        use_compile = self.cfg.get("use_torch_compile", False)
        if use_compile and self.device.type == "cuda":
            try:
                logger.info("Attempting torch.compile() on model (CUDA)...")
                # Compile the model for potential performance gains
                self.model = torch.compile(self.model)
                logger.info("Model compiled successfully.")
            except Exception as e:
                logger.warning(
                    "torch.compile failed: %s — continuing in eager mode.", e
                )
        elif use_compile and self.device.type != "cuda":
            logger.info(
                "torch.compile requested but device is '%s'; skipping compilation.",
                self.device.type,
            )
        else:
            logger.info("torch.compile not requested or not applicable.")

    def _build_optimiser_criterion(self) -> None:
        """Sets up the optimizer and loss criterion."""
        lr = self.cfg.get("learning_rate", 1e-4)
        wd = self.cfg.get("weight_decay", 1e-5)
        opt_name = self.cfg.get("optimizer", "adamw").lower()
        params_to_optimize = self.model.parameters()

        logger.info(
            "Configuring optimizer: %s (lr=%.1e, wd=%.1e)", opt_name, lr, wd
        )

        # Select optimizer based on configuration
        if opt_name == "sgd":
            self.optimizer = optim.SGD(
                params_to_optimize, lr=lr, weight_decay=wd, momentum=0.9
            )
        elif opt_name == "adam":
            self.optimizer = optim.Adam(
                params_to_optimize, lr=lr, weight_decay=wd
            )
        elif opt_name == "adamw":
            # AdamW is often a good default choice
            self.optimizer = optim.AdamW(
                params_to_optimize,
                lr=lr,
                weight_decay=wd,
                betas=tuple(self.cfg.get("adamw_betas", (0.9, 0.999))),
                eps=self.cfg.get("adamw_eps", 1e-8),
                amsgrad=self.cfg.get("adamw_amsgrad", False),
            )
        else:
            logger.warning(
                "Unknown optimizer '%s' specified, defaulting to AdamW.", opt_name
            )
            self.optimizer = optim.AdamW(
                params_to_optimize, lr=lr, weight_decay=wd
            )

        # Select loss function based on configuration
        loss_map = {
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss(),
            "smooth_l1": nn.SmoothL1Loss(
                beta=self.cfg.get("smooth_l1_beta", 1.0)
            ),
            "huber": nn.HuberLoss(delta=self.cfg.get("huber_delta", 1.0)),
        }
        loss_choice = self.cfg.get("loss_function", "mse").lower()
        self.criterion = loss_map.get(loss_choice)
        if self.criterion is None:
            logger.warning(
                "Unknown loss function '%s' specified, defaulting to MSELoss.",
                loss_choice,
            )
            self.criterion = nn.MSELoss()
        logger.info("Using loss function: %s", type(self.criterion).__name__)

        # Configure gradient clipping
        self.max_grad_norm = self.cfg.get("gradient_clip_val", 1.0)
        if self.max_grad_norm <= 0:
            logger.info("Gradient clipping disabled (max_grad_norm <= 0).")
            self.max_grad_norm = float("inf") # Effectively disable clipping
        else:
            logger.info(
                "Gradient clipping enabled with max_norm: %.2f", self.max_grad_norm
            )

    def _build_scheduler(self) -> None:
        """Sets up the learning rate scheduler."""
        scheduler_patience = self.cfg.get("lr_patience", 5)
        scheduler_factor = self.cfg.get("lr_factor", 0.5) # Renamed from gamma for clarity
        min_lr = self.cfg.get("min_lr", 1e-7)

        # Use ReduceLROnPlateau scheduler, which reduces LR when validation loss plateaus
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min", # Reduce LR when the metric stops decreasing
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr,
            verbose=False, # Set to True for more verbose output from scheduler
        )
        logger.info(
            "Using ReduceLROnPlateau LR scheduler (factor=%.2f, patience=%d, min_lr=%.1e).",
            scheduler_factor,
            scheduler_patience,
            min_lr,
        )

    def train(self) -> float:
        """
        Runs the main training loop with checkpointing and early stopping.

        Returns:
            The best validation loss (float) achieved during training.
            Returns float('inf') if training fails or makes no progress.
        """
        epochs = self.cfg.get("epochs", 100)
        patience = self.cfg.get("early_stopping_patience", 10)
        min_delta = self.cfg.get("min_delta", 1e-6) # Minimum change to qualify as improvement

        wait_counter = 0 # Counter for early stopping
        self.best_val_loss = float("inf")
        self.best_epoch = -1

        logger.info(f"--- Starting Training Loop (Max Epochs: {epochs}) ---")
        logger.info(
            f"Device: {self.device}, AMP: {self.use_amp}, "
            f"Compile: {self.cfg.get('use_torch_compile', False) and self.device.type == 'cuda'}"
        )
        logger.info(
            f"Early stopping patience: {patience} epochs, Min improvement delta: {min_delta:.1e}"
        )
        logger.info(f"Saving checkpoints and logs to: {self.save_dir}")

        # Print training log header
        header_fmt = (
            f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^12} | "
            f"{'Pearson R':^11} | {'Exp Var':^11} | {'LR':^11} | {'Time (s)':^9}"
        )
        logger.info("-" * len(header_fmt))
        logger.info(header_fmt)
        logger.info("-" * len(header_fmt))

        completed_epochs = 0
        training_successful = False
        val_loss_epoch = float('inf') # Initialize val_loss_epoch

        try:
            for ep in range(1, epochs + 1):
                completed_epochs = ep
                epoch_start_time = time.time()
                last_lr = self.optimizer.param_groups[0]["lr"]

                # Run training epoch
                train_loss_epoch = self._run_epoch(self.train_loader, train=True)

                # Run validation epoch
                val_results = self._run_epoch(self.val_loader, train=False)
                if isinstance(val_results, tuple):
                    val_loss_epoch, pearson_r, explained_var = val_results
                else:
                    # Handle case where validation might fail (though unlikely with checks in _run_epoch)
                    logger.error(f"Validation epoch {ep} did not return expected tuple.")
                    val_loss_epoch, pearson_r, explained_var = float('inf'), 0.0, 0.0

                epoch_duration = time.time() - epoch_start_time

                # Step the LR scheduler based on validation loss
                self.scheduler.step(val_loss_epoch)
                current_lr = self.optimizer.param_groups[0]["lr"]
                if current_lr < last_lr:
                    logger.info(
                        f"Learning rate reduced to {current_lr:.3e} at epoch {ep}"
                    )

                # Log epoch results (formatted)
                log_msg = (
                    f"{ep:^7d} | {train_loss_epoch:^12.4e} | {val_loss_epoch:^12.4e} | "
                    f"{pearson_r:^11.4f} | {explained_var:^11.4f} | {current_lr:^11.3e} | {epoch_duration:^9.1f}"
                )
                logger.info(log_msg)

                # Append results to the CSV log file
                try:
                    with self.log_path.open("a") as f:
                        f.write(
                            f"{ep},{train_loss_epoch:.6e},{val_loss_epoch:.6e},{pearson_r:.6f},"
                            f"{explained_var:.6f},{current_lr:.6e},{epoch_duration:.2f}\n"
                        )
                except OSError as e:
                    logger.warning(
                        f"Could not write to training log file {self.log_path}: {e}"
                    )

                # Checkpointing and Early Stopping Logic
                if val_loss_epoch < self.best_val_loss - min_delta:
                    # Improvement detected
                    self.best_val_loss = val_loss_epoch
                    self.best_epoch = ep
                    wait_counter = 0
                    # Save the best model checkpoint (without logging the save action explicitly)
                    self._checkpoint("best_model.pt", ep, self.best_val_loss)
                    # logger.info(f"  -> New best model saved at epoch {ep} (Val Loss: {self.best_val_loss:.4e})") # Removed per user request
                else:
                    # No improvement
                    wait_counter += 1
                    if wait_counter >= patience:
                        logger.info(
                            f"Early stopping triggered after epoch {ep} "
                            f"(no improvement greater than {min_delta:.1e} for {patience} epochs)."
                        )
                        training_successful = True
                        break # Exit training loop

            # Check if training completed normally (reached max epochs)
            if completed_epochs == epochs and not training_successful:
                logger.info(f"Maximum epochs ({epochs}) reached.")
                training_successful = True

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user (KeyboardInterrupt).")
            training_successful = False # Mark as not successful
        except Exception as e:
            logger.exception("An error occurred during the training loop: %s", e)
            training_successful = False # Mark as not successful

        finally:
            # Final actions after training loop finishes or is interrupted
            logger.info("-" * len(header_fmt)) # Separator line
            best_model_path = self.save_dir / "best_model.pt"
            final_model_path = self.save_dir / "final_model.pt"

            if self.best_epoch != -1:
                # If a best model was found during training
                logger.info(
                    f"Training finished. Best validation loss: {self.best_val_loss:.4e} at epoch {self.best_epoch}."
                )
                # Save the final model state regardless
                self._checkpoint(
                    "final_model.pt", completed_epochs, val_loss_epoch, is_final=True
                )

                # --- Final Test Evaluation ---
                logger.info("--- Starting Final Test Evaluation (using best model state) ---")
                if best_model_path.exists():
                    try:
                        # Load the best model checkpoint for testing
                        checkpoint = torch.load(best_model_path, map_location=self.device)
                        state_dict = checkpoint['state_dict']

                        # Handle loading into compiled vs non-compiled model
                        model_to_load = self.model
                        if hasattr(model_to_load, '_orig_mod'):
                            # If model was compiled, load into the original underlying module
                            logger.debug("Loading state dict into original uncompiled model for testing.")
                            model_to_load = model_to_load._orig_mod

                        model_to_load.load_state_dict(state_dict)
                        logger.info("Successfully loaded best model state from %s.", best_model_path.name)
                        # Run evaluation on the test set
                        self.test()
                    except Exception as e:
                        logger.error(f"Failed to load best model state ({best_model_path.name}) for testing: {e}", exc_info=True)
                        logger.warning("Testing with the *final* model state instead.")
                        # Attempt to test with the final model state if best failed to load
                        if final_model_path.exists():
                             # Assume final state is still in self.model if not explicitly loaded
                            self.test()
                        else:
                            logger.error("Neither best nor final model available for testing.")
                else:
                    # If best_model.pt doesn't exist (shouldn't happen if best_epoch != -1, but safeguard)
                    logger.warning("Best model checkpoint (best_model.pt) not found, testing with final model state.")
                    self.test() # Assume final state is still in self.model

            else:
                # If no improvement was ever detected
                logger.warning(
                    "Training completed, but no improvement detected over initial validation loss."
                )
                # Save the final state anyway for inspection if training ran at least one epoch
                if completed_epochs > 0:
                    self._checkpoint("final_model.pt", completed_epochs, val_loss_epoch, is_final=True)
                    logger.info("No best model found. Testing with final model state.")
                    self.test() # Assume final state is still in self.model
                else:
                    logger.error("Training did not complete any epochs. Skipping final testing.")

            # Return the best validation loss found
            return self.best_val_loss if self.best_epoch != -1 else float("inf")

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
    ) -> Union[float, Tuple[float, float, float]]:
        """
        Runs one pass (train or evaluation) over the given DataLoader.

        Args:
            loader: The DataLoader for the current phase (train, val, or test).
            train: Boolean indicating if this is a training phase (enables backprop).

        Returns:
            If train=True: Average training loss (float).
            If train=False: Tuple of (average loss, pearson_r, explained_variance).
        """
        self.model.train(train) # Set model to training or evaluation mode
        total_loss = 0.0
        batch_count = 0
        all_preds_list: List[torch.Tensor] = []
        all_trues_list: List[torch.Tensor] = []

        # Set up contexts for AMP and gradient calculation
        amp_enabled = self.use_amp
        autocast_context = (
            torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled)
            if amp_enabled
            else contextlib.nullcontext()
        )
        grad_context = torch.enable_grad() if train else torch.no_grad()

        phase_name = "train" if train else "eval"

        with grad_context:
            for batch_idx, batch in enumerate(loader):
                try:
                    inputs, targets = batch
                    # Move data to the target device
                    inputs = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in inputs.items()
                    }
                    targets = targets.to(self.device, non_blocking=True)

                    # Forward pass (potentially with AMP)
                    with autocast_context:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)

                    # Check for non-finite loss
                    if not torch.isfinite(loss):
                        logger.warning(
                            f"Non-finite loss detected in {phase_name} epoch, batch {batch_idx}. "
                            f"Loss: {loss.item()}. Skipping batch."
                        )
                        continue # Skip this batch

                    if train:
                        # --- Training Step ---
                        self.optimizer.zero_grad(set_to_none=True) # More efficient zeroing

                        if self.use_amp:
                            # AMP backward pass
                            self.scaler.scale(loss).backward()
                            # Unscale gradients before clipping
                            self.scaler.unscale_(self.optimizer)
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm
                            )
                            # Optimizer step (using scaler)
                            self.scaler.step(self.optimizer)
                            # Update scaler for next iteration
                            self.scaler.update()
                        else:
                            # Standard backward pass
                            loss.backward()
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm
                            )
                            # Optimizer step
                            self.optimizer.step()

                    # Accumulate loss and count batches
                    total_loss += loss.item()
                    batch_count += 1

                    # Store predictions and targets for metric calculation if evaluating
                    if not train:
                        all_preds_list.append(outputs.detach().float().cpu())
                        all_trues_list.append(targets.detach().float().cpu())

                except Exception as e:
                    # Log errors during batch processing but continue if possible
                    logger.error(
                        f"Error during {phase_name} epoch, batch {batch_idx + 1}/{len(loader)}: {e}",
                        exc_info=False, # Set to True for full traceback in logs
                    )
                    logger.debug(
                        "Detailed error trace for batch processing:", exc_info=True
                    )
                    continue # Continue to the next batch

        # After iterating through all batches in the epoch
        if batch_count == 0:
            logger.error(
                f"No batches successfully processed in {phase_name} epoch."
            )
            # Return default values indicating failure
            return (float("inf"), 0.0, 0.0) if not train else float("inf")

        avg_loss = total_loss / batch_count

        if train:
            # Return only average loss for training phase
            return avg_loss
        else:
            # --- Evaluation Metric Calculation ---
            if not all_preds_list or not all_trues_list:
                logger.warning(
                    "No valid prediction/target tensors collected during evaluation. Cannot calculate metrics."
                )
                return avg_loss, 0.0, 0.0

            try:
                # Concatenate all batch predictions and targets
                preds_all = torch.cat(all_preds_list).flatten()
                trues_all = torch.cat(all_trues_list).flatten()

                if preds_all.numel() == 0 or trues_all.numel() == 0:
                    logger.warning(
                        "Evaluation metrics calculation skipped: Empty prediction or target tensors."
                    )
                    return avg_loss, 0.0, 0.0

                # Calculate Pearson Correlation Coefficient
                mean_preds = torch.mean(preds_all)
                mean_trues = torch.mean(trues_all)
                cov = torch.mean(
                    (preds_all - mean_preds) * (trues_all - mean_trues)
                )
                std_preds = torch.std(preds_all, unbiased=False)
                std_trues = torch.std(trues_all, unbiased=False)
                pearson_r = cov / (std_preds * std_trues + 1e-8) # Add epsilon for stability

                # Calculate Explained Variance Score
                var_trues = torch.var(trues_all, unbiased=False)
                if var_trues < 1e-8: # Handle case of constant true values
                    # If predictions are also constant, variance is perfectly explained (1.0)
                    # Otherwise, it's undefined or negatively infinite.
                    explained_var = (
                        torch.tensor(1.0)
                        if torch.var(preds_all, unbiased=False) < 1e-8
                        else torch.tensor(float("-inf"))
                    )
                else:
                    # Standard formula: 1 - Var(residual) / Var(true)
                    explained_var = 1.0 - (
                        torch.var(trues_all - preds_all, unbiased=False)
                        / var_trues
                    )

                # Clamp Pearson R to valid range [-1, 1]
                pearson_r = torch.clamp(pearson_r, min=-1.0, max=1.0)

                pearson_r_float = pearson_r.item()
                explained_var_float = explained_var.item()

                # *** ADDED: Check for non-finite metric results ***
                if not np.isfinite(pearson_r_float):
                    logger.warning(f"Pearson R calculation resulted in non-finite value: {pearson_r_float}. Check for constant data or zero std dev.")
                    pearson_r_float = 0.0 # Default to 0 if non-finite
                if not np.isfinite(explained_var_float):
                    logger.warning(f"Explained Variance calculation resulted in non-finite value: {explained_var_float}. Check for constant data.")
                    explained_var_float = 0.0 # Default to 0 if non-finite

                return avg_loss, pearson_r_float, explained_var_float

            except Exception as metric_exc:
                logger.error(
                    f"Error calculating evaluation metrics: {metric_exc}",
                    exc_info=True, # Log traceback for metric errors
                )
                # Return default metrics on error
                return avg_loss, 0.0, 0.0

    def _checkpoint(self, name: str, epoch: int, val_loss: float, is_final: bool = False) -> None:
        """
        Saves model checkpoint dictionary to disk.

        Handles potential model compilation state dict wrapping.

        Args:
            name: Filename for the checkpoint (e.g., "best_model.pt").
            epoch: The epoch number at which the checkpoint is saved.
            val_loss: The validation loss associated with this checkpoint.
            is_final: Flag indicating if this is the final model save after training.
        """
        model_to_save = self.model
        state_dict_to_save = None
        try:
            # *** CORRECTED: Handle torch.compile wrapper ***
            # Check if the model was compiled and get the original module if so
            if hasattr(model_to_save, '_orig_mod'):
                logger.debug("Unwrapping compiled model state dict for checkpoint '%s'.", name)
                model_to_save = model_to_save._orig_mod # Get the original model
            state_dict_to_save = model_to_save.state_dict()
        except Exception as e:
            logger.warning(f"Could not get state_dict from primary/unwrapped model for checkpoint '{name}': {e}. "
                           f"Attempting to save raw model state dict.", exc_info=False)
            try:
                # Fallback attempt: save the state dict directly from self.model
                state_dict_to_save = self.model.state_dict()
            except Exception as e2:
                 logger.error(f"Failed to get even raw state_dict for checkpoint '{name}': {e2}. Checkpoint will not be saved.")
                 return # Do not proceed if state_dict cannot be obtained

        # Ensure state_dict_to_save is not None before proceeding
        if state_dict_to_save is None:
            logger.error(f"Could not obtain state_dict for saving checkpoint '{name}'. Aborting save.")
            return

        # Prepare checkpoint data
        save_data = {
            'state_dict': state_dict_to_save,
            'config': self.cfg, # Save configuration used for this run
            'epoch': epoch,
            'val_loss': val_loss,
            # Optionally save optimizer/scheduler state if resuming training is needed
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
            # 'amp_scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }
        save_path = self.save_dir / name

        try:
            # Save the checkpoint
            torch.save(save_data, save_path)
            # Optional: Log saving action at DEBUG level if needed
            # log_level = logging.INFO if is_final or name == "best_model.pt" else logging.DEBUG
            # logger.log(log_level, "Checkpoint saved: %s (Epoch: %d, Val Loss: %.4e)", save_path.name, epoch, val_loss)
        except OSError as e:
            logger.error(f"Failed to save checkpoint {name} to {save_path}: {e} (Disk space issue?)")
        except Exception as e:
            logger.error(f"Failed to save checkpoint {name} to {save_path}: {e}")


    def test(self) -> Optional[Tuple[float, float, float]]:
        """
        Evaluates the currently loaded model state on the test set.

        Assumes the desired model state (e.g., best checkpoint or final state)
        has already been loaded before calling this method.
        Logs results and saves them to `test_metrics.json`.

        Returns:
            A tuple (test_loss, test_pearson_r, test_explained_variance) on success,
            or None if evaluation fails.
        """
        # Note: Model state should be loaded *before* calling this method.
        # The train() method handles loading the best checkpoint.
        logger.info("--- Starting Test Set Evaluation ---")
        test_metrics_path = self.save_dir / "test_metrics.json"
        try:
            # Run evaluation epoch on the test loader
            eval_results = self._run_epoch(self.test_loader, train=False)

            # Check if evaluation was successful
            if not isinstance(eval_results, tuple) or eval_results[0] == float("inf"):
                logger.error(
                    "Test evaluation failed or produced invalid results."
                )
                return None

            test_loss, test_pearson, test_exp_var = eval_results

            # Log test results
            logger.info(
                "Test Results | Loss: %.4e | Pearson R: %.4f | Explained Var: %.4f",
                test_loss,
                test_pearson,
                test_exp_var,
            )

            # Prepare metrics dictionary for saving
            test_metrics = {
                "test_loss": test_loss,
                "test_pearson_r": test_pearson, # Consistent naming
                "test_explained_variance": test_exp_var,
            }

            # Save test metrics to JSON file
            try:
                save_success = save_json(test_metrics, test_metrics_path)
                if not save_success:
                    logger.warning(
                        "Failed to save test metrics JSON using utils.save_json."
                    )
            except Exception as e:
                logger.error(
                    f"Failed to save test metrics to {test_metrics_path.name}: {e}"
                )

            return test_loss, test_pearson, test_exp_var

        except Exception as e:
            logger.exception(
                "An unexpected error occurred during test set evaluation: %s", e
            )
            return None

    def _save_test_list(self) -> None:
        """Saves the list of file paths or identifiers used in the test set."""
        test_info_path = self.save_dir / "test_set_info.json"
        try:
            # Try to access the original dataset, potentially unwrapping Subset wrappers
            base_dataset = self.original_dataset
            while isinstance(base_dataset, Subset):
                if hasattr(base_dataset, 'dataset'):
                    base_dataset = base_dataset.dataset
                else:
                    logger.warning("Could not fully unwrap Subset dataset to find base attributes.")
                    break

            # Check if the base dataset has the expected 'valid_files' attribute
            if hasattr(base_dataset, 'valid_files') and hasattr(self, 'test_indices') and isinstance(self.test_indices, list):
                num_valid_files = len(base_dataset.valid_files)
                # Filter out any invalid indices (safeguard)
                valid_test_indices = [i for i in self.test_indices if 0 <= i < num_valid_files]

                if len(valid_test_indices) != len(self.test_indices):
                    logger.warning(
                        "Some test indices (%d) were out of bounds for base_dataset.valid_files (len %d). Saving only valid paths.",
                        len(self.test_indices) - len(valid_test_indices), num_valid_files
                    )

                # Get the actual file paths corresponding to the test indices
                test_files_paths = [base_dataset.valid_files[i] for i in valid_test_indices]

                # Convert paths to strings (preferably relative paths or just names)
                test_files_str = []
                base_data_dir = getattr(base_dataset, 'data_folder', None)
                if base_data_dir: base_data_dir = Path(base_data_dir)

                for fpath in test_files_paths:
                    try:
                        # Try to make path relative to base data dir if possible
                        if base_data_dir and fpath.is_relative_to(base_data_dir):
                            rel_path = fpath.relative_to(base_data_dir)
                            test_files_str.append(str(rel_path))
                        else:
                            # Fallback to just the filename
                            test_files_str.append(fpath.name)
                    except (ValueError, TypeError, AttributeError):
                        # Fallback if path operations fail
                        test_files_str.append(fpath.name)

                test_info = {"test_profile_paths": sorted(test_files_str)}
                save_type = "paths"

            else:
                # If file paths cannot be determined, save the raw indices
                logger.warning("Could not determine test profile paths from dataset attributes ('valid_files'). Saving test indices instead.")
                test_info = {"test_indices": sorted(self.test_indices)}
                save_type = "indices"

            # Save the information to a JSON file
            save_success = save_json(test_info, test_info_path)
            if save_success:
                logger.info(f"Saved {len(test_info[f'test_{save_type}'])} test profile {save_type} to {test_info_path.name}")
            else:
                logger.error(f"Failed to save test set info to {test_info_path.name} using utils.save_json.")

        except Exception as e:
            logger.exception(f"An error occurred while trying to save the test profile list: {e}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = ["ModelTrainer"]
