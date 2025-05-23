#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna # type: ignore
import torch
from optuna.exceptions import TrialPruned
from torch import nn, optim
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

# Local imports
from dataset import AtmosphericDataset # Assuming AtmosphericDataset is in dataset.py
from hardware import configure_dataloader_settings
from model import MultiEncoderTransformer, create_prediction_model
from utils import save_json

logger = logging.getLogger(__name__)

# Define a constant for DataLoader workers
DATALOADER_NUM_WORKERS = 1


def _split_dataset(
    dataset: AtmosphericDataset,
    val_frac: float,
    test_frac: float,
    *,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset, List[str]]:
    """
    Deterministically splits an AtmosphericDataset into train, val, test.

    Args:
        dataset: The full AtmosphericDataset instance.
        val_frac: Fraction for validation.
        test_frac: Fraction for testing.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple: (train_subset, val_subset, test_subset, test_filenames_list).
    """
    n = len(dataset)
    if not (
        0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1
    ):
        # Critical configuration error.
        logger.critical(
            "Dataset split fractions (val_frac: %f, test_frac: %f) are "
            "invalid. They must be between 0 and 1, and their sum must be "
            "less than 1. Exiting.", val_frac, test_frac
        )
        sys.exit(1)


    num_val = int(n * val_frac)
    num_test = int(n * test_frac)
    num_train = n - num_val - num_test

    if num_train <= 0:
        logger.critical(
            "Dataset size %d is too small for the requested split "
            "fractions (val: %f, test: %f), resulting in %d training "
            "samples. Exiting.", n, val_frac, test_frac, num_train
        )
        sys.exit(1)


    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    test_indices: List[int] = indices[:num_test]
    val_indices: List[int] = indices[num_test : num_test + num_val]
    train_indices: List[int] = indices[num_test + num_val:]

    logger.info(
        "Dataset split: %d train / %d val / %d test samples.",
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

    try:
        # This method is assumed to exist on AtmosphericDataset for this script to work.
        # If it doesn't, this will raise an AttributeError.
        test_filenames = dataset.get_profile_filenames_by_indices(test_indices) # type: ignore
        if test_filenames:
            logger.info("First few test filenames: %s...",test_filenames[:min(3, len(test_filenames))])
    except AttributeError:
        logger.critical(
            "AtmosphericDataset does not have method 'get_profile_filenames_by_indices'. "
            "This is required by the training script. Exiting."
            )
        sys.exit(1)
    except IndexError as e: # Should not happen if indices are from randperm(n)
        logger.critical("Error retrieving test filenames by indices: %s. Exiting.", e)
        sys.exit(1)


    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
        test_filenames,
    )


class ModelTrainer:
    """
    Orchestrates the training, validation, and testing of the model.

    Manages data splitting, DataLoader setup, model/optimizer instantiation,
    training loop with validation, early stopping, checkpointing, Optuna
    integration, and final test set evaluation.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        save_dir: Path,
        dataset: AtmosphericDataset,
        collate_fn: Callable, # Expecting an instance of PadCollate
        *,
        optuna_trial: Optional[optuna.Trial] = None,
    ) -> None:
        """
        Initializes the ModelTrainer.

        Args:
            config: Dictionary with training parameters and model config.
            device: PyTorch device for training.
            save_dir: Directory for saving model artifacts and logs.
            dataset: The complete AtmosphericDataset instance.
            collate_fn: Custom collate function for DataLoaders.
            optuna_trial: Optional Optuna Trial object for HPO.
        """
        self.cfg = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.optuna_trial = optuna_trial
        self.padding_value = float(self.cfg.get("padding_value", 0.0)) # Used by collate_fn indirectly
        self.model: MultiEncoderTransformer # type: ignore[assignment] # Initialized in _build_model

        (
            self.train_ds,
            self.val_ds,
            self.test_ds,
            self.test_filenames,
        ) = _split_dataset(
            dataset,
            val_frac=self.cfg.get("val_frac", 0.15),
            test_frac=self.cfg.get("test_frac", 0.15),
            seed=self.cfg.get("random_seed", 42),
        )
        self._build_dataloaders(collate_fn)
        self._build_model()
        self._build_optimiser()
        self._build_scheduler()

        self.use_amp = bool(
            self.cfg.get("use_amp", False) and device.type == "cuda"
        )
        self.scaler: Optional[GradScaler] = (
            GradScaler() if self.use_amp else None # Corrected GradScaler instantiation
        )
        if self.use_amp:
            logger.info("AMP enabled on CUDA.")

        self.criterion = nn.MSELoss(reduction='none')
        self.max_grad_norm = max(
            float(self.cfg.get("gradient_clip_val", 1.0)), 1e-12
        )

        self.log_path = self.save_dir / "training_log.csv"
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("epoch,train_loss,val_loss,lr,time_s\n")
        except IOError as e:
            logger.critical("Failed to write training log header to %s: %s. Exiting.", self.log_path, e)
            sys.exit(1)


        self.best_val_loss, self.best_epoch = float("inf"), -1
        self._clear_cuda_every = int(self.cfg.get("clear_cuda_cache_every", 0))
        self._save_test_set_info()

    def _build_dataloaders(self, collate_fn: Callable) -> None:
        """Initializes DataLoaders for train, validation, and test sets."""
        hw_settings = configure_dataloader_settings() # For pin_memory, persistent_workers
        batch_size = int(self.cfg.get("batch_size", 16))
        
        # Use constant for num_workers, remove from config and hw_settings for this purpose
        num_w = DATALOADER_NUM_WORKERS

        common_dl_args: Dict[str, Any] = dict( # Added type hint for common_dl_args
            batch_size=batch_size,
            num_workers=num_w,
            pin_memory=hw_settings["pin_memory"], # Still use from hw_settings
            persistent_workers=num_w > 0 and hw_settings["persistent_workers"], # Still use
            collate_fn=collate_fn,
        )
        self.train_loader = DataLoader(
            self.train_ds, shuffle=True, drop_last=False, **common_dl_args
        )
        self.val_loader = DataLoader(
            self.val_ds, shuffle=False, **common_dl_args
        )
        self.test_loader = DataLoader(
            self.test_ds, shuffle=False, **common_dl_args
        )
        logger.info(
            "DataLoaders created (batch_size=%d, num_workers=%d).",
            batch_size, num_w
        )

    def _build_model(self) -> None:
        """Creates and initializes the prediction model."""
        # create_prediction_model handles device assignment and potential exits
        # for critical hyperparameter issues.
        self.model = create_prediction_model(
            self.cfg, device=self.device
        )
        if (
            self.cfg.get("use_torch_compile", False)
            and self.device.type == "cuda"
            and hasattr(torch, "compile")
        ):
            try:
                self.model = torch.compile(self.model) # type: ignore
                logger.info("Model compiled with torch.compile().")
            except Exception as e: # Catch any exception from torch.compile
                logger.warning(
                    "torch.compile failed: %s. Using uncompiled model.", e
                )
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Model built with %s trainable parameters.", f"{num_params:,}")


    def _build_optimiser(self) -> None:
        """Initializes the optimizer based on configuration."""
        opt_name = str(self.cfg.get("optimizer", "adamw")).lower()
        lr = float(self.cfg.get("learning_rate", 1e-4))
        wd = float(self.cfg.get("weight_decay", 1e-5))

        if opt_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9
            )
        elif opt_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        elif opt_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        else:
            logger.warning(
                "Unsupported optimizer '%s' in config, defaulting to AdamW.", opt_name
            )
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        logger.info(
            "Optimizer: %s (lr=%.1e, weight_decay=%.1e)",
            self.optimizer.__class__.__name__, lr, wd
        )

    def _build_scheduler(self) -> None:
        """Initializes the learning rate scheduler."""
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.cfg.get("lr_factor", 0.5),
            patience=self.cfg.get("lr_patience", 5),
            min_lr=self.cfg.get("min_lr", 1e-7),
        )

    def train(self) -> float:
        """
        Executes the main training loop.

        Includes training steps, validation, Optuna pruning, logging,
        early stopping, and checkpointing.

        Returns:
            The best validation loss achieved.
        """
        epochs = int(self.cfg.get("epochs", 100))
        patience = int(self.cfg.get("early_stopping_patience", 10))
        min_delta = float(self.cfg.get("min_delta", 1e-10))
        
        epochs_without_improvement = 0
        completed_epochs = 0
        current_validation_loss = float('inf')


        for epoch_num in range(1, epochs + 1):
            completed_epochs = epoch_num
            epoch_start_time = time.time()

            train_loss = self._run_epoch(
                self.train_loader, train_phase=True, epoch_num=epoch_num
            )
            current_validation_loss = self._run_epoch(
                self.val_loader, train_phase=False, epoch_num=epoch_num
            )
            self.scheduler.step(current_validation_loss)

            if self.optuna_trial:
                self.optuna_trial.report(current_validation_loss, epoch_num)
                if self.optuna_trial.should_prune():
                    logger.info("Optuna trial pruned at epoch %d.", epoch_num)
                    raise TrialPruned()

            if (
                self.device.type == "cuda"
                and self._clear_cuda_every > 0
                and epoch_num % self._clear_cuda_every == 0
                and torch.cuda.is_available()
            ):
                logger.debug("Clearing CUDA cache at epoch %d.", epoch_num)
                torch.cuda.empty_cache()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_duration = time.time() - epoch_start_time
            log_line = (
                f"{epoch_num},{train_loss:.6e},{current_validation_loss:.6e},"
                f"{current_lr:.6e},{epoch_duration:.1f}\n"
            )
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(log_line)
            except IOError as e:
                logger.error("Failed to write to training log %s: %s", self.log_path, e)

            logger.info(
                "Epoch %03d | Train Loss: %.4e | Val Loss: %.4e | LR: %.2e | Time: %.1fs",
                epoch_num, train_loss, current_validation_loss, current_lr, epoch_duration
            )


            if current_validation_loss < self.best_val_loss - min_delta:
                self.best_val_loss = current_validation_loss
                self.best_epoch = epoch_num
                epochs_without_improvement = 0
                self._checkpoint("best_model.pt", epoch_num, current_validation_loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping at epoch %d after %d epochs "
                        "without improvement on validation loss.",
                        epoch_num, patience
                    )
                    break
        else: # Executed if the loop completes without a 'break'
            logger.info("Training completed %d epochs.", epochs)
            # completed_epochs is already epochs here


        self._checkpoint("final_model.pt", completed_epochs, current_validation_loss, final=True)
        self.test()

        # --- JIT Save ---
        try:
            logger.info("Attempting to JIT save the model...")
            # Ensure we get the underlying model if torch.compile was used
            # self.model would have been updated by self.test() if a checkpoint was loaded
            model_to_jit = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            
            # Script the model. No example_inputs are typically needed for torch.jit.script(nn.Module).
            # JIT will analyze the forward method's code.
            scripted_model = torch.jit.script(model_to_jit)
            jit_save_path = self.save_dir / "model_jit.pt"
            scripted_model.save(str(jit_save_path)) # Use str(Path) for compatibility
            logger.info(f"Model JIT saved successfully to {jit_save_path}")

        except Exception as e:
            logger.error(f"Failed to JIT save the model: {e}", exc_info=True)
            
        return self.best_val_loss

    def _run_epoch(
        self, loader: DataLoader, *, train_phase: bool, epoch_num: int
    ) -> float:
        """
        Executes a single epoch of training or validation/testing.

        Args:
            loader: DataLoader for the current phase.
            train_phase: True if training, False for validation/testing.
            epoch_num: Current epoch number (for logging).

        Returns:
            Average loss for the epoch over active (non-padded) elements.
        """
        self.model.train(train_phase)
        accumulated_loss = 0.0
        total_active_elements = 0

        device_type = self.device.type
        # Autocast is typically most beneficial and stable on CUDA.
        # Other backends might have varying levels of support or none.
        effective_use_amp = self.use_amp and device_type == 'cuda'

        autocast_ctx = (
            torch.autocast(
                device_type=device_type, # Should be 'cuda' if effective_use_amp is True
                dtype=torch.float16,
                enabled=effective_use_amp
            )
            # Removed else contextlib.nullcontext() for directness if not effective_use_amp
            # The enabled=effective_use_amp flag handles this.
        )

        grad_context = torch.enable_grad() if train_phase else torch.no_grad()
        phase_str = "Train" if train_phase else "Val/Test"
        # Ensure len(loader) is not zero before division
        log_interval = max(1, len(loader) // 5) if len(loader) > 0 else 1


        with grad_context:
            for batch_idx, batch_data in enumerate(loader):
                # Data from PadCollate:
                # inputs_dict, input_masks_dict (True=VALID), targets_tensor, target_mask_tensor (True=VALID)
                inputs, input_masks, targets, target_mask = batch_data

                inputs = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in inputs.items()
                }
                input_masks = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in input_masks.items()
                    if isinstance(v, torch.Tensor) # Global inputs might not have masks
                }
                targets = targets.to(self.device, non_blocking=True)
                target_mask = target_mask.to(self.device, non_blocking=True) # True means valid

                with autocast_ctx:
                    # model.forward expects input_masks where True means VALID
                    predictions = self.model(inputs, input_masks=input_masks)
                    loss_unreduced = self.criterion(predictions, targets)

                    # Ensure target_mask is correctly broadcastable to loss_unreduced shape
                    if target_mask.ndim == predictions.ndim -1 and target_mask.shape == predictions.shape[:-1]:
                        # e.g. target_mask (B,L), predictions (B,L,F)
                        active_loss_mask = target_mask.unsqueeze(-1)
                    elif target_mask.shape == loss_unreduced.shape:
                        # Mask already matches loss shape (e.g. for (B,F) or (B,L,F) if features are masked)
                        active_loss_mask = target_mask
                    elif target_mask.ndim == 1 and predictions.ndim == 2 and target_mask.size(0) == predictions.size(0):
                        # e.g. target_mask (B,), predictions (B,F)
                        active_loss_mask = target_mask.unsqueeze(-1)
                    else:
                        logger.warning(
                            "Epoch %d (%s), Batch %d: Target mask shape %s is not directly "
                            "broadcastable to prediction shape %s for loss masking. "
                            "Attempting to expand or using all elements.",
                            epoch_num, phase_str, batch_idx, target_mask.shape, predictions.shape
                        )
                        # Attempt a common case: mask is (B, L) and loss is (B, L, F)
                        if target_mask.shape == loss_unreduced.shape[:-1]: # Check if dimensions match except last one
                             active_loss_mask = target_mask.unsqueeze(-1)
                        else: # Fallback: assume all elements contribute if unsure
                             active_loss_mask = torch.ones_like(loss_unreduced, dtype=torch.bool, device=self.device)


                    masked_loss_values = loss_unreduced[active_loss_mask]
                    num_active_in_batch = masked_loss_values.numel()

                    if num_active_in_batch > 0:
                        current_batch_loss = masked_loss_values.mean()
                    else:
                        # Avoid division by zero if no active elements (e.g. fully padded batch)
                        current_batch_loss = torch.tensor(
                            0.0, device=self.device, dtype=predictions.dtype, requires_grad=train_phase
                        )


                if not torch.isfinite(current_batch_loss):
                    logger.warning(
                        "Epoch %d (%s): Non-finite loss "
                        "(%s) at batch %d. Skipping batch.",
                        epoch_num, phase_str, current_batch_loss.item(), batch_idx
                    )
                    if num_active_in_batch == 0:
                         logger.warning("Loss was non-finite AND num_active_elements was 0 for the batch.")
                    continue

                if train_phase:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler: # This implies effective_use_amp is True and device is CUDA
                        self.scaler.scale(current_batch_loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else: # Not using AMP or not on CUDA
                        current_batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.optimizer.step()

                accumulated_loss += current_batch_loss.item() * num_active_in_batch
                total_active_elements += num_active_in_batch

                if batch_idx > 0 and batch_idx % log_interval == 0 :
                    logger.debug(
                        "Epoch %d [%s] Batch %d/%d AvgLossInBatch: %.4e",
                        epoch_num, phase_str, batch_idx, len(loader),
                        current_batch_loss.item() # current_batch_loss is scalar here
                    )

        if total_active_elements == 0:
            logger.warning(
                "Epoch %d (%s): No active elements found across all batches "
                "to compute loss. Returning 0.0 for epoch loss.",
                 epoch_num, phase_str
            )
            return 0.0
        return accumulated_loss / total_active_elements

    def _checkpoint(
        self, filename: str, epoch: int, val_loss: float, *, final: bool = False
    ) -> None:
        """
        Saves a model checkpoint.

        Args:
            filename: Filename for the checkpoint.
            epoch: Epoch number for this checkpoint.
            val_loss: Validation loss for this checkpoint.
            final: True if this is the final model checkpoint.
        """
        model_to_save = (
            self.model._orig_mod # type: ignore
            if hasattr(self.model, "_orig_mod") # For torch.compile
            else self.model
        )
        save_path = self.save_dir / filename
        
        checkpoint_config = self.cfg.copy()
        # Ensure runtime nhead (if adjusted by create_prediction_model) is saved
        if hasattr(model_to_save, 'nhead') and isinstance(model_to_save.nhead, int):
            checkpoint_config['nhead'] = model_to_save.nhead
        elif hasattr(self.model, 'nhead') and isinstance(self.model.nhead, int): # Fallback to self.model if model_to_save lacks it
            checkpoint_config['nhead'] = self.model.nhead


        checkpoint = {
            "state_dict": model_to_save.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": checkpoint_config, # Use the (potentially updated by create_model) config
        }
        try:
            torch.save(checkpoint, save_path)
            # Log only if successful and for specific checkpoints for brevity
            if filename == "best_model.pt":
                logger.info("New best model saved to %s (Epoch %d, Val Loss %.4e)",
                            save_path.name, epoch, val_loss)
            elif final:
                 logger.info("Final model saved to %s (Epoch %d, Val Loss %.4e)",
                            save_path.name, epoch, val_loss)

        except IOError as e:
            logger.error(
                "Failed to save checkpoint %s at epoch %d: %s",
                filename, epoch, e
            )
        except Exception as e: # Catch other potential torch.save errors
             logger.error(
                "Unexpected error saving checkpoint %s at epoch %d: %s",
                filename, epoch, e, exc_info=True
            )


    def test(self) -> None:
        """Evaluates the best or final model on the test set."""
        best_model_filename = "best_model.pt"
        best_model_path = self.save_dir / best_model_filename
        
        config_for_testing = self.cfg.copy() # Start with current config for the model instance
        model_was_loaded_from_checkpoint = False
        
        if best_model_path.exists():
            try:
                # map_location ensures model loads to the correct device trainer is using
                checkpoint = torch.load(best_model_path, map_location=self.device)
                if 'config' in checkpoint:
                    # Prioritize config from checkpoint for model architecture if different
                    # This ensures the loaded state_dict matches the model structure.
                    config_for_testing = checkpoint['config']
                    logger.info(
                        "Using model configuration from checkpoint '%s' for testing.",
                        best_model_filename
                    )
                else: # Should not happen if checkpointing is correct
                    logger.warning("Checkpoint '%s' is missing 'config' key. Using current trainer config.", best_model_filename)


                # Re-create model using the determined config for testing
                # This ensures correct architecture before loading state_dict.
                test_model = create_prediction_model(config_for_testing, device=self.device)
                
                # Apply torch.compile if it was configured for the loaded model config
                if (
                    config_for_testing.get("use_torch_compile", False) # Use config from checkpoint
                    and self.device.type == "cuda"
                    and hasattr(torch, "compile")
                ):
                    try:
                        test_model = torch.compile(test_model) # type: ignore
                        logger.info("Test model compiled with torch.compile().")
                    except Exception as e:
                        logger.warning("torch.compile failed for test model: %s.", e)

                # Load state dict into the potentially compiled model's underlying module
                model_to_load_state_dict_into = test_model._orig_mod if hasattr(test_model, '_orig_mod') else test_model
                model_to_load_state_dict_into.load_state_dict(checkpoint["state_dict"])
                
                self.model = test_model # CRITICAL: Update self.model to the loaded one for JIT saving later
                model_was_loaded_from_checkpoint = True
                logger.info(
                    "Loaded best model from %s (Epoch %s, Val Loss: %.4e) for testing.",
                    best_model_filename,
                    checkpoint.get('epoch', 'N/A'),
                    checkpoint.get('val_loss', float('nan'))
                )
            except FileNotFoundError: # Should be caught by exists(), but defensive
                logger.warning("Best model checkpoint %s not found during load attempt. This should not happen if exists() check passed.", best_model_path)
            except Exception as e:
                logger.error(
                    "Failed to load best model from %s: %s. "
                    "Testing with the model from the end of training if available.",
                    best_model_path, e, exc_info=True
                )
                # If loading best fails, self.model is already the final trained model
                # (or initial if training failed very early and self.model was never updated)
        else:
            logger.info( # Changed from warning to info as this is an expected path
                "Best model checkpoint (%s) not found. "
                "Testing with the model from the end of training.",
                best_model_path
            )
            # self.model is already the final trained model from the loop

        if self.model is None: # Should only happen if __init__ fails before _build_model
            logger.error("No model available for testing (self.model is None). Test phase skipped.")
            return


        test_loss = self._run_epoch(self.test_loader, train_phase=False, epoch_num=-1) # epoch_num=-1 for test
        logger.info("Test Loss (on evaluated model): %.4e", test_loss)
        try:
            save_json(
                {"test_loss": test_loss}, self.save_dir / "test_metrics.json"
            )
        except Exception as e: # Catch specific IOError or general Exception
            logger.error("Failed to save test metrics: %s", e, exc_info=True)

    def _save_test_set_info(self) -> None:
        """Saves filenames of the test set profiles."""
        info_path = self.save_dir / "test_set_info.json"
        try:
            # test_filenames should be strings
            save_json({"test_filenames": sorted(self.test_filenames)}, info_path)
            logger.info("Test set filenames saved to %s", info_path.name)
        except Exception as e:
            logger.error("Failed to save test set info to %s: %s", info_path, e, exc_info=True)


__all__ = ["ModelTrainer"]