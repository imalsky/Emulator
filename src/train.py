#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna 
import torch
from optuna.exceptions import TrialPruned
from torch import nn, optim
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from dataset import AtmosphericDataset 
from hardware import configure_dataloader_settings
from model import MultiEncoderTransformer, create_prediction_model 
from utils import save_json 

logger = logging.getLogger(__name__)

# Defines the number of worker processes to use for data loading.
# A value of 1 suggests that data loading might not be a significant bottleneck
# or that system constraints favor fewer worker processes.
DATALOADER_NUM_WORKERS = 1


def _split_dataset(
    dataset: AtmosphericDataset,
    val_frac: float,
    test_frac: float,
    *,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset, List[str]]:
    """
    Deterministically splits an AtmosphericDataset into training, validation, and test subsets.

    The split is performed based on fractions for validation and test sets.
    A random seed ensures reproducibility of the splits. It also retrieves
    and returns the filenames corresponding to the test set indices, which can be
    useful for later analysis or identification of test samples.

    Args:
        dataset: The full AtmosphericDataset instance to be split.
        val_frac: The fraction of the dataset to allocate for the validation set.
        test_frac: The fraction of the dataset to allocate for the test set.
        seed: A random seed to ensure reproducible dataset splits.

    Returns:
        A tuple containing:
            - train_subset: A Subset for training.
            - val_subset: A Subset for validation.
            - test_subset: A Subset for testing.
            - test_filenames_list: A list of filenames for the profiles in the test subset.
    """
    n = len(dataset)
    # Validate that split fractions are sensible (each > 0, sum < 1).
    # If not, the configuration is critically flawed, and execution cannot continue.
    if not (
        0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1
    ):
        logger.critical(
            "Dataset split fractions (val_frac: %f, test_frac: %f) are "
            "invalid. They must be between 0 and 1, and their sum must be "
            "less than 1. Exiting.", val_frac, test_frac
        )
        sys.exit(1)

    num_val = int(n * val_frac)
    num_test = int(n * test_frac)
    num_train = n - num_val - num_test

    # Ensure that the number of training samples is positive.
    # If not, the dataset is too small for the requested split, which is a critical issue.
    if num_train <= 0:
        logger.critical(
            "Dataset size %d is too small for the requested split "
            "fractions (val: %f, test: %f), resulting in %d training "
            "samples. Exiting.", n, val_frac, test_frac, num_train
        )
        sys.exit(1)

    # Generate a random permutation of indices for shuffling the dataset.
    # Using a generator with a manual seed ensures the shuffle is deterministic.
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    # Assign indices to test, validation, and training sets based on the calculated numbers.
    test_indices: List[int] = indices[:num_test]
    val_indices: List[int] = indices[num_test : num_test + num_val]
    train_indices: List[int] = indices[num_test + num_val:]

    logger.info(
        "Dataset split: %d train / %d val / %d test samples.",
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

    # Attempt to retrieve the filenames for the test set samples.
    # This requires the dataset object to support `get_profile_filenames_by_indices`.
    # Failure here (e.g., method not found, or indices out of bounds) is critical.
    try:
        test_filenames = dataset.get_profile_filenames_by_indices(test_indices)
        if test_filenames: 
            logger.info("First few test filenames: %s...",test_filenames[:min(3, len(test_filenames))])
    except AttributeError: 
        logger.critical(
            "AtmosphericDataset does not have method 'get_profile_filenames_by_indices'. "
            "This is required by the training script. Exiting."
            )
        sys.exit(1)
    except IndexError as e:
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
    Orchestrates the training, validation, and testing of the transformer model.

    This class handles the end-to-end training pipeline, including:
    - Splitting data into train, validation, and test sets.
    - Setting up DataLoaders for efficient batch processing.
    - Instantiating the model, optimizer, and learning rate scheduler.
    - Executing the main training loop, including per-epoch training and validation.
    - Implementing early stopping to prevent overfitting.
    - Saving model checkpoints (both standard and JIT-compiled versions).
    - Integrating with Optuna for hyperparameter optimization, including trial pruning.
    - Evaluating the best performing model on the test set.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        save_dir: Path,
        dataset: AtmosphericDataset,
        collate_fn: Callable, 
        *,
        optuna_trial: Optional[optuna.Trial] = None,
    ) -> None:
        """
        Initializes the ModelTrainer.

        Args:
            config: A dictionary containing all parameters for training, model
                    architecture, and data handling.
            device: The PyTorch device (e.g., 'cuda', 'cpu') on which to perform training.
            save_dir: The directory path where model artifacts, logs, and results
                      will be saved.
            dataset: The complete AtmosphericDataset instance containing all profiles.
            collate_fn: A custom collate function (e.g., PadCollate) to prepare
                        batches for the DataLoader, handling padding for variable
                        sequence lengths.
            optuna_trial: An optional Optuna Trial object. If provided, this enables
                          integration with an Optuna hyperparameter search, allowing
                          for trial reporting and pruning.
        """
        self.cfg = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.optuna_trial = optuna_trial
        self.padding_value = float(self.cfg.get("padding_value", 0.0))
        self.model: MultiEncoderTransformer 

        # Split the dataset into training, validation, and testing subsets.
        # The split parameters are sourced from the configuration.
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
        # Set up DataLoaders, the model, optimizer, and learning rate scheduler.
        self._build_dataloaders(collate_fn)
        self._build_model()
        self._build_optimiser()
        self._build_scheduler()

        # Configure Automatic Mixed Precision (AMP) if enabled in config and on CUDA.
        # AMP can speed up training and reduce memory usage.
        self.use_amp = bool(
            self.cfg.get("use_amp", False) and device.type == "cuda"
        )
        # GradScaler is used with AMP to prevent underflow of small gradients.
        self.scaler: Optional[GradScaler] = GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("AMP enabled on CUDA.")

        # Loss function is MSELoss with reduction='none' to allow for manual masking
        # of padded elements before reducing the loss.
        self.criterion = nn.MSELoss(reduction='none') 
        # Max gradient norm for gradient clipping, preventing exploding gradients.
        self.max_grad_norm = max(
            float(self.cfg.get("gradient_clip_val", 1.0)), 1e-12
        )

        # Prepare the training log file.
        self.log_path = self.save_dir / "training_log.csv"
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("epoch,train_loss,val_loss,lr,time_s\n")
        except IOError as e:
            # Failure to write the log header is critical as subsequent logging will fail.
            logger.critical("Failed to write training log header to %s: %s. Exiting.", self.log_path, e)
            sys.exit(1)

        # Initialize tracking variables for best validation loss and early stopping.
        self.best_val_loss, self.best_epoch = float("inf"), -1
        # Configuration for periodic CUDA cache clearing.
        self._clear_cuda_every = int(self.cfg.get("clear_cuda_cache_every", 0))
        # Save information about the test set composition.
        self._save_test_set_info()

    def _build_dataloaders(self, collate_fn: Callable) -> None:
        """Initializes DataLoaders for train, validation, and test sets."""
        # Retrieve device-specific DataLoader settings (e.g., pin_memory).
        hw_settings = configure_dataloader_settings()
        batch_size = int(self.cfg.get("batch_size", 16))
        
        num_w = DATALOADER_NUM_WORKERS

        # Common arguments for all DataLoaders.
        # `persistent_workers` is used if `num_workers` > 0 to keep worker processes
        # alive across epochs, reducing overhead.
        common_dl_args: Dict[str, Any] = dict(
            batch_size=batch_size,
            num_workers=num_w,
            pin_memory=hw_settings["pin_memory"],
            persistent_workers=num_w > 0 and hw_settings["persistent_workers"],
            collate_fn=collate_fn,
        )
        # Training DataLoader shuffles data and can drop the last batch if incomplete.
        self.train_loader = DataLoader(
            self.train_ds, shuffle=True, drop_last=False, **common_dl_args
        )
        # Validation and Test DataLoaders do not shuffle data.
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
        # The model is instantiated using a factory function that takes the configuration
        # and the target device.
        self.model = create_prediction_model(
            self.cfg, device=self.device
        )
        # Optionally compile the model using `torch.compile` for potential speedups,
        # if enabled in the config, running on CUDA, and `torch.compile` is available.
        if (
            self.cfg.get("use_torch_compile", False)
            and self.device.type == "cuda"
            and hasattr(torch, "compile")
        ):
            try:
                self.model = torch.compile(self.model) 
                logger.info("Model compiled with torch.compile().")
            except Exception as e:
                logger.warning(
                    "torch.compile failed: %s. Using uncompiled model.", e
                )
        # Log the number of trainable parameters in the model.
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Model built with %s trainable parameters.", f"{num_params:,}")


    def _build_optimiser(self) -> None:
        """Initializes the optimizer based on configuration."""
        # Optimizer type, learning rate, and weight decay are read from the config.
        opt_name = str(self.cfg.get("optimizer", "adamw")).lower()
        lr = float(self.cfg.get("learning_rate", 1e-4))
        wd = float(self.cfg.get("weight_decay", 1e-5))

        # Instantiate the specified optimizer. Defaults to AdamW if an unsupported
        # optimizer name is provided.
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
        # Using ReduceLROnPlateau scheduler, which reduces the learning rate
        # when a metric (validation loss) has stopped improving.
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min", # Reduces LR when the monitored quantity stops decreasing.
            factor=self.cfg.get("lr_factor", 0.5), # Factor by which LR is reduced.
            patience=self.cfg.get("lr_patience", 5), # Epochs to wait for improvement.
            min_lr=self.cfg.get("min_lr", 1e-7), # Lower bound on the learning rate.
        )

    def train(self) -> float:
        """
        Executes the main training loop.

        This loop iterates over epochs, performs training and validation steps,
        handles learning rate scheduling, Optuna trial reporting and pruning,
        early stopping, and model checkpointing. After the loop, it evaluates
        the best model on the test set.

        Returns:
            The best validation loss achieved during training.
        """
        epochs = int(self.cfg.get("epochs", 100))
        patience = int(self.cfg.get("early_stopping_patience", 10)) # For early stopping.
        min_delta = float(self.cfg.get("min_delta", 1e-10)) # Minimum change for improvement.
        
        epochs_without_improvement = 0
        completed_epochs = 0
        # Stores the validation loss of the current epoch. Initialized to infinity.
        current_validation_loss = float('inf')


        # Main training loop.
        for epoch_num in range(1, epochs + 1):
            completed_epochs = epoch_num
            epoch_start_time = time.time()

            # Run one epoch of training.
            train_loss = self._run_epoch(
                self.train_loader, train_phase=True, epoch_num=epoch_num
            )
            # Run one epoch of validation.
            current_validation_loss = self._run_epoch(
                self.val_loader, train_phase=False, epoch_num=epoch_num
            )
            # Adjust learning rate based on validation loss.
            self.scheduler.step(current_validation_loss)

            # If part of an Optuna hyperparameter search, report the validation loss
            # and check if the trial should be pruned.
            if self.optuna_trial:
                self.optuna_trial.report(current_validation_loss, epoch_num)
                if self.optuna_trial.should_prune():
                    logger.info("Optuna trial pruned at epoch %d.", epoch_num)
                    raise TrialPruned()

            # Periodically clear CUDA cache if configured, to free up unused memory.
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
            # Log epoch metrics to CSV file.
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

            # Check for improvement in validation loss to update the best model
            # and manage early stopping.
            if current_validation_loss < self.best_val_loss - min_delta:
                self.best_val_loss = current_validation_loss
                self.best_epoch = epoch_num
                epochs_without_improvement = 0
                # Save a checkpoint of the best model found so far.
                self._checkpoint("best_model.pt", epoch_num, current_validation_loss)
            else:
                epochs_without_improvement += 1
                # If validation loss hasn't improved for `patience` epochs, stop training.
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping at epoch %d after %d epochs "
                        "without improvement on validation loss.",
                        epoch_num, patience
                    )
                    break
        else: 
            # This block executes if the loop completed naturally (no early stopping).
            logger.info("Training completed %d epochs.", epochs)
            
        # Save the final model state at the end of training.
        self._checkpoint("final_model.pt", completed_epochs, current_validation_loss, final=True)
        # Evaluate the model on the test set. This typically loads the best checkpoint.
        self.test() 
            
        return self.best_val_loss

    def _run_epoch(
        self, loader: DataLoader, *, train_phase: bool, epoch_num: int
    ) -> float:
        """
        Executes a single epoch of training or validation/testing.

        This method iterates over the provided DataLoader. For training, it computes
        gradients and updates model weights. For validation/testing, it only
        computes losses. It handles Automatic Mixed Precision (AMP) and
        gradient clipping if configured. Loss is calculated by applying a mask
        (where True indicates valid data) to the raw, unreduced MSE loss.

        Args:
            loader: DataLoader for the current phase (train, val, or test).
            train_phase: Boolean flag, True if in training phase, False otherwise.
            epoch_num: The current epoch number, used for logging.

        Returns:
            The average loss for the epoch over all valid (unmasked) elements.
        """
        # Set model to training or evaluation mode. This affects layers like Dropout.
        self.model.train(train_phase)
        accumulated_loss = 0.0
        # total_active_elements tracks the sum of unmasked elements across all batches,
        # used for calculating the true mean loss per element.
        total_active_elements = 0 

        device_type = self.device.type
        # Determine if AMP should be effectively used in this epoch.
        effective_use_amp = self.use_amp and device_type == 'cuda'

        # Context manager for AMP. Operations within this context may run in float16.
        autocast_ctx = torch.autocast(
            device_type=device_type,
            dtype=torch.float16, # Common dtype for AMP.
            enabled=effective_use_amp
        )
        # Context manager to enable or disable gradient computation.
        grad_context = torch.enable_grad() if train_phase else torch.no_grad()
        phase_str = "Train" if train_phase else "Val/Test"
        # Determine logging interval based on loader size.
        log_interval = max(1, len(loader) // 5) if len(loader) > 0 else 1

        with grad_context:
            for batch_idx, batch_data in enumerate(loader):
                # Unpack batch data: inputs, input masks, targets, and target mask.
                # Masks are boolean tensors where True indicates a valid data point
                # and False indicates a padded/invalid point.
                inputs, input_masks, targets, target_mask = batch_data

                # Move input tensors to the designated compute device.
                inputs = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in inputs.items()
                }
                # Move input masks to device. These masks have True for valid data.
                input_masks = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in input_masks.items()
                    if isinstance(v, torch.Tensor) 
                }
                targets = targets.to(self.device, non_blocking=True)
                # Target mask (True means VALID) is moved to device.
                target_mask = target_mask.to(self.device, non_blocking=True) 

                with autocast_ctx:
                    # The model's forward pass expects input_masks where True signifies VALID data.
                    # Internally, the model may convert these to PyTorch's convention (True=PAD)
                    # for specific layers like TransformerEncoderLayer.
                    predictions = self.model(inputs, input_masks=input_masks)
                    # Calculate raw MSE loss without reduction (element-wise).
                    # Shape will be (Batch, SequenceLength, NumTargetFeatures).
                    loss_unreduced = self.criterion(predictions, targets) 

                    # Determine the appropriate mask to apply to the unreduced loss.
                    # The goal is to select only the loss values corresponding to
                    # valid (unpadded) target elements.
                    if target_mask.ndim == loss_unreduced.ndim - 1 and \
                       target_mask.shape == loss_unreduced.shape[:-1]:
                        # Common case: target_mask is (B,L), loss_unreduced is (B,L,F).
                        # This means one mask value per sequence position for all features.
                        active_loss_mask = target_mask 
                    elif target_mask.shape == loss_unreduced.shape:
                        # Case: target_mask is (B,L,F), one mask value per target element.
                        active_loss_mask = target_mask
                    else:
                        # Fallback for unexpected mask shapes. Log a warning.
                        # This might occur if data preparation or collate function has an issue.
                        # The fallback uses a mask of all ones, which means all loss elements
                        # are considered valid if a more specific mask cannot be applied.
                        # This is generally undesirable but prevents a crash.
                        logger.warning(
                            "Epoch %d (%s), Batch %d: Unexpected target_mask shape %s "
                            "for loss_unreduced shape %s. Attempting to use target_mask "
                            "by unsqueezing or using all elements as fallback.",
                            epoch_num, phase_str, batch_idx, target_mask.shape, loss_unreduced.shape
                        )
                        if target_mask.ndim == loss_unreduced.ndim -1 and target_mask.shape == loss_unreduced.shape[:-1]:
                             active_loss_mask = target_mask 
                        else: 
                             active_loss_mask = torch.ones_like(loss_unreduced, dtype=torch.bool, device=self.device)
                    
                    # Apply the determined mask to select loss values from valid elements.
                    # If active_loss_mask is (B,L), masked_loss_values becomes (N, F_target),
                    # where N is the number of True elements in the original (B,L) mask.
                    # If active_loss_mask is (B,L,F), masked_loss_values becomes (K),
                    # where K is the number of True elements in the (B,L,F) mask.
                    masked_loss_values = loss_unreduced[active_loss_mask]
                    
                    # Number of elements that contributed to the loss in this batch.
                    num_active_in_batch = masked_loss_values.numel()

                    # Calculate the mean loss for the current batch over active elements.
                    if num_active_in_batch > 0:
                        current_batch_loss = masked_loss_values.mean() 
                    else:
                        # If no active elements (e.g., a batch of entirely padded sequences),
                        # the loss is zero. It needs to be a tensor for backward pass compatibility.
                        current_batch_loss = torch.tensor(
                            0.0, device=self.device, dtype=predictions.dtype, requires_grad=train_phase
                        )

                # Check for non-finite (NaN or Inf) loss, which can halt training.
                if not torch.isfinite(current_batch_loss):
                    logger.warning(
                        "Epoch %d (%s): Non-finite loss (%s) at batch %d. Skipping update.",
                        epoch_num, phase_str, current_batch_loss.item(), batch_idx,
                    )
                    if num_active_in_batch == 0:
                         logger.warning("Loss was non-finite AND num_active_elements was 0 for the batch.")
                    continue # Skip optimization step for this batch.

                # If in training phase, perform backpropagation and optimizer step.
                if train_phase:
                    self.optimizer.zero_grad(set_to_none=True) # More memory efficient.
                    if self.scaler: 
                        # AMP: Scale loss, backward pass, unscale gradients, clip, optimizer step.
                        self.scaler.scale(current_batch_loss).backward()
                        self.scaler.unscale_(self.optimizer) 
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else: 
                        # No AMP: Standard backward pass and optimizer step.
                        current_batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.optimizer.step()

                # Accumulate the sum of squared errors (from `masked_loss_values`) and
                # the number of elements that contributed to this sum.
                # This is to correctly calculate the epoch-level mean loss later.
                accumulated_loss += masked_loss_values.sum().item() 
                total_active_elements += num_active_in_batch       

                # Log progress within the epoch at specified intervals.
                if batch_idx > 0 and batch_idx % log_interval == 0 :
                    # `current_batch_loss` is already the mean for this specific batch.
                    avg_batch_loss = current_batch_loss.item() 
                    logger.debug(
                        "Epoch %d [%s] Batch %d/%d AvgLossInBatch: %.4e",
                        epoch_num, phase_str, batch_idx, len(loader),
                        avg_batch_loss
                    )
        
        # Handle cases where no active elements were found in any batch of the epoch.
        if total_active_elements == 0: 
            logger.warning(
                "Epoch %d (%s): No active elements found across all batches. Returning 0.0 for epoch loss.",
                 epoch_num, phase_str
            )
            return 0.0
        # Calculate the overall mean loss for the epoch.
        return accumulated_loss / total_active_elements

    def _checkpoint(
        self, filename: str, epoch: int, val_loss: float, *, final: bool = False
    ) -> None:
        """
        Saves a model checkpoint, including its state dictionary, epoch,
        validation loss, and the configuration used. Also saves a JIT-scripted
        version of the model for deployment or inference.

        Args:
            filename: The base name for the checkpoint file (e.g., "best_model.pt").
            epoch: The epoch number at which the checkpoint is being saved.
            val_loss: The validation loss at this epoch.
            final: A boolean flag indicating if this is the final model checkpoint
                   at the end of training.
        """
        # If `torch.compile` was used, `model._orig_mod` points to the original model.
        # Otherwise, `model_to_save` is just `self.model`.
        model_to_save = (
            self.model._orig_mod 
            if hasattr(self.model, "_orig_mod") 
            else self.model
        )
        save_path = self.save_dir / filename
        
        # Create a copy of the configuration to store in the checkpoint.
        # This ensures that the model can be reconstructed with the same hyperparameters.
        checkpoint_config = self.cfg.copy()
        # If the model has an 'nhead' attribute (common for transformers),
        # ensure this specific value (which might have been adjusted by the model
        # factory if d_model % nhead != 0) is saved in the checkpoint config.
        if hasattr(model_to_save, 'nhead') and isinstance(model_to_save.nhead, int):
            checkpoint_config['nhead'] = model_to_save.nhead
        elif hasattr(self.model, 'nhead') and isinstance(self.model.nhead, int):
            # Fallback if model_to_save doesn't have it but self.model (compiled) does.
            checkpoint_config['nhead'] = self.model.nhead

        checkpoint = {
            "state_dict": model_to_save.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": checkpoint_config,
        }
        try:
            # Save the standard PyTorch checkpoint.
            torch.save(checkpoint, save_path)
            log_msg_prefix = ""
            if filename == "best_model.pt":
                log_msg_prefix = "New best model saved"
            elif final:
                log_msg_prefix = "Final model saved"
            
            #if log_msg_prefix: 
            #     logger.info("%s: %s (Epoch %d, Val Loss %.4e)", log_msg_prefix, save_path.name, epoch, val_loss)

            # Save a JIT (Just-In-Time) compiled version of the model.
            # This is often used for optimized inference or deployment.
            jit_filename_stem = Path(filename).stem
            jit_save_path = self.save_dir / f"{jit_filename_stem}_jit.pt"
            try:
                # `model_to_save` is already the underlying nn.Module.
                scripted_model = torch.jit.script(model_to_save)
                # `save` method for scripted models requires path as string.
                scripted_model.save(str(jit_save_path)) 
                if log_msg_prefix: 
                    pass
                    #logger.info(f"JIT version of {filename} saved successfully to {jit_save_path.name}")
            except Exception as e:
                logger.error(f"Failed to JIT save model corresponding to {filename}: {e}", exc_info=True)

        except IOError as e:
            logger.error("Failed to save checkpoint %s at epoch %d: %s", filename, epoch, e)
        except Exception as e:
             logger.error("Unexpected error saving checkpoint %s at epoch %d: %s",
                          filename, epoch, e, exc_info=True)


    def test(self) -> None:
        """
        Evaluates the model on the test set.

        It attempts to load the best model checkpoint ("best_model.pt"). If found,
        that model and its associated configuration are used for testing. Otherwise,
        it uses the model state from the end of the training process. The test loss
        is computed and saved.
        """
        best_model_filename = "best_model.pt"
        best_model_path = self.save_dir / best_model_filename
        
        # Start with the current training configuration; may be overridden by checkpoint config.
        config_for_testing = self.cfg.copy()
        
        # Attempt to load the best model checkpoint.
        if best_model_path.exists():
            try:
                checkpoint = torch.load(best_model_path, map_location=self.device)
                # If the checkpoint contains a configuration, use it for testing.
                # This is crucial if hyperparameters were tuned and the best model
                # used a different config than the initial one.
                if 'config' in checkpoint:
                    config_for_testing = checkpoint['config'] 
                    logger.info("Using model configuration from checkpoint '%s' for testing.", best_model_filename)
                
                # Re-create the model instance using the (potentially different) config from the checkpoint.
                # This ensures architecture consistency.
                test_model_instance = create_prediction_model(config_for_testing, device=self.device)
                
                # If torch.compile was used during training and is configured for testing,
                # compile the test model instance.
                if (config_for_testing.get("use_torch_compile", False) and
                    self.device.type == "cuda" and hasattr(torch, "compile")):
                    try:
                        test_model_instance = torch.compile(test_model_instance) 
                        logger.info("Test model instance compiled with torch.compile().")
                    except Exception as e:
                        logger.warning("torch.compile failed for test model instance: %s.", e)

                # Load the state dictionary into the re-created model.
                # Handle cases where the model might have been compiled (access `_orig_mod`).
                model_to_load_state = test_model_instance._orig_mod if hasattr(test_model_instance, '_orig_mod') else test_model_instance
                model_to_load_state.load_state_dict(checkpoint["state_dict"])
                
                # Replace the current model instance with the loaded test model.
                self.model = test_model_instance 
                logger.info(
                    "Loaded best model from %s (Epoch %s, Val Loss: %.4e) for testing.",
                    best_model_filename, checkpoint.get('epoch', 'N/A'), checkpoint.get('val_loss', float('nan'))
                )
            except FileNotFoundError:
                 logger.warning("Best model checkpoint %s not found. Testing with current model.", best_model_path)
            except Exception as e:
                logger.error(
                    "Failed to load best model from %s: %s. Testing with current model.",
                    best_model_path, e, exc_info=True
                )
        else:
            # If no best model checkpoint is found, proceed with the model as it is
            # at the end of the training (i.e., the final model state).
            logger.info(
                "Best model checkpoint (%s) not found. Testing with model from end of training.",
                best_model_path
            )

        # Ensure a model is available for testing.
        if self.model is None: 
            logger.error("No model available for testing. Test phase skipped.")
            return

        # Run evaluation on the test set using the selected model.
        # `epoch_num=-1` is a convention to indicate the test phase in logs/metrics.
        test_loss = self._run_epoch(self.test_loader, train_phase=False, epoch_num=-1) 
        logger.info("Test Loss (on evaluated model): %.4e", test_loss)
        # Save the test loss metric to a JSON file.
        try:
            save_json({"test_loss": test_loss}, self.save_dir / "test_metrics.json")
        except Exception as e:
            logger.error("Failed to save test metrics: %s", e, exc_info=True)

    def _save_test_set_info(self) -> None:
        """Saves filenames of the profiles that constitute the test set."""
        # This helps in identifying which specific data samples were used for testing.
        info_path = self.save_dir / "test_set_info.json"
        try:
            save_json({"test_filenames": sorted(self.test_filenames)}, info_path)
            logger.info("Test set filenames saved to %s", info_path.name)
        except Exception as e:
            logger.error("Failed to save test set info to %s: %s", info_path, e, exc_info=True)

__all__ = ["ModelTrainer"]