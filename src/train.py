#!/usr/bin/env python3
"""
train.py - Optimized model training with correct padding mask handling.

Features:
- Async device prefetching for better GPU utilization
- Mixed precision training with AMP
- Gradient accumulation for large effective batch sizes
- Early stopping with patience
- Learning rate scheduling
- Hardware-specific optimizations
- Proper loss masking for padded sequences

PADDING CONVENTION:
- Mask values: True = padding position, False = valid position
- Loss computation only includes valid positions
- No output overwriting - follows industry standards
"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from torch.utils.data import DataLoader

from dataset import create_dataset
from hardware import should_pin_memory
from model import create_prediction_model, export_model
from utils import save_json, seed_everything

logger = logging.getLogger(__name__)

# Training defaults
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4
DEFAULT_OPTIMIZER = "adamw"
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_EARLY_STOPPING_PATIENCE = 20
DEFAULT_MIN_DELTA = 1e-6
DEFAULT_GRADIENT_ACCUMULATION = 1
DEFAULT_NUM_WORKERS = 8
DEFAULT_MAX_BATCH_FAILURE_RATE = 0.10


class DevicePrefetchLoader:
    """
    Wraps a DataLoader to prefetch batches to device asynchronously.
    
    Overlaps data transfer with computation for better GPU utilization.
    """
    
    def __init__(self, loader: DataLoader, device: torch.device):
        """
        Initialize prefetch loader.
        
        Args:
            loader: Base DataLoader
            device: Target device for data
        """
        self.loader = loader
        self.device = device
        self.is_cuda = self.device.type == 'cuda'
    
    def __iter__(self):
        """Iterate with prefetching for CUDA devices."""
        if self.is_cuda:
            stream = torch.cuda.Stream()
            first = True
            current_batch = None
            
            for next_batch in self.loader:
                # Asynchronously transfer next batch
                with torch.cuda.stream(stream):
                    next_batch = self._to_device(next_batch)
                
                # Yield previous batch while next is transferring
                if not first:
                    yield current_batch
                else:
                    first = False
                
                # Wait for transfer to complete
                torch.cuda.current_stream().wait_stream(stream)
                current_batch = next_batch
            
            # Yield final batch
            if current_batch is not None:
                yield current_batch
        else:
            # No prefetching for non-CUDA devices
            for batch in self.loader:
                yield self._to_device(batch)
    
    def _to_device(self, batch):
        """
        Move batch to target device.
        
        Note: Masks remain as boolean tensors (True = padding).
        """
        inputs, masks, targets, tgt_masks = batch
        non_blocking = self.is_cuda
        
        # Move inputs
        device_inputs = {}
        device_inputs["sequence"] = inputs["sequence"].to(
            self.device, non_blocking=non_blocking
        )
        if "global_features" in inputs:
            device_inputs["global_features"] = inputs["global_features"].to(
                self.device, non_blocking=non_blocking
            )
        
        # Move masks (keep as boolean)
        device_masks = {}
        device_masks["sequence"] = masks["sequence"].to(
            self.device, non_blocking=non_blocking
        )
        
        # Move targets and target masks
        device_targets = targets.to(self.device, non_blocking=non_blocking)
        device_tgt_masks = tgt_masks.to(self.device, non_blocking=non_blocking)
        
        return device_inputs, device_masks, device_targets, device_tgt_masks
    
    def __len__(self):
        """Return number of batches."""
        return len(self.loader)


class ModelTrainer:
    """
    Orchestrates model training, validation, and testing.
    
    Handles all aspects of the training pipeline including:
    - Dataset creation and DataLoader setup
    - Model initialization and optimization
    - Training loop with validation
    - Checkpointing and early stopping
    - Model export for deployment
    - Correct padding mask handling throughout
    """

    def __init__(
            self,
            config: Dict[str, Any],
            device: torch.device,
            save_dir: Path,
            processed_dir: Path,
            splits: Dict[str, List[Tuple[str, int]]],
            collate_fn: Callable,
            optuna_trial: Optional[optuna.Trial] = None,
            profiler: Optional[Any] = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
            device: Compute device
            save_dir: Directory for saving checkpoints
            processed_dir: Directory with preprocessed data
            splits: Train/val/test splits
            collate_fn: Collation function for DataLoader
            optuna_trial: Optional Optuna trial for hyperparameter search
            profiler: Optional profiler
        """
        self.cfg = config
        self.device = device
        self.save_dir = save_dir
        self.model = None
        self.current_epoch = 0
        self.trial = optuna_trial
        self.profiler = profiler

        # Extract configuration
        misc_cfg = self.cfg.get("miscellaneous_settings", {})
        train_params = self.cfg.get("training_hyperparameters", {})

        self.max_batch_failure_rate = train_params.get(
            "max_batch_failure_rate", DEFAULT_MAX_BATCH_FAILURE_RATE
        )

        if self.device.type == "cuda":
            logger.info("Applying GPU performance optimizations")

            if not train_params.get("use_amp", False):
                logger.info("Enabling AMP")
                train_params["use_amp"] = True

            current_batch_size = train_params.get("batch_size", DEFAULT_BATCH_SIZE)
            if current_batch_size < 256:
                logger.info(f"Increasing batch size from {current_batch_size} to 256")
                train_params["batch_size"] = 256

        # Enable anomaly detection if requested
        if misc_cfg.get("detect_anomaly", False):
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Anomaly detection enabled - training will be slower.")

        # Import compress_splits from utils
        from utils import compress_splits

        # Save splits for reproducibility in COMPACT format
        compressed_splits = compress_splits(splits)
        save_json(compressed_splits, self.save_dir / "dataset_splits.json", compact=True)

        # Initialize components
        self._setup_datasets(processed_dir, splits)
        self._build_dataloaders(collate_fn, misc_cfg, train_params)
        self._build_model()
        self._build_optimizer(train_params)
        self._build_scheduler(train_params)
        self._setup_training_params(train_params)
        self._setup_logging()
        self._save_metadata()

        # Check if validation set exists
        self.has_val = self.val_loader is not None and len(self.val_loader) > 0

        # Clean up memory
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("Trainer initialized. Padding convention: True = padding position")
    
    def _setup_datasets(
        self, processed_dir: Path, splits: Dict[str, List[Tuple[str, int]]]
    ) -> None:
        """Create datasets from preprocessed data."""
        # Check if using subset
        fraction = self.cfg.get("training_hyperparameters", {}).get(
            "dataset_fraction_to_use", 1.0
        )
        
        if 0.0 < fraction < 1.0:
            logger.warning(f"Using only {fraction:.0%} of the dataset.")
            random_seed = self.cfg.get("miscellaneous_settings", {}).get(
                "random_seed", 42
            )
            seed_everything(random_seed)
        
        # Dataset directories
        train_dir = processed_dir / "train"
        val_dir = processed_dir / "val"
        test_dir = processed_dir / "test"
        
        if not train_dir.exists():
            raise RuntimeError("No training directory found.")
        
        def get_indices(split_data: List[Tuple[str, int]]) -> List[int]:
            """Get indices, optionally sampling a fraction."""
            num_samples = len(split_data)
            if num_samples == 0:
                return []
            
            if 0.0 < fraction < 1.0:
                import random
                k = max(1, int(num_samples * fraction))
                return random.sample(range(num_samples), k)
            
            return list(range(num_samples))
        
        # Create datasets
        train_idx = get_indices(splits["train"])
        val_idx = get_indices(splits["validation"])
        test_idx = get_indices(splits["test"])
        
        self.train_ds = create_dataset(train_dir, self.cfg, train_idx)
        self.val_ds = (
            create_dataset(val_dir, self.cfg, val_idx) if val_dir.exists() else None
        )
        self.test_ds = (
            create_dataset(test_dir, self.cfg, test_idx) if test_dir.exists() else None
        )
        
        logger.info(
            f"Datasets ready - train:{len(self.train_ds):,}  "
            f"val:{len(self.val_ds or []):,}  test:{len(self.test_ds or []):,}"
        )
    
    def _build_dataloaders(
        self, collate_fn: Callable, misc_cfg: Dict, train_cfg: Dict
    ) -> None:
        """Create DataLoaders with optimized settings."""
        pin_memory = should_pin_memory()
        num_workers = misc_cfg.get("num_workers", DEFAULT_NUM_WORKERS)
        
        # Ensure enough workers for GPU
        if num_workers < 8 and self.device.type == "cuda":
            logger.info(f"Increasing num_workers from {num_workers} to 8 for better GPU utilization")
            num_workers = 8
        
        # Common DataLoader arguments
        dl_common = dict(
            batch_size=train_cfg.get("batch_size", DEFAULT_BATCH_SIZE),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
        
        # Add persistent workers if using multiple workers
        if num_workers > 0:
            dl_common["persistent_workers"] = True
            dl_common["prefetch_factor"] = 4
        
        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_ds, shuffle=True, drop_last=False, **dl_common
        )
        
        self.val_loader = (
            DataLoader(self.val_ds, shuffle=False, drop_last=False, **dl_common)
            if self.val_ds
            else None
        )
        
        self.test_loader = (
            DataLoader(self.test_ds, shuffle=False, drop_last=False, **dl_common)
            if self.test_ds
            else None
        )
        
        # Wrap with device prefetching for async transfers
        if self.device.type != "cpu":
            logger.info(f"Using DevicePrefetchLoader for {self.device.type.upper()} transfers")
            
            self.train_loader = DevicePrefetchLoader(self.train_loader, self.device)
            if self.val_loader:
                self.val_loader = DevicePrefetchLoader(self.val_loader, self.device)
            if self.test_loader:
                self.test_loader = DevicePrefetchLoader(self.test_loader, self.device)
    
    def _build_model(self) -> None:
        """Create and initialize the model."""
        self.model = create_prediction_model(
            self.cfg, device=self.device, compile_model=True
        )
    
    def _build_optimizer(self, tp: Dict) -> None:
        """Create optimizer with weight decay groups."""
        opt_name = tp.get("optimizer", DEFAULT_OPTIMIZER).lower()
        lr = tp.get("learning_rate", DEFAULT_LR)
        wd = tp.get("weight_decay", 1e-5)
        
        # Separate parameters for weight decay
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            
            # Don't apply weight decay to biases and normalization
            if p.dim() == 1 or "bias" in n or "norm" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        
        # Parameter groups with different weight decay
        groups = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        
        # Create optimizer
        if opt_name == "adam":
            self.optimizer = optim.Adam(groups, lr=lr)
        elif opt_name == "sgd":
            self.optimizer = optim.SGD(groups, lr=lr, momentum=0.9)
        else:  # Default to AdamW
            self.optimizer = optim.AdamW(groups, lr=lr)
        
        logger.info(f"Optimizer: {opt_name}  lr={lr:.2e}  wd={wd:.2e}")
    
    def _build_scheduler(self, tp: Dict) -> None:
        """Create learning rate scheduler."""
        scheduler_type = tp.get("scheduler_type", "reduce_on_plateau").lower()
        
        if scheduler_type == "cosine":
            epochs = tp.get("epochs", DEFAULT_EPOCHS)
            warmup_epochs = tp.get("warmup_epochs", 0)
            
            if warmup_epochs >= epochs:
                raise ValueError("warmup_epochs must be less than total epochs.")
            
            # Main cosine scheduler
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=tp.get("min_lr", 1e-8),
            )
            
            if warmup_epochs > 0:
                # Add warmup scheduler
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=1e-5,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_epochs],
                )
                logger.info(f"Cosine scheduler with {warmup_epochs} warmup epochs.")
            else:
                self.scheduler = main_scheduler
                logger.info("Cosine scheduler without warmup.")
        
        elif scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=tp.get("lr_patience", 10),
                factor=tp.get("lr_factor", 0.5),
                min_lr=tp.get("min_lr", 1e-8),
            )
            logger.info("Using ReduceLROnPlateau scheduler.")
        
        else:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")
    
    def _setup_training_params(self, tp: Dict) -> None:
        """Setup training parameters and loss function."""
        # Loss function (MSE with no reduction for masking)
        self.criterion = nn.MSELoss(reduction="none")
        
        # Gradient clipping
        self.max_grad_norm = tp.get("gradient_clip_val", DEFAULT_GRAD_CLIP)
        
        # Gradient accumulation
        self.accumulation_steps = tp.get(
            "gradient_accumulation_steps", DEFAULT_GRADIENT_ACCUMULATION
        )
        
        # Mixed precision training
        self.use_amp = tp.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        
        if self.use_amp:
            logger.info("AMP (Automatic Mixed Precision) enabled.")
    
    def _setup_logging(self) -> None:
        """Setup training log file."""
        self.log_path = self.save_dir / "training_log.csv"
        self.log_path.write_text("epoch,train_loss,val_loss,lr,time_s,improvement\n")
        
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def _save_metadata(self) -> None:
        """Save training metadata."""
        metadata = {
            "device": str(self.device),
            "use_amp": self.use_amp,
            "effective_batch_size": (
                    self.cfg["training_hyperparameters"].get("batch_size", DEFAULT_BATCH_SIZE)
                    * self.accumulation_steps
            ),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "padding_convention": "True = padding position (PyTorch standard)",
        }
        # Use compact format for metadata
        save_json(metadata, self.save_dir / "training_metadata.json", compact=True)
    
    def train(self) -> float:
        """
        Run the training loop.
        
        Returns:
            Best validation loss achieved
        """
        tp = self.cfg.get("training_hyperparameters", {})
        epochs = tp.get("epochs", DEFAULT_EPOCHS)
        patience = tp.get("early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE)
        min_delta = tp.get("min_delta", DEFAULT_MIN_DELTA)
        
        epochs_without_improvement = 0
        
        logger.info(f"Starting training for {epochs} epochs.")
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training epoch
            train_loss = self._run_epoch(self.train_loader, is_train=True)
            if train_loss is None:
                raise RuntimeError("Too many invalid batches in training.")
            
            # Validation epoch
            val_loss = (
                self._run_epoch(self.val_loader, is_train=False)
                if self.has_val
                else float("inf")
            )
            if val_loss is None:
                val_loss = float("inf")
            
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if self.has_val and val_loss is not None and val_loss != float("inf"):
                    metric = val_loss
                else:
                    metric = train_loss if train_loss is not None else float("inf")
                
                if metric != float("inf"):
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()
            
            # Report to Optuna if in hyperparameter search
            if self.trial:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    logger.info("Trial pruned by Optuna.")
                    raise optuna.exceptions.TrialPruned()
            
            # Calculate improvement and log results
            improvement = self.best_val_loss - val_loss if self.has_val else 0.0
            self._log_epoch_results(
                epoch, train_loss, val_loss, time.time() - epoch_start, improvement
            )
            
            # Check for improvement and early stopping
            if self.has_val:
                if val_loss < self.best_val_loss - min_delta:
                    # Improvement found
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    epochs_without_improvement = 0
                    self._save_best_model()
                else:
                    # No improvement
                    epochs_without_improvement += 1
                    
                    if epochs_without_improvement >= patience:
                        logger.info(
                            f"Early stopping triggered after {patience} epochs "
                            f"without improvement."
                        )
                        break
        
        # If no validation set, save final model
        if not self.has_val:
            self.best_val_loss = train_loss
            self.best_epoch = epochs
            self._save_best_model()
        
        logger.info(
            f"Training complete. Best val_loss={self.best_val_loss:.4e} "
            f"at epoch {self.best_epoch}."
        )
        
        return self.best_val_loss
    
    def test(self) -> Dict[str, float]:
        """
        Run testing on the test set.
        
        Returns:
            Dictionary with test metrics
        """
        if not self.test_loader:
            logger.warning("No test dataset available. Skipping test.")
            return {"test_loss": float("inf"), "best_epoch": self.best_epoch}
        
        # Load best model checkpoint
        checkpoint_path = self.save_dir / "best_model.pt"
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = state["state_dict"]
            
            # Handle compiled model state dict
            if hasattr(self.model, '_orig_mod'):
                # Current model is compiled, but saved state might not be
                if not any(k.startswith("_orig_mod.") for k in state_dict):
                    # Add _orig_mod. prefix to match compiled model
                    state_dict = {
                        f"_orig_mod.{k}": v
                        for k, v in state_dict.items()
                    }
            else:
                # Current model is not compiled, but saved state might be
                if any(k.startswith("_orig_mod.") for k in state_dict):
                    # Remove _orig_mod. prefix
                    state_dict = {
                        k.replace("_orig_mod.", ""): v
                        for k, v in state_dict.items()
                    }

            self.model.load_state_dict(state_dict, strict=True)
            #logger.info(f"Loaded best model from epoch {state['epoch']}")
        
        # Run test evaluation
        test_loss = self._run_epoch(self.test_loader, is_train=False)
        if test_loss is None:
            test_loss = float("inf")
        
        # Save test metrics
        metrics = {
            "test_loss": test_loss,
            "best_epoch": self.best_epoch,
        }
        save_json(metrics, self.save_dir / "test_metrics.json")
        
        logger.info(f"Test loss: {test_loss:.4e}")
        
        return metrics
    
    def _run_epoch(self, loader: Optional[DataLoader], is_train: bool) -> Optional[float]:
        """
        Run one epoch of training or validation.
        
        IMPORTANT: This method correctly handles padding masks:
        - target_masks has True for padding positions
        - valid_mask = ~target_masks (True for valid positions)
        - Loss is computed only on valid positions
        
        Args:
            loader: DataLoader for the epoch
            is_train: Whether this is a training epoch
            
        Returns:
            Average loss for the epoch, or None if too many batches failed
        """
        if not loader or len(loader) == 0:
            mode = "training" if is_train else "validation"
            logger.warning(f"DataLoader for {mode} is empty. Skipping epoch.")
            return 0.0 if is_train else float("inf")
        
        # Set model mode
        self.model.train(is_train)
        
        # Track statistics
        total_loss = 0.0
        total_elements = 0
        failed_batches = 0
        
        device_type = str(self.device.type)
        
        for batch_idx, batch in enumerate(loader):
            # Data already on device if using DevicePrefetchLoader
            inputs, masks, targets, target_masks = batch
            
            sequence = inputs["sequence"]
            global_features = inputs.get("global_features")
            sequence_mask = masks["sequence"]  # True = padding position
            
            # Forward pass with autocast
            with torch.set_grad_enabled(is_train), autocast(
                device_type=device_type, enabled=self.use_amp
            ):
                # Model forward
                predictions = self.model(sequence, global_features, sequence_mask)
                
                # Compute unreduced loss
                unreduced_loss = self.criterion(predictions, targets)
                
                # IMPORTANT: Create valid mask (inverse of padding mask)
                # target_masks has True for padding, we want True for valid
                valid_mask = (~target_masks).unsqueeze(-1).expand_as(unreduced_loss)
                
                # Count valid elements
                num_valid = valid_mask.sum()
                
                # Skip batch if all elements are padding
                if num_valid.item() == 0:
                    failed_batches += 1
                    logger.debug(f"Batch {batch_idx} has no valid elements (all padding)")
                    continue
                
                # Compute masked loss
                # Only valid positions contribute to the loss
                masked_loss = unreduced_loss * valid_mask.float()
                loss = masked_loss.sum() / num_valid.float()
            
            if is_train:
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation and optimization step
                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                    # Unscale gradients if using AMP
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    
                    # Optimizer step
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad(set_to_none=True)
            
            # Accumulate statistics
            total_loss += loss.item() * num_valid.item()
            total_elements += num_valid.item()
            
            # Step the profiler after each batch (moved from end of epoch)
            if self.profiler is not None:
                self.profiler.step()
        
        # Check failure rate
        if len(loader) > 0 and failed_batches / len(loader) > self.max_batch_failure_rate:
            logger.critical(
                f"Too many failed batches ({failed_batches}/{len(loader)}). "
                f"Aborting epoch."
            )
            return None
        
        if total_elements == 0:
            logger.error("No valid elements were processed in this epoch.")
            return None
        
        # Log padding statistics
        if failed_batches > 0:
            logger.debug(
                f"Epoch complete. {failed_batches} batches were all padding. "
                f"Processed {total_elements} valid elements."
            )
                
        return total_loss / total_elements
    
    def _log_epoch_results(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        elapsed_time: float,
        improvement: float,
    ) -> None:
        """Log epoch results to console and file."""
        lr = self.optimizer.param_groups[0]["lr"]
        
        # Console log
        msg = (
            f"Epoch {epoch:03d}  "
            f"train:{train_loss:.3e}  "
            f"val:{val_loss:.3e}  "
            f"lr:{lr:.2e}  "
            f"time:{elapsed_time:.1f}s"
        )
        
        if improvement > 0:
            msg += f"  ↓{improvement:.3e}"
        
        logger.info(msg)
        
        # File log
        with self.log_path.open("a") as f:
            f.write(
                f"{epoch},{train_loss:.6e},{val_loss:.6e},"
                f"{lr:.6e},{elapsed_time:.1f},{improvement:.6e}\n"
            )
    
    def _save_best_model(self) -> None:
        """Save the best model checkpoint and export it."""
        # Prepare checkpoint
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "epoch": self.best_epoch,
            "val_loss": self.best_val_loss,
            "config": self.cfg,
            "padding_info": {
                "convention": "True = padding position",
                "follows_pytorch_standard": True,
                "loss_excludes_padding": True,
                "model_output_not_masked": True,
            }
        }
        
        # Save checkpoint
        checkpoint_path = self.save_dir / "best_model.pt"
        torch.save(checkpoint, checkpoint_path)
        #logger.info(f"Saved best model (epoch {self.best_epoch}).")
        
        # Export model for deployment
        self._export_model()
    
    def _export_model(self) -> None:
        """Export model using torch.export."""
        try:
            # Get sample input from validation loader
            sample = next(iter(self.val_loader)) if self.val_loader else None
            if sample is None:
                logger.warning("No sample available for export.")
                return
            
            # Create fresh un-compiled model for export
            model_for_export = create_prediction_model(self.cfg, device=self.device, compile_model=False)
            
            # Load trained weights
            if hasattr(self.model, '_orig_mod'):
                state_dict = self.model._orig_mod.state_dict()
            else:
                state_dict = self.model.state_dict()
            
            model_for_export.load_state_dict(state_dict)
            
            # Prepare example input
            inputs, masks, _, _ = sample
            example = {
                "sequence": inputs["sequence"][:1],
                "sequence_mask": masks["sequence"][:1],
            }
            if "global_features" in inputs:
                example["global_features"] = inputs["global_features"][:1]
            
            # Export model
            export_path = self.save_dir / f"best_model_epoch_{self.best_epoch}"
            export_model(model_for_export, example, export_path, self.cfg)
            
        except Exception as e:
            logger.error(f"Model export failed: {e}", exc_info=True)
        finally:
            # CRITICAL: Clean up the temporary model
            if 'model_for_export' in locals():
                del model_for_export
            
            # Force garbage collection and clear GPU cache
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


__all__ = ["ModelTrainer", "DevicePrefetchLoader"]