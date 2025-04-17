#!/usr/bin/env python3
"""
train.py – Model training for multi-source transformer models.

Features:
- Optional torch.compile (on CUDA only)
- Optional Mixed Precision (torch.amp) on CUDA
- Configurable loss function (MSE, L1, SmoothL1, Huber)
- Gradient clipping
- Early stopping with min_delta
- LR scheduler (ReduceLROnPlateau)
- Pearson correlation & explained variance tracking
- Training/validation/test splits
- CSV logging of metrics each epoch
- Returns best validation loss
"""
from __future__ import annotations

import contextlib # Import nullcontext
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

# Import necessary functions (ensure these exist in your other files)
from hardware import configure_dataloader_settings, get_device_type
from model import create_prediction_model
# Import save_json if needed for saving metrics separately
from utils import save_json # Assuming save_json is in utils.py

logger = logging.getLogger(__name__)

# Helper function for dataset splitting (deterministic)
def _split_dataset(
    ds: torch.utils.data.Dataset,
    val_frac: float,
    test_frac: float,
    seed: int = 42
) -> Tuple[Subset, Subset, Subset, List[int]]:
    """Deterministic random split -> (train, val, test, test_indices)."""
    n = len(ds)
    # Ensure fractions are valid
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1):
        raise ValueError("val_frac and test_frac must be between 0 and 1, and their sum less than 1.")

    val_n = int(n * val_frac)
    test_n = int(n * test_frac)
    train_n = n - val_n - test_n

    if train_n <= 0 or val_n <= 0 or test_n <= 0:
         raise ValueError(f"Dataset size {n} is too small for fractions val={val_frac}, test={test_frac}")

    # Use generator for reproducibility
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()

    # Assign indices
    test_indices = perm[:test_n]
    val_indices = perm[test_n : test_n + val_n]
    train_indices = perm[test_n + val_n :]

    logger.info(
        "Dataset split: %d train, %d val, %d test",
        len(train_indices), len(val_indices), len(test_indices)
    )
    # Return subsets and the indices for the test set
    return Subset(ds, train_indices), Subset(ds, val_indices), Subset(ds, test_indices), test_indices


class ModelTrainer:
    """Trainer for multi-source transformer models."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        save_dir: Path,
        dataset: torch.utils.data.Dataset,
        collate_fn: Optional[Any] = None,
        val_frac: float = 0.15, # Default validation fraction
        test_frac: float = 0.15, # Default test fraction
    ) -> None:
        self.cfg = config
        self.device = device
        self.save_dir = Path(save_dir) # Ensure save_dir is a Path object
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Store original dataset reference for accessing file names if needed
        self.original_dataset = dataset

        # Data split
        (
            self.train_ds,
            self.val_ds,
            self.test_ds,
            self.test_indices, # Store test indices
        ) = _split_dataset(
            dataset,
            val_frac=self.cfg.get("val_frac", val_frac),
            test_frac=self.cfg.get("test_frac", test_frac),
            seed=self.cfg.get("random_seed", 42)
        )
        self._build_dataloaders(collate_fn)

        # Model + training components
        self._build_model()
        self._build_optimiser_criterion()
        self._build_scheduler()

        # Mixed precision scaler for CUDA only
        self.use_amp = self.cfg.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
             logger.info("Automatic Mixed Precision (AMP) enabled for CUDA.")
        elif self.cfg.get("use_amp", False) and self.device.type != "cuda":
             logger.info(f"AMP requested but device is '{self.device.type}'. AMP disabled.")


        # Prepare CSV log header
        self.log_path = self.save_dir / "training_log.csv"
        header = "epoch,train_loss,val_loss,pearson,exp_var,lr,sec\n"
        self.log_path.write_text(header)

        # Save test profile list (best effort)
        self._save_test_list()

        # Tracking variables
        self.best_val = float("inf")
        self.best_epoch = -1


    def _build_dataloaders(self, collate_fn: Any) -> None:
        """Creates DataLoaders for train, validation, and test sets."""
        hw_settings = configure_dataloader_settings()
        batch_size = self.cfg.get("batch_size", 16)
        num_workers = self.cfg.get("num_workers", hw_settings["num_workers"])

        if self.device.type == "mps" and num_workers > 1:
             logger.warning(f"Reducing num_workers from {num_workers} to 1 for MPS device.")
             num_workers = 1

        common_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=hw_settings["pin_memory"],
            persistent_workers=(hw_settings["persistent_workers"] and num_workers > 0),
            collate_fn=collate_fn
        )

        if len(self.train_ds) < batch_size:
             logger.warning(f"Training dataset size ({len(self.train_ds)}) < batch size ({batch_size}). Setting drop_last=False for train loader.")
             drop_last_train = False
        else:
             drop_last_train = self.cfg.get("drop_last_train", False)

        self.train_loader = DataLoader(self.train_ds, shuffle=True, drop_last=drop_last_train, **common_kwargs)
        self.val_loader = DataLoader(self.val_ds, shuffle=False, drop_last=False, **common_kwargs)
        self.test_loader = DataLoader(self.test_ds, shuffle=False, drop_last=False, **common_kwargs)

        if len(self.train_loader) == 0:
             raise ValueError(f"Training DataLoader is empty. Check dataset size ({len(self.train_ds)}) and batch size ({batch_size}).")

    def _build_model(self) -> None:
        """Instantiate model and optionally compile (on CUDA only)."""
        self.model: nn.Module = create_prediction_model(self.cfg).to(self.device)
        if self.cfg.get("use_torch_compile", False) and self.device.type == "cuda":
            try:
                logger.info("Attempting torch.compile() on model (CUDA)...")
                self.model = torch.compile(self.model)
                logger.info("Model compiled successfully.")
            except Exception as e:
                logger.warning("torch.compile failed: %s — continuing in eager mode.", e)
        elif self.cfg.get("use_torch_compile", False) and self.device.type != "cuda":
             logger.info("torch.compile requested but device is not CUDA; skipping compilation.")


    def _build_optimiser_criterion(self) -> None:
        """Sets up the optimiser and loss criterion."""
        lr = self.cfg.get("learning_rate", 1e-4)
        wd = self.cfg.get("weight_decay", 1e-5)
        opt_name = self.cfg.get("optimizer", "adamw").lower()
        params = self.model.parameters()

        if opt_name == "sgd":
            self.optimizer = optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        elif opt_name == "adam":
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            self.optimizer = optim.AdamW(
                params,
                lr=lr,
                weight_decay=wd,
                betas=tuple(self.cfg.get("adamw_betas", (0.9, 0.999))),
                eps=self.cfg.get("adamw_eps", 1e-8),
                amsgrad=self.cfg.get("adamw_amsgrad", False)
            )
        else:
             logger.warning(f"Unknown optimizer '{opt_name}', defaulting to AdamW.")
             self.optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)

        loss_map = {
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss(),
            "smooth_l1": nn.SmoothL1Loss(),
            "huber": nn.HuberLoss(delta=self.cfg.get("huber_delta", 1.0))
        }
        loss_choice = self.cfg.get("loss_function", "mse").lower()
        self.criterion = loss_map.get(loss_choice)
        if self.criterion is None:
            logger.warning(f"Unknown loss function '{loss_choice}', defaulting to MSELoss.")
            self.criterion = nn.MSELoss()

        self.max_grad_norm = self.cfg.get("gradient_clip_val", 1.0)


    def _build_scheduler(self) -> None:
        """Sets up the learning rate scheduler."""
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.cfg.get("lr_factor", 0.5),
            patience=self.cfg.get("lr_patience", 5),
            min_lr=self.cfg.get("min_lr", 1e-7),
            verbose=False # Set verbose=False to reduce log noise, or True to see LR changes
        )

    def train(self) -> float:
        """
        Run the main training loop.

        Returns:
            float: The best validation loss achieved during training.
        """
        epochs = self.cfg.get("epochs", 100)
        patience = self.cfg.get("early_stopping_patience", 10)
        min_delta = self.cfg.get("min_delta", 1e-10)

        wait = 0
        self.best_val = float("inf")
        self.best_epoch = -1

        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Device: {self.device}, AMP: {self.use_amp}, Compile: {self.cfg.get('use_torch_compile', False) and self.device.type == 'cuda'}")
        logger.info(f"Early stopping patience: {patience} epochs, Min delta: {min_delta:.1e}")
        logger.info(f"Saving checkpoints to: {self.save_dir}")
        logger.info("-" * 80)
        logger.info(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^12} | {'Pearson R':^10} | {'Exp Var':^10} | {'LR':^11} | {'Time (s)':^9}")
        logger.info("-" * 80)

        completed_epochs = 0
        try:
            for ep in range(1, epochs + 1):
                completed_epochs = ep
                epoch_start_time = time.time()

                train_loss = self._run_epoch(self.train_loader, train=True)
                val_loss, pearson, exp_var = self._run_epoch(self.val_loader, train=False)

                epoch_duration = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']

                log_msg = (f"{ep:^7d} | {train_loss:^12.4e} | {val_loss:^12.4e} | "
                           f"{pearson:^10.4f} | {exp_var:^10.4f} | {current_lr:^11.3e} | {epoch_duration:^9.1f}")
                logger.info(log_msg)
                with self.log_path.open("a") as f:
                     f.write(f"{ep},{train_loss:.6e},{val_loss:.6e},{pearson:.6f},"
                             f"{exp_var:.6f},{current_lr:.6e},{epoch_duration:.2f}\n")

                # Checkpointing and Early Stopping
                if val_loss < self.best_val - min_delta:
                    self.best_val = val_loss
                    self.best_epoch = ep
                    wait = 0
                    self._checkpoint("best_model.pt", ep, self.best_val)
                    logger.debug(f"New best model saved at epoch {ep} (Val Loss: {self.best_val:.4e})")
                else:
                    wait += 1
                    if wait >= patience:
                        logger.info(f"Early stopping triggered after epoch {ep} (no improvement for {patience} epochs).")
                        break

                # Scheduler Step (pass validation loss)
                self.scheduler.step(val_loss)

        except KeyboardInterrupt:
             logger.warning("Training interrupted by user (KeyboardInterrupt).")
        except Exception as e:
             logger.exception("An error occurred during training: %s", e)
             # Depending on desired behavior, maybe return inf or raise
             # return float('inf') # Indicate failure for tuning
        finally:
            # Post-Training
            if self.best_epoch != -1:
                logger.info(f"Training finished. Best validation loss: {self.best_val:.4e} at epoch {self.best_epoch}.")
                self._checkpoint("final_model.pt", completed_epochs, self.best_val, is_final=True)
                logger.info("Loading best model for final test evaluation...")
                best_model_path = self.save_dir / "best_model.pt"
                if best_model_path.exists():
                     try:
                        checkpoint = torch.load(best_model_path, map_location=self.device)
                        state_dict = checkpoint['state_dict']
                        if hasattr(self.model, '_orig_mod'): # Compiled model
                             self.model._orig_mod.load_state_dict(state_dict)
                        else:
                             self.model.load_state_dict(state_dict)
                        self.test()
                     except Exception as e:
                          logger.error(f"Failed to load best model state for testing: {e}", exc_info=True)
                          logger.info("Testing with final model state instead.")
                          self._checkpoint("final_model.pt", completed_epochs, self.best_val, is_final=True)
                          self.test()
                else:
                     logger.warning("Best model checkpoint (best_model.pt) not found. Testing with final model state.")
                     self._checkpoint("final_model.pt", completed_epochs, self.best_val, is_final=True)
                     self.test()
            else:
                 logger.warning("Training completed, but no improvement detected over initial validation loss.")
                 self._checkpoint("final_model.pt", completed_epochs, float('inf'), is_final=True)

            return self.best_val if self.best_epoch != -1 else float('inf')


    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
    ) -> Tuple[float, float, float] | float:
        """
        Run one pass (train or eval) over the given DataLoader.
        Returns average loss during training.
        Returns (loss, pearson_r, explained_variance) during evaluation.
        """
        self.model.train(train)
        total_loss = 0.0
        batch_count = 0
        all_preds: List[torch.Tensor] = []
        all_trues: List[torch.Tensor] = []

        # --- FIX 2: Use nullcontext when AMP is not active ---
        amp_enabled = self.use_amp and self.device.type == 'cuda'
        autocast_context = torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled) if amp_enabled else contextlib.nullcontext()
        # --- End FIX 2 ---

        grad_context = torch.enable_grad() if train else torch.no_grad()

        with grad_context:
            for batch_idx, batch in enumerate(loader):
                try:
                    inputs, targets = batch
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                    targets = targets.to(self.device, non_blocking=True)

                    with autocast_context: # Apply the chosen context
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)

                    if not torch.isfinite(loss):
                         logger.warning(f"Non-finite loss detected in {'train' if train else 'eval'} epoch, batch {batch_idx}. Skipping batch.")
                         continue

                    if train:
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.use_amp: # Check use_amp for scaler
                            self.scaler.scale(loss).backward()
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            self.optimizer.step()

                    total_loss += loss.item()
                    batch_count += 1
                    if not train:
                        all_preds.append(outputs.detach().float().cpu())
                        all_trues.append(targets.detach().float().cpu())

                except Exception as e:
                     logger.error(f"Error during {'train' if train else 'eval'} epoch, batch {batch_idx}: {e}", exc_info=False)
                     logger.debug("Detailed error trace:", exc_info=True)
                     continue # Skip problematic batch

        if batch_count == 0:
             phase = 'train' if train else 'eval'
             logger.error(f"No batches successfully processed in {phase} epoch.")
             return (float('inf'), 0.0, 0.0) if not train else float('inf')

        avg_loss = total_loss / batch_count

        if train:
            return avg_loss
        else:
            if not all_preds or not all_trues:
                 logger.warning("No valid prediction/target tensors collected for evaluation metrics.")
                 return avg_loss, 0.0, 0.0

            try:
                preds_all = torch.cat(all_preds).flatten()
                trues_all = torch.cat(all_trues).flatten()

                mean_preds = torch.mean(preds_all)
                mean_trues = torch.mean(trues_all)
                cov = torch.mean((preds_all - mean_preds) * (trues_all - mean_trues))
                std_preds = torch.std(preds_all, unbiased=False)
                std_trues = torch.std(trues_all, unbiased=False)
                pearson_r = cov / (std_preds * std_trues + 1e-8)

                var_trues = torch.var(trues_all, unbiased=False)
                if var_trues < 1e-8:
                     # Handle edge case: true values are constant
                     exp_var = torch.tensor(1.0) if torch.var(preds_all, unbiased=False) < 1e-8 else torch.tensor(float('-inf'))
                else:
                     exp_var = 1 - torch.var(trues_all - preds_all, unbiased=False) / var_trues

                pearson_r = torch.clamp(pearson_r, min=-1.0, max=1.0)

                return avg_loss, pearson_r.item(), exp_var.item()
            except Exception as metric_exc:
                 logger.error(f"Error calculating evaluation metrics: {metric_exc}", exc_info=True)
                 return avg_loss, 0.0, 0.0


    def _checkpoint(self, name: str, epoch: int, val_loss: float, is_final: bool = False) -> None:
        """Saves model checkpoint, handling potential compilation state dict."""
        model_to_save = self.model
        state_dict_to_save = None
        try:
             # Get state dict, unwrapping compile if needed
             if hasattr(model_to_save, '_orig_mod'):
                 model_to_save = model_to_save._orig_mod
             state_dict_to_save = model_to_save.state_dict()
        except Exception as e:
             logger.warning(f"Could not get state_dict for checkpoint '{name}': {e}. Saving raw model state.")
             try: # Fallback
                 state_dict_to_save = self.model.state_dict()
             except Exception as e2:
                  logger.error(f"Failed to get even raw state_dict for checkpoint '{name}': {e2}")
                  return

        if state_dict_to_save is None:
             logger.error(f"Could not obtain state_dict for saving checkpoint '{name}'.")
             return

        save_data = {
            'state_dict': state_dict_to_save,
            'config': self.cfg,
            'epoch': epoch,
            'val_loss': val_loss,
        }
        path = self.save_dir / name
        try:
            torch.save(save_data, path)
            if is_final:
                 logger.info(f"Final model checkpoint saved to {path}")
        except Exception as e:
             logger.error(f"Failed to save checkpoint {name} to {path}: {e}")

    def test(self) -> Tuple[float, float, float]:
        """
        Evaluate the loaded (presumably best) model on the test set.
        Returns: (test_loss, test_pearson_r, test_explained_variance)
        """
        logger.info("Evaluating model on test set...")
        test_loss, test_pearson, test_exp_var = self._run_epoch(self.test_loader, train=False)
        logger.info(
            f"Test Results | Loss: {test_loss:.4e} | Pearson R: {test_pearson:.4f} | Explained Var: {test_exp_var:.4f}"
        )
        test_metrics = {
             "test_loss": test_loss,
             "test_pearson": test_pearson,
             "test_explained_variance": test_exp_var
        }
        try:
             # Use the imported save_json utility
             if save_json is not None:
                 save_json(test_metrics, self.save_dir / "test_metrics.json")
             else:
                  (self.save_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
        except Exception as e:
             logger.error(f"Failed to save test metrics: {e}")

        return test_loss, test_pearson, test_exp_var

    def _save_test_list(self) -> None:
        """Save the list of file paths or identifiers used in the test set."""
        try:
            base_dataset = self.original_dataset
            while isinstance(base_dataset, Subset):
                if hasattr(base_dataset, 'dataset'):
                     base_dataset = base_dataset.dataset
                else:
                     break

            if hasattr(base_dataset, 'valid_files') and hasattr(self, 'test_indices') and isinstance(self.test_indices, list):
                valid_test_indices = [i for i in self.test_indices if 0 <= i < len(base_dataset.valid_files)]
                if len(valid_test_indices) != len(self.test_indices):
                     logger.warning("Some test indices were out of bounds for base_dataset.valid_files.")

                test_files = []
                base_data_dir = getattr(base_dataset, 'data_folder', None)
                for i in valid_test_indices:
                     fpath = base_dataset.valid_files[i]
                     try:
                          rel_path = fpath.relative_to(base_data_dir) if base_data_dir and fpath.is_relative_to(base_data_dir) else fpath.name
                          test_files.append(str(rel_path))
                     except Exception: # Fallback to name if relative path fails
                          test_files.append(fpath.name)

                test_info = {"test_profile_paths": test_files}
                output_path = self.save_dir / "test_profiles.json"
                if save_json is not None:
                    save_json(test_info, output_path)
                    logger.info(f"Saved {len(test_files)} test profile paths to {output_path}")
                else:
                    output_path.write_text(json.dumps(test_info, indent=2))
                    logger.warning("Used standard json to save test profile list.")

            else:
                logger.warning("Could not determine test profile paths from dataset attributes. Saving indices instead.")
                test_info = {"test_indices": self.test_indices}
                output_path = self.save_dir / "test_indices.json"
                if save_json is not None:
                     save_json(test_info, output_path)
                     logger.info(f"Saved {len(self.test_indices)} test indices to {output_path}")
                else:
                     output_path.write_text(json.dumps(test_info, indent=2))
                     logger.warning("Used standard json to save test indices list.")

        except Exception as e:
            logger.exception(f"Error occurred while trying to save test profile list: {e}")


__all__ = ["ModelTrainer"]