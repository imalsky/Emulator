#!/usr/bin/env python3
"""
train.py – streamlined training loop for the multi‑source transformer.

This module contains the ModelTrainer class, which handles the training,
validation, and testing of the transformer model, including checkpointing,
early stopping, and Optuna integration for hyperparameter tuning.
"""
from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
import torch
from optuna.exceptions import TrialPruned
from torch import nn, optim
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset

from hardware import configure_dataloader_settings
from model import create_prediction_model
from utils import save_json

logger = logging.getLogger(__name__)


def _split_dataset(
    dataset: Dataset,
    val_frac: float,
    test_frac: float,
    *,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset, List[int]]:
    """
    Deterministically splits a dataset into training, validation, and test subsets.

    Args:
        dataset: The full dataset to be split.
        val_frac: The fraction of the dataset to use for validation.
        test_frac: The fraction of the dataset to use for testing.
        seed: The random seed for shuffling to ensure reproducible splits.

    Returns:
        A tuple containing the training, validation, and test Subsets,
        and a list of indices used for the test set.
    """
    n = len(dataset)
    if not (
        0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1
    ):
        raise ValueError(
            "fractions must be between 0 and 1 and sum to < 1"
        )
    nv, nt = int(n * val_frac), int(n * test_frac)
    if n - nv - nt <= 0:
        raise ValueError("dataset too small for requested split ratio")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    test_idx, val_idx, train_idx = idx[:nt], idx[nt : nt + nv], idx[nt + nv :]
    logger.info(
        "Dataset split: %d train / %d val / %d test",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
        test_idx,
    )


class ModelTrainer:
    """
    Orchestrates the training, validation, and testing of the model.

    This class manages the entire lifecycle of model training, including
    dataset splitting, DataLoader setup, model and optimizer instantiation,
    the main training loop with validation and early stopping, checkpointing,
    Optuna integration for pruning, and final model testing.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        save_dir: Path,
        dataset: Dataset,
        collate_fn: Optional[Callable] = None,
        *,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        optuna_trial: Optional[optuna.Trial] = None,
    ) -> None:
        """
        Initializes the ModelTrainer.

        Args:
            config: Configuration dictionary for training parameters.
            device: The PyTorch device (CPU or CUDA) for training.
            save_dir: Directory to save model checkpoints, logs, and artifacts.
            dataset: The complete dataset instance.
            collate_fn: Custom collate function for DataLoaders.
            val_frac: Fraction of data for the validation set.
            test_frac: Fraction of data for the test set.
            optuna_trial: Optuna Trial object if part of a hyperparameter search.
        """
        self.cfg, self.device = config, device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.optuna_trial = optuna_trial
        (
            self.train_ds,
            self.val_ds,
            self.test_ds,
            self.test_indices,
        ) = _split_dataset(
            dataset,
            val_frac=self.cfg.get("val_frac", val_frac),
            test_frac=self.cfg.get("test_frac", test_frac),
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
            GradScaler() if self.use_amp else None
        )
        if self.use_amp:
            logger.info("AMP enabled on CUDA")
        self.criterion = nn.MSELoss()
        self.max_grad_norm = max(
            float(self.cfg.get("gradient_clip_val", 1.0)), 1e-12
        )
        self.log_path = self.save_dir / "training_log.csv"
        self.log_path.write_text("epoch,train_loss,val_loss,lr,time_s\n")
        self.best_val, self.best_epoch = float("inf"), -1
        self._clear_every = int(self.cfg.get("clear_cuda_cache_every", 0))
        self._save_test_list()

    def _build_dataloaders(self, collate_fn: Optional[Callable]) -> None:
        """Initializes training, validation, and test DataLoaders."""
        hw = configure_dataloader_settings()
        bs = int(self.cfg.get("batch_size", 16))
        workers = max(0, int(self.cfg.get("num_workers", hw["num_workers"])))
        common = dict(
            batch_size=bs,
            num_workers=workers,
            pin_memory=hw["pin_memory"],
            persistent_workers=workers > 0 and hw["persistent_workers"],
            collate_fn=collate_fn,
        )
        self.train_loader = DataLoader(
            self.train_ds, shuffle=True, drop_last=False, **common
        )
        self.val_loader = DataLoader(self.val_ds, shuffle=False, **common)
        self.test_loader = DataLoader(self.test_ds, shuffle=False, **common)
        logger.info("DataLoaders ready (batch=%d, workers=%d)", bs, workers)

    def _build_model(self) -> None:
        """Creates and initializes the prediction model."""
        self.model = create_prediction_model(self.cfg).to(self.device)
        if (
            self.cfg.get("use_torch_compile", False)
            and self.device.type == "cuda"
            and hasattr(torch, "compile")
        ):
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                logger.warning(
                    f"torch.compile failed: {e}. Using uncompiled model."
                )
        logger.info(
            "Model built (%d parameters)",
            sum(p.numel() for p in self.model.parameters()),
        )

    def _build_optimiser(self) -> None:
        """Initializes the optimizer based on the configuration."""
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
        else:
            if opt_name != "adamw":
                logger.warning(
                    "Unsupported optimizer '%s', defaulting to AdamW.", opt_name
                )
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        logger.info(
            "Optimizer: %s lr=%.1e wd=%.1e",
            self.optimizer.__class__.__name__,
            lr,
            wd,
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
        Executes the main training loop over a configured number of epochs.

        Includes training steps, validation, Optuna pruning, logging,
        early stopping, and checkpointing.

        Returns:
            The best validation loss achieved during training.
        """
        epochs = int(self.cfg.get("epochs", 100))
        patience = int(self.cfg.get("early_stopping_patience", 10))
        min_delta = float(self.cfg.get("min_delta", 1e-10))
        wait = 0
        final_epoch_completed = 0 # Tracks the last epoch run

        for ep in range(1, epochs + 1):
            final_epoch_completed = ep
            t0 = time.time()
            train_loss = self._run_epoch(
                self.train_loader, train=True, epoch=ep
            )
            val_loss = self._run_epoch(
                self.val_loader, train=False, epoch=ep
            )
            self.scheduler.step(val_loss)

            if self.optuna_trial:
                self.optuna_trial.report(val_loss, ep)
                if self.optuna_trial.should_prune():
                    raise TrialPruned()

            if (
                self.device.type == "cuda"
                and self._clear_every > 0
                and ep % self._clear_every == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

            lr_val = self.optimizer.param_groups[0]["lr"]
            dt = time.time() - t0
            logger.info(
                "Epoch %03d | train %.4e | val %.4e | lr %.4e | %.1fs",
                ep,
                train_loss,
                val_loss,
                lr_val,
                dt,
            )
            with open(self.log_path, "a") as f:
                f.write(
                    f"{ep},{train_loss:.6e},{val_loss:.6e},"
                    f"{lr_val:.6e},{dt:.1f}\n"
                )

            if val_loss < self.best_val - min_delta:
                self.best_val, self.best_epoch, wait = val_loss, ep, 0
                self._checkpoint("best_model.pt", ep, val_loss)
            else:
                wait += 1
                if wait >= patience:
                    logger.info("Early stopping after %d epochs", ep)
                    break
        else:
            # This else block executes if the loop completed without a 'break'
            final_epoch_completed = epochs


        self._checkpoint(
            "final_model.pt", final_epoch_completed, val_loss, final=True
        )
        self.test()
        return self.best_val

    def _run_epoch(
        self, loader: DataLoader, *, train: bool, epoch: int
    ) -> float:
        """
        Executes a single epoch of training or validation.

        Args:
            loader: The DataLoader for the current phase (train or val).
            train: Boolean indicating if this is a training phase.
            epoch: The current epoch number (for logging purposes).

        Returns:
            The average loss for the epoch.
        """
        self.model.train(train)
        total_loss, count = 0.0, 0
        amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if self.use_amp
            else contextlib.nullcontext()
        )
        grad_ctx = torch.enable_grad() if train else torch.no_grad()

        with grad_ctx:
            for inputs, targets in loader:
                inputs = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in inputs.items()
                }
                targets = targets.to(self.device, non_blocking=True)
                with amp_ctx:
                    pred = self.model(inputs)
                    loss = self.criterion(pred, targets)
                if not torch.isfinite(loss):
                    logger.warning(
                        f"Epoch {epoch}: Non-finite loss. Skipping batch."
                    )
                    continue
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.optimizer.step()
                total_loss += loss.item() * targets.size(0)
                count += targets.size(0)
        return total_loss / max(count, 1)

    def _checkpoint(
        self, name: str, epoch: int, val_loss: float, *, final: bool = False
    ) -> None:
        """
        Saves a model checkpoint.

        Args:
            name: The filename for the checkpoint (e.g., "best_model.pt").
            epoch: The epoch number at which the checkpoint is saved.
            val_loss: The validation loss at this checkpoint.
            final: Boolean indicating if this is the final model checkpoint.
        """
        mod = (
            self.model._orig_mod
            if hasattr(self.model, "_orig_mod")
            else self.model
        )
        path = self.save_dir / name
        checkpoint_data = {
            "state_dict": mod.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": self.cfg,
        }
        try:
            torch.save(checkpoint_data, path)
            if final:
                logger.info("Final model saved → %s", path.name)
        except Exception as e:
            logger.error(
                f"Failed to save checkpoint {name}: {e}", exc_info=True
            )

    def test(self) -> None:
        """Evaluates the best model on the held-out test set."""
        best_model_path = self.save_dir / "best_model.pt"
        if best_model_path.exists():
            try:
                ckpt = torch.load(best_model_path, map_location=self.device)
                mod = (
                    self.model._orig_mod
                    if hasattr(self.model, "_orig_mod")
                    else self.model
                )
                mod.load_state_dict(ckpt["state_dict"])
                logger.info(
                    f"Loaded best model (epoch {ckpt.get('epoch', '?')}) for testing."
                )
            except Exception as e:
                logger.error(
                    f"Failed to load best model for testing: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                "Best model checkpoint not found. Testing with final model state."
            )
        loss = self._run_epoch(self.test_loader, train=False, epoch=-1)
        logger.info("Test loss (on best model): %.4e", loss)
        try:
            save_json(
                {"test_loss": loss}, self.save_dir / "test_metrics.json"
            )
        except Exception as e:
            logger.error(f"Failed to save test metrics: {e}", exc_info=True)

    def _save_test_list(self) -> None:
        """Saves the indices used for the test set to a JSON file."""
        try:
            save_json(
                {"test_indices": sorted(self.test_indices)},
                self.save_dir / "test_set_info.json",
            )
        except Exception as e:
            logger.error(f"Failed to save test set info: {e}", exc_info=True)


__all__ = ["ModelTrainer"]
