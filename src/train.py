#!/usr/bin/env python3
"""
train.py – streamlined training loop for the multi‑source transformer.

"""
from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.amp.grad_scaler import GradScaler # Added import
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset

from hardware import configure_dataloader_settings
from model import create_prediction_model
from utils import save_json

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _split_dataset(
    dataset: Dataset,
    val_frac: float,
    test_frac: float,
    *,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset, List[int]]:
    """Deterministically split *dataset* into train / val / test subsets."""
    n = len(dataset)
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("fractions must be between 0 and 1 and sum to < 1")

    nv, nt = int(n * val_frac), int(n * test_frac)
    if n - nv - nt <= 0:
        raise ValueError("dataset too small for requested split ratio")

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    test_idx, val_idx, train_idx = idx[:nt], idx[nt : nt + nv], idx[nt + nv :]

    logger.info("Dataset split: %d train / %d val / %d test", len(train_idx), len(val_idx), len(test_idx))

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx), test_idx

# --------------------------------------------------------------------------- #
# trainer                                                                     #
# --------------------------------------------------------------------------- #

class ModelTrainer:
    """Orchestrates training / validation / testing including checkpoints."""
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
    ) -> None:
        self.cfg, self.device = config, device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # dataset split
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

        # data loaders
        self._build_dataloaders(collate_fn)
        # model + optimiser + scheduler
        self._build_model()
        self._build_optimiser()
        self._build_scheduler()

        # AMP
        self.use_amp = bool(self.cfg.get("use_amp", False) and device.type == "cuda")

        # Apply the fix for FutureWarning here:
        self.scaler: Optional[GradScaler] = GradScaler(device.type) if self.use_amp else None
        if self.use_amp:
            logger.info("AMP enabled on CUDA")

        # loss & grad‑clip
        self.criterion = nn.MSELoss()
        self.max_grad_norm = max(float(self.cfg.get("gradient_clip_val", 1.0)), 1e-12)

        # logging
        self.log_path = self.save_dir / "training_log.csv"
        self.log_path.write_text("epoch,train_loss,val_loss,lr,time_s\n")
        self.best_val, self.best_epoch = float("inf"), -1

        # clear‑cache setting - Use default 0 (off) if not specified
        self._clear_every = int(self.cfg.get("clear_cuda_cache_every", 0))

        # save list of test files/indices
        self._save_test_list(dataset)

    # ------------------------------------------------------------------ #
    # build helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_dataloaders(self, collate_fn: Optional[Callable]) -> None:
        hw = configure_dataloader_settings()
        bs = int(self.cfg.get("batch_size", 16))
        workers = max(0, int(self.cfg.get("num_workers", hw["num_workers"])))

        common = dict(
            batch_size=bs,
            num_workers=workers,
            pin_memory=hw["pin_memory"],
            # Ensure persistent_workers is False if workers is 0
            persistent_workers=workers > 0 and hw["persistent_workers"],
            collate_fn=collate_fn,
        )
        self.train_loader = DataLoader(self.train_ds, shuffle=True, drop_last=False, **common)
        self.val_loader = DataLoader(self.val_ds, shuffle=False, **common)
        self.test_loader = DataLoader(self.test_ds, shuffle=False, **common)
        logger.info("DataLoaders ready (batch=%d, workers=%d)", bs, workers)

    def _build_model(self) -> None:
        self.model = create_prediction_model(self.cfg).to(self.device)
        if self.cfg.get("use_torch_compile", False) and self.device.type == "cuda":
            # Keep original silent suppression as per user's code
            with contextlib.suppress(Exception):
                self.model = torch.compile(self.model)  # type: ignore[attr-defined]
                
        logger.info("Model built (%d parameters)", sum(p.numel() for p in self.model.parameters()))

    def _build_optimiser(self) -> None:
        opt_name = str(self.cfg.get("optimizer", "adamw")).lower()
        lr = float(self.cfg.get("learning_rate", 1e-4))
        wd = float(self.cfg.get("weight_decay", 1e-5))
        if opt_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        elif opt_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        # Default to AdamW if optimizer name is not 'sgd' or 'adam'
        else:
            if opt_name != "adamw":
                 logger.warning("Unsupported optimizer '%s', defaulting to AdamW.", opt_name)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        logger.info("Optimizer: %s lr=%.1e wd=%.1e", self.optimizer.__class__.__name__, lr, wd)


    def _build_scheduler(self) -> None:
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.cfg.get("lr_factor", 0.5),
            patience=self.cfg.get("lr_patience", 5),
            min_lr=self.cfg.get("min_lr", 1e-7),
            # Keep original verbose setting (default False)
        )

    # ------------------------------------------------------------------ #
    # high‑level train / test                                            #
    # ------------------------------------------------------------------ #

    def train(self) -> float:
        epochs = int(self.cfg.get("epochs", 100))
        patience = int(self.cfg.get("early_stopping_patience", 10))
        min_delta = float(self.cfg.get("min_delta", 1e-10))
        wait = 0
        for ep in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step(val_loss)

            # --- optional cache clear ---
            if (
                self.device.type == "cuda"
                and self._clear_every > 0
                and ep % self._clear_every == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

            # --- csv + console log ---
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
            # Keep original logging method (read + write)
            self.log_path.write_text(
                self.log_path.read_text()
                + f"{ep},{train_loss:.6e},{val_loss:.6e},{lr_val:.6e},{dt:.1f}\n"
            )

            # --- early stopping & checkpoint ---
            if val_loss < self.best_val - min_delta:
                # Original had no logging here, keeping it that way
                self.best_val, self.best_epoch, wait = val_loss, ep, 0
                self._checkpoint("best_model.pt", ep, val_loss)
            else:
                wait += 1
                if wait >= patience:
                    logger.info("Early stopping after %d epochs", ep)
                    break

        # --- final checkpoint & test ---
        # Use last recorded epoch 'ep' and 'val_loss'
        self._checkpoint("final_model.pt", ep, val_loss, final=True)
        self.test()
        # Original had no final logging here
        return self.best_val

    # ------------------------------------------------------------------ #
    # epoch loop                                                         #
    # ------------------------------------------------------------------ #

    def _run_epoch(self, loader: DataLoader, *, train: bool) -> float:
        self.model.train(train)
        total, cnt = 0.0, 0

        # Determine autocast context based on device and config
        # Keep original float16 default
        amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if self.use_amp
            else contextlib.nullcontext()
        )
        # Enable/disable gradient calculation
        grad_ctx = torch.enable_grad() if train else torch.no_grad()

        # Keep original loop without tqdm
        with grad_ctx:
            for inputs, targets in loader:
                # Keep original data moving without extra try/except
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                targets = targets.to(self.device, non_blocking=True)
                with amp_ctx:
                    # Keep original forward/loss calc without extra try/except
                    pred = self.model(inputs)
                    loss = self.criterion(pred, targets)

                # Check for non-finite loss
                if not torch.isfinite(loss):
                    logger.info("Non finite loss!")
                    continue

                if train:
                    # Backward pass & optimization step
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler: # Check if scaler exists (i.e., use_amp is True)
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer) # Unscale before clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()

                # Accumulate loss
                total += loss.item()
                cnt += 1

        # Calculate average loss for the epoch
        return total / max(cnt, 1)

    # ------------------------------------------------------------------ #
    # checkpoint / test                                                  #
    # ------------------------------------------------------------------ #

    def _checkpoint(self, name: str, epoch: int, val_loss: float, *, final: bool = False) -> None:
        """Save model & metadata; compatible with torch.compile‑wrapped models."""
        # Ensure the model object for saving state_dict is the original one if compiled
        mod = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        path = self.save_dir / name
        checkpoint_data = {
            "state_dict": mod.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": self.cfg, # Save config used for this training run
        }
        # Keep original saving without extra try/except
        torch.save(checkpoint_data, path)
        # Original logged only final save, keeping it that way
        if final:
            logger.info("Final model saved → %s", path.name)


    def test(self) -> None:
        """Evaluate the final model on the held-out test set."""
        # Keep original test execution without extra logging/eval mode setting
        loss = self._run_epoch(self.test_loader, train=False)
        logger.info("Test loss: %.4e", loss)
        # Keep original saving without extra try/except or logging
        save_json({"test_loss": loss}, self.save_dir / "test_metrics.json")


    # ------------------------------------------------------------------ #
    # utilities                                                          #
    # ------------------------------------------------------------------ #

    def _save_test_list(self, dataset: Dataset) -> None:
        """Saves the indices used for the test set."""
        save_json({"test_indices": sorted(self.test_indices)}, self.save_dir / "test_set_info.json")


__all__ = ["ModelTrainer"]