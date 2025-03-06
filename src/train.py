#!/usr/bin/env python3
"""
train.py - Training infrastructure for sequence prediction models.
"""

import time
import json
import logging
import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from model import create_prediction_model

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config, device, save_path, dataset, test_size=0.15, val_size=0.15):
        """Initialize the trainer with configuration and dataset."""
        self.config = config
        self.device = device
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ModelTrainer with device: {device}")
        
        # Setup mixed precision training - CPU or CUDA only
        self.use_amp = config.get("use_amp", False) or config.get("use_mixed_precision", False)
        if self.device.type != 'cuda':
            self.use_amp = False
        
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        logger.info(f"Mixed precision training enabled: {self.use_amp}")
        
        # Split dataset
        self._prepare_datasets(dataset, test_size, val_size)
        self._create_dataloaders()
        
        # Initialize model
        self.model = create_prediction_model(config).to(device)
        logger.info(f"Model created with {self.model.count_parameters():,} parameters")
        
        # Setup training components
        self._setup_training_components()
        
        # Training state
        self.best_val_loss = float("inf")
        self.best_state = None
        self.best_epoch = -1

    def _prepare_datasets(self, dataset, test_size, val_size):
        """Split dataset into training, validation, and test sets."""
        dataset_size = len(dataset)
        test_count = int(test_size * dataset_size)
        val_count = int(val_size * dataset_size)
        train_count = dataset_size - test_count - val_count

        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_count, val_count, test_count], generator=generator
        )
        
        logger.info(f"Dataset split: {train_count} train, {val_count} validation, {test_count} test samples")

    def _create_dataloaders(self):
        """Create DataLoaders for training, validation, and testing."""
        output_seq_length = self.config.get("output_seq_length", 0)
        
        # Determine batch size based on sequence length
        if output_seq_length > 40000:
            default_batch = 2
        elif output_seq_length > 20000:
            default_batch = 4
        elif output_seq_length > 10000:
            default_batch = 16
        else:
            default_batch = 32
            
        batch_size = self.config.get("batch_size", default_batch)
        num_workers = self.config.get("num_workers", min(4, os.cpu_count() or 1))
        
        dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": num_workers > 0
        }
        
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, **dataloader_kwargs)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **dataloader_kwargs)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **dataloader_kwargs)

    def _setup_training_components(self):
        """Set up optimizer, loss function, and learning rate scheduler."""
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-5)
        
        # Optimizer
        optimizer_name = self.config.get("optimizer", "adamw").lower()
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:  # Default to AdamW
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Loss function
        loss_type = self.config.get("loss_function", "mse").lower()
        if loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif loss_type == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            factor=self.config.get("gamma", 0.5),
            patience=self.config.get("lr_patience", 5),
            min_lr=self.config.get("min_lr", 1e-7),
            verbose=False
        )
        
        # Gradient clipping
        self.gradient_clip_val = self.config.get("gradient_clip_val", 1.0)

    def save_model(self, path, metadata=None):
        """Save model state and metadata to disk."""
        save_dict = {
            'state_dict': self.model.state_dict(),
            'config': self.config,
        }
        
        if metadata:
            save_dict.update(metadata)
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_dict, path)

    def train(self, num_epochs=None, early_stopping_patience=None, min_delta=None):
        """Train model with early stopping."""
        # Get parameters from config if not provided
        num_epochs = num_epochs or self.config.get("epochs", 100)
        early_stopping_patience = early_stopping_patience or self.config.get("early_stopping_patience", 10)
        min_delta = min_delta or self.config.get("min_delta", 1e-6)
        
        logger.info(f"Starting training for {num_epochs} epochs (patience={early_stopping_patience})")
        
        # Initialize tracking variables
        self.best_val_loss = float("inf")
        self.best_state = None
        self.best_epoch = -1
        patience_counter = 0
        completed_epochs = 0
        
        # Create training log file
        log_path = self.save_path / "training_log.csv"
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,lr,time\n")
        
        # Print header for training progress
        logger.info("-" * 80)
        logger.info(f"{'Epoch':^8} | {'Train Loss':^18} | {'Val Loss':^18} | {'LR':^10} | {'Time':^6}")
        logger.info("-" * 80)
        start_time = time.time()
        
        try:
            for epoch in range(num_epochs):
                epoch_start = time.time()
                completed_epochs = epoch + 1
                
                # Train epoch
                train_loss = self._train_epoch()
                
                # Validation phase
                val_loss = self.evaluate()
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Calculate metrics
                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]["lr"]
                improved = val_loss < (self.best_val_loss - min_delta)
                
                # Print progress line
                logger.info(f"{epoch+1:^8d} | {train_loss:^18.3e} | {val_loss:^18.3e} | {current_lr:^10.3e} | {epoch_time:^6.1f}")
                
                # Save logs to CSV
                with open(log_path, "a") as f:
                    f.write(f"{epoch+1},{train_loss:.6e},{val_loss:.6e},{current_lr:.6e},{epoch_time:.2f}\n")
                
                # Check for improvement
                if improved:
                    self.best_val_loss = val_loss
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self.best_epoch = epoch
                    patience_counter = 0
                    
                    # Save checkpoint for best model
                    self.save_model(str(self.save_path / "best_model.pt"), 
                                    metadata={'val_loss': val_loss, 'epoch': epoch})
                else:
                    patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping triggered after epoch {epoch+1}")
                        break
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # Ensure best model is restored
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)
                
                # Save final model
                final_model_path = str(self.save_path / "final_model.pt")
                self.save_model(
                    final_model_path,
                    metadata={
                        'val_loss': self.best_val_loss,
                        'best_epoch': self.best_epoch,
                        'total_epochs': completed_epochs
                    }
                )
                
                print(f"\nTraining completed. Best model from epoch {self.best_epoch + 1}")
                print(f"Best validation loss: {self.best_val_loss:.6e}")
                
                return final_model_path
            else:
                logger.warning("No best model was found during training")
                return None

    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Extract inputs, targets and coordinates from batch
                if len(batch) == 3:
                    inputs, targets, coordinates = batch
                    coordinates = coordinates.to(device=self.device, dtype=torch.float32)
                else:
                    inputs, targets = batch
                    coordinates = None
                
                # Move data to device
                inputs = inputs.to(device=self.device, dtype=torch.float32)
                targets = targets.to(device=self.device, dtype=torch.float32)
                
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast(device_type=self.device.type, enabled=True):
                        outputs = self.model(inputs, coordinates)
                        loss = self.criterion(outputs, targets)
                    
                    # Skip batch if loss is NaN
                    if not torch.isfinite(loss).item():
                        continue
                        
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.gradient_clip_val
                        )
                    
                    # Update weights with gradient rescaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                
                else:
                    # Standard precision training
                    try:
                        outputs = self.model(inputs, coordinates)
                        loss = self.criterion(outputs, targets)
                        
                        # Skip batch if loss is NaN
                        if not torch.isfinite(loss).item():
                            continue
                            
                        loss.backward()
                        
                        # Gradient clipping
                        if self.gradient_clip_val > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=self.gradient_clip_val
                            )
                        
                        self.optimizer.step()
                    except RuntimeError as e:
                        # Clear CUDA cache if applicable
                        if self.device.type == 'cuda' and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Skip this batch
                        continue
                
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                logger.warning(f"Error in batch processing: {str(e)}")
                continue  # Skip this batch but continue training
                    
        # Return average loss
        return total_loss / max(1, batch_count)

    @torch.no_grad()
    def evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(self.val_loader):
            try:
                # Extract inputs, targets and coordinates from batch
                if len(batch) == 3:
                    inputs, targets, coordinates = batch
                    coordinates = coordinates.to(device=self.device, dtype=torch.float32)
                else:
                    inputs, targets = batch
                    coordinates = None
                
                # Move data to device
                inputs = inputs.to(device=self.device, dtype=torch.float32)
                targets = targets.to(device=self.device, dtype=torch.float32)
                
                try:
                    # Forward pass
                    if self.use_amp and self.device.type == 'cuda':
                        with torch.amp.autocast(device_type=self.device.type, enabled=True):
                            outputs = self.model(inputs, coordinates)
                            loss = self.criterion(outputs, targets)
                    else:
                        outputs = self.model(inputs, coordinates)
                        loss = self.criterion(outputs, targets)
                    
                    # Skip batch if loss is NaN
                    if not torch.isfinite(loss).item():
                        continue
                        
                    total_loss += loss.item()
                    batch_count += 1
                except RuntimeError:
                    if self.device.type == 'cuda' and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
            except Exception:
                continue
        
        # Return average loss
        return total_loss / max(1, batch_count)

    @torch.no_grad()
    def test(self):
        """Evaluate model on test set."""
        logger.info("Starting test evaluation")
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        for batch in self.test_loader:
            try:
                # Extract inputs, targets and coordinates from batch
                if len(batch) == 3:
                    inputs, targets, coordinates = batch
                    coordinates = coordinates.to(device=self.device, dtype=torch.float32)
                else:
                    inputs, targets = batch
                    coordinates = None
                
                # Move data to device
                inputs = inputs.to(device=self.device, dtype=torch.float32)
                targets = targets.to(device=self.device, dtype=torch.float32)
                
                try:
                    # Forward pass
                    if self.use_amp and self.device.type == 'cuda':
                        with torch.amp.autocast(device_type=self.device.type, enabled=True):
                            outputs = self.model(inputs, coordinates)
                            loss = self.criterion(outputs, targets)
                    else:
                        outputs = self.model(inputs, coordinates)
                        loss = self.criterion(outputs, targets)
                    
                    # Skip batch if loss is NaN
                    if not torch.isfinite(loss).item():
                        continue
                        
                    total_loss += loss.item()
                    batch_count += 1
                except RuntimeError:
                    if self.device.type == 'cuda' and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                    
            except Exception:
                continue
        
        # Return test loss
        test_loss = total_loss / max(1, batch_count)
        print(f"\nTest Loss: {test_loss:.3e}")
        return test_loss