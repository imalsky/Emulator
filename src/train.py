#!/usr/bin/env python3
"""
train.py - Model training for multi-source transformer architecture

Implements training, validation, and testing for models with separate encoders
for different data types (global features, atmospheric profiles, and spectral sequences).
"""

import time
import json
import logging
import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from model import create_prediction_model
from hardware import get_device_type, configure_dataloader_settings

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trainer for multi-source transformer models.
    
    Handles dataset splitting, model training with early stopping,
    and evaluation on validation and test sets.
    """
    
    def __init__(self, config, device, save_path, dataset, 
                 test_size=0.15, val_size=0.15, collate_fn=None):
        """
        Initialize the trainer with configuration and dataset.
        
        Parameters
        ----------
        config : dict
            Model and training configuration
        device : torch.device
            Device to train on (CPU/GPU)
        save_path : str or Path
            Directory to save model checkpoints
        dataset : torch.utils.data.Dataset
            Dataset to train on
        test_size : float
            Fraction of dataset to use for testing
        val_size : float
            Fraction of dataset to use for validation
        collate_fn : callable, optional
            Custom batch collation function for multi-source data
        """
        self.config = config
        self.device = device
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.collate_fn = collate_fn
        
        # Store original dataset for file path access
        self.original_dataset = dataset
        
        logger.info(f"Initializing ModelTrainer with device: {device}")
        
        # Configure mixed precision training
        device_type = get_device_type()
        self.use_amp = (config.get("use_amp", False) and device_type == 'cuda')
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        logger.info(f"Mixed precision training enabled: {self.use_amp}")
        
        # Initialize datasets and model
        self._prepare_datasets(dataset, test_size, val_size)
        self._create_dataloaders()
        self._initialize_model()
        self._setup_training_components()
        
        # Initialize tracking variables
        self.best_val_loss = float("inf")
        self.best_state = None
        self.best_epoch = -1
        
        # Save test profile information
        self._save_test_profiles()

    def _initialize_model(self):
        """Initialize model with appropriate configuration."""
        logger.info("Creating prediction model with multi-source architecture")
        self.model = create_prediction_model(self.config)
        self.model = self.model.to(self.device)
        logger.info(f"Model created and moved to {self.device}")
    
    def _prepare_datasets(self, dataset, test_size, val_size):
        """Split dataset into training, validation, and test sets."""
        dataset_size = len(dataset)
        test_count = int(test_size * dataset_size)
        val_count = int(val_size * dataset_size)
        train_count = dataset_size - test_count - val_count
        
        # Generate indices for the split with fixed seed for reproducibility
        generator = torch.Generator().manual_seed(self.config.get("random_seed", 42))
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        
        # Store test indices for filename lookup
        self.test_indices = indices[:test_count]
        val_indices = indices[test_count:test_count+val_count]
        train_indices = indices[test_count+val_count:]
        
        # Create the subset datasets
        self.train_dataset = torch.utils.data.Subset(dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(dataset, self.test_indices)
        
        logger.info(f"Dataset split: {train_count} train, {val_count} validation, {test_count} test samples")

    def _create_dataloaders(self):
        """Create DataLoaders for training, validation, and testing."""
        # Use specified batch size without sequence length adjustments
        batch_size = self.config.get("batch_size", 16)
        
        # Get hardware-specific dataloader settings
        dataloader_settings = configure_dataloader_settings()
        num_workers = self.config.get("num_workers", dataloader_settings["num_workers"])
        
        dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": dataloader_settings["pin_memory"],
            "persistent_workers": dataloader_settings["persistent_workers"] and num_workers > 0,
            "collate_fn": self.collate_fn
        }
        
        # Create dataloaders
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, **dataloader_kwargs)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **dataloader_kwargs)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **dataloader_kwargs)

    def _setup_training_components(self):
        """Set up optimizer, loss function, and learning rate scheduler."""
        # Get hyperparameters from config
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-5)
        
        # Initialize optimizer
        optimizer_name = self.config.get("optimizer", "adamw").lower()
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:  # Default to AdamW
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Initialize loss function
        loss_type = self.config.get("loss_function", "mse").lower()
        if loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif loss_type == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()

        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            factor=self.config.get("gamma", 0.5),
            patience=self.config.get("lr_patience", 5),
            min_lr=self.config.get("min_lr", 1e-7),
        )
        
        # Gradient clipping value
        self.gradient_clip_val = self.config.get("gradient_clip_val", 1.0)

    def _find_base_dataset(self, dataset):
        """Recursively find the base dataset through potentially multiple Subset layers."""
        if hasattr(dataset, 'dataset'):
            return self._find_base_dataset(dataset.dataset)
        return dataset

    def _save_test_profiles(self):
        """Save the names of test profiles to a file."""
        try:
            # Find the base dataset (unwrap any Subset layers)
            base_dataset = self._find_base_dataset(self.original_dataset)
            test_profile_names = []
            
            # Try to get file paths from dataset attributes
            if hasattr(base_dataset, 'valid_files') and hasattr(self, 'test_indices'):
                test_profile_paths = [base_dataset.valid_files[i] for i in self.test_indices]
                test_profile_names = [p.name for p in test_profile_paths]
            elif hasattr(base_dataset, 'filenames') and hasattr(self, 'test_indices'):
                test_profile_names = [base_dataset.filenames[i] for i in self.test_indices]
            else:
                # Create placeholder names if necessary
                test_profile_names = [f"test_sample_{i}" for i in range(len(self.test_dataset))]
            
            # Save to file
            test_profiles_path = self.save_path / "test_profiles.json"
            with open(test_profiles_path, 'w') as f:
                json.dump(test_profile_names, f, indent=2)
            
            logger.info(f"Saved {len(test_profile_names)} test profile names")
        
        except Exception as e:
            logger.error(f"Error saving test profile names: {e}")

    def save_model(self, path, metadata=None):
        """
        Save model state and metadata to disk.
        
        Parameters
        ----------
        path : str
            Path to save the model
        metadata : dict, optional
            Additional metadata to save with the model
        """
        save_dict = {
            'state_dict': self.model.state_dict(),
            'config': self.config,
        }
        
        if metadata:
            save_dict.update(metadata)
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_dict, path)

    def train(self, num_epochs=None, early_stopping_patience=None, min_delta=None):
        """
        Train model with early stopping.
        
        Parameters
        ----------
        num_epochs : int, optional
            Maximum number of epochs to train for
        early_stopping_patience : int, optional
            Number of epochs with no improvement after which training will be stopped
        min_delta : float, optional
            Minimum change in validation loss to qualify as improvement
            
        Returns
        -------
        str or None
            Path to the saved model if training was successful, else None
        """
        # Get training parameters from config if not provided
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
        
        try:
            for epoch in range(num_epochs):
                epoch_start = time.time()
                completed_epochs = epoch + 1
                
                # Train and evaluate for this epoch
                train_loss = self._train_epoch()
                val_loss = self.evaluate()
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Calculate metrics and log progress
                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]["lr"]
                improved = val_loss < (self.best_val_loss - min_delta)
                
                logger.info(f"{epoch+1:^8d} | {train_loss:^18.3e} | {val_loss:^18.3e} | "
                         f"{current_lr:^10.3e} | {epoch_time:^6.1f}")
                
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
                        logger.info(f"Early stopping triggered after epoch {epoch+1}")
                        break
                        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            
        finally:
            # Ensure best model is restored and saved
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
                
                logger.info(f"Training completed. Best model from epoch {self.best_epoch + 1}")
                logger.info(f"Best validation loss: {self.best_val_loss:.6e}")
                
                return final_model_path
            else:
                logger.warning("No best model was found during training")
                return None

    def _train_epoch(self):
        """
        Train for one epoch.
        
        Returns
        -------
        float
            Average loss for this epoch
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Extract inputs and targets
                inputs, targets = batch
                
                # Move data to device
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device, dtype=torch.float32) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device, dtype=torch.float32)
                
                targets = targets.to(self.device, dtype=torch.float32)
                
                # Reset gradients
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    # Mixed precision training
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    
                    # Skip batch if loss is not finite
                    if not torch.isfinite(loss):
                        logger.warning(f"Batch {batch_idx}: Non-finite loss, skipping")
                        continue
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.gradient_clip_val
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard training
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    # Skip batch if loss is not finite
                    if not torch.isfinite(loss):
                        logger.warning(f"Batch {batch_idx}: Non-finite loss, skipping")
                        continue
                    
                    # Backward pass and optimization
                    loss.backward()
                    
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.gradient_clip_val
                        )
                    
                    self.optimizer.step()
                
                # Accumulate statistics
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Return average loss
        if batch_count == 0:
            logger.error("No batches were successfully processed")
            return float('inf')
            
        return total_loss / batch_count

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate model on validation set.
        
        Returns
        -------
        float
            Average validation loss
        """
        return self._evaluate_on_loader(self.val_loader, "validation")

    @torch.no_grad()
    def test(self):
        """
        Evaluate model on test set.
        
        Returns
        -------
        float
            Average test loss
        """
        test_loss = self._evaluate_on_loader(self.test_loader, "test")
        logger.info(f"Test Loss: {test_loss:.3e}")
        return test_loss

    @torch.no_grad()
    def _evaluate_on_loader(self, data_loader, phase_name):
        """
        Evaluate model on the specified data loader.
        
        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Data loader to evaluate on
        phase_name : str
            Name of evaluation phase for logging
            
        Returns
        -------
        float
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(data_loader):
            try:
                # Extract inputs and targets
                inputs, targets = batch
                
                # Move data to device
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device, dtype=torch.float32) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device, dtype=torch.float32)
                
                targets = targets.to(self.device, dtype=torch.float32)
                
                # Forward pass
                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Skip batch if loss is not finite
                if not torch.isfinite(loss):
                    continue
                    
                # Accumulate statistics
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                logger.warning(f"{phase_name.capitalize()} error in batch {batch_idx}: {e}")
                continue
        
        # Return average loss
        if batch_count == 0:
            logger.error(f"No valid batches in {phase_name} evaluation")
            return float('inf')
            
        return total_loss / batch_count