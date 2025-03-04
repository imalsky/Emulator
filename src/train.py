#!/usr/bin/env python3
"""
model_trainer.py

Enhanced training infrastructure for sequence prediction models.
Features:
- Memory-efficient training for ultra-large sequences (40,000+ points)
- Automatic mixed precision training (disabled on MPS devices)
- Support for any coordinate variables
- Early stopping based on validation loss
- Test set evaluation
- Minimalist logging for clean training output
"""

import time
import json
import logging
import os
import traceback
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import numpy as np

logger = logging.getLogger(__name__)


def is_mps_device(device):
    """Check if device is Apple Silicon MPS."""
    return device.type == 'mps' or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())


class ModelTrainer:
    """Trainer for sequence prediction models optimized for ultra-large sequences."""
    
    def __init__(
        self,
        config,
        device,
        save_path,
        dataset,
        test_size=0.15,
        val_size=0.15
    ):
        """Initialize the trainer with configuration and dataset."""
        try:
            self.config = config
            self.device = device
            self.save_path = Path(save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Initializing ModelTrainer with config: {config.get('model_type', 'default')} model, output_seq_length: {config.get('output_seq_length', 'unknown')}")
            logger.info(f"Using device: {device}")
            
            # Set up gradient scaler for mixed precision training 
            # Define use_amp before model initialization as it might be needed by the model
            self.use_amp = config.get("use_amp", False) or config.get("use_mixed_precision", False)
            
            # Disable mixed precision on MPS devices - it's not fully supported
            if is_mps_device(device):
                logger.info("Mixed precision is not supported on MPS devices - disabling")
                self.use_amp = False
                
            self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
            logger.info(f"Mixed precision training enabled: {self.use_amp}")
            
            # Prepare datasets and dataloaders
            logger.info("Preparing datasets...")
            self._prepare_datasets(dataset, test_size, val_size)
            
            logger.info("Creating dataloaders...")
            self._create_dataloaders()
            
            # Initialize model
            logger.info("Initializing model...")
            try:
                from model import create_prediction_model
                self.model = create_prediction_model(config).to(device)
                logger.info(f"Model created successfully with {self.model.count_parameters():,} parameters")
            except Exception as e:
                logger.error(f"Failed to create model: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # Setup training components
            logger.info("Setting up training components...")
            self._setup_training_components()
            
            # Training state
            self.best_val_loss = float("inf")
            self.best_state = None
            self.best_epoch = -1
            
            logger.info(f"Model trainer initialized with {self.model.count_parameters():,} parameters")
            
            # Check if model supports coordinate variables
            if hasattr(self.model, 'use_coordinate') and self.model.use_coordinate:
                coord_vars = config.get("coordinate_variable", [])
                if coord_vars:
                    logger.info(f"Model is configured to use coordinate variable(s): {coord_vars}")
                else:
                    logger.warning("Model is configured to use coordinates but none are specified in the config")
                    
        except Exception as e:
            logger.error(f"Error during ModelTrainer initialization: {e}")
            logger.error(traceback.format_exc())
            raise

    def _prepare_datasets(self, dataset, test_size, val_size):
        """Split dataset into training, validation, and test sets."""
        try:
            # Calculate split sizes
            dataset_size = len(dataset)
            test_count = int(test_size * dataset_size)
            val_count = int(val_size * dataset_size)
            train_count = dataset_size - test_count - val_count

            logger.info(f"Splitting dataset of size {dataset_size} into {train_count} train, {val_count} validation, {test_count} test samples")

            # Split dataset with fixed random seed
            generator = torch.Generator().manual_seed(42)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, [train_count, val_count, test_count], generator=generator
            )
            
            logger.info(f"Dataset split: {train_count} train, {val_count} validation, {test_count} test samples")
        except Exception as e:
            logger.error(f"Error in _prepare_datasets: {e}")
            logger.error(traceback.format_exc())
            raise

    def _create_dataloaders(self):
        """Create DataLoaders for training, validation, and testing."""
        try:
            # Get batch size from config, with smaller default for very large sequences
            output_seq_length = self.config.get("output_seq_length", 0)
            
            # Choose batch size based on sequence length
            if output_seq_length > 40000:
                default_batch = 2  # Ultra-conservative for extremely large sequences
            elif output_seq_length > 20000:
                default_batch = 4  # Very conservative
            elif output_seq_length > 10000:
                default_batch = 16  # Conservative
            else:
                default_batch = 32  # Standard
                
            batch_size = self.config.get("batch_size", default_batch)
            
            # Number of workers (default to system CPU count or less)
            num_workers = self.config.get("num_workers", min(4, os.cpu_count() or 1))
            
            dataloader_kwargs = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": self.device.type == "cuda",
                "persistent_workers": num_workers > 0
            }
            
            logger.info(f"Creating dataloaders with batch_size={batch_size}, num_workers={num_workers}")
            
            self.train_loader = DataLoader(self.train_dataset, shuffle=True, **dataloader_kwargs)
            self.val_loader = DataLoader(self.val_dataset, shuffle=False, **dataloader_kwargs)
            self.test_loader = DataLoader(self.test_dataset, shuffle=False, **dataloader_kwargs)
            
            logger.info(f"Created dataloaders - train: {len(self.train_loader)} batches, val: {len(self.val_loader)} batches, test: {len(self.test_loader)} batches")
        except Exception as e:
            logger.error(f"Error in _create_dataloaders: {e}")
            logger.error(traceback.format_exc())
            raise

    def _setup_training_components(self):
        """Set up optimizer, loss function, and learning rate scheduler."""
        try:
            # Optimizer
            lr = self.config.get("learning_rate", 1e-4)
            weight_decay = self.config.get("weight_decay", 1e-5)
            
            # Choose optimizer based on config
            optimizer_name = self.config.get("optimizer", "adamw").lower()
            if optimizer_name == "adam":
                self.optimizer = optim.Adam(
                    self.model.parameters(), lr=lr, weight_decay=weight_decay
                )
            elif optimizer_name == "sgd":
                self.optimizer = optim.SGD(
                    self.model.parameters(), 
                    lr=lr, 
                    weight_decay=weight_decay,
                    momentum=0.9
                )
            else:  # Default to AdamW
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=lr, weight_decay=weight_decay
                )

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
                self.optimizer,
                mode='min',
                factor=self.config.get("gamma", 0.5),
                patience=self.config.get("lr_patience", 5),
                min_lr=self.config.get("min_lr", 1e-7),
                verbose=False  # Disable verbose LR updates
            )
            
            logger.info(f"Training setup: optimizer={optimizer_name}, loss={loss_type}, lr={lr:.2e}, use_amp={self.use_amp}")
            
            # Maximum gradient clipping value
            self.gradient_clip_val = self.config.get("gradient_clip_val", 1.0)
        except Exception as e:
            logger.error(f"Error in _setup_training_components: {e}")
            logger.error(traceback.format_exc())
            raise

    def save_model(self, path, metadata=None, verbose=False):
        """Save model state and metadata to disk."""
        try:
            save_dict = {
                'state_dict': self.model.state_dict(),
                'config': self.config,
            }
            
            if metadata:
                save_dict.update(metadata)
                
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(save_dict, path)
            if verbose:
                logger.info(f"Saved model to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {e}")
            logger.error(traceback.format_exc())
            raise

    def train(self, num_epochs=None, early_stopping_patience=None, min_delta=None):
        """Train model with early stopping and mixed precision for memory efficiency."""
        try:
            # Get parameters from config if not provided
            if num_epochs is None:
                num_epochs = self.config.get("epochs", 100)
                
            if early_stopping_patience is None:
                early_stopping_patience = self.config.get("early_stopping_patience", 10)
                
            if min_delta is None:
                min_delta = self.config.get("min_delta", 1e-6)
            
            logger.info(f"Starting training for {num_epochs} epochs (early_stopping_patience={early_stopping_patience}, min_delta={min_delta:.2e})")
            
            # Initialize tracking variables
            self.best_val_loss = float("inf")
            self.best_state = None
            self.best_epoch = -1
            patience_counter = 0
            
            # Create training log file
            log_path = self.save_path / "training_log.csv"
            with open(log_path, "w") as f:
                f.write("epoch,train_loss,val_loss,lr,time\n")
            
            # Print header for training progress
            print("\n{:^50}".format("TRAINING PROGRESS"))
            print("{:^50}".format("-" * 40))
            print(f"{'Epoch':^6}|{'Train Loss':^15}|{'Val Loss':^15}|{'LR':^8}|{'Time':^4}")
            print("{:^50}".format("-" * 40))
            
            start_time = time.time()
            
            try:
                # Check if num_epochs is valid
                if num_epochs <= 0:
                    logger.error(f"Invalid number of epochs: {num_epochs}, must be greater than 0")
                    return None
                    
                # Verify that dataloader yields batches
                try:
                    # Check if train loader has data
                    if len(self.train_loader) == 0:
                        logger.error("Training loader has no batches!")
                        return None
                    
                    # Try to get a sample batch
                    test_batch = next(iter(self.train_loader))
                    #logger.info(f"Sample batch shape: {[t.shape if isinstance(t, torch.Tensor) else type(t) for t in test_batch]}")
                except Exception as e:
                    logger.error(f"Error testing dataloader: {e}")
                    return None
                    
                #logger.info(f"Beginning training loop, {num_epochs} epochs")
                
                for epoch in range(num_epochs):
                    epoch_start = time.time()
                    
                    # Train epoch - no verbose output
                    try:
                        train_loss = self._train_epoch(verbose=False)
                    except Exception as e:
                        logger.error(f"Error during training phase in epoch {epoch+1}: {e}")
                        logger.error(traceback.format_exc())
                        raise
                    
                    # Validation phase - no verbose output
                    try:
                        val_loss = self.evaluate(verbose=False)
                    except Exception as e:
                        logger.error(f"Error during validation phase in epoch {epoch+1}: {e}")
                        logger.error(traceback.format_exc())
                        raise
                    
                    # Update learning rate scheduler
                    self.scheduler.step(val_loss)
                    
                    # Calculate metrics
                    epoch_time = time.time() - epoch_start
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    improved = val_loss < (self.best_val_loss - min_delta)
                    
                    # Print concise progress line
                    print(f"{epoch+1:6d}|{train_loss:.6e}|{val_loss:.6e}|{current_lr:.2e}|{epoch_time:4.1f}")
                    
                    # Save logs to CSV
                    with open(log_path, "a") as f:
                        f.write(f"{epoch+1},{train_loss:.6e},{val_loss:.6e},{current_lr:.6e},{epoch_time:.2f}\n")
                    
                    # Check for improvement
                    if improved:
                        self.best_val_loss = val_loss
                        self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        self.best_epoch = epoch
                        patience_counter = 0
                        
                        # Save checkpoint for best model (no verbose output)
                        self.save_model(
                            str(self.save_path / "best_model.pt"),
                            metadata={'val_loss': val_loss, 'epoch': epoch},
                            verbose=False
                        )
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
                    logger.info(f"Restoring best model from epoch {self.best_epoch + 1}")
                    self.model.load_state_dict(self.best_state)
                    
                    # Save final model
                    final_model_path = str(self.save_path / "final_model.pt")
                    self.save_model(
                        final_model_path,
                        metadata={
                            'val_loss': self.best_val_loss,
                            'best_epoch': self.best_epoch,
                            'total_epochs': epoch + 1,
                            'training_time': time.time() - start_time
                        },
                        verbose=True
                    )
                    
                    # Print summary
                    total_time = time.time() - start_time
                    print("\n{:^50}".format("-" * 40))
                    print(f"Training completed in {total_time:.2f} seconds")
                    print(f"Best model from epoch {self.best_epoch + 1}")
                    print(f"Best validation loss: {self.best_val_loss:.6e}")
                    print("{:^50}".format("-" * 40))
                    
                    return final_model_path
                else:
                    logger.warning("No best model was found during training")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in train method: {e}")
            logger.error(traceback.format_exc())
            return None

    def _train_epoch(self, verbose=False):
        """Train for one epoch with minimal logging."""
        try:
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
                    
                    # Forward pass with mixed precision if enabled (not on MPS)
                    if self.use_amp and self.scaler is not None and not is_mps_device(self.device):
                        with torch.amp.autocast(device_type='cuda', enabled=True):
                            # Forward pass depends on whether model uses coordinates
                            outputs = self.model(inputs, coordinates)
                            loss = self.criterion(outputs, targets)
                        
                        # Skip batch if loss is NaN
                        if not torch.isfinite(loss).item():
                            if verbose:
                                logger.warning(f"NaN loss detected in batch {batch_idx}, skipping batch")
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
                        outputs = self.model(inputs, coordinates)
                        loss = self.criterion(outputs, targets)
                        
                        # Skip batch if loss is NaN
                        if not torch.isfinite(loss).item():
                            if verbose:
                                logger.warning(f"NaN loss detected in batch {batch_idx}, skipping batch")
                            continue
                            
                        loss.backward()
                        
                        # Gradient clipping
                        if self.gradient_clip_val > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=self.gradient_clip_val
                            )
                        
                        self.optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    logger.error(traceback.format_exc())
                    continue  # Skip this batch but continue training
            
            # Return average loss
            if batch_count == 0:
                logger.error("No batches were processed successfully in this epoch!")
                return float('inf')
                
            avg_loss = total_loss / batch_count
            return avg_loss
            
        except Exception as e:
            logger.error(f"Error in _train_epoch: {e}")
            logger.error(traceback.format_exc())
            raise

    @torch.no_grad()
    def evaluate(self, verbose=False):
        """Evaluate model on validation set with minimal logging."""
        try:
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
                    
                    # Forward pass (regular precision for MPS devices)
                    if self.use_amp and not is_mps_device(self.device) and self.scaler is not None:
                        with torch.amp.autocast(device_type='cuda', enabled=True):
                            outputs = self.model(inputs, coordinates)
                            loss = self.criterion(outputs, targets)
                    else:
                        outputs = self.model(inputs, coordinates)
                        loss = self.criterion(outputs, targets)
                    
                    # Skip batch if loss is NaN
                    if not torch.isfinite(loss).item():
                        if verbose:
                            logger.warning(f"NaN loss detected in validation batch {batch_idx}, skipping batch")
                        continue
                        
                    total_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing validation batch {batch_idx}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            # Return average loss
            if batch_count == 0:
                logger.error("No validation batches were processed successfully!")
                return float('inf')
                
            avg_loss = total_loss / batch_count
            return avg_loss
            
        except Exception as e:
            logger.error(f"Error in evaluate: {e}")
            logger.error(traceback.format_exc())
            raise

    @torch.no_grad()
    def test(self, save_predictions=True):
        """Evaluate model on test set with enhanced metrics."""
        try:
            logger.info("Starting test evaluation")
            self.model.eval()
            total_loss = 0.0
            batch_count = 0
            
            # Collect predictions for detailed analysis
            all_preds = []
            all_targets = []
            test_coordinates = []  # Save coordinates for visualization
            
            for batch_idx, batch in enumerate(self.test_loader):
                try:
                    # Extract inputs, targets and coordinates from batch
                    if len(batch) == 3:
                        inputs, targets, coordinates = batch
                        coordinates = coordinates.to(device=self.device, dtype=torch.float32)
                        # Save coordinates for visualization if needed
                        if save_predictions and len(test_coordinates) == 0:
                            test_coordinates = coordinates[0].cpu().numpy()  # Just save the first batch's coordinates
                    else:
                        inputs, targets = batch
                        coordinates = None
                    
                    # Move data to device
                    inputs = inputs.to(device=self.device, dtype=torch.float32)
                    targets = targets.to(device=self.device, dtype=torch.float32)
                    
                    # Forward pass - disable mixed precision on MPS
                    if self.use_amp and not is_mps_device(self.device) and self.scaler is not None:
                        with torch.amp.autocast(device_type='cuda', enabled=True):
                            outputs = self.model(inputs, coordinates)
                            loss = self.criterion(outputs, targets)
                    else:
                        outputs = self.model(inputs, coordinates)
                        loss = self.criterion(outputs, targets)
                    
                    # Skip batch if loss is NaN
                    if not torch.isfinite(loss).item():
                        logger.warning(f"NaN loss detected in test batch {batch_idx}, skipping batch")
                        continue
                        
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Collect predictions and targets for analysis
                    if save_predictions:
                        all_preds.append(outputs.cpu().numpy())
                        all_targets.append(targets.cpu().numpy())
                        
                except Exception as e:
                    logger.error(f"Error processing test batch {batch_idx}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            # If no batches were processed successfully
            if batch_count == 0:
                logger.error("No test batches were processed successfully!")
                return {"test_loss": float('inf')}
                
            test_loss = total_loss / batch_count
            metrics = {"test_loss": test_loss}
            
            # Detailed metrics if predictions were collected
            if save_predictions and all_preds:
                try:
                    logger.info("Calculating detailed metrics from predictions")
                    # Concatenate predictions and targets
                    predictions = np.concatenate(all_preds, axis=0)
                    targets = np.concatenate(all_targets, axis=0)
                    
                    logger.info(f"Collected predictions shape: {predictions.shape}, targets shape: {targets.shape}")
                    
                    # Calculate additional metrics
                    try:
                        from utils import calculate_metrics
                        additional_metrics = calculate_metrics(predictions, targets)
                        metrics.update(additional_metrics)
                        logger.info(f"Additional metrics calculated: {list(additional_metrics.keys())}")
                    except Exception as e:
                        logger.error(f"Error calculating metrics: {e}")
                        logger.error(traceback.format_exc())
                    
                    # Save a sample prediction
                    if len(predictions) > 0:
                        sample_idx = 0
                        sample_pred = predictions[sample_idx]
                        sample_target = targets[sample_idx]
                        
                        # Save numpy arrays
                        np.save(self.save_path / "sample_prediction.npy", sample_pred)
                        np.save(self.save_path / "sample_target.npy", sample_target)
                        logger.info(f"Saved sample prediction and target arrays")
                        
                        # Save coordinates if available
                        if len(test_coordinates) > 0:
                            np.save(self.save_path / "sample_coordinates.npy", test_coordinates)
                            logger.info(f"Saved coordinate array of shape {test_coordinates.shape}")
                        
                        # Try to visualize if matplotlib is available
                        try:
                            from utils import plot_comparison
                            
                            logger.info("Generating prediction vs target plot")
                            # Use coordinates for x-axis if available
                            if len(test_coordinates) > 0:
                                fig = plot_comparison(
                                    sample_pred, sample_target,
                                    x_values=test_coordinates,
                                    title="Sample Prediction vs Target",
                                    save_path=str(self.save_path / "sample_prediction.png")
                                )
                            else:
                                fig = plot_comparison(
                                    sample_pred, sample_target,
                                    title="Sample Prediction vs Target",
                                    save_path=str(self.save_path / "sample_prediction.png")
                                )
                            logger.info("Plot saved successfully")
                        except Exception as e:
                            logger.warning(f"Visualization failed: {e}")
                            logger.warning(traceback.format_exc())
                
                except Exception as e:
                    logger.error(f"Error processing test predictions: {e}")
                    logger.error(traceback.format_exc())
            
            # Save test metrics with proper formatting
            try:
                metrics_path = self.save_path / "test_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2, sort_keys=True)
                logger.info(f"Test metrics saved to {metrics_path}")
            except Exception as e:
                logger.error(f"Error saving test metrics: {e}")
            
            # Print test results in a nice format
            print("\n{:^50}".format("TEST RESULTS"))
            print("{:^50}".format("-" * 40))
            print(f"Test Loss: {test_loss:.6e}")
            
            for k, v in metrics.items():
                if k != "test_loss":
                    print(f"  {k}: {v:.6e}")
            print("{:^50}".format("-" * 40))
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Error in test method: {e}")
            logger.error(traceback.format_exc())
            return {"test_loss": float('inf'), "error": str(e)}