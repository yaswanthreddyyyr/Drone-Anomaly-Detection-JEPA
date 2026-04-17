"""
JEPA-DRONE Training Module

Handles the training loop for the JEPA model, including:
- Learning rate scheduling
- EMA updates for target encoder
- Logging and checkpointing
- Early stopping
"""

import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .jepa import JEPA, create_jepa_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing learning rate scheduler with linear warmup.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class JEPATrainer:
    """
    Trainer class for JEPA model.
    
    Handles the complete training pipeline including:
    - Model initialization
    - Optimizer and scheduler setup
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        model: JEPA,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: str = "auto",
        output_dir: str = "outputs"
    ):
        """
        Initialize trainer.
        
        Args:
            model: JEPA model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            output_dir: Directory for checkpoints and logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        
        # Setup device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        if config:
            with open(self.run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
        
        # Training parameters
        training_config = self.config.get("training", {})
        self.epochs = training_config.get("epochs", 100)
        self.lr = training_config.get("learning_rate", 0.001)
        self.weight_decay = training_config.get("weight_decay", 0.0001)
        self.warmup_epochs = training_config.get("warmup_epochs", 5)
        
        # Setup optimizer (only context encoder and predictor)
        self.optimizer = optim.AdamW(
            [
                {"params": self.model.context_encoder.parameters()},
                {"params": self.model.predictor.parameters()}
            ],
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=self.warmup_epochs,
            max_epochs=self.epochs
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        
        # Logging
        self.log_file = open(self.run_dir / "training.log", "w")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs}")
        
        for batch in pbar:
            # Get features
            features = batch["features"].to(self.device)
            
            # Forward pass
            loss = self.model(features)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update target encoder (EMA)
            self.model.update_target_encoder()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            features = batch["features"].to(self.device)
            loss = self.model(features)
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        
        torch.save(checkpoint, self.run_dir / filename)
        logger.info(f"Saved checkpoint to {self.run_dir / filename}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self) -> Dict:
        """
        Full training loop.
        
        Returns:
            Training history dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info("Starting JEPA Training")
        logger.info(f"{'='*60}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Learning rate: {self.lr}")
        logger.info(f"Training samples: {len(self.train_loader.dataset):,}")
        if self.val_loader:
            logger.info(f"Validation samples: {len(self.val_loader.dataset):,}")
        logger.info(f"Output directory: {self.run_dir}")
        logger.info(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Logging
            epoch_time = time.time() - epoch_start
            log_msg = (
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )
            logger.info(log_msg)
            self.log_file.write(log_msg + "\n")
            self.log_file.flush()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"{'='*60}\n")
        
        # Close log file
        self.log_file.close()
        
        # Save training history
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "total_time_seconds": total_time
        }
        
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        return history


def train_jepa(
    config_path: str = "configs/config.yaml",
    data_dir: str = "processed_data",
    output_dir: str = "outputs",
    device: str = "auto",
    resume: Optional[str] = None
) -> Tuple[JEPA, Dict]:
    """
    Train JEPA model from configuration.
    
    Args:
        config_path: Path to configuration file
        data_dir: Path to processed data
        output_dir: Output directory for checkpoints
        device: Device to train on
        resume: Path to checkpoint to resume from
        
    Returns:
        Trained model and training history
    """
    import yaml
    from ..data.dataset import DroneChunkDataset
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = DroneChunkDataset(data_dir, split="train_normal", return_labels=False)
    val_dataset = DroneChunkDataset(data_dir, split="validation", return_labels=False)
    
    logger.info(f"Training samples: {len(train_dataset):,}")
    logger.info(f"Validation samples: {len(val_dataset):,}")
    
    # Create data loaders
    batch_size = config["training"]["batch_size"]
    num_workers = config.get("num_workers", 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating JEPA model...")
    model = create_jepa_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = JEPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=output_dir
    )
    
    # Resume from checkpoint if specified
    if resume:
        trainer.load_checkpoint(resume)
    
    # Train
    history = trainer.train()
    
    return model, history


# Export
__all__ = ["JEPATrainer", "CosineWarmupScheduler", "train_jepa"]
