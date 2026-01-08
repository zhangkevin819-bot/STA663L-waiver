"""
Advanced training loop with mixed precision, gradient accumulation,
and sophisticated learning rate scheduling.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging import MetricsLogger, log_execution_time, logger


class WarmupCosineScheduler(LambdaLR):
    """Cosine learning rate schedule with linear warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        
        super().__init__(optimizer, self.lr_lambda)
    
    def lr_lambda(self, step: int) -> float:
        """Calculate learning rate multiplier."""
        if step < self.warmup_steps:
            # Linear warmup
            return step / self.warmup_steps
        
        # Cosine decay
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs")
                self.should_stop = True
                return True
        
        return False


class Trainer:
    """Comprehensive training orchestrator with modern techniques."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        accumulation_steps: int = 1
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.mixed_precision = mixed_precision
        
        # Metrics tracking
        self.metrics_logger = MetricsLogger("training")
        self.global_step = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        scheduler: Optional[LambdaLR] = None
    ) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
                
                self.global_step += 1
            
            # Track metrics
            epoch_loss += loss.item() * self.accumulation_steps
            predictions = outputs.argmax(dim=1)
            epoch_correct += (predictions == targets).sum().item()
            epoch_total += targets.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{epoch_loss / (batch_idx + 1):.4f}",
                'acc': f"{100 * epoch_correct / epoch_total:.2f}%"
            })
        
        metrics = {
            'train_loss': epoch_loss / len(train_loader),
            'train_accuracy': epoch_correct / epoch_total
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            val_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == targets).sum().item()
            val_total += targets.size(0)
        
        metrics = {
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_correct / val_total
        }
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        scheduler: Optional[LambdaLR] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_dir: Optional[Path] = None
    ) -> dict[str, list[float]]:
        """Complete training loop."""
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            with log_execution_time(f"Training epoch {epoch + 1}"):
                train_metrics = self.train_epoch(train_loader, scheduler)
            
            # Validation
            with log_execution_time(f"Validation epoch {epoch + 1}"):
                val_metrics = self.validate(val_loader)
            
            # Update history
            for key, value in {**train_metrics, **val_metrics}.items():
                history[key].append(value)
            
            # Log metrics
            self.metrics_logger.log_metrics(
                {**train_metrics, **val_metrics},
                step=epoch
            )
            
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                if checkpoint_dir:
                    self.save_checkpoint(
                        checkpoint_dir / "best_model.pt",
                        epoch,
                        val_metrics
                    )
            
            # Early stopping
            if early_stopping:
                if early_stopping(val_metrics['val_loss']):
                    logger.info("Early stopping triggered")
                    break
        
        return history
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict[str, float]
    ) -> None:
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path) -> dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
