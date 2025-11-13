"""
Training script for Position-based Network

Trains the model to predict next move given current position and grid state.
This is more aligned with step-by-step decision making.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime

from position_net import (
    PositionNetwork,
    PositionDataset,
    collate_position_batch,
    compute_position_accuracy,
    position_network_loss
)


class PositionNetTrainer:
    """Trainer for position-based network."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        learning_rate=1e-3,
        checkpoint_dir='model/checkpoints',
        log_dir='model/logs',
        use_amp=True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Mixed precision training scaler
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_top5': [],
            'val_loss': [],
            'val_acc': [],
            'val_top5': []
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_top5 = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for inputs, targets, valid_masks in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            valid_masks = valid_masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits, _ = self.model(inputs, valid_masks)
                    loss = position_network_loss(logits, targets, valid_masks)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                logits, _ = self.model(inputs, valid_masks)
                loss = position_network_loss(logits, targets, valid_masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Metrics
            acc, top5 = compute_position_accuracy(logits, targets, valid_masks)
            total_loss += loss.item()
            total_acc += acc
            total_top5 += top5
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}',
                'top5': f'{top5:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        avg_top5 = total_top5 / num_batches
        
        return avg_loss, avg_acc, avg_top5
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_top5 = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for inputs, targets, valid_masks in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                valid_masks = valid_masks.to(self.device)
                
                # Forward pass
                logits, _ = self.model(inputs, valid_masks)
                
                # Compute loss
                loss = position_network_loss(logits, targets, valid_masks)
                
                # Metrics
                acc, top5 = compute_position_accuracy(logits, targets, valid_masks)
                total_loss += loss.item()
                total_acc += acc
                total_top5 += top5
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc:.4f}',
                    'top5': f'{top5:.4f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        avg_top5 = total_top5 / num_batches
        
        return avg_loss, avg_acc, avg_top5
    
    def train(self, num_epochs):
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
        """
        print(f"\nStarting position-based training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision (AMP): {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print("=" * 70)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc, train_top5 = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_top5 = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_top5'].append(train_top5)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_top5'].append(val_top5)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Top-5: {train_top5:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | Top-5: {val_top5:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("=" * 70)
            
            # Save checkpoint if best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            # Save regular checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Save final model
        self.save_checkpoint(num_epochs, is_best=False, final=True)
        
        # Plot training curves
        self.plot_training_curves()
        
        print("\nTraining completed!")
    
    def save_checkpoint(self, epoch, is_best=False, final=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        if final:
            path = os.path.join(self.checkpoint_dir, 'position_final.pt')
        elif is_best:
            path = os.path.join(self.checkpoint_dir, 'position_best.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'position_epoch_{epoch}.pt')
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded from: {checkpoint_path}")
        return checkpoint['epoch']
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        ax2.plot(epochs, self.history['train_top5'], 'b--', alpha=0.5, label='Train Top-5')
        ax2.plot(epochs, self.history['val_top5'], 'r--', alpha=0.5, label='Val Top-5')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, f'position_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Training curves saved: {plot_path}")
        plt.close()


def main():
    """Main training function."""
    
    # Configuration
    config = {
        'data_path': 'gym/output/datasets/position_dataset.jsonl',
        'batch_size': 256,  # More samples per puzzle, can use larger batches
        'val_split': 0.1,
        'num_epochs': 2,
        'learning_rate': 1e-4,
        'hidden_dim': 1024,
        'num_layers': 4,
        'use_transformer': True,  # Transformer vs LSTM
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_amp': True,
        'checkpoint_dir': 'model/checkpoints',
        'log_dir': 'model/logs',
        'seed': 42
    }
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    print("=" * 70)
    print("POSITION-BASED NETWORK TRAINING - STEP-BY-STEP DECISIONS")
    print("=" * 70)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading dataset...")
    full_dataset = PositionDataset(config['data_path'])
    
    # Split into train/val
    val_size = int(len(full_dataset) * config['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_position_batch,
        num_workers=8,
        pin_memory=True if config['device'] == 'cuda' else False,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_position_batch,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False,
        persistent_workers=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = PositionNetwork(
        input_dim=9,  # [x, y, waypoint, wall_up, wall_down, wall_left, wall_right, is_visited, is_current]
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        use_transformer=config['use_transformer']
    )
    
    # Create trainer
    trainer = PositionNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        checkpoint_dir=config['checkpoint_dir'],
        log_dir=config['log_dir'],
        use_amp=config.get('use_amp', True)
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])
    
    print("\nTraining complete! Best validation loss: {:.4f}".format(trainer.best_val_loss))
    print(f"Best validation accuracy: {max(trainer.history['val_acc']):.4f}")
    print(f"Best validation top-5: {max(trainer.history['val_top5']):.4f}")


if __name__ == '__main__':
    main()
