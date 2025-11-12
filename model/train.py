"""
Training script for Pointer Network on Hamiltonian Path Puzzles

Implements supervised learning with:
- Variable grid sizes and checkpoint counts
- Dynamic batching for different sequence lengths
- Learning rate scheduling
- Model checkpointing
- Training/validation monitoring
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

from model.ptrnet import (
    PointerNetwork,
    HamiltonianPuzzleDataset,
    collate_fn,
    compute_accuracy,
    pointer_network_loss
)


class PtrNetTrainer:
    """Trainer for Pointer Network model."""
    
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
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Mixed precision training scaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch, teacher_forcing_ratio=0.5):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for inputs, targets, lengths in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            lengths = lengths.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pointers, _ = self.model(inputs, targets, teacher_forcing_ratio)
                    loss = pointer_network_loss(pointers, targets, lengths)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                pointers, _ = self.model(inputs, targets, teacher_forcing_ratio)
                loss = pointer_network_loss(pointers, targets, lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
            
            # Metrics
            acc = compute_accuracy(pointers, targets, lengths)
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        return avg_loss, avg_acc
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for inputs, targets, lengths in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)
                
                # Forward pass (no teacher forcing during validation)
                pointers, _ = self.model(inputs, targets, teacher_forcing_ratio=0.0)
                
                # Compute loss
                loss = pointer_network_loss(pointers, targets, lengths)
                
                # Metrics
                acc = compute_accuracy(pointers, targets, lengths)
                total_loss += loss.item()
                total_acc += acc
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        return avg_loss, avg_acc
    
    def train(self, num_epochs, teacher_forcing_schedule=None):
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            teacher_forcing_schedule: Function(epoch) -> ratio, or None for constant 0.5
        """
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision (AMP): {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print("=" * 70)
        
        for epoch in range(1, num_epochs + 1):
            # Determine teacher forcing ratio
            if teacher_forcing_schedule:
                tf_ratio = teacher_forcing_schedule(epoch)
            else:
                tf_ratio = max(0.5 - (epoch - 1) * 0.02, 0.0)  # Decay from 0.5 to 0
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch, tf_ratio)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  Teacher Forcing Ratio: {tf_ratio:.2f}")
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
            path = os.path.join(self.checkpoint_dir, 'final_model.pt')
        elif is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        
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
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Training curves saved: {plot_path}")
        plt.close()


def main():
    """Main training function."""
    
    # Configuration
    config = {
        'data_path': 'gym/output/datasets/ptrnet_dataset.jsonl',
        'batch_size': 64,  # Increased for RTX 3090 Ti (24GB VRAM)
        'val_split': 0.1,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_amp': True,  # Mixed precision training for 2x speedup
        'checkpoint_dir': 'model/checkpoints',
        'log_dir': 'model/logs',
        'seed': 42
    }
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    print("=" * 70)
    print("POINTER NETWORK TRAINING - HAMILTONIAN PATH PUZZLES")
    print("=" * 70)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading dataset...")
    full_dataset = HamiltonianPuzzleDataset(config['data_path'])
    
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
        collate_fn=collate_fn,
        num_workers=8,  # Increased for better CPU utilization
        pin_memory=True if config['device'] == 'cuda' else False,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False,
        persistent_workers=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = PointerNetwork(
        input_dim=8,  # [x, y, waypoint_type, wall_up, wall_down, wall_left, wall_right, is_visited]
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Create trainer
    trainer = PtrNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        checkpoint_dir=config['checkpoint_dir'],
        log_dir=config['log_dir'],
        use_amp=config.get('use_amp', True)  # Enable mixed precision
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])
    
    print("\nTraining complete! Best validation loss: {:.4f}".format(trainer.best_val_loss))


if __name__ == '__main__':
    main()
