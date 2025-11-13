"""
Position-based Network for Step-by-Step Decision Making

Instead of sequence-to-sequence, this network:
- Input: Current grid state with position and visited cells (N cells × 9 features)
- Output: Probability distribution over next cell to move to (N cells)

This is more aligned with how an agent would actually solve the puzzle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Tuple, Dict, Any


class PositionNetwork(nn.Module):
    """
    Position-based network for predicting next move.
    
    Architecture:
    - Grid encoder: Processes all cells to understand the environment
    - Attention mechanism: Focuses on current position and viable moves
    - Action head: Outputs probability over which cell to move to next
    
    Args:
        input_dim: Dimension of input features (9: x, y, waypoint, 4 walls, is_visited, is_current)
        hidden_dim: Hidden dimension for processing
        num_layers: Number of transformer/LSTM layers
    """
    
    def __init__(self, input_dim=9, hidden_dim=1024, num_layers=4, use_transformer=True):
        super(PositionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_transformer = use_transformer
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        if use_transformer:
            # Transformer encoder for understanding grid structure
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                activation='gelu'
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            # LSTM alternative
            self.encoder = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Action head - predicts which cell to move to
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, inputs, valid_mask=None):
        """
        Forward pass.
        
        Args:
            inputs: (batch_size, num_cells, input_dim) - grid cell features
            valid_mask: (batch_size, num_cells) - mask for valid (unvisited) cells
            
        Returns:
            logits: (batch_size, num_cells) - unnormalized scores for each cell
            probs: (batch_size, num_cells) - probability distribution over cells
        """
        batch_size = inputs.size(0)
        num_cells = inputs.size(1)
        
        # Embed all cells
        embedded = self.input_embedding(inputs)  # (B, N, H)
        
        # Encode grid structure
        if self.use_transformer:
            encoded = self.encoder(embedded)  # (B, N, H)
        else:
            encoded, _ = self.encoder(embedded)  # (B, N, 2H)
            encoded = self.projection(encoded)  # (B, N, H)
        
        # Compute scores for each cell
        logits = self.action_head(encoded).squeeze(-1)  # (B, N)
        
        # Mask invalid cells (already visited or unreachable)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float('-inf'))
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)  # (B, N)
        
        return logits, probs
    
    def predict_next_move(self, inputs, valid_mask=None, temperature=1.0):
        """
        Predict next move (for inference).
        
        Args:
            inputs: (1, num_cells, input_dim) - single sample
            valid_mask: (1, num_cells) - mask for valid cells
            temperature: Sampling temperature (1.0 = normal, <1 = more confident)
            
        Returns:
            next_cell_idx: Index of next cell to move to
            confidence: Probability of selected cell
        """
        self.eval()
        with torch.no_grad():
            logits, probs = self.forward(inputs, valid_mask)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
            
            # Select cell with highest probability
            next_cell_idx = probs.argmax(dim=-1).item()
            confidence = probs[0, next_cell_idx].item()
            
            return next_cell_idx, confidence


class PositionDataset(Dataset):
    """
    Dataset for position-based training.
    Each sample represents one decision point.
    """
    
    def __init__(self, data_path, max_samples=None):
        """
        Args:
            data_path: Path to JSONL file
            max_samples: Maximum number of samples to load
        """
        self.samples = []
        
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line)
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} position samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        inputs = torch.tensor(sample['input'], dtype=torch.float32)
        target = torch.tensor(sample['output'], dtype=torch.long)
        
        # Create valid mask (cells that haven't been visited and aren't current)
        # is_visited is at index 7, is_current is at index 8
        visited = inputs[:, 7]  # All visited cells
        valid_mask = (visited == 0)  # Can only move to unvisited cells
        
        return inputs, target, valid_mask


def collate_position_batch(batch):
    """
    Collate function for variable-size grids.
    Pads to maximum grid size in batch.
    """
    inputs, targets, valid_masks = zip(*batch)
    
    # Find max grid size
    max_cells = max(inp.size(0) for inp in inputs)
    
    # Pad inputs and masks
    padded_inputs = []
    padded_masks = []
    
    for inp, mask in zip(inputs, valid_masks):
        num_cells = inp.size(0)
        
        if num_cells < max_cells:
            # Pad with zeros
            pad_size = max_cells - num_cells
            inp = torch.cat([inp, torch.zeros(pad_size, inp.size(1))], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_size, dtype=torch.bool)], dim=0)
        
        padded_inputs.append(inp)
        padded_masks.append(mask)
    
    inputs = torch.stack(padded_inputs)
    valid_masks = torch.stack(padded_masks)
    targets = torch.tensor(targets, dtype=torch.long)
    
    return inputs, targets, valid_masks


def compute_position_accuracy(logits, targets, valid_masks):
    """
    Compute accuracy of next move predictions.
    
    Args:
        logits: (B, N) - predicted logits
        targets: (B,) - target cell indices
        valid_masks: (B, N) - valid cell masks
        
    Returns:
        accuracy: Percentage of correct predictions
        top5_accuracy: Percentage where target is in top 5 predictions
    """
    predictions = logits.argmax(dim=-1)  # (B,)
    correct = (predictions == targets).float().mean().item()
    
    # Top-5 accuracy
    top5_preds = logits.topk(5, dim=-1)[1]  # (B, 5)
    top5_correct = (top5_preds == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
    
    return correct, top5_correct


def position_network_loss(logits, targets, valid_masks):
    """
    Cross-entropy loss for position prediction.
    
    Args:
        logits: (B, N) - predicted logits
        targets: (B,) - target cell indices
        valid_masks: (B, N) - valid cell masks
        
    Returns:
        loss: Cross-entropy loss
    """
    return F.cross_entropy(logits, targets)
