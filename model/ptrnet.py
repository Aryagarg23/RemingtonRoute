"""
Pointer Network Implementation for Hamiltonian Path Puzzles

Based on "Pointer Networks" (Vinyals et al., 2015)
https://arxiv.org/abs/1506.03134

This implementation uses supervised learning to train a Pointer Network
to solve Hamiltonian path puzzles with checkpoints and walls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Tuple, Dict, Any


class PointerNetwork(nn.Module):
    """
    Pointer Network for solving Hamiltonian path puzzles.
    
    Architecture:
    - Encoder: LSTM that processes grid cell features
    - Decoder: LSTM with attention mechanism that points to input positions
    
    Args:
        input_dim: Dimension of input features (8: x, y, waypoint_type, 4 walls, is_visited)
        hidden_dim: Hidden dimension for LSTM layers
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    
    def __init__(self, input_dim=8, hidden_dim=256, num_layers=2, dropout=0.2):
        super(PointerNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism parameters
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # For encoder outputs
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # For decoder state
        self.v = nn.Linear(hidden_dim, 1, bias=False)  # Attention score
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, targets=None, teacher_forcing_ratio=0.5):
        """
        Forward pass through the network.
        
        Args:
            inputs: (batch_size, seq_len, input_dim) - grid cell features
            targets: (batch_size, seq_len) - target sequence of indices (for training)
            teacher_forcing_ratio: Probability of using teacher forcing during training
            
        Returns:
            pointers: (batch_size, seq_len, seq_len) - attention distributions
            indices: (batch_size, seq_len) - predicted indices
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        
        # Clone inputs to track visited state (last dimension is is_visited)
        current_inputs = inputs.clone()
        
        # Embed inputs
        embedded = self.dropout(torch.tanh(self.input_embedding(current_inputs)))  # (B, L, H)
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(embedded)  # (B, L, H)
        encoder_outputs = self.dropout(encoder_outputs)
        
        # Initialize decoder input (use first encoder output or learned embedding)
        decoder_input = encoder_outputs[:, 0:1, :]  # (B, 1, H)
        decoder_hidden = (hidden, cell)
        
        pointers = []
        indices = []
        mask = torch.zeros(batch_size, seq_len, device=inputs.device).bool()
        
        # Decode sequence
        for step in range(seq_len):
            # Re-encode with updated visited states if not first step
            if step > 0:
                embedded = self.dropout(torch.tanh(self.input_embedding(current_inputs)))
                encoder_outputs, _ = self.encoder(embedded)
                encoder_outputs = self.dropout(encoder_outputs)
            
            # Decoder step
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = self.dropout(decoder_output)  # (B, 1, H)
            
            # Attention mechanism (Bahdanau-style)
            # u_i = v^T tanh(W1 * e_i + W2 * d_t)
            e_transformed = self.W1(encoder_outputs)  # (B, L, H)
            d_transformed = self.W2(decoder_output)  # (B, 1, H)
            
            # Compute attention scores
            scores = self.v(torch.tanh(e_transformed + d_transformed))  # (B, L, 1)
            scores = scores.squeeze(-1)  # (B, L)
            
            # Mask already selected positions
            scores = scores.masked_fill(mask, float('-inf'))
            
            # Compute attention distribution
            pointer = F.softmax(scores, dim=1)  # (B, L)
            pointers.append(pointer)
            
            # Select next position
            if self.training and targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                index = targets[:, step]
            else:
                # Use predicted index
                index = pointer.argmax(dim=1)
            
            indices.append(index)
            
            # Update mask
            mask.scatter_(1, index.unsqueeze(1), True)
            
            # Update visited state in input features (last dimension)
            for b in range(batch_size):
                current_inputs[b, index[b], -1] = 1.0
            
            # Update decoder input (use selected encoder output)
            decoder_input = torch.gather(
                encoder_outputs,
                1,
                index.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)
            )  # (B, 1, H)
        
        # Stack outputs
        pointers = torch.stack(pointers, dim=1)  # (B, seq_len, seq_len)
        indices = torch.stack(indices, dim=1)  # (B, seq_len)
        
        return pointers, indices
    
    def beam_search(self, inputs, beam_width=5):
        """
        Beam search decoding for inference.
        
        Args:
            inputs: (1, seq_len, input_dim) - single sample
            beam_width: Number of beams to maintain
            
        Returns:
            best_sequence: List of indices
            best_score: Log probability of best sequence
        """
        batch_size = inputs.size(0)
        assert batch_size == 1, "Beam search only supports batch_size=1"
        
        seq_len = inputs.size(1)
        
        # Clone inputs to track visited state
        current_inputs = inputs.clone()
        
        # Embed and encode
        embedded = torch.tanh(self.input_embedding(current_inputs))
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        
        # Initialize beams: (sequence, score, decoder_hidden, mask, current_inputs)
        beams = [([], 0.0, (hidden, cell), torch.zeros(1, seq_len, device=inputs.device).bool(), current_inputs.clone())]
        
        for step in range(seq_len):
            candidates = []
            
            for sequence, score, decoder_hidden, mask, beam_inputs in beams:
                # Re-encode with updated visited states if not first step
                if step > 0:
                    embedded = torch.tanh(self.input_embedding(beam_inputs))
                    encoder_outputs, _ = self.encoder(embedded)
                
                # Get decoder input
                if step == 0:
                    decoder_input = encoder_outputs[:, 0:1, :]
                else:
                    last_idx = sequence[-1]
                    decoder_input = encoder_outputs[:, last_idx:last_idx+1, :]
                
                # Decoder step
                decoder_output, new_hidden = self.decoder(decoder_input, decoder_hidden)
                
                # Attention
                e_transformed = self.W1(encoder_outputs)
                d_transformed = self.W2(decoder_output)
                scores = self.v(torch.tanh(e_transformed + d_transformed)).squeeze(-1)
                scores = scores.masked_fill(mask, float('-inf'))
                log_probs = F.log_softmax(scores, dim=1)
                
                # Get top-k candidates
                top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)
                
                for i in range(beam_width):
                    idx = top_indices[0, i].item()
                    log_prob = top_log_probs[0, i].item()
                    
                    if mask[0, idx]:
                        continue  # Skip already visited
                    
                    new_sequence = sequence + [idx]
                    new_score = score + log_prob
                    new_mask = mask.clone()
                    new_mask[0, idx] = True
                    
                    # Update visited state in inputs
                    new_beam_inputs = beam_inputs.clone()
                    new_beam_inputs[0, idx, -1] = 1.0
                    
                    candidates.append((new_sequence, new_score, new_hidden, new_mask, new_beam_inputs))
            
            # Select top beams
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Return best sequence
        best_sequence, best_score, _, _, _ = beams[0]
        return best_sequence, best_score


class HamiltonianPuzzleDataset(Dataset):
    """
    Dataset for Hamiltonian path puzzles.
    
    Loads data from JSONL files generated by generate_dataset.py
    """
    
    def __init__(self, data_path, max_samples=None):
        """
        Args:
            data_path: Path to JSONL file
            max_samples: Maximum number of samples to load (None = all)
        """
        self.samples = []
        
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line)
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        inputs = torch.tensor(sample['input'], dtype=torch.float32)
        targets = torch.tensor(sample['output'], dtype=torch.long)
        
        return inputs, targets


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the maximum length in the batch.
    """
    inputs, targets = zip(*batch)
    
    # Find max length
    max_len = max(inp.size(0) for inp in inputs)
    
    # Pad inputs
    padded_inputs = []
    padded_targets = []
    lengths = []
    
    for inp, tgt in zip(inputs, targets):
        seq_len = inp.size(0)
        lengths.append(seq_len)
        
        if seq_len < max_len:
            # Pad with zeros
            pad_size = max_len - seq_len
            inp = torch.cat([inp, torch.zeros(pad_size, inp.size(1))], dim=0)
            tgt = torch.cat([tgt, torch.zeros(pad_size, dtype=torch.long)], dim=0)
        
        padded_inputs.append(inp)
        padded_targets.append(tgt)
    
    inputs = torch.stack(padded_inputs)
    targets = torch.stack(padded_targets)
    lengths = torch.tensor(lengths)
    
    return inputs, targets, lengths


def compute_accuracy(pointers, targets, lengths):
    """
    Compute accuracy of predictions.
    
    Args:
        pointers: (B, L, L) attention distributions
        targets: (B, L) target indices
        lengths: (B,) actual sequence lengths
        
    Returns:
        accuracy: Percentage of correct predictions
    """
    batch_size = pointers.size(0)
    predictions = pointers.argmax(dim=2)  # (B, L)
    
    correct = 0
    total = 0
    
    for i in range(batch_size):
        length = lengths[i].item()
        correct += (predictions[i, :length] == targets[i, :length]).sum().item()
        total += length
    
    return correct / total if total > 0 else 0.0


def pointer_network_loss(pointers, targets, lengths):
    """
    Compute cross-entropy loss for pointer network.
    
    Args:
        pointers: (B, L, L) attention distributions
        targets: (B, L) target indices
        lengths: (B,) actual sequence lengths
        
    Returns:
        loss: Mean cross-entropy loss
    """
    batch_size = pointers.size(0)
    max_len = pointers.size(1)
    
    # Create mask for valid positions
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    
    # Reshape for cross-entropy: (B*L, L)
    pointers_flat = pointers.view(-1, pointers.size(2))
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1)
    
    # Compute loss only for valid positions
    loss = F.cross_entropy(pointers_flat[mask_flat], targets_flat[mask_flat])
    
    return loss
