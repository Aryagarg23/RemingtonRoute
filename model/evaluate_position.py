"""
Evaluation script for Position-based Network

Tests the model's ability to solve puzzles step-by-step.
"""

import torch
import numpy as np
import os
import json
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from position_net import PositionNetwork
from gym.hamiltonian_puzzle_env import PuzzleDataGenerator


class PositionSolver:
    """
    Solves puzzles step-by-step using the position-based network.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def solve_puzzle(self, puzzle, max_steps=None, temperature=1.0):
        """
        Solve a puzzle step by step.
        
        Args:
            puzzle: Puzzle dictionary
            max_steps: Maximum steps (None = grid size)
            temperature: Sampling temperature
            
        Returns:
            path: List of (row, col) positions
            success: Whether puzzle was solved
            info: Additional information
        """
        rows, cols = puzzle['rows'], puzzle['cols']
        checkpoints = puzzle['checkpoints']
        walls = puzzle['walls']
        
        # Flatten grid
        grid_cells = [(r, c) for r in range(rows) for c in range(cols)]
        cell_to_idx = {cell: idx for idx, cell in enumerate(grid_cells)}
        idx_to_cell = {idx: cell for cell, idx in cell_to_idx.items()}
        
        # Start position
        current_pos = checkpoints['start']
        goal_pos = checkpoints['goal']
        required_checkpoints = set(checkpoints['checkpoints'])
        
        # Track state
        path = [current_pos]
        visited = {current_pos}
        collected_checkpoints = set()
        
        if max_steps is None:
            max_steps = rows * cols
        
        for step in range(max_steps):
            # Build input features
            inputs = self._build_input_features(
                grid_cells, current_pos, visited,
                checkpoints, walls, rows, cols
            )
            
            # Convert to tensor
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Create valid mask (can only move to unvisited cells)
            valid_mask = torch.zeros(1, len(grid_cells), dtype=torch.bool, device=self.device)
            for idx, cell in enumerate(grid_cells):
                if cell not in visited:
                    valid_mask[0, idx] = True
            
            # Predict next move
            with torch.no_grad():
                next_idx, confidence = self.model.predict_next_move(
                    inputs_tensor, valid_mask, temperature
                )
            
            next_pos = idx_to_cell[next_idx]
            
            # Check if move is valid (adjacent cell)
            if not self._is_adjacent(current_pos, next_pos, walls):
                return path, False, {
                    'reason': 'invalid_move',
                    'step': step,
                    'from': current_pos,
                    'to': next_pos
                }
            
            # Update state
            path.append(next_pos)
            visited.add(next_pos)
            current_pos = next_pos
            
            # Check if checkpoint collected
            if next_pos in required_checkpoints:
                collected_checkpoints.add(next_pos)
            
            # Check if goal reached
            if next_pos == goal_pos:
                # Verify all checkpoints collected
                if collected_checkpoints == required_checkpoints:
                    return path, True, {
                        'steps': len(path) - 1,
                        'cells_visited': len(visited),
                        'total_cells': len(grid_cells)
                    }
                else:
                    return path, False, {
                        'reason': 'missing_checkpoints',
                        'collected': len(collected_checkpoints),
                        'required': len(required_checkpoints)
                    }
        
        return path, False, {
            'reason': 'max_steps_reached',
            'steps': len(path) - 1
        }
    
    def _build_input_features(self, grid_cells, current_pos, visited,
                             checkpoints, walls, rows, cols):
        """Build input feature array for current state."""
        inputs = []
        
        for (r, c) in grid_cells:
            x_norm = r / (rows - 1) if rows > 1 else 0.0
            y_norm = c / (cols - 1) if cols > 1 else 0.0
            
            # Waypoint encoding
            if (r, c) == checkpoints['start']:
                w_label = 1
            elif (r, c) == checkpoints['goal']:
                w_label = 3
            elif (r, c) in checkpoints['checkpoints']:
                w_label = 2
            else:
                w_label = 0
            
            # Directional walls
            wall_up = int(frozenset([(r, c), (r - 1, c)]) in walls) if r > 0 else 1
            wall_down = int(frozenset([(r, c), (r + 1, c)]) in walls) if r < rows - 1 else 1
            wall_left = int(frozenset([(r, c), (r, c - 1)]) in walls) if c > 0 else 1
            wall_right = int(frozenset([(r, c), (r, c + 1)]) in walls) if c < cols - 1 else 1
            
            # State
            is_visited = 1 if (r, c) in visited else 0
            is_current = 1 if (r, c) == current_pos else 0
            
            inputs.append([
                x_norm, y_norm, w_label,
                wall_up, wall_down, wall_left, wall_right,
                is_visited, is_current
            ])
        
        return inputs
    
    def _is_adjacent(self, pos1, pos2, walls):
        """Check if two positions are adjacent and not blocked by wall."""
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Check if adjacent
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
        
        # Check if wall blocks
        edge = frozenset([pos1, pos2])
        return edge not in walls


def evaluate_model(model_path, num_samples=100, device='cuda', visualize=0):
    """
    Evaluate model on random puzzles.
    
    Args:
        model_path: Path to model checkpoint
        num_samples: Number of puzzles to test
        device: Device to run on
        visualize: Number of solutions to visualize
    """
    print("=" * 70)
    print("POSITION-BASED NETWORK EVALUATION")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = PositionNetwork(
        input_dim=9,
        hidden_dim=1024,
        num_layers=4,
        use_transformer=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded (epoch {checkpoint['epoch']})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create solver
    solver = PositionSolver(model, device)
    
    # Generate test puzzles
    print(f"\nGenerating {num_samples} test puzzles...")
    puzzle_gen = PuzzleDataGenerator()
    
    results = {
        'total': 0,
        'solved': 0,
        'invalid_move': 0,
        'missing_checkpoints': 0,
        'max_steps': 0,
        'path_lengths': [],
        'optimal_lengths': []
    }
    
    for i in tqdm(range(num_samples), desc="Evaluating"):
        # Generate puzzle
        puzzle = puzzle_gen.generate_puzzle(
            rows=6, cols=6,
            num_checkpoints=7,
            wall_probability=0.0
        )
        
        # Solve
        path, success, info = solver.solve_puzzle(puzzle)
        
        results['total'] += 1
        if success:
            results['solved'] += 1
            results['path_lengths'].append(len(path) - 1)
            results['optimal_lengths'].append(len(puzzle['solution_path']) - 1)
        else:
            reason = info.get('reason', 'unknown')
            results[reason] = results.get(reason, 0) + 1
        
        # Visualize
        if visualize > 0 and i < visualize:
            visualize_solution(puzzle, path, success, info, i)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total puzzles: {results['total']}")
    print(f"Solved: {results['solved']} ({results['solved']/results['total']*100:.1f}%)")
    print(f"Failed - Invalid move: {results.get('invalid_move', 0)}")
    print(f"Failed - Missing checkpoints: {results.get('missing_checkpoints', 0)}")
    print(f"Failed - Max steps: {results.get('max_steps', 0)}")
    
    if results['solved'] > 0:
        avg_len = np.mean(results['path_lengths'])
        avg_opt = np.mean(results['optimal_lengths'])
        print(f"\nAverage path length: {avg_len:.1f}")
        print(f"Average optimal length: {avg_opt:.1f}")
        print(f"Efficiency: {avg_opt/avg_len*100:.1f}%")


def visualize_solution(puzzle, path, success, info, idx):
    """Visualize a solution."""
    rows, cols = puzzle['rows'], puzzle['cols']
    checkpoints = puzzle['checkpoints']
    walls = puzzle['walls']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw grid
    for r in range(rows + 1):
        ax.plot([0, cols], [r, r], 'k-', lw=0.5)
    for c in range(cols + 1):
        ax.plot([c, c], [0, rows], 'k-', lw=0.5)
    
    # Draw walls
    for wall in walls:
        cells = list(wall)
        if len(cells) == 2:
            (r1, c1), (r2, c2) = cells
            if r1 == r2:  # Vertical wall
                x = max(c1, c2)
                ax.plot([x, x], [r1, r1 + 1], 'r-', lw=4)
            else:  # Horizontal wall
                y = max(r1, r2)
                ax.plot([c1, c1 + 1], [y, y], 'r-', lw=4)
    
    # Draw path
    if len(path) > 1:
        path_y = [r + 0.5 for r, c in path]
        path_x = [c + 0.5 for r, c in path]
        color = 'green' if success else 'orange'
        ax.plot(path_x, path_y, color=color, lw=2, alpha=0.7, marker='o', markersize=4)
    
    # Draw checkpoints
    for cp in checkpoints['checkpoints']:
        r, c = cp
        circle = mpatches.Circle((c + 0.5, r + 0.5), 0.3, color='blue', alpha=0.5)
        ax.add_patch(circle)
    
    # Draw start/goal
    r, c = checkpoints['start']
    ax.text(c + 0.5, r + 0.5, 'S', ha='center', va='center', fontsize=16, fontweight='bold')
    r, c = checkpoints['goal']
    ax.text(c + 0.5, r + 0.5, 'G', ha='center', va='center', fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f"Puzzle {idx} - {'SOLVED' if success else 'FAILED'}\n{info}")
    
    plt.tight_layout()
    plt.savefig(f'model/logs/solution_{idx}.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate position-based network")
    parser.add_argument("--model", type=str, default="model/checkpoints/position_best.pt")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--visualize", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        num_samples=args.num_samples,
        device=args.device,
        visualize=args.visualize
    )
