"""
Position-based Dataset Generator for Step-by-Step Decision Making

Instead of sequence-to-sequence, each training sample is:
- State: current position + grid state + visited cells
- Action: which cell to move to next

This allows the model to learn "given where I am, what's the best next move"
"""

import os
import json
import random
from tqdm import tqdm
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym.hamiltonian_puzzle_env import PuzzleDataGenerator, HamiltonianPathGenerator


def puzzle_to_position_samples(puzzle):
    """
    Convert a puzzle into multiple position-based samples.
    Each step along the solution path becomes a training sample.
    
    For each step:
    - Input: grid state with current position marked + visited cells
    - Output: index of next cell to move to
    
    Returns:
        List of (input, output) tuples
    """
    rows, cols = puzzle['rows'], puzzle['cols']
    path = puzzle['solution_path']
    walls = puzzle['walls']
    checkpoints = puzzle['checkpoints']
    
    # Flatten grid (row-major order)
    grid_cells = [(r, c) for r in range(rows) for c in range(cols)]
    cell_to_idx = {cell: idx for idx, cell in enumerate(grid_cells)}
    
    samples = []
    
    # Create a sample for each step in the path (except last step which has no next move)
    for step_idx in range(len(path) - 1):
        current_pos = path[step_idx]
        next_pos = path[step_idx + 1]
        visited_so_far = set(path[:step_idx + 1])  # Include current position
        
        # Build input features for all cells
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
            
            # Directional wall encoding
            wall_up = int(frozenset([(r, c), (r - 1, c)]) in walls) if r > 0 else 1
            wall_down = int(frozenset([(r, c), (r + 1, c)]) in walls) if r < rows - 1 else 1
            wall_left = int(frozenset([(r, c), (r, c - 1)]) in walls) if c > 0 else 1
            wall_right = int(frozenset([(r, c), (r, c + 1)]) in walls) if c < cols - 1 else 1
            
            # Mark if visited
            is_visited = 1 if (r, c) in visited_so_far else 0
            
            # Mark if current position (NEW FEATURE)
            is_current = 1 if (r, c) == current_pos else 0
            
            inputs.append([
                x_norm, y_norm, w_label, 
                wall_up, wall_down, wall_left, wall_right, 
                is_visited, is_current
            ])
        
        # Output is the index of the next cell
        output = cell_to_idx[next_pos]
        
        samples.append({
            "input": inputs,
            "output": output,
            "step": step_idx,
            "path_length": len(path)
        })
    
    return samples


def generate_position_dataset(
    num_puzzles=10000,
    grid_size_range=(6, 6),
    checkpoint_range=(7, 7),
    wall_prob_range=(0.00, 0.00),
    output_dir="gym/output/datasets/",
    variable_dimensions=True
):
    """
    Generate position-based training dataset.
    
    Each puzzle generates multiple training samples (one per step).
    A 6x6 puzzle with 36 steps creates 35 training samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "position_dataset.jsonl")
    
    puzzle_gen = PuzzleDataGenerator()
    
    total_samples = 0
    successful_puzzles = 0
    
    print(f"Generating position-based dataset from {num_puzzles} puzzles...")
    print(f"Grid size range: {grid_size_range}")
    print(f"Checkpoint range: {checkpoint_range}")
    print(f"Wall probability range: {wall_prob_range}")
    
    with open(json_path, 'w') as f:
        for i in tqdm(range(num_puzzles), desc="Generating puzzles"):
            # Random puzzle parameters
            if variable_dimensions:
                rows = random.randint(*grid_size_range)
                cols = random.randint(*grid_size_range)
            else:
                size = random.randint(*grid_size_range)
                rows, cols = size, size
            
            num_checkpoints = random.randint(*checkpoint_range)
            wall_prob = random.uniform(*wall_prob_range)
            
            try:
                # Generate puzzle
                puzzle = puzzle_gen.generate_puzzle(
                    rows=rows,
                    cols=cols,
                    num_checkpoints=num_checkpoints,
                    wall_probability=wall_prob
                )
                
                # Convert to position-based samples
                samples = puzzle_to_position_samples(puzzle)
                
                # Write each sample
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
                    total_samples += 1
                
                successful_puzzles += 1
                
            except Exception as e:
                # Skip puzzles that fail to generate
                continue
    
    print(f"\n✓ Dataset generation complete!")
    print(f"  Successful puzzles: {successful_puzzles}/{num_puzzles}")
    print(f"  Total training samples: {total_samples}")
    print(f"  Avg samples per puzzle: {total_samples/successful_puzzles:.1f}")
    print(f"  Saved to: {json_path}")
    
    return json_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate position-based training dataset")
    parser.add_argument("--num-puzzles", type=int, default=10000, help="Number of puzzles to generate")
    parser.add_argument("--min-size", type=int, default=6, help="Minimum grid size")
    parser.add_argument("--max-size", type=int, default=6, help="Maximum grid size")
    parser.add_argument("--min-checkpoints", type=int, default=7, help="Minimum checkpoints")
    parser.add_argument("--max-checkpoints", type=int, default=7, help="Maximum checkpoints")
    parser.add_argument("--min-walls", type=float, default=0.0, help="Minimum wall probability")
    parser.add_argument("--max-walls", type=float, default=0.0, help="Maximum wall probability")
    parser.add_argument("--output-dir", type=str, default="gym/output/datasets/", help="Output directory")
    parser.add_argument("--variable-dims", action="store_true", help="Allow non-square grids")
    
    args = parser.parse_args()
    
    generate_position_dataset(
        num_puzzles=args.num_puzzles,
        grid_size_range=(args.min_size, args.max_size),
        checkpoint_range=(args.min_checkpoints, args.max_checkpoints),
        wall_prob_range=(args.min_walls, args.max_walls),
        output_dir=args.output_dir,
        variable_dimensions=args.variable_dims
    )
