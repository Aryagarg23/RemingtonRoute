"""
Enhanced Dataset Generator for Pointer Network Training

Generates puzzles with:
- Variable grid sizes (5x5 to 10x10)
- Variable checkpoint counts (2 to 8)
- Variable wall densities (0.1 to 0.3)
- Randomized difficulty levels
"""

import os
import json
import random
from tqdm import tqdm
import numpy as np
import time
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym.hamiltonian_puzzle_env import PuzzleDataGenerator, HamiltonianPathGenerator


def puzzle_to_ptrnet_sample(puzzle):
    """
    Converts a Hamiltonian puzzle dict into a Pointer Network sample.
    Each grid cell is encoded as:
        [x_norm, y_norm, waypoint_label, wall_up, wall_down, wall_left, wall_right, is_visited]
    where:
        waypoint_label: 0=none, 1=start, 2=checkpoint, 3=goal
        wall_up/down/left/right: 1 if there's a wall blocking that direction, 0 otherwise
        is_visited: Initially 0 for all cells (will be updated during decoding)
    """
    rows, cols = puzzle['rows'], puzzle['cols']
    path = puzzle['solution_path']
    walls = puzzle['walls']
    checkpoints = puzzle['checkpoints']

    # Flatten grid (row-major order)
    grid_cells = [(r, c) for r in range(rows) for c in range(cols)]
    cell_to_idx = {cell: idx for idx, cell in enumerate(grid_cells)}

    # Build input features
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

        # Visited state (initially all unvisited)
        is_visited = 0

        inputs.append([x_norm, y_norm, w_label, wall_up, wall_down, wall_left, wall_right, is_visited])

    # Build output: sequence of indices along solution path
    outputs = [cell_to_idx[cell] for cell in path]

    return {"input": inputs, "output": outputs}


def generate_variable_ptrnet_dataset(
    num_samples=10000,
    grid_size_range=(6, 6),
    checkpoint_range=(7, 7),
    wall_prob_range=(0.00, 0.00),
    output_dir="gym/output/datasets/",
    difficulty_distribution=None,
    variable_dimensions=True
):
    """
    Generates puzzles with variable grid sizes, checkpoint counts, and wall densities.
    
    Args:
        num_samples: Number of puzzles to generate
        grid_size_range: (min, max) grid size
        checkpoint_range: (min, max) number of checkpoints
        wall_prob_range: (min, max) wall probability
        output_dir: Output directory
        difficulty_distribution: Dict with 'easy', 'medium', 'hard' proportions (None = uniform)
        variable_dimensions: If True, rows and cols can be different
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "ptrnet_dataset_variable.jsonl")
    puzzle_path = os.path.join(output_dir, "ptrnet_puzzles_variable.jsonl")
    stats_path = os.path.join(output_dir, "dataset_statistics.json")
    
    # Default difficulty distribution (uniform)
    if difficulty_distribution is None:
        difficulty_distribution = {'easy': 0.33, 'medium': 0.34, 'hard': 0.33}
    
    # Statistics tracking
    stats = {
        'total_samples': num_samples,
        'grid_sizes': {},
        'checkpoint_counts': {},
        'wall_densities': [],
        'sequence_lengths': [],
        'difficulty_levels': {'easy': 0, 'medium': 0, 'hard': 0}
    }
    
    print(f"Generating {num_samples} puzzles with variable parameters...")
    print(f"Grid sizes: {grid_size_range}")
    print(f"Checkpoints: {checkpoint_range}")
    print(f"Wall probability: {wall_prob_range}")
    print("=" * 70)
    
    with open(json_path, "w") as f_samples, open(puzzle_path, "w") as f_puzzles:
        for i in tqdm(range(num_samples), desc="Creating puzzles"):
            # Determine difficulty level
            rand_val = random.random()
            if rand_val < difficulty_distribution['easy']:
                difficulty = 'easy'
            elif rand_val < difficulty_distribution['easy'] + difficulty_distribution['medium']:
                difficulty = 'medium'
            else:
                difficulty = 'hard'
            
            stats['difficulty_levels'][difficulty] += 1
            
            # Generate parameters based on difficulty
            if difficulty == 'easy':
                if variable_dimensions:
                    rows = random.randint(grid_size_range[0], (grid_size_range[0] + grid_size_range[1]) // 2)
                    cols = random.randint(grid_size_range[0], (grid_size_range[0] + grid_size_range[1]) // 2)
                else:
                    rows = cols = random.randint(grid_size_range[0], (grid_size_range[0] + grid_size_range[1]) // 2)
                num_checkpoints = random.randint(checkpoint_range[0], (checkpoint_range[0] + checkpoint_range[1]) // 2)
                wall_prob = random.uniform(wall_prob_range[0], (wall_prob_range[0] + wall_prob_range[1]) / 2)
            elif difficulty == 'medium':
                if variable_dimensions:
                    rows = random.randint((grid_size_range[0] + grid_size_range[1]) // 2, grid_size_range[1])
                    cols = random.randint((grid_size_range[0] + grid_size_range[1]) // 2, grid_size_range[1])
                else:
                    rows = cols = random.randint((grid_size_range[0] + grid_size_range[1]) // 2, grid_size_range[1])
                num_checkpoints = random.randint((checkpoint_range[0] + checkpoint_range[1]) // 2, checkpoint_range[1])
                wall_prob = random.uniform((wall_prob_range[0] + wall_prob_range[1]) / 2, wall_prob_range[1])
            else:  # hard
                if variable_dimensions:
                    rows = random.randint(grid_size_range[1] - 2, grid_size_range[1])
                    cols = random.randint(grid_size_range[1] - 2, grid_size_range[1])
                else:
                    rows = cols = random.randint(grid_size_range[1] - 1, grid_size_range[1])
                num_checkpoints = random.randint(checkpoint_range[1] - 3, checkpoint_range[1])
                wall_prob = random.uniform(wall_prob_range[1] * 0.8, wall_prob_range[1])
            
            # Generate puzzle
            puzzle = PuzzleDataGenerator.generate_puzzle(rows, cols, num_checkpoints, wall_prob)
            sample = puzzle_to_ptrnet_sample(puzzle)
            
            # Add metadata
            puzzle['difficulty'] = difficulty
            puzzle['grid_size'] = f"{rows}x{cols}"
            
            # Write to files
            f_samples.write(json.dumps(sample) + "\n")
            f_puzzles.write(json.dumps(puzzle, default=list) + "\n")
            
            # Update statistics
            grid_key = f"{rows}x{cols}"
            stats['grid_sizes'][grid_key] = stats['grid_sizes'].get(grid_key, 0) + 1
            stats['checkpoint_counts'][num_checkpoints] = stats['checkpoint_counts'].get(num_checkpoints, 0) + 1
            stats['wall_densities'].append(wall_prob)
            stats['sequence_lengths'].append(len(sample['output']))
    
    # Compute summary statistics
    stats['avg_wall_density'] = np.mean(stats['wall_densities'])
    stats['avg_sequence_length'] = np.mean(stats['sequence_lengths'])
    stats['min_sequence_length'] = int(np.min(stats['sequence_lengths']))
    stats['max_sequence_length'] = int(np.max(stats['sequence_lengths']))
    
    # Save statistics
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("Dataset Generation Complete!")
    print(f"{'=' * 70}")
    print(f"\nDataset saved to: {json_path}")
    print(f"Puzzles saved to: {puzzle_path}")
    print(f"Statistics saved to: {stats_path}")
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Sequence length range: {stats['min_sequence_length']} - {stats['max_sequence_length']}")
    print(f"  Average sequence length: {stats['avg_sequence_length']:.1f}")
    print(f"  Average wall density: {stats['avg_wall_density']:.3f}")
    
    print(f"\nDifficulty Distribution:")
    for level, count in stats['difficulty_levels'].items():
        print(f"  {level.capitalize()}: {count} ({count/num_samples*100:.1f}%)")
    
    print(f"\nGrid Size Distribution:")
    for size, count in sorted(stats['grid_sizes'].items()):
        print(f"  {size}: {count} samples")
    
    print(f"\nCheckpoint Count Distribution:")
    for cp, count in sorted(stats['checkpoint_counts'].items()):
        print(f"  {cp} checkpoints: {count} samples")


def generate_fixed_size_dataset(
    num_samples=1000,
    rows=7,
    cols=7,
    checkpoint_range=(3, 8),
    wall_probability=0.15,
    output_dir="gym/output/datasets/"
):
    """
    Generate dataset with fixed grid size (for compatibility with existing code).
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "ptrnet_dataset.jsonl")
    puzzle_path = os.path.join(output_dir, "ptrnet_puzzles.jsonl")

    print(f"Generating {num_samples} puzzles ({rows}x{cols})...")
    with open(json_path, "w") as f_samples, open(puzzle_path, "w") as f_puzzles:
        for _ in tqdm(range(num_samples), desc="Creating puzzles"):
            num_checkpoints = random.randint(*checkpoint_range)
            puzzle = PuzzleDataGenerator.generate_puzzle(rows, cols, num_checkpoints, wall_probability)
            sample = puzzle_to_ptrnet_sample(puzzle)

            f_samples.write(json.dumps(sample) + "\n")
            f_puzzles.write(json.dumps(puzzle, default=list) + "\n")

    print(f"\nDataset saved to: {json_path}")
    print(f"Puzzles saved to: {puzzle_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Pointer Network training dataset')
    parser.add_argument('--mode', choices=['fixed', 'variable'], default='variable',
                        help='Generation mode: fixed or variable grid sizes')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default='gym/output/datasets/',
                        help='Output directory')
    
    # Fixed mode parameters
    parser.add_argument('--rows', type=int, default=7,
                        help='Grid rows (fixed mode only)')
    parser.add_argument('--cols', type=int, default=7,
                        help='Grid columns (fixed mode only)')
    parser.add_argument('--wall-prob', type=float, default=0.15,
                        help='Wall probability (fixed mode only)')
    
    # Variable mode parameters
    parser.add_argument('--grid-min', type=int, default=6,
                        help='Minimum grid size (variable mode)')
    parser.add_argument('--grid-max', type=int, default=6,
                        help='Maximum grid size (variable mode)')
    parser.add_argument('--cp-min', type=int, default=7,
                        help='Minimum checkpoints')
    parser.add_argument('--cp-max', type=int, default=7,
                        help='Maximum checkpoints')
    parser.add_argument('--wall-min', type=float, default=0.0,
                        help='Minimum wall probability (variable mode)')
    parser.add_argument('--wall-max', type=float, default=0.0,
                        help='Maximum wall probability (variable mode)')
    parser.add_argument('--variable-dims', action='store_true', default=True,
                        help='Allow different row and column sizes')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.mode == 'fixed':
        generate_fixed_size_dataset(
            num_samples=args.num_samples,
            rows=args.rows,
            cols=args.cols,
            checkpoint_range=(args.cp_min, args.cp_max),
            wall_probability=args.wall_prob,
            output_dir=args.output_dir
        )
    else:  # variable
        generate_variable_ptrnet_dataset(
            num_samples=args.num_samples,
            grid_size_range=(args.grid_min, args.grid_max),
            checkpoint_range=(args.cp_min, args.cp_max),
            wall_prob_range=(args.wall_min, args.wall_max),
            output_dir=args.output_dir,
            variable_dimensions=args.variable_dims
        )
    
    elapsed = time.time() - start_time
    print(f"\nTotal generation time: {elapsed:.2f} seconds")
    print(f"Generation speed: {args.num_samples/elapsed:.2f} samples/second")
