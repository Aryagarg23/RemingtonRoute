import os
import json
import random
from tqdm import tqdm
import numpy as np
import time

# Import your environment and utilities
from ..hamiltonian_puzzle_env import PuzzleDataGenerator, HamiltonianPathGenerator


# ==============================
# 1. Convert Puzzle → Ptr-Net Sample
# ==============================
def puzzle_to_ptrnet_sample(puzzle):
    """
    Converts a Hamiltonian puzzle dict into a Pointer Network sample.
    Each grid cell is encoded as:
        [x_norm, y_norm, waypoint_label, barrier_flag]
    where:
        waypoint_label: 0=none, 1=start, 2=checkpoint, 3=goal
        barrier_flag:   1 if blocked on any side
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
        x_norm = r / (rows - 1)
        y_norm = c / (cols - 1)

        # Waypoint encoding
        if (r, c) == checkpoints['start']:
            w_label = 1
        elif (r, c) == checkpoints['goal']:
            w_label = 3
        elif (r, c) in checkpoints['checkpoints']:
            w_label = 2
        else:
            w_label = 0

        # Barrier encoding
        is_blocked = any(
            frozenset([(r, c), n]) in walls
            for n in HamiltonianPathGenerator._get_neighbors(r, c, rows, cols)
        )

        inputs.append([x_norm, y_norm, w_label, int(is_blocked)])

    # Build output: sequence of indices along solution path
    outputs = [cell_to_idx[cell] for cell in path]

    return {"input": inputs, "output": outputs}


# ==============================
# 2. Dataset Generation Function
# ==============================
def generate_ptrnet_dataset(
    num_samples=1000,
    rows=7,
    cols=7,
    checkpoint_range=(3, 6),
    wall_probability=0.15,
    output_dir="../output/datasets/",
    save_format="jsonl"
):
    """
    Generates multiple puzzles and saves them in a format suitable
    for training a Pointer Network.
    Also saves the full puzzle data for visualization.
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
            f_puzzles.write(json.dumps(puzzle, default=list) + "\n")  # store real puzzle

    print(f"\nDataset saved to: {json_path}")
    print(f"Puzzles saved to: {puzzle_path}")


# ==============================
# 3. Example Usage
# ==============================
if __name__ == "__main__":
    start_time = time.time()
    generate_ptrnet_dataset(
        num_samples=1000,
        rows=7,
        cols=7,
        checkpoint_range=(3, 8),
        wall_probability=0.15,
        output_dir="gym/output/datasets/",
        save_format="jsonl"  # or "npz"
    )
    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.2f} seconds")
