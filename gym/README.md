# Gym Module: Hamiltonian Puzzle Environment and Visualization

This module provides tools for generating, visualizing, and interacting with Hamiltonian path puzzles.
## Overview

The `gym` module contains classes and utilities for creating puzzles where an agent must navigate a grid from start to goal, visiting numbered checkpoints in order, while avoiding walls. These puzzles are inspired by LinkedIn's daily games and are used to train AI models like Pointer Networks for sequence prediction.

## Key Components

### 1. HamiltonianPuzzleEnv (`hamiltonian_puzzle_env.py`)
A Gymnasium-compatible environment for reinforcement learning.

**Classes:**
- `HamiltonianPathGenerator`: Generates randomized Hamiltonian paths for grids.
- `PuzzleDataGenerator`: Creates puzzle data including checkpoints and walls.
- `HamiltonianPuzzleEnv`: The main RL environment.

**Usage:**
```python
from gym.hamiltonian_puzzle_env import HamiltonianPuzzleEnv

env = HamiltonianPuzzleEnv(rows=7, cols=7, num_checkpoints=3)
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

**Features:**
- Configurable grid size, checkpoints, wall probability.
- Supports rendering with matplotlib.
- Rewards for visiting checkpoints and reaching goal.

### 2. PuzzleVisualizer (`puzzle_visualizer.py`)
Tools for visualizing puzzles and generating datasets.

**Classes:**
- `PuzzleVisualizer`: Static methods for drawing puzzles.
- `PuzzleDatasetGenerator`: Generates and saves puzzle visualizations.

**Usage:**
```python
from gym.puzzle_visualizer import PuzzleVisualizer, PuzzleDatasetGenerator

# Visualize a single puzzle
PuzzleVisualizer.visualize_puzzle(puzzle_data, save_path="puzzle.png")

# Generate dataset
generator = PuzzleDatasetGenerator(num_samples=100)
generator.generate_dataset()
```

**Features:**
- Side-by-side incomplete/complete puzzle images.
- Customizable output directories and parameters.

## Installation

Ensure you have the required dependencies:
- `gymnasium`
- `numpy`
- `matplotlib`

Install via pip if needed.

## Example Workflow

1. **Generate Puzzles:**
   ```python
   from gym.hamiltonian_puzzle_env import PuzzleDataGenerator
   puzzle = PuzzleDataGenerator.generate_puzzle(7, 7, 3, 0.1)
   ```

2. **Visualize:**
   ```python
   from gym.puzzle_visualizer import PuzzleVisualizer
   PuzzleVisualizer.visualize_puzzle(puzzle)
   ```

## Data Format

### Puzzle Structure
Puzzle data is a dict with:
- `rows`, `cols`: Grid dimensions.
- `solution_path`: List of (row, col) tuples for the Hamiltonian path.
- `checkpoints`: Dict with 'start', 'goal', 'checkpoints' (lists of positions).
- `walls`: Set of frozensets representing blocked edges.
- `wall_set`: Alias for walls.

### ML Encoding (8D Feature Vector)
Each grid cell is encoded as 8 features for machine learning:
- `[x_norm, y_norm, waypoint_type, wall_up, wall_down, wall_left, wall_right, is_visited]`

Where:
- `x_norm`, `y_norm`: Normalized coordinates [0, 1]
- `waypoint_type`: 0=empty, 1=start, 2=checkpoint, 3=goal
- `wall_up/down/left/right`: 1=blocked, 0=open
- `is_visited`: 0=unvisited, 1=visited (updated during path construction)

