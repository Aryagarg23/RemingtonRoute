# RemingtonRoute - Hamiltonian Puzzle Environment

A Gymnasium environment for solving Hamiltonian path puzzles with machine learning visualization tools.

## Project Structure

```
gym/
├── __init__.py
├── hamiltonian_puzzle_env.py      # Main environment and puzzle utilities
├── documentation.md               # This documentation
├── generation/
│   ├── __init__.py
│   └── generate_dataset.py        # Dataset generation for ML training
├── visualization/
│   ├── __init__.py
│   ├── main_visualizer.py         # Comprehensive 3-panel visualizer
│   ├── puzzle_visualizer.py       # Core puzzle drawing utilities
│   ├── visualize_dataset.py       # Individual puzzle visualization
│   └── visualize_ptrnet_dataset.py # PTRNet dataset visualization
└── output/                        # Standardized output directory
    ├── datasets/                  # Generated datasets (JSONL format)
    ├── comprehensive/             # Multi-panel visualizations
    ├── ptrnet/                    # PTRNet-specific visualizations
    └── puzzle/                    # Individual puzzle visualizations
```

## Core Components

### Environment (`hamiltonian_puzzle_env.py`)
- **HamiltonianPuzzleEnv**: Gymnasium environment for reinforcement learning
- **HamiltonianPathGenerator**: Generates valid Hamiltonian paths through grids
- **PuzzleDataGenerator**: Creates puzzle data with checkpoints and walls

### Dataset Generation (`generation/generate_dataset.py`)
Generates training datasets for Pointer Network (PTRNet) models. Creates puzzles in two formats:
- **ptrnet_dataset.jsonl**: ML training data with grid cells encoded as `[x_norm, y_norm, waypoint_type, is_blocked]`
- **ptrnet_puzzles.jsonl**: Full puzzle data for visualization

Output location: `output/datasets/`

### Visualization Tools (`visualization/`)

#### `main_visualizer.py`
Comprehensive 3-panel visualization showing:
1. **PTR Network Input**: ML training data representation
2. **Incomplete Puzzle**: Puzzle layout without solution
3. **Complete Solution**: Full puzzle with solution path

#### `puzzle_visualizer.py`
Core visualization utilities for drawing puzzles with matplotlib.

#### `visualize_dataset.py`
Loads and visualizes individual puzzles from the dataset.

#### `visualize_ptrnet_dataset.py`
Visualizes PTRNet training data, showing the grid encoding used for machine learning.

## Usage Examples

### Generate Dataset
```bash
cd /home/arya/projects/RemingtonRoute
python -m gym.generation.generate_dataset
```

### Run Visualizations
```bash
# Comprehensive 3-panel view
python -m gym.visualization.main_visualizer --sample 0

# Individual puzzle visualization
python -m gym.visualization.visualize_dataset

# PTRNet dataset visualization
python -m gym.visualization.visualize_ptrnet_dataset --sample 0
```

## Data Formats

### Puzzle Structure
```python
{
    'rows': int,           # Grid height
    'cols': int,           # Grid width
    'solution_path': list, # Hamiltonian path as [(r,c), ...]
    'checkpoints': {       # Start, goal, and intermediate checkpoints
        'start': (r,c),
        'goal': (r,c),
        'checkpoints': [(r,c), ...]
    },
    'walls': set,          # Wall edges as frozensets
}
```

### PTRNet Sample Format
```python
{
    'input': list,         # Flattened grid: [x_norm, y_norm, waypoint_type, is_blocked]
    'output': list         # Solution path as cell indices
}
```

Where waypoint_type is:
- 0: empty cell
- 1: start
- 2: checkpoint
- 3: goal

## Output Organization

All visualizations are saved to standardized subdirectories under `output/`:
- `output/datasets/` - Generated training data
- `output/comprehensive/` - Multi-panel visualizations
- `output/ptrnet/` - PTRNet-specific plots
- `output/puzzle/` - Individual puzzle visualizations

## Dependencies

- gymnasium: Reinforcement learning environment
- matplotlib: Visualization and plotting
- numpy: Numerical computations
- tqdm: Progress bars
- pathlib: Path handling

## Notes

This project demonstrates using machine learning (specifically Pointer Networks) to solve problems that can be solved with simple backtracking algorithms. The "AI-powered Remington" approach showcases the irony of applying complex ML models to problems with trivial algorithmic solutions.
