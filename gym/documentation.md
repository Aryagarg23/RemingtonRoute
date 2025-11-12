# RemingtonRoute - Hamiltonian Puzzle Environment

A Gymnasium environment for solving Hamiltonian path puzzles with machine learning visualization tools.

## Project Structure

```
gym/
├── __init__.py
├── hamiltonian_puzzle_env.py      # Main environment and puzzle utilities
├── generator_main.py              # Main generation script (cleans & generates all)
├── documentation.md               # This documentation
├── generation/
│   ├── __init__.py
│   └── generate_dataset.py        # Dataset generation for ML training
├── visualization/
│   ├── __init__.py
│   ├── main_visualizer.py         # Comprehensive 2x2 grid visualizer
│   ├── puzzle_visualizer.py       # Core puzzle drawing utilities
│   ├── visualize_dataset.py       # Individual puzzle visualization
│   └── visualize_ptrnet_dataset.py # PTRNet dataset visualization
└── output/                        # Standardized output directory
    ├── datasets/                  # Generated datasets (JSONL format)
    ├── comprehensive/             # 2x2 grid visualizations (human + PTR views)
    ├── ptrnet/                    # PTRNet-specific visualizations
    └── puzzle/                    # Individual puzzle visualizations
```

## Quick Start

### One-Command Generation
```bash
# Clean all output, generate datasets, create visualizations
python gym/generator_main.py
```

This runs the complete pipeline:
1. Cleans `gym/output/` directory
2. Generates 1000 training samples
3. Creates 5 samples of each visualization type

## Core Components

### Generator (`generator_main.py`)
Main orchestration script that manages the complete generation pipeline.

**RemingtonRouteGenerator Class**:
- `clean_output_directory()`: Removes and recreates output structure
- `generate_dataset()`: Creates training data
- `generate_comprehensive_visualizations()`: 2x2 grid views
- `generate_ptrnet_visualizations()`: PTRNet-specific views
- `generate_puzzle_visualizations()`: Side-by-side puzzle views
- `run_all()`: Executes complete pipeline

### Environment (`hamiltonian_puzzle_env.py`)
- **HamiltonianPuzzleEnv**: Gymnasium environment for reinforcement learning
- **HamiltonianPathGenerator**: Generates valid Hamiltonian paths through grids
- **PuzzleDataGenerator**: Creates puzzle data with checkpoints and walls

### Dataset Generation (`generation/generate_dataset.py`)
Generates training datasets for Pointer Network (PTRNet) models. Creates puzzles in two formats:
- **ptrnet_dataset.jsonl**: ML training data with grid cells encoded as 7-element vectors
- **ptrnet_puzzles.jsonl**: Full puzzle data for visualization

**Encoding Format**: Each cell is represented as `[x_norm, y_norm, waypoint_type, wall_up, wall_down, wall_left, wall_right]`
- `x_norm, y_norm`: Normalized coordinates (0.0 to 1.0)
- `waypoint_type`: 0=empty, 1=start, 2=checkpoint, 3=goal
- `wall_up/down/left/right`: 1 if wall blocks that direction, 0 otherwise (directional encoding)

Output location: `output/datasets/`

### Visualization Tools (`visualization/`)

#### `main_visualizer.py`
Comprehensive 2x2 grid visualization showing:
- **Top-left**: Human perspective - Complete solution with path
- **Top-right**: Human perspective - Incomplete puzzle (no path shown)
- **Bottom-left**: PTR Network view - With solution path
- **Bottom-right**: PTR Network view - Grid only (no path)

**Visual Features**:
- Unified coordinate system (0-7 on both axes)
- Clean axes (no tick labels)
- Cell indices displayed (0-48)
- Unified legend across all subplots
- Matching colors and markers in all views
- Green circle (S) for start
- Red square (G) for goal  
- Yellow plus with numbers for checkpoints (1, 2, 3...)
- Red walls (directional, correctly positioned)
- Blue solution path (when shown)

#### `puzzle_visualizer.py`
Core visualization utilities for drawing puzzles with matplotlib.
- Standard coordinate system (0-7, no axis inversion)
- White cell backgrounds with overlaid markers
- Cell indices displayed
- Clean axes without tick labels

#### `visualize_dataset.py`
Loads and visualizes individual puzzles from the dataset.

#### `visualize_ptrnet_dataset.py`
Visualizes PTRNet training data, showing the grid encoding used for machine learning.

## Usage Examples

### Generate Dataset
```python
from gym.generation.generate_dataset import generate_ptrnet_dataset

generate_ptrnet_dataset(
    num_samples=1000,
    rows=7,
    cols=7,
    checkpoint_range=(3, 6),
    wall_probability=0.15,
    output_dir='gym/output/datasets/'
)
```

### Run Visualizations
```python
# Comprehensive 2x2 grid view
from gym.visualization.main_visualizer import MainVisualizer

viz = MainVisualizer(
    ptrnet_dataset_path='gym/output/datasets/ptrnet_dataset.jsonl',
    puzzle_dataset_path='gym/output/datasets/ptrnet_puzzles.jsonl'
)
viz.visualize_sample(0, 'gym/output/comprehensive/sample_000.png')

# PTRNet dataset visualization
from gym.visualization.visualize_ptrnet_dataset import PTRNetDatasetVisualizer

viz = PTRNetDatasetVisualizer('gym/output/datasets/ptrnet_dataset.jsonl')
viz.visualize_sample(0, 'gym/output/ptrnet/sample_000.png')

# Puzzle visualization
from gym.visualization.puzzle_visualizer import PuzzleVisualizer
import json

with open('gym/output/datasets/ptrnet_puzzles.jsonl', 'r') as f:
    puzzle = json.loads(f.readline())
PuzzleVisualizer.visualize_puzzle(puzzle, save_path='gym/output/puzzle/sample_000.png')
```

### Data Formats

#### Puzzle Structure
```python
{
    'rows': int,           # Grid height (default: 7)
    'cols': int,           # Grid width (default: 7)
    'solution_path': list, # Hamiltonian path as [[r,c], ...]
    'checkpoints': {       # Start, goal, and intermediate checkpoints
        'start': [r, c],
        'goal': [r, c],
        'checkpoints': [[r, c], ...]  # Traversed in sequence
    },
    'walls': set,          # Wall edges as frozensets
    'wall_set': list       # Walls as [[[r1,c1], [r2,c2]], ...]
}
```

#### PTRNet Sample Format
```python
{
    'input': [             # List of 49 cells (7x7 grid, row-major order)
        [x_norm, y_norm, waypoint_type, wall_up, wall_down, wall_left, wall_right],
        # ... 48 more cells
    ],
    'output': [            # Solution path as cell indices (0-48)
        idx0, idx1, idx2, ...
    ]
}
```

**Cell Encoding Details**:
- `x_norm`: Row coordinate normalized to [0, 1]
- `y_norm`: Column coordinate normalized to [0, 1]
- `waypoint_type`: 0=empty, 1=start, 2=checkpoint, 3=goal
- `wall_up`: 1 if wall blocks movement to cell above (row-1), 0 otherwise
- `wall_down`: 1 if wall blocks movement to cell below (row+1), 0 otherwise
- `wall_left`: 1 if wall blocks movement to cell on left (col-1), 0 otherwise
- `wall_right`: 1 if wall blocks movement to cell on right (col+1), 0 otherwise

**Grid Boundaries**: Cells at grid edges have boundary walls encoded as 1 (e.g., cell at row=0 has wall_up=1). These are encoded for ML model awareness but filtered out during visualization to show only internal puzzle walls.

**Checkpoint Order**: Checkpoints must be traversed in ascending order (checkpoint[0], then checkpoint[1], etc.) from start to goal.

### Visualization Style

**Coordinate System**:
- X-axis: 0 to 7 (left to right)
- Y-axis: 0 to 7 (bottom to top)
- No axis tick labels (clean display)
- Grid lines for cell boundaries

**Visual Elements** (consistent across human and PTR views):
- **White cell backgrounds** with black borders
- **Cell indices**: Gray numbers (0-48) in each cell
- **Start marker**: Green circle with white 'S' label
- **Goal marker**: Red square with white 'G' label
- **Checkpoint markers**: Yellow plus (+) with black numbered labels (1, 2, 3...)
- **Walls**: Red lines (linewidth=4) on cell edges
- **Solution path**: Blue line (alpha=0.6) when displayed
- **Unified legend**: Single legend at bottom showing all elements

## Output Organization

All generated files are saved to standardized subdirectories under `output/`:
- `output/datasets/` - Training data (ptrnet_dataset.jsonl, ptrnet_puzzles.jsonl)
- `output/comprehensive/` - 2x2 grid visualizations (human + PTR views)
- `output/ptrnet/` - PTRNet-specific visualizations
- `output/puzzle/` - Side-by-side puzzle visualizations (incomplete | complete)

## Verification & Data Integrity

### Encoding Parity
The PTRNet encoding has been verified to maintain **100% parity** with the original puzzle data:
- ✅ All waypoints (start, goal, checkpoints) correctly encoded
- ✅ All walls accurately represented with directional flags
- ✅ Complete solution paths preserved
- ✅ Checkpoint traversal order maintained

### Visualization Consistency
Both human and PTR Network views use identical:
- Coordinate systems (0-7 on both axes)
- Color schemes and markers
- Wall positioning and rendering
- Legend elements

This ensures visual validation of the encoding accuracy.

## Dependencies

- gymnasium: Reinforcement learning environment
- matplotlib: Visualization and plotting
- numpy: Numerical computations
- tqdm: Progress bars
- pathlib: Path handling

## Key Features

### Directional Wall Encoding
Unlike simplified approaches that only flag "has adjacent walls", this implementation preserves **directional wall information**:
- Each cell encodes which specific edges are blocked (up, down, left, right)
- ML models receive accurate directional constraint information
- Critical for learning optimal pathfinding strategies

### Sequential Checkpoints
Puzzles enforce sequential checkpoint traversal:
- Solution paths must visit checkpoints in order: start → checkpoint[0] → checkpoint[1] → ... → goal
- Adds complexity and structure to the pathfinding problem
- Better reflects real-world routing constraints

### Hamiltonian Path Constraint
All puzzles are generated with valid Hamiltonian paths:
- Solution visits every cell exactly once
- Ensures solvability and consistency
- Provides ground truth for supervised learning
