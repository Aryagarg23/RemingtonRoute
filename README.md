# RemingtonRoute 🧠🔫

In a world demanding "AI," the simple, correct, and computationally-trivial backtracking algorithm is no longer "innovative." It's too fast, too efficient, and worst of all, too explainable.

RemingtonRoute is our answer to this "problem."

We are building the "AI-powered Remington" that costs $10 million, requires a 50-page user manual, and has a 2% chance of missing... to solve a problem that a 100 dollar "glock" (a backtracking algorithm) solves every single time.

We are teaching a machine to guess the answer, because "guessing" (probabilistic modeling) is what "AI" does.

## What is this?

A Gymnasium environment for Hamiltonian path puzzles, complete with machine learning visualization tools. Uses Pointer Networks to solve problems that backtracking algorithms handle trivially.

**Key Features:**
- **Directional wall encoding** - Each cell encodes which specific edges are blocked (up/down/left/right)
- **Sequential checkpoints** - Must be traversed in order from start to goal
- **Hamiltonian paths** - Solutions visit every cell exactly once
- **Unified visualizations** - Human and ML views use identical coordinate systems and styling

## Quick Start

### Setup
```bash
# Clone and enter the repository
cd RemingtonRoute

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Everything (Recommended)
```bash
# Clean up, generate datasets, and create visualizations - all in one command
python gym/generator_main.py
```

This single command will:
1. Clean all previous output files
2. Generate 1000 training samples
3. Create 5 comprehensive visualizations (2x2 grids)
4. Create 5 PTRNet visualizations
5. Create 5 puzzle visualizations

### Manual Generation (Advanced)

#### Generate Training Data
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

#### Create Visualizations
```python
# Comprehensive 2x2 grid: Human vs PTR Network perspectives
from gym.visualization.main_visualizer import MainVisualizer

viz = MainVisualizer(
    ptrnet_dataset_path='gym/output/datasets/ptrnet_dataset.jsonl',
    puzzle_dataset_path='gym/output/datasets/ptrnet_puzzles.jsonl'
)
viz.visualize_sample(0, 'gym/output/comprehensive/sample_000.png')
```
```

## Project Structure

```
RemingtonRoute/
├── gym/                           # Main Python package
│   ├── hamiltonian_puzzle_env.py  # Gymnasium environment
│   ├── generation/                # Dataset generation
│   │   └── generate_dataset.py
│   ├── visualization/             # Visualization tools
│   │   ├── main_visualizer.py     # 2x2 grid comprehensive view
│   │   ├── puzzle_visualizer.py   # Core drawing utilities
│   │   ├── visualize_dataset.py
│   │   └── visualize_ptrnet_dataset.py
│   ├── output/                    # Generated files
│   │   ├── datasets/              # JSONL training data
│   │   ├── comprehensive/         # 2x2 grid visualizations
│   │   ├── ptrnet/               # PTRNet visualizations
│   │   └── puzzle/               # Puzzle visualizations
│   └── documentation.md          # Full API documentation
├── Extension/                     # Browser extension
├── Frontend/                      # Web interface
└── README.md                      # This file
```

## Data Format

Each PTRNet sample encodes a 7×7 grid as 49 cells with 7 features each:
```
[x_norm, y_norm, waypoint_type, wall_up, wall_down, wall_left, wall_right]
```

- **Coordinates**: Normalized to [0, 1]
- **Waypoints**: 0=empty, 1=start, 2=checkpoint, 3=goal
- **Directional walls**: 1 if blocked, 0 if open

## Visualization Features

All views (human and PTR Network) share:
- ✅ Unified coordinate system (0-7, standard orientation)
- ✅ Clean axes (no tick labels)
- ✅ Cell indices (0-48)
- ✅ Identical markers (green circle for start, red square for goal, yellow plus for checkpoints)
- ✅ Matching colors and wall positions
- ✅ Single unified legend

## Documentation

See `gym/documentation.md` for:
- Detailed API documentation
- Data format specifications
- Usage examples
- Visualization style guide

## The Irony

This project exists to highlight the absurdity of applying expensive, complex machine learning solutions to problems with simple, efficient algorithmic solutions.

**Key Technical Insight:** The visualizations reveal the crucial difference between:
- **Edge-based walls** that block specific movement between cells
- **Cell-based encoding** where the ML model sees simplified movement constraints

Sometimes the most "innovative" approach is just using the right algorithm for the job.
