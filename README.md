# RemingtonRoute 🧠🔫

In a world demanding "AI," the simple, correct, and computationally-trivial backtracking algorithm is no longer "innovative." It's too fast, too efficient, and worst of all, too explainable.

RemingtonRoute is our answer to this "problem."

We are building the "AI-powered Remington" that costs $10 million, requires a 50-page user manual, and has a 2% chance of missing... to solve a problem that a 100 dollar "glock" (a backtracking algorithm) solves every single time.

We are teaching a machine to guess the answer, because "guessing" (probabilistic modeling) is what "AI" does.

## What is this?

A Gymnasium environment for Hamiltonian path puzzles, complete with machine learning visualization tools. Uses Pointer Networks to solve problems that backtracking algorithms handle trivially.

**Features:**
- Directional wall encoding (4-way: up/down/left/right)
- Sequential checkpoints
- Variable grid sizes
- Pointer Network with supervised learning

## Quick Start

```bash
# Generate dataset and visualizations
python gym/generator_main.py

# Train Pointer Network
python model/generate_dataset.py --mode variable --num-samples 10000
python model/train.py
python model/evaluate.py --visualize 10
```

## Pointer Network

Supervised learning with variable grid sizes (5×5 to 12×12), 5-15 checkpoints.

```bash
python model/generate_dataset.py --mode variable --num-samples 10000
python model/train.py
python model/evaluate.py --beam-search --visualize 10
```

## Project Structure

```
RemingtonRoute/
├── gym/                           # Main Python package
│   ├── hamiltonian_puzzle_env.py  # Gymnasium environment
│   ├── generator_main.py          # One-command generation pipeline
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
│   └── documentation.md           # Full API documentation
├── model/                         # Pointer Network ML
│   ├── ptrnet.py                  # Network architecture
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation & inference
│   ├── generate_dataset.py        # Variable-size data generator
│   ├── quickstart.py              # One-command ML pipeline
│   ├── requirements.txt           # Python dependencies
│   ├── checkpoints/               # Saved models
│   ├── logs/                      # Training logs
│   └── README.md                  # ML documentation
├── Extension/                     # Browser extension
├── Frontend/                      # Web interface
└── README.md                      # This file
```

## Data Format

Cell encoding: `[x_norm, y_norm, waypoint_type, wall_up, wall_down, wall_left, wall_right, is_visited]`
- Waypoints: 0=empty, 1=start, 2=checkpoint, 3=goal
- Walls: 1=blocked, 0=open
- Visited: 0=unvisited, 1=visited (updated during path construction)
