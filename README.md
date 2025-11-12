# RemingtonRoute 🧠🔫

In a world demanding "AI," the simple, correct, and computationally-trivial backtracking algorithm is no longer "innovative." It's too fast, too efficient, and worst of all, too explainable.

RemingtonRoute is our answer to this "problem."

We are building the "AI-powered Remington" that costs $10 million, requires a 50-page user manual, and has a 2% chance of missing... to solve a problem that a 100 dollar "glock" (a backtracking algorithm) solves every single time.

We are teaching a machine to guess the answer, because "guessing" (probabilistic modeling) is what "AI" does.

## What is this?

A Gymnasium environment for Hamiltonian path puzzles, complete with machine learning visualization tools. Uses Pointer Networks to solve problems that backtracking algorithms handle trivially.

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

### Generate Training Data
```bash
python -m gym.generation.generate_dataset
```

### Run Visualizations
```bash
# Comprehensive 2x2 grid: Human vs ML perspectives
python -m gym.visualization.main_visualizer --sample 0

# Individual puzzle visualization
python -m gym.visualization.visualize_dataset

# PTRNet dataset visualization
python -m gym.visualization.visualize_ptrnet_dataset --sample 0
```

## Project Structure

- `gym/` - Main Python package
  - `hamiltonian_puzzle_env.py` - Gymnasium environment
  - `generation/` - Dataset generation tools
  - `visualization/` - Visualization and plotting tools
  - `output/` - Generated datasets and visualizations
- `extension/` - Browser extension component

## Documentation

See `gym/documentation.md` for detailed API documentation and data formats.

## The Irony

This project exists to highlight the absurdity of applying expensive, complex machine learning solutions to problems with simple, efficient algorithmic solutions. Sometimes the most "innovative" approach is just using the right algorithm for the job.
