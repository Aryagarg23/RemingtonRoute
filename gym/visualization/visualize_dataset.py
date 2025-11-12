import json
from pathlib import Path
from gym.visualization.puzzle_visualizer import PuzzleVisualizer

# Path to your saved puzzles file
puzzle_path = Path(__file__).parent.parent.parent / "gym" / "output" / "datasets" / "ptrnet_puzzles.jsonl"

# Load one puzzle (you can pick another line to view a different one)
with open(puzzle_path, "r") as f:
    first_puzzle = json.loads(next(f))

# Convert stringified sets back to Python sets if needed
if isinstance(first_puzzle.get("walls"), list):
    first_puzzle["walls"] = {frozenset(map(tuple, w)) for w in first_puzzle["walls"]}

# Visualize it!
save_path = Path(__file__).parent.parent.parent / "gym" / "output" / "puzzle" / "first_puzzle_visualization.png"
PuzzleVisualizer.visualize_puzzle(first_puzzle, save_path=str(save_path))
print(f"Visualization saved to {save_path}")
