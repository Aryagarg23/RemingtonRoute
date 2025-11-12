import json
from visualize_puzzle import PuzzleVisualizer

# Path to your saved puzzles file
puzzle_path = "ptrnet_dataset/ptrnet_puzzles.jsonl"

# Load one puzzle (you can pick another line to view a different one)
with open(puzzle_path, "r") as f:
    first_puzzle = json.loads(next(f))

# Convert stringified sets back to Python sets if needed
if isinstance(first_puzzle.get("walls"), list):
    first_puzzle["walls"] = {frozenset(map(tuple, w)) for w in first_puzzle["walls"]}

# Visualize it!
PuzzleVisualizer.visualize_puzzle(first_puzzle)
