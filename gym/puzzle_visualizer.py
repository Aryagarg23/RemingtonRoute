"""
Puzzle Visualization Utilities.

This module provides classes for visualizing Hamiltonian puzzles,
including static drawing and dataset generation.

Classes:
    PuzzleVisualizer: Static methods for drawing and visualizing puzzles.
    PuzzleDatasetGenerator: Generates datasets of puzzle visualizations.
"""

import os
import random
import time
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from .hamiltonian_puzzle_env import HamiltonianPuzzleEnv


class PuzzleVisualizer:
    """Handles visualization of Hamiltonian puzzles using matplotlib.

    Provides static methods to draw individual puzzles or side-by-side
    incomplete/complete views. Supports saving to files or displaying.
    """

    @staticmethod
    def draw_puzzle(puzzle: Dict[str, Any], show_solution: bool = False,
                    ax: Optional[plt.Axes] = None, title: str = "") -> plt.Axes:
        """
        Draws a single puzzle on a given matplotlib axis.

        Args:
            puzzle: Puzzle data with 'solution_path', 'checkpoints', 'walls', 'rows', 'cols'.
            show_solution: If True, draw the full solution path.
            ax: Axis to draw on. If None, creates a new figure.
            title: Optional title for the plot.

        Returns:
            The matplotlib Axes object.
        """
        path = puzzle['solution_path']
        checkpoints = puzzle['checkpoints']
        walls = puzzle['walls']
        rows, cols = puzzle['rows'], puzzle['cols']

        if ax is None:
            fig, ax = plt.subplots(figsize=(cols, rows))

        # Grid setup
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xticks([x - 0.5 for x in range(1, cols)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, rows)], minor=True)
        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title(title)

        # Draw walls
        for wall in walls:
            a, b = list(wall)
            r1, c1 = a
            r2, c2 = b
            if r1 == r2:  # vertical wall
                ax.plot([min(c1, c2) + 0.5] * 2, [r1 - 0.5, r1 + 0.5],
                        'r-', linewidth=4, solid_capstyle='butt')
            elif c1 == c2:  # horizontal wall
                ax.plot([c1 - 0.5, c1 + 0.5], [min(r1, r2) + 0.5] * 2,
                        'r-', linewidth=4, solid_capstyle='butt')

        # Draw solution path if requested
        if show_solution and len(path) > 1:
            r_coords, c_coords = zip(*path)
            ax.plot(c_coords, r_coords, 'b-', alpha=0.6, linewidth=2, label='Solution Path')

        # Draw start and goal
        start = checkpoints['start']
        goal = checkpoints['goal']
        ax.plot(start[1], start[0], 'go', markersize=12)
        ax.text(start[1], start[0], 'S', ha='center', va='center', color='white', weight='bold')

        ax.plot(goal[1], goal[0], 'rs', markersize=12)
        ax.text(goal[1], goal[0], 'G', ha='center', va='center', color='white', weight='bold')

        # Draw intermediate checkpoints
        for i, cp in enumerate(checkpoints['checkpoints']):
            ax.plot(cp[1], cp[0], 'yP', markersize=12)
            ax.text(cp[1], cp[0], str(i + 1), ha='center', va='center', color='black', weight='bold')

        return ax

    @staticmethod
    def visualize_puzzle(puzzle: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """
        Visualizes a puzzle side by side: incomplete vs complete.
        Optionally saves it to a file.

        Args:
            puzzle: The puzzle data.
            save_path: If provided, saves the figure to this path.
        """
        num_checkpoints = len(puzzle['checkpoints']['checkpoints'])
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Left: incomplete puzzle
        PuzzleVisualizer.draw_puzzle(
            puzzle, show_solution=False, ax=axes[0],
            title=f"Incomplete ({num_checkpoints} CPs)"
        )

        # Right: complete puzzle
        PuzzleVisualizer.draw_puzzle(
            puzzle, show_solution=True, ax=axes[1],
            title=f"Complete Solution ({num_checkpoints} CPs)"
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()


class PuzzleDatasetGenerator:
    """Generates and visualizes Hamiltonian puzzles for datasets.

    Creates random puzzles and saves their visualizations as images.
    Useful for generating training data for ML models.

    Args:
        output_dir: Directory to save images.
        num_samples: Number of puzzles to generate.
        rows: Grid rows.
        cols: Grid columns.
        checkpoint_range: Tuple (min, max) for random checkpoint count.
        wall_probability: Probability of walls.
    """
    def __init__(self, output_dir: str = "dataset_samples_side_by_side/",
                 num_samples: int = 20, rows: int = 7, cols: int = 7,
                 checkpoint_range: tuple = (13, 13), wall_probability: float = 0.15):
        """
        Initialize the dataset generator.

        Args:
            output_dir: Directory to save images.
            num_samples: Number of samples to generate.
            rows: Grid rows.
            cols: Grid columns.
            checkpoint_range: Min/max checkpoints.
            wall_probability: Wall generation probability.
        """
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.rows = rows
        self.cols = cols
        self.checkpoint_range = checkpoint_range
        self.wall_probability = wall_probability
        os.makedirs(output_dir, exist_ok=True)

    def generate_dataset(self) -> None:
        """Generates a dataset of puzzles and saves visualizations."""
        start_time = time.time()

        for i in range(self.num_samples):
            num_checkpoints = random.randint(*self.checkpoint_range)

            # Create environment
            env = HamiltonianPuzzleEnv(
                rows=self.rows, cols=self.cols,
                num_checkpoints=num_checkpoints,
                wall_probability=self.wall_probability,
                render_mode=None
            )

            env.reset()
            puzzle = env.puzzle_data

            save_path = os.path.join(self.output_dir, f"sample_{i}_side_by_side.png")
            PuzzleVisualizer.visualize_puzzle(puzzle, save_path=save_path)

            env.close()

        end_time = time.time()
        print(f"✅ Generated {self.num_samples} samples in {end_time - start_time:.2f} seconds.")
        print(f"📁 Images saved in: {self.output_dir}")


if __name__ == "__main__":
    generator = PuzzleDatasetGenerator(
        num_samples=5,
        rows=7, cols=7,
        checkpoint_range=(13, 13),
        wall_probability=0.15
    )
    generator.generate_dataset()
