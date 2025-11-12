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
from ..hamiltonian_puzzle_env import HamiltonianPuzzleEnv


class PuzzleVisualizer:
    """Handles visualization of Hamiltonian puzzles using matplotlib.

    Provides static methods to draw individual puzzles or side-by-side
    incomplete/complete views. Supports saving to files or displaying.
    """

    @staticmethod
    def draw_puzzle(puzzle: Dict[str, Any], show_solution: bool = False,
                    ax: Optional[plt.Axes] = None, title: str = "", add_legend: bool = True) -> plt.Axes:
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

        # Grid setup - match PTR Network coordinate system
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks(range(cols + 1))
        ax.set_yticks(range(rows + 1))
        ax.set_xticklabels([])  # No axis labels
        ax.set_yticklabels([])  # No axis labels
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

        # Draw grid cells as white rectangles
        import matplotlib.patches as patches
        for r in range(rows):
            for c in range(cols):
                rect = patches.Rectangle((c, rows - 1 - r), 1, 1, linewidth=1,
                                   edgecolor='black', facecolor='white')
                ax.add_patch(rect)
                
                # Add cell index number (like PTR Network)
                cell_idx = r * cols + c
                ax.text(c + 0.5, rows - 1 - r + 0.5, str(cell_idx),
                       ha='center', va='center', fontsize=6, fontweight='bold', color='gray', zorder=1)

        # Draw walls using PTR Network coordinate system
        for wall in walls:
            a, b = list(wall)
            r1, c1 = a
            r2, c2 = b
            
            # Convert to visual coordinates (matching PTR Network)
            y1 = rows - 1 - r1
            y2 = rows - 1 - r2
            
            if r1 == r2:  # vertical wall (same row, different columns)
                # Wall is between two horizontally adjacent cells
                x_wall = max(c1, c2)  # Right edge of left cell
                y_bottom = y1
                y_top = y1 + 1
                ax.plot([x_wall, x_wall], [y_bottom, y_top], 'r-', linewidth=4, solid_capstyle='butt', zorder=5)
            elif c1 == c2:  # horizontal wall (same column, different rows)
                # Wall is between two vertically adjacent cells
                x_left = c1
                x_right = c1 + 1
                y_wall = max(y1, y2)  # Top edge of bottom cell
                ax.plot([x_left, x_right], [y_wall, y_wall], 'r-', linewidth=4, solid_capstyle='butt', zorder=5)

        # Draw solution path if requested
        if show_solution and len(path) > 1:
            path_x = [c + 0.5 for r, c in path]
            path_y = [rows - 1 - r + 0.5 for r, c in path]
            ax.plot(path_x, path_y, 'b-', alpha=0.6, linewidth=2, label='Solution Path', zorder=10)

        # Draw start and goal
        start = checkpoints['start']
        goal = checkpoints['goal']
        
        start_x = start[1] + 0.5
        start_y = rows - 1 - start[0] + 0.5
        ax.plot(start_x, start_y, 'go', markersize=12, zorder=15)
        ax.text(start_x, start_y, 'S', ha='center', va='center', color='white', weight='bold', zorder=16)

        goal_x = goal[1] + 0.5
        goal_y = rows - 1 - goal[0] + 0.5
        ax.plot(goal_x, goal_y, 'rs', markersize=12, zorder=15)
        ax.text(goal_x, goal_y, 'G', ha='center', va='center', color='white', weight='bold', zorder=16)

        # Draw intermediate checkpoints
        for i, cp in enumerate(checkpoints['checkpoints']):
            cp_x = cp[1] + 0.5
            cp_y = rows - 1 - cp[0] + 0.5
            ax.plot(cp_x, cp_y, 'yP', markersize=12, zorder=15)
            ax.text(cp_x, cp_y, str(i + 1), ha='center', va='center', color='black', weight='bold', zorder=16)

        # Add legend
        if add_legend:
            legend_elements = [
                plt.Line2D([0], [0], color='red', linewidth=4, label='Walls'),
                plt.Line2D([0], [0], color='blue', linewidth=2, alpha=0.6, label='Solution Path'),
                plt.scatter([0], [0], c='green', marker='o', s=100, label='Start'),
                plt.scatter([0], [0], c='red', marker='s', s=100, label='Goal'),
                plt.scatter([0], [0], c='yellow', marker='P', s=100, label='Checkpoint')
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

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
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: incomplete puzzle
        PuzzleVisualizer.draw_puzzle(
            puzzle, show_solution=False, ax=axes[0],
            title=f"Incomplete Puzzle"
        )

        # Right: complete puzzle
        PuzzleVisualizer.draw_puzzle(
            puzzle, show_solution=True, ax=axes[1],
            title=f"Complete Solution"
        )

        # Remove individual legends
        for ax in axes:
            if ax.get_legend():
                ax.get_legend().remove()

        # Add unified figure legend at bottom
        legend_elements = [
            plt.Line2D([0], [0], color='red', linewidth=4, label='Walls'),
            plt.Line2D([0], [0], color='blue', linewidth=2, alpha=0.6, label='Solution Path'),
            plt.scatter([0], [0], c='green', marker='o', s=100, label='Start'),
            plt.scatter([0], [0], c='red', marker='s', s=100, label='Goal'),
            plt.scatter([0], [0], c='yellow', marker='P', s=100, label='Checkpoint')
        ]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                  ncol=5, fontsize=12, frameon=True)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Make room for legend
        
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
