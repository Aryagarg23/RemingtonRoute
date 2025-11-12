#!/usr/bin/env python3
"""
Main Visualizer - Shows Human vs PTR Network perspectives in a 2x2 grid.
Top row: Human perspective (complete/incomplete)
Bottom row: PTR Network (ML) perspective (with/without solution)
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import argparse

from .puzzle_visualizer import PuzzleVisualizer
from .visualize_ptrnet_dataset import PTRNetDatasetVisualizer


class MainVisualizer:
    """Main visualizer that combines PTR Network, Incomplete, and Solution views."""

    def __init__(self, ptrnet_dataset_path: str, puzzle_dataset_path: str):
        self.ptrnet_path = Path(ptrnet_dataset_path)
        self.puzzle_path = Path(puzzle_dataset_path)
        self.ptrnet_viz = PTRNetDatasetVisualizer(str(self.ptrnet_path))

        # Load corresponding puzzle data
        self.puzzle_data = []
        self._load_puzzle_data()

    def _load_puzzle_data(self):
        """Load puzzle data corresponding to PTRNet data."""
        with open(self.puzzle_path, 'r') as f:
            for line in f:
                if line.strip():
                    puzzle = json.loads(line.strip())
                    # Convert stringified sets back to Python sets if needed
                    if isinstance(puzzle.get("walls"), list):
                        puzzle["walls"] = {frozenset(map(tuple, w)) for w in puzzle["walls"]}
                    self.puzzle_data.append(puzzle)

    def visualize_sample(self, sample_idx: int, save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive 2x2 visualization showing:
        Top row: Human perspective (complete solution | incomplete puzzle)
        Bottom row: PTR Network perspective (with solution | grid only)
        """
        if sample_idx >= len(self.ptrnet_viz.data) or sample_idx >= len(self.puzzle_data):
            print(f"Sample index {sample_idx} out of range.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Top row: Human views
        # 1. Complete Solution View (Human)
        puzzle = self.puzzle_data[sample_idx]
        PuzzleVisualizer.draw_puzzle(puzzle, show_solution=True, ax=axes[0, 0], add_legend=False)
        axes[0, 0].set_title('Human: Complete Solution', fontsize=14, fontweight='bold')

        # Add legend for complete solution
        complete_legend = [
            plt.Line2D([0], [0], color='red', linewidth=4, label='Walls'),
            plt.Line2D([0], [0], color='blue', linewidth=2, alpha=0.6, label='Solution Path'),
            plt.scatter([0], [0], c='green', marker='o', s=100, label='Start'),
            plt.scatter([0], [0], c='red', marker='s', s=100, label='Goal'),
            plt.scatter([0], [0], c='yellow', marker='P', s=100, label='Checkpoint')
        ]
        axes[0, 0].legend(handles=complete_legend, loc='upper right', fontsize=8)

        # 2. Incomplete Puzzle View (Human)
        PuzzleVisualizer.draw_puzzle(puzzle, show_solution=False, ax=axes[0, 1], add_legend=False)
        axes[0, 1].set_title('Human: Incomplete Puzzle', fontsize=14, fontweight='bold')

        # Add legend for incomplete puzzle
        incomplete_legend = [
            plt.Line2D([0], [0], color='red', linewidth=4, label='Walls'),
            plt.scatter([0], [0], c='green', marker='o', s=100, label='Start'),
            plt.scatter([0], [0], c='red', marker='s', s=100, label='Goal'),
            plt.scatter([0], [0], c='yellow', marker='P', s=100, label='Checkpoint')
        ]
        axes[0, 1].legend(handles=incomplete_legend, loc='upper right', fontsize=8)

        # Bottom row: PTR Network views
        # 3. PTR Network with Solution
        ptrnet_sample = self.ptrnet_viz.data[sample_idx]
        self._draw_ptrnet_view(ptrnet_sample, axes[1, 0], show_solution=True)
        axes[1, 0].set_title('PTR Network: With Solution Path', fontsize=14, fontweight='bold')

        # 4. PTR Network without Solution
        self._draw_ptrnet_view(ptrnet_sample, axes[1, 1], show_solution=False)
        axes[1, 1].set_title('PTR Network: Grid Only', fontsize=14, fontweight='bold')

        # Overall title
        fig.suptitle(f'Comprehensive Puzzle Visualization - Sample {sample_idx}\nTop: Human Perspective | Bottom: PTR Network (ML) Perspective',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comprehensive visualization to {save_path}")
        else:
            plt.show()

    def _draw_ptrnet_view(self, sample: dict, ax: plt.Axes, show_solution: bool = True) -> None:
        """Draw the PTR Network representation view."""
        input_data = sample['input']

        # Color mapping for different waypoint types
        waypoint_colors = {
            0: 'white',        # none
            1: 'lightgreen',   # start
            2: 'yellow',       # checkpoint
            3: 'lightblue'     # goal
        }

        # Plot the grid
        for i in range(7):
            for j in range(7):
                cell_idx = i * 7 + j
                if cell_idx < len(input_data):
                    cell_data = input_data[cell_idx]
                    waypoint_type = int(cell_data[2])
                    is_blocked = int(cell_data[3])

                    # Base color from waypoint type
                    color = waypoint_colors.get(waypoint_type, 'white')

                    # Darker shade if blocked
                    if is_blocked:
                        if color == 'white':
                            color = 'lightgray'
                        elif color == 'lightgreen':
                            color = 'darkgreen'
                        elif color == 'yellow':
                            color = 'orange'
                        elif color == 'lightblue':
                            color = 'blue'

                    # Draw cell
                    rect = plt.Rectangle((j, 6-i), 1, 1, linewidth=1,
                                       edgecolor='black', facecolor=color)
                    ax.add_patch(rect)

                    # Add cell index
                    ax.text(j + 0.5, 6-i + 0.5, str(cell_idx),
                           ha='center', va='center', fontsize=6, fontweight='bold')

        # Draw the path sequence
        if show_solution:
            output_sequence = sample['output']
            path_x = []
            path_y = []
            for idx in output_sequence:
                row = idx // 7
                col = idx % 7
                path_x.append(col + 0.5)
                path_y.append(6 - row + 0.5)

            # Draw path lines
            ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.8, zorder=10)

            # Draw path points
            ax.scatter(path_x, path_y, c='red', s=50, zorder=11)

        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.set_aspect('equal')
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.grid(True, alpha=0.3)

        # Add legend for PTRNet view
        import matplotlib.patches as patches
        legend_elements = [
            patches.Patch(facecolor='white', edgecolor='black', label='Empty'),
            patches.Patch(facecolor='lightgreen', edgecolor='black', label='Start'),
            patches.Patch(facecolor='yellow', edgecolor='black', label='Checkpoint'),
            patches.Patch(facecolor='lightblue', edgecolor='black', label='Goal'),
            patches.Patch(facecolor='lightgray', edgecolor='black', label='Empty (blocked)'),
            patches.Patch(facecolor='darkgreen', edgecolor='black', label='Start (blocked)'),
            patches.Patch(facecolor='orange', edgecolor='black', label='Checkpoint (blocked)'),
            patches.Patch(facecolor='blue', edgecolor='black', label='Goal (blocked)')
        ]
        if show_solution:
            legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=2, label='Solution Path'))
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    def visualize_multiple_samples(self, num_samples: int = 5, save_dir: Optional[str] = None) -> None:
        """Visualize multiple samples."""
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(exist_ok=True)

        for i in range(min(num_samples, len(self.ptrnet_viz.data), len(self.puzzle_data))):
            save_path = save_dir_path / f'comprehensive_sample_{i:03d}.png' if save_dir else None
            self.visualize_sample(i, save_path)
            if not save_path:
                plt.close()


def main():
    parser = argparse.ArgumentParser(description='Main visualizer for PTRNet puzzles')
    parser.add_argument('--ptrnet-dataset', default='gym/output/datasets/ptrnet_dataset.jsonl',
                       help='Path to PTRNet dataset file')
    parser.add_argument('--puzzle-dataset', default='gym/output/datasets/ptrnet_puzzles.jsonl',
                       help='Path to puzzle dataset file')
    parser.add_argument('--sample', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--save-dir', default='gym/output/comprehensive',
                       help='Directory to save visualizations (default: gym/output/comprehensive)')

    args = parser.parse_args()

    visualizer = MainVisualizer(args.ptrnet_dataset, args.puzzle_dataset)

    print(f"Loaded {len(visualizer.ptrnet_viz.data)} PTRNet samples and {len(visualizer.puzzle_data)} puzzles")

    if args.save_dir:
        visualizer.visualize_multiple_samples(args.num_samples, args.save_dir)
    else:
        visualizer.visualize_sample(args.sample)


if __name__ == '__main__':
    main()