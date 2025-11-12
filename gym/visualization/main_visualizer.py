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

from gym.visualization.puzzle_visualizer import PuzzleVisualizer
from gym.visualization.visualize_ptrnet_dataset import PTRNetDatasetVisualizer


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

        # Define unified legend elements
        import matplotlib.patches as patches
        unified_legend = [
            plt.Line2D([0], [0], color='red', linewidth=4, label='Walls'),
            plt.Line2D([0], [0], color='blue', linewidth=2, alpha=0.6, label='Solution Path'),
            plt.scatter([0], [0], c='green', marker='o', s=100, label='Start'),
            plt.scatter([0], [0], c='red', marker='s', s=100, label='Goal'),
            plt.scatter([0], [0], c='yellow', marker='P', s=100, label='Checkpoint')
        ]

        # Top row: Human views
        # 1. Complete Solution View (Human)
        puzzle = self.puzzle_data[sample_idx]
        PuzzleVisualizer.draw_puzzle(puzzle, show_solution=True, ax=axes[0, 0], add_legend=False)
        axes[0, 0].set_title('Human: Complete Solution', fontsize=14, fontweight='bold')

        # 2. Incomplete Puzzle View (Human)
        PuzzleVisualizer.draw_puzzle(puzzle, show_solution=False, ax=axes[0, 1], add_legend=False)
        axes[0, 1].set_title('Human: Incomplete Puzzle', fontsize=14, fontweight='bold')

        # Bottom row: PTR Network views
        # 3. PTR Network with Solution
        ptrnet_sample = self.ptrnet_viz.data[sample_idx]
        self._draw_ptrnet_view(ptrnet_sample, axes[1, 0], puzzle, show_solution=True)
        axes[1, 0].set_title('PTR Network: With Solution Path', fontsize=14, fontweight='bold')

        # 4. PTR Network without Solution
        self._draw_ptrnet_view(ptrnet_sample, axes[1, 1], puzzle, show_solution=False)
        axes[1, 1].set_title('PTR Network: Grid Only', fontsize=14, fontweight='bold')

        # Add unified legend at the bottom
        fig.legend(handles=unified_legend, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                  ncol=5, fontsize=12, frameon=True)

        # Overall title
        fig.suptitle(f'Comprehensive Puzzle Visualization - Sample {sample_idx}\nTop: Human Perspective | Bottom: PTR Network (ML) Perspective',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comprehensive visualization to {save_path}")
        else:
            plt.show()

    def _draw_ptrnet_view(self, sample: dict, ax: plt.Axes, puzzle: dict, show_solution: bool = True) -> None:
        """Draw the PTR Network representation view with same visual style as human view."""
        input_data = sample['input']
        
        # Get actual grid dimensions from puzzle
        rows = puzzle['rows']
        cols = puzzle['cols']

        # Plot the grid - all cells are white background
        import matplotlib.patches as patches
        for i in range(rows):
            for j in range(cols):
                cell_idx = i * cols + j
                if cell_idx < len(input_data):
                    cell_data = input_data[cell_idx]

                    # Draw cell with white background
                    rect = patches.Rectangle((j, (rows-1)-i), 1, 1, linewidth=1,
                                       edgecolor='black', facecolor='white')
                    ax.add_patch(rect)

                    # Draw directional walls (if new encoding with 8 features)
                    if len(cell_data) >= 7:
                        wall_up = int(cell_data[3])
                        wall_down = int(cell_data[4])
                        wall_left = int(cell_data[5])
                        wall_right = int(cell_data[6])
                        # cell_data[7] is is_visited (not used for static wall visualization)
                        
                        # Draw walls as thick red lines on the cell edges
                        # Convert grid coordinates (i,j) to visual coordinates
                        c = j  # column (x-coordinate)
                        r = (rows-1) - i  # row (y-coordinate for bottom edge, inverted)
                        
                        if wall_up and i > 0:  # Wall to cell above
                            # Horizontal line at top edge
                            ax.plot([c, c + 1], [r + 1, r + 1], 'r-', linewidth=4, solid_capstyle='butt', zorder=5)
                        if wall_down and i < rows-1:  # Wall to cell below
                            # Horizontal line at bottom edge
                            ax.plot([c, c + 1], [r, r], 'r-', linewidth=4, solid_capstyle='butt', zorder=5)
                        if wall_left and j > 0:  # Wall to cell on left
                            # Vertical line at left edge
                            ax.plot([j, j], [r, r + 1], 'r-', linewidth=4, solid_capstyle='butt', zorder=5)
                        if wall_right and j < cols-1:  # Wall to cell on right
                            # Vertical line at right edge
                            ax.plot([j + 1, j + 1], [r, r + 1], 'r-', linewidth=4, solid_capstyle='butt', zorder=5)

                    # Add cell index
                    ax.text(j + 0.5, (rows-1)-i + 0.5, str(cell_idx),
                           ha='center', va='center', fontsize=6, fontweight='bold')

        # Draw waypoint markers (matching human visualizer style)
        # Create checkpoint position to index mapping
        checkpoint_positions = puzzle['checkpoints']['checkpoints']
        
        for i in range(rows):
            for j in range(cols):
                cell_idx = i * cols + j
                if cell_idx < len(input_data):
                    cell_data = input_data[cell_idx]
                    waypoint_type = int(cell_data[2])
                    
                    x = j + 0.5
                    y = (rows-1) - i + 0.5
                    
                    if waypoint_type == 1:  # Start - green circle
                        ax.plot(x, y, 'go', markersize=12, zorder=15)
                        ax.text(x, y, 'S', ha='center', va='center', color='white', weight='bold', fontsize=8, zorder=16)
                    elif waypoint_type == 3:  # Goal - red square
                        ax.plot(x, y, 'rs', markersize=12, zorder=15)
                        ax.text(x, y, 'G', ha='center', va='center', color='white', weight='bold', fontsize=8, zorder=16)
                    elif waypoint_type == 2:  # Checkpoint - yellow plus with number
                        ax.plot(x, y, 'yP', markersize=12, zorder=15)
                        # Find checkpoint index - convert to tuple for comparison
                        cell_pos = (i, j)
                        for cp_idx, cp_pos in enumerate(checkpoint_positions):
                            # Convert list to tuple for comparison
                            if tuple(cp_pos) == cell_pos:
                                ax.text(x, y, str(cp_idx + 1), ha='center', va='center', 
                                       color='black', weight='bold', fontsize=8, zorder=16)
                                break

        # Draw the path sequence
        if show_solution:
            output_sequence = sample['output']
            path_x = []
            path_y = []
            for idx in output_sequence:
                row = idx // cols
                col = idx % cols
                path_x.append(col + 0.5)
                path_y.append((rows-1) - row + 0.5)

            # Draw path lines - blue to match human visualizer
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.6, zorder=10)

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.set_xticks(range(cols+1))
        ax.set_yticks(range(rows+1))
        ax.set_xticklabels([])  # No axis labels
        ax.set_yticklabels([])  # No axis labels
        ax.grid(True, alpha=0.3)

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