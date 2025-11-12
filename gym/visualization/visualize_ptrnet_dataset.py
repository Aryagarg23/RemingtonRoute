#!/usr/bin/env python3
"""
Visualization script for the Pointer Network dataset.
Shows the grid puzzles and their solution paths.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import argparse

class PTRNetDatasetVisualizer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.data = []
        self.load_data()

    def load_data(self):
        """Load the JSONL dataset"""
        with open(self.dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))

    def reconstruct_grid(self, input_data):
        """Reconstruct 7x7 grid from flattened input data"""
        grid = np.zeros((7, 7, 4))  # x, y, type, is_checkpoint
        for i, cell_data in enumerate(input_data):
            row = i // 7
            col = i % 7
            grid[row, col] = cell_data
        return grid

    def visualize_sample(self, sample_idx, save_path=None):
        """Visualize a single sample"""
        if sample_idx >= len(self.data):
            print(f"Sample index {sample_idx} out of range. Max: {len(self.data)-1}")
            return

        sample = self.data[sample_idx]
        input_data = sample['input']
        output_sequence = sample['output']

        grid = self.reconstruct_grid(input_data)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

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
                        # Make blocked cells darker
                        if color == 'white':
                            color = 'lightgray'
                        elif color == 'lightgreen':
                            color = 'darkgreen'
                        elif color == 'yellow':
                            color = 'orange'
                        elif color == 'lightblue':
                            color = 'blue'

                    # Draw cell
                    rect = patches.Rectangle((j, 6-i), 1, 1, linewidth=1,
                                           edgecolor='black', facecolor=color)
                    ax.add_patch(rect)

                    # Add cell index
                    ax.text(j + 0.5, 6-i + 0.5, str(cell_idx),
                           ha='center', va='center', fontsize=6, fontweight='bold')

        # Plot the path sequence
        path_x = []
        path_y = []
        for idx in output_sequence:
            row = idx // 7
            col = idx % 7
            path_x.append(col + 0.5)
            path_y.append(6 - row + 0.5)

        # Draw path lines
        ax.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.7, zorder=10)

        # Draw path points
        ax.scatter(path_x, path_y, c='red', s=100, zorder=11)

        # Mark start and end
        if output_sequence:
            start_idx = output_sequence[0]
            end_idx = output_sequence[-1]
            start_row = start_idx // 7
            start_col = start_idx % 7
            end_row = end_idx // 7
            end_col = end_idx % 7

            ax.scatter([start_col + 0.5], [6 - start_row + 0.5],
                      c='green', s=200, marker='*', zorder=12, label='Start')
            ax.scatter([end_col + 0.5], [6 - end_row + 0.5],
                      c='purple', s=200, marker='X', zorder=12, label='End')

        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.set_aspect('equal')
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.grid(True, alpha=0.3)

        # Legend
        legend_elements = [
            patches.Patch(facecolor='white', edgecolor='black', label='Empty'),
            patches.Patch(facecolor='lightgreen', edgecolor='black', label='Start'),
            patches.Patch(facecolor='yellow', edgecolor='black', label='Checkpoint'),
            patches.Patch(facecolor='lightblue', edgecolor='black', label='Goal'),
            patches.Patch(facecolor='lightgray', edgecolor='black', label='Empty (blocked)'),
            patches.Patch(facecolor='darkgreen', edgecolor='black', label='Start (blocked)'),
            patches.Patch(facecolor='orange', edgecolor='black', label='Checkpoint (blocked)'),
            patches.Patch(facecolor='blue', edgecolor='black', label='Goal (blocked)'),
            plt.Line2D([0], [0], color='red', linewidth=3, label='Path'),
            plt.scatter([0], [0], c='green', s=100, marker='*', label='Start'),
            plt.scatter([0], [0], c='purple', s=100, marker='X', label='End')
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title(f'PTRNet Dataset Sample {sample_idx}\nPath Length: {len(output_sequence)}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

    def visualize_multiple_samples(self, num_samples=5, save_dir=None):
        """Visualize multiple samples"""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        for i in range(min(num_samples, len(self.data))):
            save_path = save_dir / f'sample_{i:03d}.png' if save_dir else None
            self.visualize_sample(i, save_path)
            if not save_path:
                plt.close()  # Close figure if not saving

    def print_sample_info(self, sample_idx):
        """Print detailed information about a sample"""
        if sample_idx >= len(self.data):
            print(f"Sample index {sample_idx} out of range. Max: {len(self.data)-1}")
            return

        sample = self.data[sample_idx]
        input_data = sample['input']
        output_sequence = sample['output']

        print(f"=== Sample {sample_idx} ===")
        print(f"Input length: {len(input_data)}")
        print(f"Output sequence length: {len(output_sequence)}")
        print(f"Output sequence: {output_sequence}")

        # Count different waypoint types
        waypoint_counts = {}
        blocked_count = 0
        for cell in input_data:
            waypoint_type = int(cell[2])
            is_blocked = int(cell[3])
            waypoint_counts[waypoint_type] = waypoint_counts.get(waypoint_type, 0) + 1
            if is_blocked:
                blocked_count += 1

        print(f"Waypoint type counts: {waypoint_counts}")
        print(f"Total blocked cells: {blocked_count}")

        # Validate path - should visit all 49 cells in order
        expected_sequence = list(range(49))
        path_valid = (output_sequence == expected_sequence)
        print(f"Path is sequential (0-48): {path_valid}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Visualize PTRNet dataset')
    parser.add_argument('--dataset', default='gym/output/datasets/ptrnet_dataset.jsonl',
                       help='Path to dataset file')
    parser.add_argument('--sample', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--save-dir', default='gym/output/ptrnet',
                       help='Directory to save visualizations (default: gym/output/ptrnet)')
    parser.add_argument('--info', action='store_true',
                       help='Print sample information instead of visualizing')

    args = parser.parse_args()

    visualizer = PTRNetDatasetVisualizer(args.dataset)

    print(f"Loaded {len(visualizer.data)} samples from {args.dataset}")

    if args.info:
        for i in range(min(args.num_samples, len(visualizer.data))):
            visualizer.print_sample_info(i)
    else:
        if args.save_dir:
            visualizer.visualize_multiple_samples(args.num_samples, args.save_dir)
        else:
            visualizer.visualize_sample(args.sample)


if __name__ == '__main__':
    main()