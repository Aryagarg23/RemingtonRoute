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
        """Reconstruct grid from flattened input data (now 8 features per cell)"""
        # Infer grid size from number of cells
        num_cells = len(input_data)
        grid_size = int(np.sqrt(num_cells))
        
        # Features: x, y, type, wall_up, wall_down, wall_left, wall_right, is_visited
        grid = np.zeros((grid_size, grid_size, 8))
        for i, cell_data in enumerate(input_data):
            row = i // grid_size
            col = i % grid_size
            grid[row, col] = cell_data
        return grid

    def visualize_sample(self, sample_idx, save_path=None):
        """Visualize a single sample with consistent formatting"""
        if sample_idx >= len(self.data):
            print(f"Sample index {sample_idx} out of range. Max: {len(self.data)-1}")
            return

        sample = self.data[sample_idx]
        input_data = sample['input']
        output_sequence = sample['output']

        # Infer grid size from input data
        num_cells = len(input_data)
        grid_size = int(np.sqrt(num_cells))

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Plot grid cells with white backgrounds
        for i in range(grid_size):
            for j in range(grid_size):
                cell_idx = i * grid_size + j
                if cell_idx < len(input_data):
                    # Draw cell with white background
                    rect = patches.Rectangle((j, grid_size-1-i), 1, 1, linewidth=1,
                                           edgecolor='black', facecolor='white')
                    ax.add_patch(rect)

                    # Add cell index
                    ax.text(j + 0.5, grid_size-1-i + 0.5, str(cell_idx),
                           ha='center', va='center', fontsize=6, fontweight='bold', 
                           color='gray', zorder=1)

        # Draw directional walls (8D features: last one is is_visited)
        for i in range(grid_size):
            for j in range(grid_size):
                cell_idx = i * grid_size + j
                if cell_idx < len(input_data):
                    cell_data = input_data[cell_idx]
                    
                    if len(cell_data) >= 7:
                        wall_up = int(cell_data[3])
                        wall_down = int(cell_data[4])
                        wall_left = int(cell_data[5])
                        wall_right = int(cell_data[6])
                        # cell_data[7] is is_visited (not used for visualization)
                        
                        # Draw walls at cell edges
                        if wall_up and i > 0:
                            ax.plot([j, j + 1], [grid_size - i, grid_size - i], 'r-', linewidth=4, 
                                   solid_capstyle='butt', zorder=5)
                        if wall_down and i < grid_size-1:
                            ax.plot([j, j + 1], [grid_size-1 - i, grid_size-1 - i], 'r-', linewidth=4, 
                                   solid_capstyle='butt', zorder=5)
                        if wall_left and j > 0:
                            ax.plot([j, j], [grid_size-1 - i, grid_size - i], 'r-', linewidth=4, 
                                   solid_capstyle='butt', zorder=5)
                        if wall_right and j < grid_size-1:
                            ax.plot([j + 1, j + 1], [grid_size-1 - i, grid_size - i], 'r-', linewidth=4, 
                                   solid_capstyle='butt', zorder=5)

        # Draw waypoint markers
        for i in range(grid_size):
            for j in range(grid_size):
                cell_idx = i * grid_size + j
                if cell_idx < len(input_data):
                    cell_data = input_data[cell_idx]
                    waypoint_type = int(cell_data[2])
                    
                    x = j + 0.5
                    y = grid_size-1 - i + 0.5
                    
                    if waypoint_type == 1:  # Start
                        ax.plot(x, y, 'go', markersize=12, zorder=15)
                        ax.text(x, y, 'S', ha='center', va='center', 
                               color='white', weight='bold', fontsize=8, zorder=16)
                    elif waypoint_type == 3:  # Goal
                        ax.plot(x, y, 'rs', markersize=12, zorder=15)
                        ax.text(x, y, 'G', ha='center', va='center', 
                               color='white', weight='bold', fontsize=8, zorder=16)
                    elif waypoint_type == 2:  # Checkpoint
                        ax.plot(x, y, 'yP', markersize=12, zorder=15)

        # Plot the path sequence
        path_x = []
        path_y = []
        for idx in output_sequence:
            row = idx // grid_size
            col = idx % grid_size
            path_x.append(col + 0.5)
            path_y.append(grid_size-1 - row + 0.5)

        # Draw path lines (blue to match comprehensive view)
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.6, zorder=10)

        # Set up axes
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.set_xticks(range(grid_size+1))
        ax.set_yticks(range(grid_size+1))
        ax.set_xticklabels([])  # No axis labels
        ax.set_yticklabels([])  # No axis labels
        ax.grid(True, alpha=0.3)
        ax.set_title(f'PTR Network Encoding ({grid_size}x{grid_size})', fontsize=14, fontweight='bold')

        # Unified legend at bottom
        legend_elements = [
            plt.Line2D([0], [0], color='red', linewidth=4, label='Walls'),
            plt.Line2D([0], [0], color='blue', linewidth=2, alpha=0.6, label='Solution Path'),
            plt.scatter([0], [0], c='green', marker='o', s=100, label='Start'),
            plt.scatter([0], [0], c='red', marker='s', s=100, label='Goal'),
            plt.scatter([0], [0], c='yellow', marker='P', s=100, label='Checkpoint')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                 ncol=5, fontsize=10, frameon=True)

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