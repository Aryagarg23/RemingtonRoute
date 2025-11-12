#!/usr/bin/env python3
"""
Main generator script for RemingtonRoute.
Cleans up old files, generates fresh datasets, and creates sample visualizations.
"""

import os
import sys
import shutil
import json
from pathlib import Path

# Ensure we can import from gym package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gym.generation.generate_dataset import generate_ptrnet_dataset
from gym.visualization.main_visualizer import MainVisualizer
from gym.visualization.visualize_ptrnet_dataset import PTRNetDatasetVisualizer
from gym.visualization.puzzle_visualizer import PuzzleVisualizer


class RemingtonRouteGenerator:
    """Main generator class for creating datasets and visualizations."""
    
    def __init__(self, output_dir='gym/output'):
        self.output_dir = Path(output_dir)
        self.datasets_dir = self.output_dir / 'datasets'
        self.comprehensive_dir = self.output_dir / 'comprehensive'
        self.ptrnet_dir = self.output_dir / 'ptrnet'
        self.puzzle_dir = self.output_dir / 'puzzle'
        
    def clean_output_directory(self):
        """Remove all generated files and recreate directory structure."""
        print("=" * 60)
        print("CLEANING OUTPUT DIRECTORY")
        print("=" * 60)
        
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            print(f"✓ Removed {self.output_dir}")
        
        # Recreate directory structure
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.comprehensive_dir.mkdir(parents=True, exist_ok=True)
        self.ptrnet_dir.mkdir(parents=True, exist_ok=True)
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created fresh directory structure")
        print()
        
    def generate_dataset(self, num_samples=1000):
        """Generate training dataset."""
        print("=" * 60)
        print("GENERATING DATASET")
        print("=" * 60)
        
        generate_ptrnet_dataset(
            num_samples=num_samples,
            rows=7,
            cols=7,
            checkpoint_range=(3, 6),
            wall_probability=0.15,
            output_dir=str(self.datasets_dir) + '/',
            save_format='jsonl'
        )
        
        print(f"✓ Generated {num_samples} training samples")
        print()
        
    def generate_comprehensive_visualizations(self, num_samples=5):
        """Generate comprehensive 2x2 grid visualizations."""
        print("=" * 60)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 60)
        
        viz = MainVisualizer(
            ptrnet_dataset_path=str(self.datasets_dir / 'ptrnet_dataset.jsonl'),
            puzzle_dataset_path=str(self.datasets_dir / 'ptrnet_puzzles.jsonl')
        )
        
        for i in range(num_samples):
            save_path = self.comprehensive_dir / f'sample_{i:03d}.png'
            viz.visualize_sample(i, str(save_path))
            print(f"  ✓ Sample {i}")
        
        print()
        
    def generate_ptrnet_visualizations(self, num_samples=5):
        """Generate PTRNet-specific visualizations."""
        print("=" * 60)
        print("GENERATING PTRNET VISUALIZATIONS")
        print("=" * 60)
        
        viz = PTRNetDatasetVisualizer(str(self.datasets_dir / 'ptrnet_dataset.jsonl'))
        
        for i in range(num_samples):
            save_path = self.ptrnet_dir / f'sample_{i:03d}.png'
            viz.visualize_sample(i, str(save_path))
            print(f"  ✓ Sample {i}")
        
        print()
        
    def generate_puzzle_visualizations(self, num_samples=5):
        """Generate puzzle-only visualizations."""
        print("=" * 60)
        print("GENERATING PUZZLE VISUALIZATIONS")
        print("=" * 60)
        
        # Load puzzles
        puzzles = []
        with open(self.datasets_dir / 'ptrnet_puzzles.jsonl', 'r') as f:
            for line in f:
                if line.strip():
                    puzzle = json.loads(line.strip())
                    # Convert wall_set back to frozenset format
                    if 'wall_set' in puzzle:
                        walls = set()
                        for wall in puzzle['wall_set']:
                            walls.add(frozenset([tuple(wall[0]), tuple(wall[1])]))
                        puzzle['walls'] = walls
                    puzzles.append(puzzle)
        
        for i in range(num_samples):
            save_path = self.puzzle_dir / f'sample_{i:03d}.png'
            PuzzleVisualizer.visualize_puzzle(puzzles[i], save_path=str(save_path))
            print(f"  ✓ Sample {i}")
        
        print()
        
    def print_summary(self):
        """Print generation summary."""
        # Count files
        dataset_count = 0
        with open(self.datasets_dir / 'ptrnet_dataset.jsonl', 'r') as f:
            for line in f:
                if line.strip():
                    dataset_count += 1
        
        comp_files = len(list(self.comprehensive_dir.glob('*.png')))
        ptrnet_files = len(list(self.ptrnet_dir.glob('*.png')))
        puzzle_files = len(list(self.puzzle_dir.glob('*.png')))
        
        print("=" * 60)
        print("GENERATION SUMMARY")
        print("=" * 60)
        print()
        print("📊 DATASETS:")
        print(f"  • Training samples: {dataset_count}")
        print()
        print("🎨 VISUALIZATIONS:")
        print(f"  • Comprehensive (2x2 grid): {comp_files} samples")
        print(f"  • PTRNet views: {ptrnet_files} samples")
        print(f"  • Puzzle views: {puzzle_files} samples")
        print()
        print("📁 OUTPUT STRUCTURE:")
        print(f"  {self.output_dir}/")
        print(f"  ├── datasets/          ({len(list(self.datasets_dir.glob('*')))} files)")
        print(f"  ├── comprehensive/     ({comp_files} files)")
        print(f"  ├── ptrnet/           ({ptrnet_files} files)")
        print(f"  └── puzzle/           ({puzzle_files} files)")
        print()
        print("✅ All files generated successfully!")
        print("=" * 60)
        
    def run_all(self, num_dataset_samples=1000, num_viz_samples=5):
        """Run complete generation pipeline."""
        self.clean_output_directory()
        self.generate_dataset(num_samples=num_dataset_samples)
        self.generate_comprehensive_visualizations(num_samples=num_viz_samples)
        self.generate_ptrnet_visualizations(num_samples=num_viz_samples)
        self.generate_puzzle_visualizations(num_samples=num_viz_samples)
        self.print_summary()


def main():
    """Main entry point."""
    generator = RemingtonRouteGenerator(output_dir='gym/output')
    generator.run_all(num_dataset_samples=1000, num_viz_samples=5)


if __name__ == '__main__':
    main()
