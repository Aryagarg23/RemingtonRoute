"""
Inference and Evaluation Script for Pointer Network

Tests trained model on new puzzles and visualizes results.
"""

import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

from model.ptrnet import PointerNetwork, HamiltonianPuzzleDataset


class PtrNetEvaluator:
    """Evaluator for trained Pointer Network models."""
    
    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model
        self.model = PointerNetwork(
            input_dim=8,
            hidden_dim=256,
            num_layers=2,
            dropout=0.0  # No dropout during inference
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    
    def predict(self, puzzle_input, use_beam_search=False, beam_width=5):
        """
        Predict solution path for a puzzle.
        
        Args:
            puzzle_input: List of cell features or tensor
            use_beam_search: Whether to use beam search
            beam_width: Beam width for beam search
            
        Returns:
            predicted_path: List of indices
            confidence: Confidence scores
        """
        if not isinstance(puzzle_input, torch.Tensor):
            puzzle_input = torch.tensor(puzzle_input, dtype=torch.float32)
        
        puzzle_input = puzzle_input.unsqueeze(0).to(self.device)  # Add batch dimension
        
        with torch.no_grad():
            if use_beam_search:
                path, score = self.model.beam_search(puzzle_input, beam_width)
                return path, score
            else:
                pointers, indices = self.model(puzzle_input, teacher_forcing_ratio=0.0)
                path = indices[0].cpu().numpy().tolist()
                confidences = pointers[0].max(dim=1).values.cpu().numpy()
                return path, confidences
    
    def evaluate_dataset(self, dataset_path, num_samples=100):
        """
        Evaluate model on a dataset.
        
        Args:
            dataset_path: Path to dataset JSONL file
            num_samples: Number of samples to evaluate
            
        Returns:
            metrics: Dict with evaluation metrics
        """
        dataset = HamiltonianPuzzleDataset(dataset_path, max_samples=num_samples)
        
        total_samples = len(dataset)
        exact_matches = 0
        partial_matches = []
        avg_confidence = []
        
        print(f"\nEvaluating on {total_samples} samples...")
        
        for i in range(total_samples):
            inputs, targets = dataset[i]
            
            # Predict
            predicted_path, confidences = self.predict(inputs)
            
            # Compute metrics
            targets_list = targets.numpy().tolist()
            
            # Exact match
            if predicted_path == targets_list:
                exact_matches += 1
            
            # Partial match (percentage of correct positions)
            correct = sum(p == t for p, t in zip(predicted_path, targets_list))
            partial_matches.append(correct / len(targets_list))
            
            # Average confidence
            if isinstance(confidences, np.ndarray):
                avg_confidence.append(confidences.mean())
            else:
                avg_confidence.append(confidences)
        
        metrics = {
            'total_samples': total_samples,
            'exact_match_accuracy': exact_matches / total_samples,
            'avg_partial_match': np.mean(partial_matches),
            'avg_confidence': np.mean(avg_confidence)
        }
        
        print(f"\nEvaluation Results:")
        print(f"  Total samples: {metrics['total_samples']}")
        print(f"  Exact match accuracy: {metrics['exact_match_accuracy']:.4f}")
        print(f"  Average partial match: {metrics['avg_partial_match']:.4f}")
        print(f"  Average confidence: {metrics['avg_confidence']:.4f}")
        
        return metrics
    
    def visualize_prediction(self, puzzle_data, predicted_path, save_path=None):
        """
        Visualize a puzzle with predicted and ground truth paths.
        
        Args:
            puzzle_data: Puzzle dict from JSONL file
            predicted_path: List of predicted cell indices
            save_path: Path to save visualization
        """
        rows = puzzle_data['rows']
        cols = puzzle_data['cols']
        true_path = puzzle_data['solution_path']
        walls = puzzle_data['walls']
        checkpoints = puzzle_data['checkpoints']
        
        # Convert indices to coordinates
        grid_cells = [(r, c) for r in range(rows) for c in range(cols)]
        predicted_coords = [grid_cells[idx] for idx in predicted_path]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot ground truth
        self._plot_puzzle(ax1, rows, cols, true_path, walls, checkpoints, "Ground Truth")
        
        # Plot prediction
        self._plot_puzzle(ax2, rows, cols, predicted_coords, walls, checkpoints, "Prediction")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_puzzle(self, ax, rows, cols, path, walls, checkpoints, title):
        """Helper function to plot a puzzle."""
        # Draw grid
        for i in range(rows + 1):
            ax.plot([0, cols], [i, i], 'k-', linewidth=0.5, alpha=0.3)
        for j in range(cols + 1):
            ax.plot([j, j], [0, rows], 'k-', linewidth=0.5, alpha=0.3)
        
        # Draw walls
        for wall in walls:
            wall_list = list(wall)
            (r1, c1), (r2, c2) = wall_list[0], wall_list[1]
            
            if r1 == r2:  # Vertical wall
                x = max(c1, c2)
                y = r1
                ax.plot([x, x], [y, y + 1], 'brown', linewidth=3)
            else:  # Horizontal wall
                x = c1
                y = max(r1, r2)
                ax.plot([x, x + 1], [y, y], 'brown', linewidth=3)
        
        # Draw path
        path_x = [c + 0.5 for r, c in path]
        path_y = [rows - r - 0.5 for r, c in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.6, label='Path')
        
        # Draw checkpoints
        start = checkpoints['start']
        goal = checkpoints['goal']
        cps = checkpoints['checkpoints']
        
        ax.plot(start[1] + 0.5, rows - start[0] - 0.5, 'go', markersize=12, label='Start')
        ax.plot(goal[1] + 0.5, rows - goal[0] - 0.5, 'rs', markersize=12, label='Goal')
        
        for cp in cps:
            ax.plot(cp[1] + 0.5, rows - cp[0] - 0.5, 'y+', markersize=15, 
                   markeredgewidth=3, label='Checkpoint' if cp == cps[0] else '')
        
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Pointer Network model')
    parser.add_argument('--model', type=str, default='model/checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='gym/output/datasets/ptrnet_dataset.jsonl',
                        help='Path to dataset for evaluation')
    parser.add_argument('--puzzles', type=str, default='gym/output/datasets/ptrnet_puzzles.jsonl',
                        help='Path to puzzle data for visualization')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--visualize', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='model/evaluations',
                        help='Output directory for visualizations')
    parser.add_argument('--beam-search', action='store_true',
                        help='Use beam search for inference')
    parser.add_argument('--beam-width', type=int, default=5,
                        help='Beam width for beam search')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = PtrNetEvaluator(args.model)
    
    # Evaluate on dataset
    metrics = evaluator.evaluate_dataset(args.dataset, args.num_samples)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")
    
    # Visualize some predictions
    if args.visualize > 0:
        print(f"\nGenerating {args.visualize} visualizations...")
        
        dataset = HamiltonianPuzzleDataset(args.dataset, max_samples=args.visualize)
        
        with open(args.puzzles, 'r') as f:
            puzzles = [json.loads(line) for i, line in enumerate(f) if i < args.visualize]
        
        for i in range(min(args.visualize, len(dataset))):
            inputs, targets = dataset[i]
            puzzle_data = puzzles[i]
            
            # Predict
            predicted_path, _ = evaluator.predict(inputs, args.beam_search, args.beam_width)
            
            # Visualize
            save_path = os.path.join(args.output_dir, f'prediction_{i:03d}.png')
            evaluator.visualize_prediction(puzzle_data, predicted_path, save_path)
        
        print(f"Visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
