"""
Quick Start Example - Train and Evaluate Pointer Network

This script demonstrates the complete workflow:
1. Generate variable-size dataset
2. Train the model
3. Evaluate and visualize results
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """Run a shell command and print results."""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    print(f"Command: {cmd}\n")
    
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"\n✓ {description} completed in {elapsed:.1f}s")
    else:
        print(f"\n✗ {description} failed!")
        sys.exit(1)
    
    return result.returncode == 0


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   POINTER NETWORK TRAINING - HAMILTONIAN PATH PUZZLES           ║
    ║                                                                  ║
    ║   This script will:                                             ║
    ║   1. Generate 10,000 training puzzles (variable sizes)          ║
    ║   2. Train a Pointer Network for 50 epochs                      ║
    ║   3. Evaluate the model and generate visualizations             ║
    ║                                                                  ║
    ║   Estimated time: 3-4 hours on GPU, 8-10 hours on CPU           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Step 1: Generate dataset
    run_command(
        "python model/generate_dataset.py --mode variable --num-samples 10000",
        "Generate Variable-Size Dataset (10,000 samples)"
    )
    
    # Step 2: Train model
    run_command(
        "python model/train.py",
        "Train Pointer Network (50 epochs)"
    )
    
    # Step 3: Evaluate model
    run_command(
        "python model/evaluate.py --num-samples 200 --visualize 10",
        "Evaluate Model and Generate Visualizations"
    )
    
    print("\n" + "=" * 70)
    print("✓ COMPLETE!")
    print("=" * 70)
    print("\nResults:")
    print("  - Model checkpoint: model/checkpoints/best_model.pt")
    print("  - Training curves: model/logs/")
    print("  - Evaluation metrics: model/evaluations/evaluation_metrics.json")
    print("  - Visualizations: model/evaluations/prediction_*.png")
    print("\nTo train with different parameters, edit model/train.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
