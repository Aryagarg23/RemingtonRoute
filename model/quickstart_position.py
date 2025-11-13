"""
Quickstart script for Position-based Network Training

Runs the complete pipeline:
1. Generate position-based training dataset
2. Train the model
3. Evaluate on test puzzles

Usage:
    python model/quickstart_position.py
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """Run a command and print output."""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    print(f"Command: {cmd}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: Command failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ Completed in {elapsed:.1f} seconds")
    return elapsed


def main():
    print("=" * 70)
    print("POSITION-BASED NETWORK - FULL TRAINING PIPELINE")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Generate position-based dataset (10,000 puzzles)")
    print("  2. Train the position-based network (50 epochs)")
    print("  3. Evaluate on test puzzles (100 samples)")
    print("\n" + "=" * 70)
    
    total_start = time.time()
    
    # Step 1: Generate dataset
    dataset_time = run_command(
        "python model/generate_position_dataset.py --num-puzzles 10000 --min-size 6 --max-size 6 --min-checkpoints 7 --max-checkpoints 7",
        "Generate Position-based Dataset"
    )
    
    # Step 2: Train model
    train_time = run_command(
        "python model/train_position.py",
        "Train Position-based Network"
    )
    
    # Step 3: Evaluate
    eval_time = run_command(
        "python model/evaluate_position.py --num-samples 100 --visualize 10",
        "Evaluate Model Performance"
    )
    
    # Summary
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nTiming Summary:")
    print(f"  Dataset generation: {dataset_time/60:.1f} minutes")
    print(f"  Model training: {train_time/60:.1f} minutes")
    print(f"  Evaluation: {eval_time/60:.1f} minutes")
    print(f"  Total: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print("\nModel checkpoints saved to: model/checkpoints/")
    print("Training logs saved to: model/logs/")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
