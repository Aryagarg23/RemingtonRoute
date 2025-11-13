# Position-Based Training System

## Overview

The training has been completely redesigned from **sequence-to-sequence** to **position-based decision making**.

### Key Difference

**OLD (Pointer Network):**
- Model sees entire grid at once
- Predicts complete sequence of moves
- Like planning the entire path upfront

**NEW (Position Network):**
- Model sees current position + grid state
- Predicts ONE next move
- Like making decisions step-by-step

## Architecture

### Input (9 features per cell)
```
[x_norm, y_norm, waypoint_type, 
 wall_up, wall_down, wall_left, wall_right,
 is_visited, is_current]  ← NEW: marks current position
```

### Model
- **Transformer Encoder** (4 layers, 1024 hidden dim)
- Processes all cells to understand grid structure
- Attention mechanism focuses on current position
- Outputs probability distribution over next cells

### Output
- Single prediction: which cell to move to next
- Masking ensures only unvisited cells are selected

## Training Data

Each puzzle generates multiple training samples:

**Example: 6×6 grid with 36-step solution**
- Creates **35 training samples** (one per step)
- Sample 0: "From start position, move to cell X"
- Sample 1: "From cell X (with X visited), move to cell Y"
- ...
- Sample 34: "From second-to-last cell, move to goal"

**10,000 puzzles → ~350,000 training samples**

## Files

### Core Files
- `generate_position_dataset.py` - Creates position-based dataset
- `position_net.py` - Position-based network architecture
- `train_position.py` - Training script
- `evaluate_position.py` - Step-by-step puzzle solver
- `quickstart_position.py` - Full pipeline

### Old Files (Pointer Network)
- `generate_dataset.py` - Sequence-to-sequence dataset
- `ptrnet.py` - Pointer network (sequence model)
- `train.py` - Old training script
- `evaluate.py` - Old evaluation

## Usage

### Quick Start (Recommended)
```bash
python model/quickstart_position.py
```

This runs:
1. Dataset generation (10,000 puzzles)
2. Training (50 epochs, ~1-2 hours on RTX 3090 Ti)
3. Evaluation (100 test puzzles)

### Manual Steps

**1. Generate Dataset**
```bash
python model/generate_position_dataset.py \
    --num-puzzles 10000 \
    --min-size 6 --max-size 6 \
    --min-checkpoints 7 --max-checkpoints 7
```

**2. Train Model**
```bash
python model/train_position.py
```

**3. Evaluate**
```bash
python model/evaluate_position.py \
    --model model/checkpoints/position_best.pt \
    --num-samples 100 \
    --visualize 10
```

## Training Details

### Configuration
```python
{
    'batch_size': 256,          # Larger batches (more samples per puzzle)
    'learning_rate': 1e-4,      # Lower LR for transformer
    'hidden_dim': 1024,         # 4x larger than before
    'num_layers': 4,            # Transformer layers
    'use_transformer': True,    # vs LSTM
    'use_amp': True,           # Mixed precision
    'num_epochs': 50
}
```

### Expected Performance
- **Training Accuracy**: 85-95% (predicting next move)
- **Top-5 Accuracy**: 95-99% (correct move in top 5)
- **Puzzle Solve Rate**: 70-90% (complete puzzle solutions)

### GPU Utilization
- RTX 3090 Ti: 85-95% utilization
- Batch size 256 with AMP
- ~8-10GB VRAM usage
- Training time: 1.5-2 hours (50 epochs)

## How It Works

### Training Example
```
State at step 5:
├─ Current position: (2, 3)
├─ Visited cells: {(0,0), (0,1), (1,1), (2,1), (2,2), (2,3)}
├─ Grid features: walls, checkpoints, etc.
└─ Target: Next cell is (2, 4)

Model learns: "Given I'm at (2,3) with these visited cells, 
               the best next move is (2,4)"
```

### Inference (Solving a Puzzle)
```python
solver = PositionSolver(model, device='cuda')
path, success, info = solver.solve_puzzle(puzzle)

# Solver iteratively:
# 1. Builds current state features
# 2. Asks model "what's next move?"
# 3. Updates position and visited set
# 4. Repeats until goal reached
```

## Advantages

1. **More Natural**: Learns decision-making, not path planning
2. **Better Generalization**: Sees many states per puzzle
3. **Interpretable**: Can inspect what model "thinks" at each step
4. **Robust**: Even if one move is wrong, can recover
5. **Extensible**: Easy to add RL fine-tuning later

## Comparison

| Aspect | Pointer Network | Position Network |
|--------|----------------|------------------|
| Input | Full grid | Full grid + position |
| Output | Full sequence | Single next move |
| Training samples | 10K | 350K |
| Learning | Path planning | Decision making |
| At inference | Plans ahead | Step-by-step |
| Sample efficiency | Low | High |
| Interpretability | Medium | High |

## Next Steps

1. **Run quickstart**: `python model/quickstart_position.py`
2. **Monitor training**: Watch accuracy climb to 85%+
3. **Check solve rate**: Should hit 70-90% on test puzzles
4. **Fine-tune**: Adjust hyperparameters if needed

## Reinforcement Learning (Future)

This position-based architecture is **RL-ready**:
- Current design: Supervised (imitation learning)
- Easy upgrade: Replace with PPO/A2C for exploration
- Reward shaping: Checkpoints, path efficiency, etc.
