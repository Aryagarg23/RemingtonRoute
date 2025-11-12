# Pointer Network for Hamiltonian Paths

Encoder-Decoder LSTM with attention. Based on [Vinyals et al., 2015](https://arxiv.org/abs/1506.03134).

## Architecture

- Encoder: 2-layer LSTM (hidden=256)
- Decoder: 2-layer LSTM with attention
- Input: 8 features per cell `[x, y, waypoint, wall_up, wall_down, wall_left, wall_right, is_visited]`
- Output: Sequence of cell indices

## Usage

```bash
# Generate data (grid 5-12, checkpoints 5-15)
python model/generate_dataset.py --mode variable --num-samples 10000

# Train (50 epochs, ~1.5-2 hrs on RTX 3090 Ti with AMP)
python model/train.py

# Evaluate
python model/evaluate.py --beam-search --visualize 10
```

## Config

Edit `model/train.py` config dict:
- Hidden dim: 256
- Layers: 2
- Dropout: 0.2
- Learning rate: 1e-3
- Batch size: 64 (optimized for RTX 3090 Ti)
- Mixed precision: Enabled (AMP)
- Teacher forcing: 0.5 → 0.0 (decay)

## Files

- `ptrnet.py` - Network architecture
- `train.py` - Training script
- `evaluate.py` - Inference & visualization
- `generate_dataset.py` - Data generator
