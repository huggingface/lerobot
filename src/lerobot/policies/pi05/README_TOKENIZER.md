# FAST Tokenizer Training for LeRobotDataset

This directory contains tools for training a FAST (Factorized Action Sequence Tokenizer) on LeRobot datasets.

## Files

- **`train_fast_tokenizer.py`**: Main training script (refactored for LeRobotDataset)
- **`train_fast_tokenizer_example.md`**: Usage examples and parameter documentation
- **`MIGRATION_NOTES.md`**: Migration guide from B1K to LeRobotDataset

## Quick Start

```bash
# Basic usage
python train_fast_tokenizer.py \
    --repo_id "lerobot/aloha_sim_insertion_human" \
    --action_horizon 10 \
    --encoded_dims "0:14"

# With delta transform
python train_fast_tokenizer.py \
    --repo_id "lerobot/aloha_sim_insertion_human" \
    --action_horizon 10 \
    --encoded_dims "0:14" \
    --delta_dims "0,1,2,3,4,5,6,7,8,9,10,11,12,13" \
    --state_key "observation.state" \
    --vocab_size 1024
```

## What is FAST?

FAST is a tokenizer for robotic action sequences that:
1. Applies DCT (Discrete Cosine Transform) to action chunks
2. Quantizes DCT coefficients 
3. Uses BPE (Byte-Pair Encoding) to compress the quantized sequence
4. Achieves high compression ratios (e.g., 10-20x) while maintaining accuracy

This enables efficient storage and processing of long action sequences in vision-language-action models.

## Requirements

- Python 3.10+
- LeRobot dataset (either local or from HuggingFace Hub)
- transformers (for AutoProcessor)
- numpy
- torch
- tyro

## Workflow

```
LeRobotDataset → Extract Episodes → Apply Delta Transform 
    ↓
Select Dimensions → Normalize (q01, q99) → Create Chunks
    ↓
Train FAST Tokenizer → Compute Stats → Save
```

## Parameters Guide

### Essential Parameters

- **`repo_id`**: HuggingFace dataset repository ID
  - Example: `"lerobot/aloha_sim_insertion_human"`
  
- **`action_horizon`**: Length of action sequences to tokenize
  - Typical: 10-16 steps
  
- **`encoded_dims`**: Which action dimensions to encode
  - Format: `"start:end,start:end"`
  - Example: `"0:7"` = dimensions 0-6
  - Example: `"0:3,7:10"` = dimensions 0-2 and 7-9

### Optional Parameters

- **`delta_dims`**: Apply delta transform (action - state) to these dimensions
  - Format: `"0,1,2,3,4,5"`
  - Use for position-based actions
  
- **`state_key`**: Dataset key containing state observations
  - Default: `"observation.state"`
  
- **`vocab_size`**: BPE vocabulary size
  - Default: 1024
  - Larger = better compression but more memory
  
- **`scale`**: DCT quantization scale
  - Default: 10.0
  - Smaller = finer quantization, larger = coarser

- **`sample_fraction`**: Fraction of action chunks to use per episode
  - Default: 0.1 (10%)
  - Increase for small datasets, decrease for large datasets

## Output

The script creates a directory (default: `./fast_tokenizer_{repo_id}`) containing:

1. **Tokenizer files**: Can be loaded with `AutoProcessor.from_pretrained()`
2. **`metadata.json`**: Contains:
   - Training configuration
   - Compression statistics
   - Dataset information

## Example Output

```
Loading dataset: lerobot/aloha_sim_insertion_human
Dataset loaded: 50 episodes, 5000 frames
Encoding 14 dimensions: 0:14
Delta dimensions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
Action horizon: 10
Processing 50 episodes...
Collected 4500 action chunks
Extracted 14 encoded dimensions

Before normalization - overall stats:
  Min: -2.3451, Max: 3.1234, Mean: 0.0234, Std: 0.8765

Applied quantile normalization [q01, q99] → [-1, 1]

After normalization - overall stats:
  Min: -1.0000, Max: 1.0000, Mean: 0.0156, Std: 0.4321

Training FAST tokenizer on 4500 action chunks...
Action chunk shape: (4500, 10, 14)
Vocab size: 1024
DCT scale: 10.0
✓ Tokenizer training complete!

Compression Statistics:
  Average compression ratio: 14.23x
  Mean token length: 9.8
  P99 token length: 15
  Min token length: 6
  Max token length: 18

✅ Saved FAST tokenizer to ./fast_tokenizer_lerobot_aloha_sim_insertion_human
```

## Using the Trained Tokenizer

```python
from transformers import AutoProcessor

# Load tokenizer
tokenizer = AutoProcessor.from_pretrained(
    "./fast_tokenizer_lerobot_aloha_sim_insertion_human",
    trust_remote_code=True
)

# Encode action chunk [horizon, action_dim]
action_chunk = np.random.randn(10, 14)  # Example
tokens = tokenizer(action_chunk[None])[0]  # Returns token IDs

# Decode tokens back to actions
reconstructed = tokenizer.decode(tokens)
```

## Tips

1. **Start Small**: Use `--max_episodes 10` for initial testing
2. **Check Dimensions**: Verify encoded dimensions match your robot's action space
3. **Delta Transform**: Use for position-based actions, not velocity-based
4. **Normalization**: Ensure dataset has proper statistics computed
5. **Compression Ratio**: Aim for 10-20x for good balance of compression and accuracy

## Troubleshooting

**Issue**: "No normalization stats found"
- **Solution**: Compute dataset statistics first, or use raw actions

**Issue**: "Episode too short for action horizon"
- **Solution**: Reduce `--action_horizon` or filter short episodes

**Issue**: "State key not found"
- **Solution**: Check dataset features and use correct `--state_key`

**Issue**: Memory error with large datasets
- **Solution**: Reduce `--sample_fraction` or `--max_episodes`

## Citation

If you use FAST in your research, please cite:

```bibtex
@article{black2023fast,
  title={FAST: Factorized Action Sequence Tokenizer for Vision-Language-Action Models},
  author={Black, Kevin and others},
  journal={arXiv preprint},
  year={2023}
}
```



