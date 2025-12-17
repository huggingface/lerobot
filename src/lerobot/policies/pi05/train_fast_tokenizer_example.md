# Train FAST Tokenizer - Usage Examples

This script trains a FAST (Factorized Action Sequence Tokenizer) on LeRobotDataset action data.

## Basic Usage

```bash
python src/lerobot/policies/pi05/train_fast_tokenizer.py \
    --repo_id "lerobot/aloha_sim_insertion_human" \
    --action_horizon 10 \
    --encoded_dims "0:7" \
    --vocab_size 1024 \
    --scale 10.0
```

## Parameters

### Required
- `--repo_id`: LeRobot dataset repository ID (e.g., "lerobot/aloha_sim_insertion_human")

### Optional
- `--root`: Root directory for dataset (default: ~/.cache/huggingface/lerobot)
- `--action_horizon`: Number of future actions in each chunk (default: 10)
- `--max_episodes`: Maximum number of episodes to use (default: None = all)
- `--sample_fraction`: Fraction of chunks to sample per episode (default: 0.1)
- `--encoded_dims`: Comma-separated dimension ranges to encode (default: "0:6,7:23")
  - Example: "0:7" encodes dimensions 0-6
  - Example: "0:3,6:9" encodes dimensions 0-2 and 6-8
- `--delta_dims`: Comma-separated dimension indices for delta transform (default: None)
  - Example: "0,1,2,3,4,5" applies delta transform to first 6 dimensions
  - Delta transform: action[i] - state[i] for specified dimensions
- `--state_key`: Dataset key for state observations (default: "observation.state")
- `--vocab_size`: FAST vocabulary size / BPE vocab size (default: 1024)
- `--scale`: DCT scaling factor (default: 10.0)
- `--output_dir`: Directory to save tokenizer (default: ./fast_tokenizer_{repo_id})

## Examples

### Example 1: Train on full action space

```bash
python src/lerobot/policies/pi05/train_fast_tokenizer.py \
    --repo_id "lerobot/pusht" \
    --action_horizon 16 \
    --encoded_dims "0:2" \
    --vocab_size 512 \
    --max_episodes 100
```

### Example 2: Train with delta transform

```bash
python src/lerobot/policies/pi05/train_fast_tokenizer.py \
    --repo_id "lerobot/aloha_sim_insertion_human" \
    --action_horizon 10 \
    --encoded_dims "0:14" \
    --delta_dims "0,1,2,3,4,5,6,7,8,9,10,11,12,13" \
    --state_key "observation.state" \
    --vocab_size 1024 \
    --scale 10.0 \
    --sample_fraction 0.2
```

### Example 3: Train on subset of dimensions

```bash
python src/lerobot/policies/pi05/train_fast_tokenizer.py \
    --repo_id "lerobot/aloha_sim_insertion_human" \
    --action_horizon 10 \
    --encoded_dims "0:7" \
    --vocab_size 1024 \
    --output_dir "./my_tokenizer"
```

## Output

The script saves:
1. **Tokenizer files**: Trained FAST tokenizer (can be loaded with `AutoProcessor.from_pretrained()`)
2. **metadata.json**: Contains:
   - Configuration parameters
   - Compression statistics (compression ratio, token lengths)
   - Training dataset information

## Understanding the Process

1. **Load Dataset**: Loads the LeRobotDataset from HuggingFace
2. **Extract Action Chunks**: Creates sliding windows of actions with specified horizon
3. **Apply Delta Transform**: (Optional) Computes action deltas relative to current state
4. **Select Encoded Dimensions**: Extracts only the dimensions to be encoded
5. **Normalize**: Applies quantile normalization ([q01, q99] → [-1, 1])
6. **Train Tokenizer**: Trains BPE tokenizer on DCT coefficients
7. **Compute Stats**: Reports compression ratio and token length statistics
8. **Save**: Saves tokenizer and metadata

## Notes

- **Normalization**: The script uses quantile normalization (q01, q99) from the dataset's statistics
- **Sampling**: To speed up training, you can sample a fraction of chunks per episode
- **Delta Transform**: Applied per-dimension to make actions relative to current state
- **Compression**: FAST uses DCT + BPE to compress action sequences efficiently

