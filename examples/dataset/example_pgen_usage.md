# Example: Synthetic Data Generation with Sampling

## Quick Start

### 1. Test with 100 frames and 1 second sampling
```bash
python examples/dataset/annotate_pgen.py \
    --data-dir /fsx/jade_choghari/.cache/huggingface/lerobot/lerobot/svla_so101_pickplace \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --num-samples 100 \
    --sample-interval 1.0 \
    --output-dir ./outputs/test_pgen
```

**Expected behavior** (assuming 30 fps):
- Total frames: 100
- Frames sampled: ~4 (every 30 frames = 1 second)
- Efficiency: 96% fewer VLM calls
- Output: All 100 frames get `task_index_high_level`, but only 4 unique dialogues generated

### 2. Process full dataset with different sampling rates

#### Conservative (every 2 seconds)
```bash
python examples/dataset/annotate_pgen.py \
    --data-dir /fsx/jade_choghari/.cache/huggingface/lerobot/lerobot/svla_so101_pickplace \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --sample-interval 2.0 \
    --output-dir ./outputs/pgen_2s
```

#### Standard (every 1 second) - **RECOMMENDED**
```bash
python examples/dataset/annotate_pgen.py \
    --data-dir /fsx/jade_choghari/.cache/huggingface/lerobot/lerobot/svla_so101_pickplace \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --sample-interval 1.0 \
    --output-dir ./outputs/pgen_1s
```

#### Fine-grained (every 0.5 seconds)
```bash
python examples/dataset/annotate_pgen.py \
    --data-dir /fsx/jade_choghari/.cache/huggingface/lerobot/lerobot/svla_so101_pickplace \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --sample-interval 0.5 \
    --output-dir ./outputs/pgen_0.5s
```

## Performance Estimates

For a dataset with:
- 100 episodes
- 10 seconds per episode (average)
- 30 fps
- Total frames: 30,000

| Sampling Interval | Frames Sampled | % Sampled | Speedup | Time Estimate |
|-------------------|----------------|-----------|---------|---------------|
| Every frame (0.033s) | 30,000 | 100% | 1x | ~10 hours |
| 0.5 seconds | 2,000 | 6.7% | 15x | ~40 min |
| **1.0 seconds** | **1,000** | **3.3%** | **30x** | **~20 min** |
| 2.0 seconds | 500 | 1.7% | 60x | ~10 min |

*Note: Times are approximate and depend on GPU, model size, and generation speed*

## Understanding the Output

### Console Output Example
```
[cyan]Generating synthetic data for 30000 frames...[/cyan]
[cyan]Sampling interval: 1.0s (fps: 30)[/cyan]
Generating synthetic dialogue: 100%|████████| 30000/30000 [20:15<00:00, 24.68it/s]
[green]✓ Sampled 1000 frames out of 30000 (3.3%)[/green]
[green]✓ Generated 450 unique high-level tasks[/green]
```

### What happens:
1. **Frame 0 (t=0.0s)**: Generate dialogue → Task index 0
2. **Frames 1-29 (t=0.033s-0.967s)**: Reuse task index 0
3. **Frame 30 (t=1.0s)**: Generate new dialogue → Task index 1
4. **Frames 31-59 (t=1.033s-1.967s)**: Reuse task index 1
5. And so on...

### Result:
- Every frame has a `task_index_high_level`
- Only sampled frames have unique dialogues generated
- Intermediate frames inherit from the most recent sample
- Maintains temporal coherence within episodes

## Checking Your Results

After running, verify the output:

```bash
# Check the generated tasks
python -c "
import pandas as pd
from pathlib import Path

tasks = pd.read_parquet('outputs/test_pgen/meta/tasks_high_level.parquet')
print(f'Total unique tasks: {len(tasks)}')
print(f'Sample tasks:')
print(tasks[['user_prompt', 'robot_utterance', 'skill']].head())
"

# Check debug output
head outputs/test_pgen/meta/syn_annotations.jsonl

# Load and verify dataset
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id='local_with_high_level_tasks', 
                    root='outputs/test_pgen')
print(f'Dataset has {len(ds)} frames')
print(f'Features: {list(ds.features.keys())}')
assert 'task_index_high_level' in ds.features
print('✓ task_index_high_level feature added successfully!')
"
```

## Common Use Cases

### Development/Testing
```bash
--sample-interval 2.0  # Fast iteration
--num-samples 500      # Small subset
```

### Production Training
```bash
--sample-interval 1.0  # Good coverage
# Process all samples (no --num-samples)
```

### High-Quality Dataset
```bash
--sample-interval 0.5  # Fine-grained
--temperature 0.6      # More consistent
--model Qwen/Qwen3-VL-30B-A3B-Instruct  # Larger model
```

