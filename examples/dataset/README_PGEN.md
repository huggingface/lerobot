# Synthetic Data Generation for Hierarchical Robot Policies

This directory contains `annotate_pgen.py`, a script for generating synthetic user prompts and robot utterances for hierarchical policy training using Vision-Language Models (VLMs).

## Overview

The script implements the synthetic data generation pipeline described in the Hi-Robot paper:

1. **Load** a LeRobot dataset with skill annotations (from `annotate.py`)
2. **Generate** synthetic dialogue using Qwen VLM:
   - User prompts (ℓ_t): Natural requests that lead to specific skills
   - Robot utterances (u_t): Acknowledgments and clarifications
3. **Save** results as a new dataset feature `task_index_high_level`

## Prerequisites

1. First, annotate your dataset with skills using `annotate.py`:

```bash
python examples/dataset/annotate.py \
    --repo-id lerobot/svla_so101_pickplace \
    --video-key observation.images.base \
    --model Qwen/Qwen2-VL-7B-Instruct
```

This creates `meta/skills.json` with skill segmentation for each episode.

## Usage

### Basic Usage

```bash
python examples/dataset/annotate_pgen.py \
    --repo-id lerobot/svla_so101_pickplace \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --sample-interval 1.0 \
    --output-dir ./outputs/pgen_dataset
```

**Note**: The script processes **all episodes** in the dataset. It generates dialogue every 1 second (`--sample-interval 1.0`) using temporal sampling. Frames between samples reuse the last generated dialogue. This makes the process efficient while ensuring complete dataset coverage.

### Advanced Options

```bash
python examples/dataset/annotate_pgen.py \
    --repo-id lerobot/svla_so101_pickplace \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --temperature 0.8 \
    --sample-interval 0.5 \
    --num-image-views-per-sample 2 \
    --output-dir ./outputs/pgen_dataset \
    --push-to-hub
```

This example uses a more powerful model and samples every 0.5 seconds for finer granularity.

### Fast Testing (larger interval)

```bash
python examples/dataset/annotate_pgen.py \
    --repo-id lerobot/svla_so101_pickplace \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --sample-interval 5.0 \
    --output-dir ./outputs/pgen_quick_test
```

Use a larger interval (5.0 seconds) for rapid iteration during development. All episodes are still processed.

### Using Local Dataset

```bash
python examples/dataset/annotate_pgen.py \
    --data-dir /fsx/jade_choghari/.cache/huggingface/lerobot/lerobot/svla_so101_pickplace \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --output-dir ./outputs/pgen_dataset
```

## Output Files

The script produces several outputs:

1. **`meta/tasks_high_level.parquet`**: High-level tasks with user prompts and robot utterances
   - Columns: task_index, user_prompt, robot_utterance, skill, scenario_type, response_type

2. **`meta/syn_annotations.jsonl`**: Debug file with all generated dialogues
   - One JSON object per line with full context for each frame

3. **Modified dataset**: New dataset with `task_index_high_level` feature added to all parquet files

## Scenario and Response Types

The generator produces diverse interaction types:

### Scenario Types
- **specific_object**: Direct specification of objects/actions
- **negative_task**: Instructions about what NOT to do
- **situated_correction**: Adjustments based on current state
- **implicit_request**: Implied needs without direct commands
- **constraint_based**: Specific constraints or preferences

### Response Types
- **confirmation**: Simple acknowledgment ("OK, I'll do X")
- **clarification**: Seeking confirmation ("Just to confirm...")
- **acknowledgment**: Action acknowledgment ("Got it, doing X")
- **constraint_acknowledgment**: Acknowledging constraints ("Sure, I'll X while Y")

## Example Generated Data

```json
{
  "episode_id": 0,
  "frame_index": 45,
  "timestamp": 2.5,
  "skill_current": "robot arm picks up pink lego brick",
  "skill_history": ["robot arm moves towards pink lego brick"],
  "task_description": "pink lego brick into the transparent box",
  "scenario_type": "specific_object",
  "response_type": "confirmation",
  "user_prompt": "Can you grab the pink brick?",
  "robot_utterance": "Sure, I'll pick up the pink lego brick."
}
```

## Accessing the Data

After running the script, access the synthetic data in your code:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import pandas as pd

# Load modified dataset
dataset = LeRobotDataset(repo_id="lerobot/svla_so101_pickplace_with_high_level_tasks")

# Access frame with high-level task
frame = dataset[100]
high_level_task_idx = frame["task_index_high_level"].item()

# Load high-level tasks
tasks_df = pd.read_parquet(dataset.root / "meta" / "tasks_high_level.parquet")
task_info = tasks_df.iloc[high_level_task_idx]

print(f"User prompt: {task_info['user_prompt']}")
print(f"Robot utterance: {task_info['robot_utterance']}")
print(f"Skill: {task_info['skill']}")
```

## Architecture

The script is modular and extensible:

```python
# Core components
class QwenPgen:
    """VLM wrapper for generation"""
    def call_qwen(images, prompt) -> dict
    
def construct_prompt(task, history, skill) -> str
    """Build prompt for VLM"""
    
def annotate_sample(pgen, images, ...) -> dict
    """Generate dialogue for one sample"""
    
def generate_synthetic_data(dataset, pgen, ...) -> tuple
    """Process entire dataset"""
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--repo-id` | - | HuggingFace dataset ID |
| `--data-dir` | - | Local dataset path |
| `--model` | Qwen/Qwen2-VL-7B-Instruct | VLM model name |
| `--device` | cuda | Device (cuda/cpu) |
| `--dtype` | bfloat16 | Model precision |
| `--temperature` | 0.7 | Sampling temperature |
| `--sample-interval` | 1.0 | Generate dialogue every N seconds (all episodes processed) |
| `--num-image-views-per-sample` | 1 | Number of cameras |
| `--output-dir` | None | Output directory |
| `--push-to-hub` | False | Push to HuggingFace Hub |

## Sampling Strategy

The script uses **temporal sampling** to efficiently generate dialogue:

- **Default**: Generate dialogue every 1 second (`--sample-interval 1.0`)
- **Efficiency**: If a dataset runs at 30fps, this samples ~3% of frames
- **Propagation**: Frames between samples reuse the last generated task_index
- **Episode-aware**: Always samples the first frame of each episode

### Example with 30 fps dataset:
```bash
# Sample every 1 second (every 30 frames)
--sample-interval 1.0  # ~3,000 generations for a 100 episode dataset (3 sec/episode)

# Sample every 0.5 seconds (every 15 frames)
--sample-interval 0.5  # ~6,000 generations (more granular)

# Sample every 2 seconds (every 60 frames)
--sample-interval 2.0  # ~1,500 generations (more efficient)
```

### Why sampling works:
- Skills typically last 1-3 seconds
- Dialogue doesn't need to change every frame
- Reduces computational cost by 30-100x
- Still provides good coverage for training

## Tips

1. **Quick testing**: Use larger `--sample-interval` (e.g., 5.0 or 10.0) for rapid iteration
2. **Monitor GPU**: VLM inference is memory-intensive
3. **Check outputs**: Review `syn_annotations.jsonl` for quality
4. **Adjust temperature**: Higher = more diverse, lower = more consistent
5. **Multiple views**: Use `--num-image-views-per-sample 2+` for better context
6. **Tune sampling**: Start with 1.0s, increase for speed (testing), decrease for granularity (production)

## Troubleshooting

### No skills.json found
Run `annotate.py` first to generate skill annotations.

### Out of memory
- Reduce batch size to 1
- Use smaller model (Qwen2-VL-7B instead of Qwen3-VL-30B)
- Process fewer samples at a time

### Poor quality generations
- Adjust temperature (try 0.6-0.9)
- Check that skills.json has good annotations
- Ensure images are loading correctly

## Citation

Based on the Hi-Robot paper's synthetic data generation approach:
```
@article{hirobot2024,
  title={Hi-Robot: Hierarchical Robot Learning with Vision-Language Models},
  year={2024}
}
```

