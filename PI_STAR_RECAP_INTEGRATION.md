# π*₀.₆ RECAP Integration with LeRobot

This document describes how the π*₀.₆ RECAP policy has been integrated with the LeRobot framework.

## Overview

The π*₀.₆ RECAP policy has been adapted to work seamlessly with LeRobot's:
- Dataset format (`LeRobotDataset`)
- Training pipeline
- Evaluation framework
- Visualization tools

## Project Structure

```
lerobot/
├── src/lerobot/
│   ├── policies/
│   │   └── pi_star_recap/           # π*₀.₆ RECAP implementation
│   │       ├── __init__.py
│   │       ├── configuration_pi_star_recap.py
│   │       ├── modeling_pi_star_recap.py
│   │       ├── processor_pi_star_recap.py
│   │       └── README.md
│   └── configs/
│       └── policies.py              # Auto-registered via @PreTrainedConfig
├── examples/
│   ├── train_pi_star_recap.py       # Training example
│   └── prepare_recap_dataset.py     # Dataset preparation
└── PI_STAR_RECAP_INTEGRATION.md     # This file
```

## Key Integration Points

### 1. Policy Registration

The policy is automatically registered with LeRobot's factory:

```python
@PreTrainedConfig.register_subclass("pi_star_recap")
@dataclass
class PiStarRECAPConfig(PreTrainedConfig):
    ...
```

This allows using `--policy.type=pi_star_recap` in training scripts.

### 2. Dataset Format

RECAP extends LeRobot's standard dataset with additional fields:

| Field | Type | Description |
|-------|------|-------------|
| `data_type` | str | "demo", "auto", or "intervention" |
| `rewards` | float | Episode reward (for IQL) |
| `dones` | bool | Episode termination |
| `intervention_mask` | bool[] | Optional: intervention points |

### 3. Training Loop Integration

The policy follows LeRobot's `PreTrainedPolicy` interface:

```python
class PiStarRECAPPolicy(PreTrainedPolicy):
    def forward(self, batch) -> dict:
        # Returns losses
        return {
            "loss": total_loss,
            "v_loss": v_loss,
            "q_loss": q_loss,
            "policy_loss": policy_loss,
        }
    
    def select_action(self, batch) -> torch.Tensor:
        # Returns action for inference
        return action
    
    def update(self):
        # Optimizer step
        ...
```

### 4. Multi-Optimizer Support

π*₀.₆ uses separate optimizers for IQL components:

```python
# Q-network optimizer
q_optimizer = AdamW(q_networks.parameters(), lr=3e-4)

# V-network optimizer  
v_optimizer = AdamW(v_network.parameters(), lr=3e-4)

# Policy optimizer
policy_optimizer = AdamW(policy.parameters(), lr=3e-5)
```

All are managed through the policy's `update()` method.

## Usage Examples

### 1. Basic Training

```bash
python lerobot/scripts/train.py \
    --policy.type=pi_star_recap \
    --dataset.repo_id=your_username/your_dataset \
    --output_dir=outputs/pi_star_recap \
    --policy.vlm_model_name=google/gemma-2b-it \
    --policy.iql_expectile=0.7 \
    --policy.iql_temperature=0.5
```

### 2. Using Example Script

```bash
python examples/train_pi_star_recap.py \
    --dataset.repo_id=your_username/your_dataset \
    --output_dir=outputs/pi_star_recap
```

### 3. Preparing Dataset

```bash
python examples/prepare_recap_dataset.py \
    --output_dir=./my_recap_dataset \
    --num_demo=100 \
    --num_auto=50 \
    --num_intervention=50
```

## Configuration Parameters

### IQL Hyperparameters

```yaml
policy:
  iql_expectile: 0.7        # τ: expectile for V-network
  iql_temperature: 0.5      # β: advantage weighting temperature
  iql_discount: 0.99        # γ: discount factor
```

### RECAP Data Weights

```yaml
policy:
  demo_weight: 1.0           # Demonstration episodes
  auto_weight: 1.0           # Autonomous execution
  intervention_weight: 2.0   # Expert interventions
  balance_data_types: true   # Balance sampling
```

### Training Strategy

```yaml
policy:
  freeze_vlm: true           # Freeze entire VLM
  train_expert_only: true    # Only train action expert
  q_lr: 3e-4                # Q-network learning rate
  v_lr: 3e-4                # V-network learning rate
  policy_lr: 3e-5           # Policy learning rate (lower)
```

## Advantages of LeRobot Integration

### 1. Standardized Data Loading

```python
# LeRobot handles all data loading
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("your_username/your_dataset")
# Automatically handles RECAP fields
```

### 2. Built-in Visualization

```python
# Use LeRobot's visualization tools
from lerobot.scripts.visualize_dataset import visualize_dataset

visualize_dataset("your_username/your_dataset")
```

### 3. Model Hub Integration

```python
# Push to HuggingFace Hub
from lerobot.utils.hub import push_to_hub

push_to_hub(
    repo_id="your_username/pi_star_recap_model",
    folder_path="outputs/pi_star_recap/checkpoints/latest"
)
```

### 4. Evaluation Framework

```bash
# Evaluate with LeRobot's eval script
python lerobot/scripts/eval.py \
    --policy.type=pi_star_recap \
    --policy.pretrained_model_path=outputs/pi_star_recap/checkpoints/latest \
    --dataset.repo_id=your_username/your_dataset
```

## Improvements Over Standalone Implementation

| Feature | Standalone π*₀.₆ | LeRobot Integration |
|---------|-----------------|---------------------|
| Dataset format | Custom | Standard LeRobotDataset |
| Data loading | Manual | Optimized DataLoader |
| Visualization | Custom | LeRobot visualization |
| Evaluation | Custom | LeRobot eval framework |
| Model hub | Manual | HuggingFace integration |
| Multi-GPU | Manual | Distributed training support |
| Logging | Basic | TensorBoard + WandB |

## Migration Guide

### From Standalone to LeRobot

**Before (Standalone):**
```python
from pi_star_model import create_pi_star_model
from recap_trainer import RECAPTrainer

model = create_pi_star_model()
trainer = RECAPTrainer(model, dataloader, ...)
trainer.train()
```

**After (LeRobot):**
```python
from lerobot.policies.pi_star_recap import PiStarRECAPPolicy, PiStarRECAPConfig
from lerobot.scripts.train import train

config = PiStarRECAPConfig()
train(config)
```

Or via command line:
```bash
python lerobot/scripts/train.py --policy.type=pi_star_recap ...
```

## Troubleshooting

### Issue: Policy not found

```
Error: Unknown policy type 'pi_star_recap'
```

**Solution:** Ensure you're in the LeRobot directory and have installed it:
```bash
cd lerobot
pip install -e .
```

### Issue: Import errors

```
ModuleNotFoundError: No module named 'lerobot.policies.pi_star_recap'
```

**Solution:** Check that the files are in the correct location:
```bash
ls src/lerobot/policies/pi_star_recap/
# Should show: __init__.py, configuration_*.py, modeling_*.py, etc.
```

### Issue: CUDA out of memory

**Solution:** Enable gradient checkpointing:
```yaml
policy:
  gradient_checkpointing: true
  batch_size: 4  # Reduce if needed
```

## Future Enhancements

- [ ] Integration with LeRobot's async inference
- [ ] Multi-GPU distributed training
- [ ] Real-time intervention recording
- [ ] Automatic advantage visualization
- [ ] Integration with LeRobot's SimX environment

## References

- LeRobot Documentation: https://github.com/huggingface/lerobot
- π*₀.₆ Paper: https://arxiv.org/abs/2511.14759
- RECAP Method: Section 3.2 of paper
- IQL Paper: https://arxiv.org/abs/2110.06169
