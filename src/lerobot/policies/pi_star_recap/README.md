# π*₀.₆ RECAP Policy for LeRobot

This is a LeRobot-compatible implementation of **π*₀.₆** with **RECAP** (RL with Experience and Corrections via Advantage-conditioned Policies) from the paper:

> **"π*₀.₆: a VLA That Learns From Experience"**  
> Physical Intelligence, 2025  
> [arXiv:2511.14759](https://arxiv.org/abs/2511.14759)

## Features

- **IQL (Implicit Q-Learning)**: Stable offline RL training
- **Advantage-Conditioned Policy**: Policy learns from A(s,a) = Q(s,a) - V(s)
- **RECAP Data Mixing**: Supports heterogeneous data sources
- **LeRobot Integration**: Compatible with LeRobotDataset and training pipeline

## Quick Start

### 1. Installation

```bash
# Install LeRobot (if not already installed)
pip install -e .

# Install additional dependencies for π*₀.₆
pip install transformers accelerate
```

### 2. Configuration

```python
from lerobot.policies.pi_star_recap import PiStarRECAPConfig

config = PiStarRECAPConfig(
    vlm_model_name="google/gemma-2b-it",
    chunk_size=16,
    n_action_steps=16,
    # IQL parameters
    iql_expectile=0.7,
    iql_temperature=0.5,
    iql_discount=0.99,
    # RECAP weights
    demo_weight=1.0,
    auto_weight=1.0,
    intervention_weight=2.0,
)
```

### 3. Training

```bash
python lerobot/scripts/train.py \
    --policy.type=pi_star_recap \
    --dataset.repo_id=your_username/your_dataset \
    --output_dir=outputs/pi_star_recap_experiment
```

### 4. Evaluation

```bash
python lerobot/scripts/eval.py \
    --policy.type=pi_star_recap \
    --policy.pretrained_model_path=outputs/pi_star_recap_experiment/checkpoints/latest/pretrained_model \
    --dataset.repo_id=your_username/your_dataset \
    --eval.n_episodes=10
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    π*₀.₆ RECAP Policy                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              VLM Backbone (Gemma/Qwen)                    │  │
│  │                 [Frozen during RL]                        │  │
│  └─────────────────────────┬────────────────────────────────┘  │
│                            │                                     │
│  ┌─────────────────────────▼────────────────────────────────┐  │
│  │                    IQL Components                         │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐   │  │
│  │  │  Q-Networks │ │  V-Network  │ │    Advantage     │   │  │
│  │  │    (x2)     │ │  (Expectile)│ │   A(s,a)=Q-V     │   │  │
│  │  └─────────────┘ └─────────────┘ └──────────────────┘   │  │
│  └─────────────────────────┬────────────────────────────────┘  │
│                            │                                     │
│  ┌─────────────────────────▼────────────────────────────────┐  │
│  │         RECAP Policy (Advantage-Conditioned)              │  │
│  │              Flow Matching + Transformer                  │  │
│  │                                                          │  │
│  │   Data Sources:                                          │  │
│  │   • Demonstrations (weight: 1.0)                         │  │
│  │   • Autonomous (weight: 1.0)                             │  │
│  │   • Interventions (weight: 2.0) ⚡                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Format

### Standard LeRobot Dataset

Your dataset should follow the standard LeRobot format with additional RECAP fields:

```python
# In your dataset
{
    "observation.images": [...],
    "observation.state": [...],
    "action": [...],
    # RECAP-specific fields
    "data_type": "demo" | "auto" | "intervention",
    "reward": float,  # Episode reward
    "done": bool,     # Episode termination
    "intervention_mask": [...],  # Optional: intervention points
}
```

### Data Type Weights

| Type | Weight | Description |
|------|--------|-------------|
| `demo` | 1.0 | Expert teleoperation demonstrations |
| `auto` | 1.0 | Autonomous policy execution |
| `intervention` | 2.0 | Expert interventions during autonomy |

## Configuration Options

### IQL Hyperparameters

```python
config = PiStarRECAPConfig(
    iql_expectile=0.7,      # τ: Higher = focus on better actions
    iql_temperature=0.5,    # β: Lower = more deterministic
    iql_discount=0.99,      # γ: Future reward discount
)
```

### Training Strategy

```python
config = PiStarRECAPConfig(
    freeze_vlm=True,           # Freeze entire VLM
    freeze_vision_encoder=True,# Or just freeze vision
    train_expert_only=True,    # Only train action expert
    
    # Different learning rates
    q_lr=3e-4,      # Q-network
    v_lr=3e-4,      # V-network
    policy_lr=3e-5, # Policy (lower for stability)
)
```

### Flow Matching

```python
config = PiStarRECAPConfig(
    num_inference_steps=10,  # Flow matching steps (lower = faster)
    use_advantage_conditioning=True,
    advantage_scale=1.0,     # Scale for evaluation
)
```

## Training Tips

### 1. Data Collection

```python
# Label your data correctly
data_type = "intervention"  # If expert took over
# This gives 2x weight during training
```

### 2. Monitoring Training

Watch for these metrics in TensorBoard:

- `v_loss`: Value function loss (should decrease)
- `q_loss`: Q-function loss (should stabilize)
- `policy_loss`: Policy loss (weighted by advantage)
- `advantage_mean`: Average advantage (should increase)

### 3. Hyperparameter Tuning

| Scenario | Expectile (τ) | Temperature (β) |
|----------|---------------|-----------------|
| Conservative | 0.8 | 0.3 |
| Balanced | 0.7 | 0.5 |
| Exploratory | 0.6 | 0.7 |

### 4. Multi-Stage Training

```bash
# Stage 1: Pre-train with demonstrations only
python lerobot/scripts/train.py \
    --policy.type=pi_star_recap \
    --dataset.repo_id=user/demo_data_only \
    --output_dir=outputs/stage1

# Stage 2: Fine-tune with mixed data (RECAP)
python lerobot/scripts/train.py \
    --policy.type=pi_star_recap \
    --dataset.repo_id=user/mixed_data \
    --policy.pretrained_model_path=outputs/stage1/checkpoints/latest \
    --output_dir=outputs/stage2_recap
```

## Comparison with Other Policies

| Policy | Architecture | RL | Advantages |
|--------|-------------|-----|-----------|
| ACT | Transformer | ❌ | Simple, fast |
| Diffusion | U-Net | ❌ | Multi-modal actions |
| pi0 | VLA + Flow | ❌ | Language-conditioned |
| **pi_star_recap** | VLA + IQL | ✅ | Self-improving, heterogeneous data |

## Citation

```bibtex
@article{pi_star_06,
  title={π*₀.₆: a VLA That Learns From Experience},
  author={Physical Intelligence Team},
  journal={arXiv preprint arXiv:2511.14759},
  year={2025}
}
```

## License

Apache 2.0 (same as LeRobot)
