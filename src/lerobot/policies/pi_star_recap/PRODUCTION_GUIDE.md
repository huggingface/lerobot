# π*₀.₆ RECAP - Production Deployment Guide

Complete guide for training and deploying π*₀.₆ RECAP at scale.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Distributed Training](#distributed-training)
6. [Evaluation](#evaluation)
7. [Deployment](#deployment)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)

## Overview

π*₀.₆ is a Vision-Language-Action (VLA) model trained with RECAP (RL with Experience and Corrections via Advantage-conditioned Policies).

### Key Features

- **VLA Architecture**: PaliGemma + Action Expert with Flow Matching
- **Offline RL**: IQL (Implicit Q-Learning) with expectile regression
- **Multi-modal Data**: Supports demo, auto-collected, and intervention data
- **Production Ready**: FSDP, mixed precision, checkpointing

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      π*₀.₆ RECAP                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PaliGemma VLM (Frozen)                  │   │
│  │   - Vision Encoder: SigLIP                           │   │
│  │   - Language Model: Gemma 2B                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Action Expert (Trainable)               │   │
│  │   - DiT (Diffusion Transformer) Blocks              │   │
│  │   - Advantage Conditioning                          │   │
│  │   - Flow Matching                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│           ┌───────────────┼───────────────┐                 │
│           ▼               ▼               ▼                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Q-Network  │  │  Q-Network  │  │  V-Network  │         │
│  │   (Twin)    │  │   (Twin)    │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

```bash
# Python 3.10+
python --version

# CUDA 12.1+
nvidia-smi

# Install LeRobot with dependencies
pip install -e ".[pi0]"

# Additional dependencies for training
pip install wandb pyyaml
```

### Hardware Requirements

| Configuration | GPUs | Memory | Training Time (100k steps) |
|--------------|------|--------|---------------------------|
| Single GPU   | 1x A100 80GB | 80GB | ~24 hours |
| Multi-GPU    | 4x A100 40GB | 160GB | ~6 hours |
| FSDP (Sharded) | 8x A100 40GB | 320GB | ~3 hours |

## Configuration

### Basic Config

```yaml
# configs/pi_star_recap_base.yaml
name: "pi_star_recap"

model:
  vlm_model_name: "google/paligemma-3b-pt-224"
  freeze_vlm: true
  action_expert_hidden_size: 1024
  action_expert_num_layers: 6
  
iql:
  discount: 0.99
  expectile: 0.7
  temperature: 0.5

recap:
  demo_weight: 1.0
  auto_weight: 1.0
  intervention_weight: 2.0

training:
  batch_size: 32
  learning_rate: 1e-4
  num_training_steps: 100000
```

### Advanced Config

```yaml
# configs/pi_star_recap_large.yaml
# For large-scale training

model:
  vlm_model_name: "google/paligemma-3b-pt-224"
  freeze_vlm: true
  action_expert_hidden_size: 2048
  action_expert_num_layers: 12
  action_expert_num_heads: 16
  
distributed:
  use_distributed: true
  fsdp_strategy: "full_shard"
  fsdp_mixed_precision: "bf16"

training:
  batch_size: 64
  gradient_accumulation_steps: 4
  use_amp: true
  amp_dtype: "bfloat16"
```

## Training

### Single GPU Training

```bash
python examples/train_pi_star_recap.py \
    --config configs/pi_star_recap.yaml \
    --dataset_path path/to/dataset \
    --output_dir ./outputs/pi_star_recap \
    --batch_size 32 \
    --num_epochs 100
```

### Multi-GPU Training (DDP)

```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/train_pi_star_recap.py \
    --config configs/pi_star_recap.yaml \
    --dataset_path path/to/dataset \
    --output_dir ./outputs/pi_star_recap_ddp
```

### Multi-Node Training

```bash
# On each node
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    examples/train_pi_star_recap.py \
    --config configs/pi_star_recap.yaml \
    --dataset_path path/to/dataset
```

## Distributed Training

### FSDP Configuration

```python
from lerobot.policies.pi_star_recap.distributed import setup_fsdp

policy = setup_fsdp(
    policy,
    device_id=local_rank,
    mixed_precision="bf16",
    strategy="full_shard",
)
```

### Memory Optimization

| Strategy | Memory | Speed | Use Case |
|----------|--------|-------|----------|
| `full_shard` | Lowest | Slowest | Large models |
| `shard_grad_op` | Medium | Medium | Medium models |
| `no_shard` | Highest | Fastest | Small models |

### Checkpoint Sharding

```python
# Save sharded checkpoint (each rank saves its shard)
policy.save_checkpoint_sharded(
    path=f"checkpoint_step_{step}/",
    distributed_manager=distributed_manager,
)

# Load sharded checkpoint
policy.load_checkpoint_sharded(
    path="checkpoint_step_10000/",
)
```

## Evaluation

### Dataset Evaluation

```bash
python examples/eval_pi_star_recap.py \
    --checkpoint outputs/pi_star_recap/checkpoint_100000.pt \
    --dataset_path path/to/eval_dataset \
    --output_dir ./eval_results
```

### Environment Rollout

```python
from lerobot.envs import make_env
from lerobot.policies.pi_star_recap import PiStarRECAPPolicy

# Load policy
policy = PiStarRECAPPolicy(config)
policy.load_checkpoint("checkpoint.pt")
policy = policy.cuda()

# Create environment
env = make_env("aloha", task="transfer_cube")

# Rollout
obs = env.reset()
for step in range(500):
    action = policy.select_action(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Success Rate | Task completion rate | > 80% |
| Mean Reward | Average episode reward | > 50 |
| Q-Value | Estimated value | Stable, increasing |
| Advantage | Q - V | Reasonable range |

## Deployment

### Model Export

```python
# Export to TorchScript
scripted_policy = torch.jit.script(policy)
torch.jit.save(scripted_policy, "policy_ts.pt")

# Export to ONNX
torch.onnx.export(
    policy,
    dummy_input,
    "policy.onnx",
    input_names=["images", "state"],
    output_names=["action"],
)
```

### Inference Server

```python
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
policy = load_policy()  # Load once at startup

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    obs = preprocess(data)
    
    with torch.no_grad():
        action = policy.select_action(obs)
    
    return jsonify({'action': action.cpu().numpy().tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### TensorRT Optimization

```python
import torch_tensorrt

# Compile with TensorRT
trt_model = torch_tensorrt.compile(
    policy,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],
    enabled_precisions={torch.float16},
)

# Save
torch.jit.save(trt_model, "policy_trt.pt")
```

## Monitoring

### Wandb Integration

```python
import wandb

wandb.init(
    project="pi-star-recap",
    config=config.to_dict(),
)

# Log metrics
wandb.log({
    'train/loss': loss,
    'train/q_value': q_value.mean(),
    'train/advantage': advantage.mean(),
})
```

### Key Metrics to Monitor

1. **Training Losses**
   - Total loss (should decrease)
   - V-loss (expectile regression)
   - Q-loss (Bellman error)
   - Policy loss (flow matching)

2. **Q-Values**
   - Mean Q (should increase)
   - Q distribution (check for collapse)

3. **Advantages**
   - Mean advantage (should be reasonable)
   - Advantage distribution

4. **Learning Rate**
   - Cosine decay schedule
   - Warmup progress

### Alerting

```python
# Example: Alert if Q-values collapse
if q_value.mean() < -100:
    send_alert("Q-values collapsed!")

# Example: Alert if loss spikes
if loss > 10 * previous_loss:
    send_alert("Loss spike detected!")
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Symptom**: CUDA OOM error

**Solutions**:
```yaml
# Reduce batch size
training:
  batch_size: 16  # Instead of 32

# Enable gradient checkpointing
model:
  use_gradient_checkpointing: true

# Use FSDP
distributed:
  use_distributed: true
  fsdp_strategy: "full_shard"
```

#### 2. Training Instability

**Symptom**: Loss spikes, NaN values

**Solutions**:
```yaml
# Reduce learning rate
training:
  action_expert_lr: 5e-5  # Instead of 1e-4

# Increase gradient clipping
training:
  max_grad_norm: 0.5  # Instead of 1.0

# Check data normalization
# Ensure observations are in reasonable range
```

#### 3. Poor Sample Efficiency

**Symptom**: Slow learning, low success rate

**Solutions**:
```yaml
# Adjust IQL hyperparameters
iql:
  expectile: 0.9  # Higher for better value estimation
  temperature: 0.1  # Lower for sharper advantage weighting

# Adjust data weights
recap:
  intervention_weight: 5.0  # Higher weight on interventions
```

#### 4. Slow Inference

**Symptom**: High latency during deployment

**Solutions**:
```python
# Reduce inference steps
config.num_inference_steps = 5  # Instead of 10

# Use TorchScript
policy = torch.jit.script(policy)

# Use TensorRT
policy = optimize_with_tensorrt(policy)
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model outputs
policy.eval()
with torch.no_grad():
    losses = policy.compute_loss(batch)
    print(f"V-loss: {losses['v_loss']}")
    print(f"Q-loss: {losses['q_loss']}")
    print(f"Policy loss: {losses['policy_loss']}")
```

### Performance Profiling

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    policy.training_step(batch)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Best Practices

### 1. Data Preparation

- Ensure data is normalized
- Balance data types (demo/auto/intervention)
- Filter out low-quality demonstrations

### 2. Training

- Start with smaller models
- Use gradient accumulation for large batches
- Monitor validation metrics
- Save checkpoints frequently

### 3. Evaluation

- Evaluate on held-out data
- Test in simulation before real robot
- Measure success rate, not just loss

### 4. Deployment

- Quantize model for faster inference
- Use TorchScript/ONNX for portability
- Implement proper error handling

## Reference

### Paper

"π*₀.₆: A VLA That Learns From Experience"
Physical Intelligence, 2025
https://arxiv.org/abs/2511.14759

### Related Work

- π₀: "A Vision-Language-Action Flow Model for General Robot Control"
- IQL: "Offline Reinforcement Learning with Implicit Q-Learning"
- DiT: "Scalable Diffusion Models with Transformers"

### Contact

For issues and questions:
- GitHub Issues: https://github.com/GWinfinity/lerobot_loong/issues
- Email: your-email@example.com
