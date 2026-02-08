# QC-FQL: Q-Chunking with Fitted Q-Learning

Implementation of **"Reinforcement Learning with Action Chunking"** (Li et al., 2025)

Paper: https://arxiv.org/abs/2507.07969

## Overview

QC-FQL is an offline-to-online RL algorithm that combines **action chunking** with **flow-based policies** and **optimal transport** to achieve strong sample efficiency on long-horizon, sparse-reward tasks.

### Key Ideas

1. **Action Chunking**: Instead of predicting single actions, predict sequences of k actions. This provides:
   - Temporally coherent exploration
   - Unbiased n-step TD backups (n = chunk size)
   - Better handling of non-Markovian behavior in data

2. **Flow-Matching Behavior Policy**: Captures complex behavior distributions using flow matching, which learns a velocity field to transform noise into data samples.

3. **Noise-Conditioned Policy**: The learned policy takes Gaussian noise as input and outputs action chunks, enabling:
   - Stochastic policy via noise sampling
   - Easy Q-maximization

4. **Wasserstein Constraint**: Distillation loss serves as an upper bound on the 2-Wasserstein distance between the learned policy and behavior policy.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         QC-FQL                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │ Flow-Matching       │    │ Noise-Conditioned   │             │
│  │ Behavior Policy π_b │    │ Policy π_θ          │             │
│  │                     │    │                     │             │
│  │ Trained with        │    │ Trained to          │             │
│  │ flow matching loss  │    │ maximize Q +        │             │
│  │                     │    │ distillation        │             │
│  └─────────────────────┘    └─────────────────────┘             │
│           │                            │                        │
│           ▼                            ▼                        │
│  ┌─────────────────────────────────────────────────────┐       │
│  │           Critic Ensemble Q(s, a_chunk)             │       │
│  │                                                     │       │
│  │   - Takes state and action chunk as input          │       │
│  │   - Provides unbiased n-step backup                │       │
│  │   - Conservative estimation via min over ensemble  │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Algorithm

### Training Loop

```
1. Pretrain behavior policy π_b with flow matching on offline data

2. For each training step:
   a. Train behavior policy:
      L_behavior = E[||v_t(a_t) - (a_1 - a_0)||²]
   
   b. Train critic ensemble:
      L_critic = E[(Q(s, a) - (r + γⁿ * min Q'(s', π(s'))))²]
      
   c. Train policy:
      L_policy = -E[Q(s, π(s, ε))] + λ * ||π(s, ε) - π_b(s)||²
      
   d. Update target networks: θ' ← τθ + (1-τ)θ'
```

### Loss Components

1. **Flow Matching Loss**: Train velocity field to match conditional flow
   - a_t = (1-t) * a_0 + t * a_1
   - Target: a_1 - a_0
   - Prediction: v_t(a_t; s)

2. **TD Loss**: Standard Q-learning on action chunks
   - Unbiased n-step backup (n = chunk size)
   - Conservative estimation via ensemble min

3. **Policy Loss**: Q-maximization + distillation
   - Q-maximization: Sample noise ε, predict action, maximize Q
   - Distillation: Match policy output to behavior samples

## Usage

### Basic Example

```python
from lerobot.policies.qc_fql import QCFQLPolicy, QCFQLConfig

# Create configuration
config = QCFQLConfig(
    action_chunk_size=4,      # k=4: predict 4 actions at a time
    num_critics=10,           # Ensemble of 10 critics
    distillation_weight=1.0,  # λ=1.0: Wasserstein constraint strength
    state_dim=128,
    action_dim=16,
    discount=0.99,
)

# Create policy
policy = QCFQLPolicy(config)

# Training
for batch in dataloader:
    # Compute losses
    losses = policy.forward(batch)
    
    # Update behavior policy
    loss_behavior = losses["loss_behavior"]
    optimizer_behavior.zero_grad()
    loss_behavior.backward()
    optimizer_behavior.step()
    
    # Update critic
    loss_critic = losses["loss_critic"]
    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()
    
    # Update target networks
    policy.update_target_networks()
    
    # Update policy
    loss_policy = losses["loss_policy"]
    optimizer_policy.zero_grad()
    loss_policy.backward()
    optimizer_policy.step()

# Inference
action = policy.select_action({"observation.state": obs})
```

### Training from Command Line

```bash
python examples/train_qc_fql.py \
    --dataset_path path/to/lerobot/dataset \
    --output_dir ./checkpoints/qc_fql \
    --action_chunk_size 4 \
    --num_critics 10 \
    --distillation_weight 1.0 \
    --num_pretrain_steps 100000 \
    --num_online_steps 100000 \
    --batch_size 256
```

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `action_chunk_size` | 4 | Number of actions in a chunk (k) |
| `num_critics` | 10 | Number of critics in ensemble |
| `distillation_weight` | 1.0 | Weight for distillation loss (λ) |
| `discount` | 0.99 | Discount factor (γ) |
| `critic_target_update_tau` | 0.005 | EMA weight for target networks |
| `flow_matching_sigma` | 0.001 | Noise scale for flow matching |
| `flow_matching_num_inference_steps` | 10 | Integration steps for sampling |

### Recommended Settings

For **long-horizon manipulation tasks**:
- `action_chunk_size=4`: Good balance between exploration and reactivity
- `num_critics=10`: Conservative estimation with reasonable compute
- `distillation_weight=1.0`: Standard Wasserstein constraint

For **shorter tasks**:
- `action_chunk_size=2`: More reactive policy
- `num_critics=5`: Faster training

For **complex tasks**:
- `action_chunk_size=8`: More temporally coherent exploration
- `num_critics=20`: More conservative Q-estimation
- `distillation_weight=2.0`: Stronger behavior constraint

## Key Components

### 1. FlowMatchingBehaviorPolicy

Captures behavior distribution using flow matching:

```python
# Training
velocity = policy.forward(state, action_chunk, time)
loss = MSE(velocity, target_action - noise)

# Sampling
action_chunk = policy.sample(state, num_steps=10)
```

### 2. ChunkedCritic

Q-function operating on action chunks:

```python
q_value = critic(state, action_chunk)  # (batch, 1)
```

### 3. NoiseConditionedPolicy

Policy network with noise conditioning:

```python
action_chunk = policy(state, noise)  # (batch, chunk_size, action_dim)
```

## Implementation Notes

### Action Chunking

- Actions are grouped into chunks of size `k`
- During execution, actions are applied open-loop for `k` steps
- The critic learns Q(s, a_chunk) enabling unbiased n-step backups

### Flow Matching

- Uses conditional flow matching for efficient training
- Velocity field is conditioned on state and time
- Sampling via Euler integration (configurable steps)

### Distillation Loss

- Upper bound on 2-Wasserstein distance
- Implemented as MSE between policy output and behavior samples
- Weight λ controls strength of behavior constraint

## Comparison with Other Methods

| Method | Action Space | Critic | Exploration |
|--------|-------------|--------|-------------|
| SAC | Single action | Q(s,a) | Independent noise |
| FQL | Single action | Q(s,a) | Flow matching |
| QC (best-of-n) | Action chunk | Q(s,a_chunk) | Best-of-n sampling |
| **QC-FQL** | **Action chunk** | **Q(s,a_chunk)** | **Flow + distillation** |

## Citation

```bibtex
@article{li2025reinforcement,
  title={Reinforcement Learning with Action Chunking},
  author={Li, Qiyang and Zhou, Zhiyuan and Levine, Sergey},
  journal={arXiv preprint arXiv:2507.07969},
  year={2025}
}
```

## References

1. Li et al., "Reinforcement Learning with Action Chunking", 2025
2. Lipman et al., "Flow Matching for Generative Modeling", 2022
3. Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning", 2021
4. Haarnoja et al., "Soft Actor-Critic", 2018
