#!/usr/bin/env python3
"""
π*₀.₆ RECAP: Production Grade Implementation

Vision-Language-Action model trained with RECAP:
- RL with Experience and Corrections via Advantage-conditioned Policies

Features:
- FSDP (Fully Sharded Data Parallel) support
- Mixed precision training (bfloat16)
- Checkpoint management
- Efficient inference

Reference: "π*₀.₆: A VLA That Learns From Experience" (Physical Intelligence, 2025)
https://arxiv.org/abs/2511.14759
"""

from .configuration_pi_star_recap import (
    PiStarRECAPConfig,
    ModelConfig,
    IQLConfig,
    RECAPConfig,
    TrainingConfig,
    DistributedConfig,
    DataType,
)
from .modeling_pi_star_recap import (
    PiStarRECAPPolicy,
    ActionExpert,
    QNetwork,
    VNetwork,
    DiTBlock,
    TimestepEmbedder,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "PiStarRECAPConfig",
    "ModelConfig",
    "IQLConfig", 
    "RECAPConfig",
    "TrainingConfig",
    "DistributedConfig",
    "DataType",
    
    # Model
    "PiStarRECAPPolicy",
    "ActionExpert",
    "QNetwork",
    "VNetwork",
    "DiTBlock",
    "TimestepEmbedder",
]
