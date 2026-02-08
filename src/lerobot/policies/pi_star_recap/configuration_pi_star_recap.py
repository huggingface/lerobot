#!/usr/bin/env python3
"""
π*₀.₆ RECAP Configuration - Production Grade

Reference: "π*₀.₆: A VLA That Learns From Experience" (Physical Intelligence, 2025)
https://arxiv.org/abs/2511.14759
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum

from lerobot.configs.policies import PreTrainedConfig


class DataType(Enum):
    """RECAP data types"""
    DEMO = "demo"
    AUTO = "auto"
    INTERVENTION = "intervention"


@dataclass
class ModelConfig:
    """VLA Model architecture config"""
    vlm_model_name: str = "google/paligemma-3b-pt-224"
    freeze_vlm: bool = True
    freeze_vision_encoder: bool = False
    train_expert_only: bool = True  # Only train action expert
    
    # Action Expert
    action_expert_hidden_size: int = 1024
    action_expert_num_layers: int = 6
    action_expert_num_heads: int = 8
    action_expert_mlp_ratio: float = 4.0
    action_expert_dropout: float = 0.1
    
    # Q/V Networks
    qv_hidden_size: int = 512
    qv_num_layers: int = 3
    num_q_networks: int = 2  # Twin Q
    
    # Flow Matching
    flow_matching_sigma: float = 0.001
    flow_matching_num_steps: int = 10


@dataclass
class IQLConfig:
    """IQL (Implicit Q-Learning) hyperparameters"""
    discount: float = 0.99
    expectile: float = 0.7
    temperature: float = 0.5
    
    # Loss weights
    v_loss_weight: float = 1.0
    q_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0


@dataclass
class RECAPConfig:
    """RECAP data mixing config"""
    # Data type weights
    demo_weight: float = 1.0
    auto_weight: float = 1.0
    intervention_weight: float = 2.0
    
    # Advantage conditioning
    use_advantage_conditioning: bool = True
    eval_advantage_scale: float = 1.0
    
    # Advantage clamping
    advantage_min: float = -10.0
    advantage_max: float = 10.0


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Batch sizes
    batch_size: int = 32
    eval_batch_size: int = 64
    
    # Learning rates
    vlm_lr: float = 1e-5
    action_expert_lr: float = 1e-4
    qv_lr: float = 3e-4
    
    # Optimizer
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0
    
    # Training duration
    num_warmup_steps: int = 5000
    num_training_steps: int = 100000
    
    # Target network update
    target_update_tau: float = 0.005
    target_update_period: int = 1
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # bfloat16, float16
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Evaluation
    eval_every_n_steps: int = 1000
    save_every_n_steps: int = 5000


@dataclass
class DistributedConfig:
    """Distributed training configuration"""
    use_distributed: bool = False
    backend: str = "nccl"
    fsdp_strategy: str = "full_shard"  # full_shard, shard_grad_op, no_shard
    fsdp_wrap_policy: str = "transformer_layer"
    
    # Mixed precision for distributed
    fsdp_mixed_precision: str = "bf16"  # bf16, fp16, fp32
    
    # Checkpoint sharding
    state_dict_type: str = "sharded"  # sharded, full


@PreTrainedConfig.register_subclass("pi_star_recap")
@dataclass
class PiStarRECAPConfig(PreTrainedConfig):
    """
    Complete configuration for π*₀.₆ RECAP policy - Production Grade
    
    π*₀.₆ is a VLA (Vision-Language-Action) model trained with RECAP:
    - RL with Experience and Corrections via Advantage-conditioned Policies
    - Combines IQL (Implicit Q-Learning) with Flow Matching
    - Supports multi-modal data: demo, auto-collected, intervention
    """
    
    # Model identifier
    name: str = "pi_star_recap"
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    iql: IQLConfig = field(default_factory=IQLConfig)
    recap: RECAPConfig = field(default_factory=RECAPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    # Input/Output specification
    image_size: int = 224
    num_obs_steps: int = 2
    chunk_size: int = 10  # Action chunk size
    max_action_dim: int = 14
    
    # Inference
    num_inference_steps: int = 10
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.iql.expectile > 0.5, "expectile must be > 0.5 for IQL"
        assert 0 < self.iql.discount <= 1, "discount must be in (0, 1]"
        assert self.training.batch_size > 0, "batch_size must be positive"
        
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "name": self.name,
            "model": vars(self.model),
            "iql": vars(self.iql),
            "recap": vars(self.recap),
            "training": vars(self.training),
            "distributed": vars(self.distributed),
            "image_size": self.image_size,
            "num_obs_steps": self.num_obs_steps,
            "chunk_size": self.chunk_size,
            "max_action_dim": self.max_action_dim,
            "num_inference_steps": self.num_inference_steps,
            "device": self.device,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "PiStarRECAPConfig":
        """Create config from dictionary"""
        return cls(
            name=config_dict.get("name", "pi_star_recap"),
            model=ModelConfig(**config_dict.get("model", {})),
            iql=IQLConfig(**config_dict.get("iql", {})),
            recap=RECAPConfig(**config_dict.get("recap", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            distributed=DistributedConfig(**config_dict.get("distributed", {})),
            image_size=config_dict.get("image_size", 224),
            num_obs_steps=config_dict.get("num_obs_steps", 2),
            chunk_size=config_dict.get("chunk_size", 10),
            max_action_dim=config_dict.get("max_action_dim", 14),
            num_inference_steps=config_dict.get("num_inference_steps", 10),
            device=config_dict.get("device", "cuda"),
        )
