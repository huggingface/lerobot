#!/usr/bin/env python3
"""
Configuration for π*₀.₆ RECAP Policy

RECAP: RL with Experience and Corrections via Advantage-conditioned Policies
Based on: "π*₀.₆: a VLA That Learns From Experience" (Physical Intelligence, 2025)

Integrated with LeRobot framework
"""

from dataclasses import dataclass, field
from typing import Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi_star_recap")
@dataclass
class PiStarRECAPConfig(PreTrainedConfig):
    """
    Configuration for π*₀.₆ with RECAP training
    
    Key features:
    - IQL (Implicit Q-Learning) for offline RL
    - Advantage-conditioned flow matching policy
    - Support for heterogeneous data (demo, auto, intervention)
    """
    
    # VLM Backbone configuration
    vlm_model_name: str = "google/gemma-2b-it"
    vlm_variant: str = "gemma_2b"
    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"
    
    # Observation and action configuration
    n_obs_steps: int = 1  # Number of observation steps
    chunk_size: int = 16  # Action prediction horizon
    n_action_steps: int = 16  # Number of action steps to execute
    
    max_state_dim: int = 32
    max_action_dim: int = 32
    
    # Flow matching parameters
    num_inference_steps: int = 10  # Flow matching steps during inference
    flow_beta_schedule: str = "cosine"  # "cosine" or "linear"
    
    # IQL hyperparameters
    iql_expectile: float = 0.7  # τ for expectile regression (0.5 = median, 0.7 = high value)
    iql_temperature: float = 0.5  # β for advantage weighting
    iql_discount: float = 0.99  # γ for future reward discount
    num_q_networks: int = 2  # Clipped double Q-learning
    
    # RECAP specific
    use_advantage_conditioning: bool = True  # Condition policy on advantage
    advantage_scale: float = 1.0  # Scale factor for advantage
    
    # Data mixing weights for RECAP
    demo_weight: float = 1.0
    auto_weight: float = 1.0
    intervention_weight: float = 2.0  # Interventions are most valuable
    balance_data_types: bool = True  # Balance sampling across data types
    
    # Image configuration
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    
    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    
    # Training configuration
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None
    
    # Freezing configuration
    freeze_vlm: bool = True  # Freeze entire VLM
    freeze_vision_encoder: bool = True  # Freeze vision encoder specifically
    train_expert_only: bool = True  # Only train action expert
    
    # Optimizer settings
    # Q-network optimizer
    q_lr: float = 3e-4
    q_betas: tuple[float, float] = (0.9, 0.999)
    q_weight_decay: float = 0.01
    
    # V-network optimizer
    v_lr: float = 3e-4
    v_betas: tuple[float, float] = (0.9, 0.999)
    v_weight_decay: float = 0.01
    
    # Policy optimizer (lower LR for stability)
    policy_lr: float = 3e-5
    policy_betas: tuple[float, float] = (0.9, 0.95)
    policy_weight_decay: float = 0.01
    policy_grad_clip_norm: float = 1.0
    
    # Scheduler settings
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6
    
    # Loss weights
    q_loss_weight: float = 1.0
    v_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    
    # Target network update
    target_update_tau: float = 0.005  # Soft update coefficient
    target_update_period: int = 1  # Update every N steps
    
    # Evaluation
    eval_advantage_scale: float = 1.0  # Scale for evaluation
    eval_num_samples: int = 1  # Number of action samples for evaluation
    
    tokenizer_max_length: int = 48
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate IQL parameters
        assert 0.0 < self.iql_expectile < 1.0, "Expectile must be in (0, 1)"
        assert self.iql_temperature > 0.0, "Temperature must be positive"
        assert 0.0 < self.iql_discount <= 1.0, "Discount must be in (0, 1]"
        
        # Validate data weights
        assert self.demo_weight >= 0.0, "Demo weight must be non-negative"
        assert self.auto_weight >= 0.0, "Auto weight must be non-negative"
        assert self.intervention_weight >= 0.0, "Intervention weight must be non-negative"
