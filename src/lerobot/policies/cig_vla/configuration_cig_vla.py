from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("cig_vla")
@dataclass
class CIGVLAConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16
    max_state_dim: int = 32
    max_action_dim: int = 7
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    qwen_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    torch_dtype: str = "bfloat16"
    freeze_vision_tower: bool = True
    gradient_checkpointing: bool = True
    use_peft: bool = False
    enable_qwen_lora: bool = True
    grounding_architecture: str = "learned_queries"
    grounding_hidden_dim: int = 512
    grounding_num_heads: int = 8
    grounding_num_layers: int = 2
    bottleneck_mode: str = "interaction_tuple"
    detach_bottleneck_for_main_action: bool = True
    detach_bottleneck_for_causal_branch: bool = True
    controller_hidden_dim: int = 384
    controller_num_layers: int = 6
    controller_num_heads: int = 8
    num_inference_steps: int = 10
    translation_goal_loss_weight: float = 1.0
    approach_direction_loss_weight: float = 0.5
    translation_magnitude_loss_weight: float = 0.5
    rotation_goal_loss_weight: float = 0.0
    gripper_transition_loss_weight: float = 0.25
    action_loss_weight: float = 1.0
    enable_causal_intervention: bool = False
    causal_loss_weight: float = 0.1
    translation_goal_shift_m: float = 0.01
    causal_action_prefix_steps: int = 4
    causal_response_margin: float = 0.01
    action_semantics: str = "libero_safety"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    optimizer_lr: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if self.grounding_architecture not in {"last_token", "mean_pool", "learned_queries"}:
            raise ValueError("Invalid grounding_architecture")

    @property
    def observation_delta_indices(self):
        return [0]

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    def delta_indices_for_feature(self, key: str):
        if key == "actions":
            return self.action_delta_indices
        return None

    @property
    def reward_delta_indices(self):
        return None

    def validate_features(self):
        return None

    def get_optimizer_preset(self):
        return AdamWConfig(lr=self.optimizer_lr)

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr, decay_lr=1e-6, num_warmup_steps=1000, num_decay_steps=30000
        )
