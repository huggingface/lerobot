from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("vla_jepa")
@dataclass
class VLAJEPAConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    qwen_model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    jepa_encoder_name: str = "facebook/vjepa2-vitl-fpc64-256"

    tokenizer_padding_side: str = "left"
    prompt_template: str = "{instruction}\n\nPredict {actions} and condition future prediction with {e_actions}."
    special_action_token: str = "<|action_{}|>"
    embodied_action_token: str = "<|embodied_action|>"

    action_dim: int = 7
    state_dim: int = 7
    future_action_window_size: int = 15
    past_action_window_size: int = 0
    num_action_tokens_per_timestep: int = 4
    num_embodied_action_tokens_per_instruction: int = 8
    num_inference_timesteps: int = 10

    action_hidden_size: int = 1024
    action_model_type: str = "DiT-B"
    action_num_layers: int = 12
    action_num_heads: int = 12
    action_attention_head_dim: int = 64
    action_dropout: float = 0.1
    action_num_timestep_buckets: int = 1000
    action_noise_beta_alpha: float = 1.5
    action_noise_beta_beta: float = 1.0
    action_noise_s: float = 0.999

    num_video_frames: int = 4
    predictor_depth: int = 6
    predictor_num_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    predictor_dropout: float = 0.0
    world_model_loss_weight: float = 0.1
    enable_world_model: bool = True

    resize_images_to: tuple[int, int] | None = None
    torch_dtype: str = "bfloat16"

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("`n_action_steps` must be <= `chunk_size`.")
        if self.future_action_window_size + 1 > self.chunk_size:
            raise ValueError("`chunk_size` must cover the predicted action horizon.")
        if self.num_video_frames < 2:
            raise ValueError("`num_video_frames` must be >= 2 for JEPA prediction.")

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("VLAJEPA requires at least one visual input feature.")
        if self.action_feature is None:
            raise ValueError("VLAJEPA requires an action output feature.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
