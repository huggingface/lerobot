from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("safediff_vla")
@dataclass
class SafeDiffVLAConfig(PreTrainedConfig):
    """Configuration for the external SafeDiff-VLA trajectory planner."""

    n_obs_steps: int = 1
    action_horizon: int = 16
    execute_horizon: int = 4
    backbone_name: str | None = None
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    tokenizer_max_length: int = 48
    pad_language_to: str = "longest"
    freeze_backbone: bool = True
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    num_diffusion_steps: int = 10
    beta_schedule: str = "cosine"
    prediction_type: str = "epsilon"
    num_candidates: int = 4
    use_vla_prior_init: bool = True

    latent_dim: int = 256
    planner_hidden_dim: int = 512
    task_critic_hidden_dim: int = 256
    risk_critic_hidden_dim: int = 256
    timestep_embedding_dim: int = 64

    training_mode: str = "joint"
    lambda_diff: float = 1.0
    lambda_task: float = 1.0
    lambda_risk: float = 1.0
    lambda_prior: float = 0.05
    use_task_critic: bool = True
    use_safety_critic: bool = True

    use_diffusion_refinement: bool = True
    use_critic_guidance: bool = False
    critic_guidance_scale: float = 0.1
    critic_gradient_clip: float = 1.0
    adaptive_planning: bool = False
    risk_threshold: float = 0.5
    uncertainty_threshold: float = 0.5
    enable_inference_metrics: bool = False

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.execute_horizon <= self.action_horizon:
            raise ValueError("execute_horizon must be in [1, action_horizon]")
        if self.num_diffusion_steps < 1 or self.num_candidates < 1:
            raise ValueError("num_diffusion_steps and num_candidates must be positive")
        if self.use_lora and self.freeze_backbone:
            raise ValueError("use_lora requires freeze_backbone=False")
        if self.beta_schedule not in {"linear", "cosine"}:
            raise ValueError("beta_schedule must be 'linear' or 'cosine'")
        if self.prediction_type != "epsilon":
            raise ValueError("prediction_type must be 'epsilon'")
        if self.training_mode not in {"diffusion", "critics", "joint"}:
            raise ValueError("training_mode must be 'diffusion', 'critics', or 'joint'")

    def validate_features(self) -> None:
        if self.action_feature is None:
            raise ValueError("SafeDiff-VLA requires an action output feature")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr / 10,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
