from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import CosineAnnealingSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("dot")
@dataclass
class DOTConfig(PreTrainedConfig):
    """Configuration for DOT (Decision Transformer) policy.

    You need to change some parameters in this configuration to make it work for your problem:

    FPS/prediction horizon related features - may need to adjust:
    - train_horizon: the number of steps to predict during training
    - inference_horizon: the number of steps to predict during validation
    - alpha: exponential factor for weighting of each next action
    - train_alpha: exponential factor for action weighting during training

    For inference speed optimization:
    - predict_every_n: number of frames to predict in the future
    - return_every_n: instead of returning next predicted actions, returns nth future action
    """

    # Input / output structure.
    n_obs_steps: int = 3
    train_horizon: int = 20
    inference_horizon: int = 20
    lookback_obs_steps: int = 10
    lookback_aug: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Not sure if there is a better way to do this with new config system.
    override_dataset_stats: bool = False
    new_dataset_stats: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            "action": {"max": [512.0] * 2, "min": [0.0] * 2},
            "observation.environment_state": {"max": [512.0] * 16, "min": [0.0] * 16},
            "observation.state": {"max": [512.0] * 2, "min": [0.0] * 2},
        }
    )

    # Architecture.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    pre_norm: bool = True
    lora_rank: int = 20
    merge_lora: bool = False

    dim_model: int = 128
    n_heads: int = 8
    dim_feedforward: int = 512
    n_decoder_layers: int = 8
    rescale_shape: tuple[int, int] = (96, 96)

    # Augmentation.
    crop_scale: float = 0.8
    state_noise: float = 0.01
    noise_decay: float = 0.999995

    # Training and loss computation.
    dropout: float = 0.1

    # Weighting and inference.
    alpha: float = 0.75
    train_alpha: float = 0.9
    predict_every_n: int = 1
    return_every_n: int = 1

    # Training preset
    optimizer_lr: float = 1.0e-4
    optimizer_min_lr: float = 1.0e-4
    optimizer_lr_cycle_steps: int = 300000
    optimizer_weight_decay: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if self.predict_every_n > self.inference_horizon:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be less than or equal to horizon ({self.inference_horizon})."
            )
        if self.return_every_n > self.inference_horizon:
            raise ValueError(
                f"return_every_n ({self.return_every_n}) must be less than or equal to horizon ({self.inference_horizon})."
            )
        if self.predict_every_n > self.inference_horizon // self.return_every_n:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be less than or equal to horizon //  return_every_n({self.inference_horizon // self.return_every_n})."
            )
        if self.train_horizon < self.inference_horizon:
            raise ValueError(
                f"train_horizon ({self.train_horizon}) must be greater than or equal to horizon ({self.inference_horizon})."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return CosineAnnealingSchedulerConfig(
            min_lr=self.optimizer_min_lr, T_max=self.optimizer_lr_cycle_steps
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        far_past_obs = list(
            range(
                -self.lookback_aug - self.lookback_obs_steps, self.lookback_aug + 1 - self.lookback_obs_steps
            )
        )
        recent_obs = list(range(2 - self.n_obs_steps, 1))

        return far_past_obs + recent_obs

    @property
    def action_delta_indices(self) -> list:
        far_past_actions = list(
            range(
                -self.lookback_aug - self.lookback_obs_steps, self.lookback_aug + 1 - self.lookback_obs_steps
            )
        )
        recent_actions = list(range(2 - self.n_obs_steps, self.train_horizon))

        return far_past_actions + recent_actions

    @property
    def reward_delta_indices(self) -> None:
        return None
