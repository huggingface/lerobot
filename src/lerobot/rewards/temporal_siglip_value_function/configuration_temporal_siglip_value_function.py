"""Configuration for the experimental temporal SigLIP2 value function."""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@RewardModelConfig.register_subclass("temporal_siglip_value_function")
@dataclass
class TemporalSiglipVFConfig(RewardModelConfig):
    siglip_path: str = "google/siglip2-so400m-patch14-384"
    image_resolution: tuple[int, int] = (384, 384)
    tokenizer_max_length: int = 64
    history_steps: int = 6
    frame_gap: int = 30
    state_key: str = "observation.state"
    state_dim: int = 32
    hidden_size: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    num_value_bins: int = 201
    value_support_min: float = -1.0
    value_support_max: float = 0.0
    hl_gauss_sigma_ratio: float = 0.75
    target_method: str = "dirac_delta"
    use_one_hot_terminal: bool = True

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
        }
    )

    @property
    def observation_delta_indices(self) -> list[int]:
        return [-self.frame_gap * index for index in range(self.history_steps - 1, -1, -1)]

    def validate_features(self) -> None:
        if not any(feature.type == FeatureType.VISUAL for feature in self.input_features.values()):
            raise ValueError("TemporalSiglipVFConfig requires visual input features")
        if self.state_key not in self.input_features:
            raise ValueError(f"TemporalSiglipVFConfig requires {self.state_key!r}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=1e-4, weight_decay=1e-4, grad_clip_norm=1.0)

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=500,
            num_decay_steps=40000,
            peak_lr=1e-4,
            decay_lr=1e-6,
        )
