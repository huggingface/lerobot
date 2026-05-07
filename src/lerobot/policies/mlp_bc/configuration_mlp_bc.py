from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("mlp_bc")
@dataclass
class MLPBCConfig(PreTrainedConfig):
    """Plain MLP behavior-cloning policy.

    Maps `observation.state` (shape [state_dim]) to `action` (shape [action_dim]) via a
    feed-forward MLP trained with L1 regression. No images, no temporal modeling, no
    action chunking — the simplest baseline for state-only ("blind") tasks.
    """

    n_obs_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    hidden_dims: tuple[int, ...] = (256, 256, 256)
    dropout: float = 0.1
    use_layernorm: bool = True

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_obs_steps != 1:
            raise ValueError(
                f"MLPBCConfig only supports n_obs_steps=1, got {self.n_obs_steps}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError(
                "MLPBCPolicy requires `observation.state` (FeatureType.STATE) as input."
            )
        if self.action_feature is None:
            raise ValueError("MLPBCPolicy requires `action` (FeatureType.ACTION) as output.")
        if self.image_features:
            raise ValueError(
                "MLPBCPolicy is a state-only policy; image features are not supported. "
                f"Got image keys: {list(self.image_features)}."
            )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
