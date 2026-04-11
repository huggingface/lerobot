import dataclasses

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.optim.optimizers import AdamConfig


@PreTrainedConfig.register_subclass("flow_matching")
@dataclasses.dataclass
class FlowMatchingConfig(PreTrainedConfig):
    action_dim: int = 14
    qpos_dim: int = 13
    num_cameras: int = 1
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    hidden_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    max_horizon: int = 100

    # lerobot specific settings for dictionary mapping
    image_keys: list[str] = dataclasses.field(default_factory=lambda: ["observation.images.cam_high"])
    state_key: str = "observation.state"
    action_key: str = "action"

    # Flow Matching specific parameters
    uncond_prob: float = 0.1
    num_sampling_steps: int = 10
    omega: float = 1.0

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if self.input_features is None:
            return

        if self.state_key not in self.input_features:
            raise ValueError(f"Missing required state feature key: {self.state_key}.")

        state_feature = self.input_features[self.state_key]
        if state_feature.type is not FeatureType.STATE:
            raise ValueError(f"Feature `{self.state_key}` must be of type STATE.")
        if state_feature.shape[0] != self.qpos_dim:
            raise ValueError(
                f"State feature `{self.state_key}` first dimension must be {self.qpos_dim}, "
                f"got {state_feature.shape[0]}."
            )

        for image_key in self.image_keys:
            if image_key not in self.input_features:
                raise ValueError(f"Missing required image feature key: {image_key}.")
            image_feature = self.input_features[image_key]
            if image_feature.type is not FeatureType.VISUAL:
                raise ValueError(f"Feature `{image_key}` must be of type VISUAL.")

        if self.output_features is None or self.action_key not in self.output_features:
            raise ValueError(f"Missing required action output feature key: {self.action_key}.")

        action_feature = self.output_features[self.action_key]
        if action_feature.type is not FeatureType.ACTION:
            raise ValueError(f"Feature `{self.action_key}` must be of type ACTION.")
        if action_feature.shape[0] != self.action_dim:
            raise ValueError(
                f"Action feature `{self.action_key}` first dimension must be {self.action_dim}, "
                f"got {action_feature.shape[0]}."
            )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.max_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
