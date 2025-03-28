from dataclasses import dataclass
from typing import List

from lerobot.common.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass(name="hilserl_classifier")
@dataclass
class ClassifierConfig(PreTrainedConfig):
    """Configuration for the Classifier model."""

    name: str = "hilserl_classifier"
    num_classes: int = 2
    hidden_dim: int = 256
    dropout_rate: float = 0.1
    model_name: str = "helper2424/resnet10"
    device: str = "cpu"
    model_type: str = "cnn"  # "transformer" or "cnn"
    num_cameras: int = 2
    learning_rate: float = 1e-4
    normalization_mode = None
    # output_features: Dict[str, PolicyFeature] = field(
    #     default_factory=lambda: {"next.reward": PolicyFeature(type=FeatureType.REWARD, shape=(1,))}
    # )

    @property
    def observation_delta_indices(self) -> List | None:
        return None

    @property
    def action_delta_indices(self) -> List | None:
        return None

    @property
    def reward_delta_indices(self) -> List | None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(
            lr=self.learning_rate,
            weight_decay=0.01,
            grad_clip_norm=1.0,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        """Validate feature configurations."""
        # Classifier doesn't need specific feature validation
        pass
