from lerobot.common.optim.optimizers import OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.configs.policies import PretrainedConfig
from dataclasses import dataclass, field

from lerobot.configs.types import NormalizationMode

@PretrainedConfig.register_subclass("pi0")
@dataclass
class PI0Config(PretrainedConfig):

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    state_dim: int = 24
    action_dim: int = 24

    # Decoding
    num_steps: int = 10

    # Action expert
    action_expert_width: int = 1024
    action_expert_depth: int = 18
    action_expert_mlp_dim: int = 4096
    action_expert_num_heads: int = 8
    action_expert_num_kv_heads: int = 1
    action_expert_head_dim: int = 256
    action_expert_projection_lora: bool | None = None
    action_expert_projection_kv_lora: bool | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        
    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
    
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        raise NotImplementedError

    def validate_features(self) -> None:
        raise NotImplementedError