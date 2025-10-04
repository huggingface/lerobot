from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)


@PreTrainedConfig.register_subclass("smolandfast")
@dataclass
class SMOLANDFASTConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 10
    n_action_steps: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "ENV": NormalizationMode.MIN_MAX,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    n_state_bins = 256

    # Decoding
    max_decoding_steps: int = 512
    fast_skip_tokens: int = 280  # Skip last 280 tokens
    max_input_seq_len: int = 512  # 512

    # Utils
    use_cache: bool = True

    # Training presets
    vision_model_optimizer_lr: float = 2e-5
    connector_optimizer_lr: float = 2e-4
    text_model_optimizer_lr: float = 2e-4
    optimizer_lr: float = 2e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    checkpoint_path: str = None

    vlm_checkpoint: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

    precision: str = "float32"
    freeze_vision_encoder: bool = True
    freeze_connector: bool = True
    scale_factor: int = 4
    do_image_spliting: bool = False
    drop_n_last_frames: bool = True

    grad_clip_norm: float = 1

    # Allows padding/truncation of generated action tokens during detokenization to ensure decoding.
    # In the original version, tensors of 0s were generated if shapes didn't match for stable decoding.
    relaxed_action_decoding: bool = True

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
        pass

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(0, self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
