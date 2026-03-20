#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("siglip_decoder")
@dataclass
class SiglipDecoderConfig(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    dtype: str = "float32"

    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    empty_cameras: int = 0
    target_image_key: str | None = None

    # Encoder loading / freezing controls.
    load_pi0fast_vision_only_from: str | None = None
    load_from_model_safetensors: bool = True
    include_multi_modal_projector: bool = True

    # Decoder architecture.
    decoder_hidden_dim: int = 1024
    decoder_n_layers: int = 4
    decoder_n_heads: int = 8
    decoder_mlp_ratio: float = 4.0
    decoder_dropout: float = 0.1

    # Reconstruction objective.
    recon_loss_type: str = "l1"  # l1 | mse | charbonnier
    charbonnier_eps: float = 1e-3
    output_activation: str = "tanh"  # tanh | identity

    # Reconstruction eval on held-out episodes.
    eval_holdout_ratio: float = 0.1
    eval_num_batches: int = 8
    eval_visualization_num_frames: int = 64
    eval_video_fps: int = 8
    eval_compute_lpips: bool = False

    # Optimizer / schedule.
    optimizer_lr: float = 2.5e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.recon_loss_type not in ["l1", "mse", "charbonnier"]:
            raise ValueError(f"Invalid recon_loss_type: {self.recon_loss_type}")

        if self.output_activation not in ["tanh", "identity"]:
            raise ValueError(f"Invalid output_activation: {self.output_activation}")

        if not (0.0 <= self.eval_holdout_ratio < 1.0):
            raise ValueError(f"eval_holdout_ratio must be in [0, 1). Got {self.eval_holdout_ratio}")
        if self.eval_num_batches < 1:
            raise ValueError(f"eval_num_batches must be >= 1. Got {self.eval_num_batches}")
        if self.eval_visualization_num_frames < 2:
            raise ValueError(
                f"eval_visualization_num_frames must be >= 2. Got {self.eval_visualization_num_frames}"
            )
        if self.eval_video_fps < 1:
            raise ValueError(f"eval_video_fps must be >= 1. Got {self.eval_video_fps}")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(3,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return [0]

    @property
    def reward_delta_indices(self) -> None:
        return None
