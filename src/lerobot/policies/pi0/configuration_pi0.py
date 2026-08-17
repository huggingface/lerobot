#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi0")
@dataclass
class PI0Config(PreTrainedConfig):
    """Configuration class for the PI0 flow-matching vision-language-action policy.

    PI0 is a PyTorch port of Physical Intelligence's openpi model: a PaliGemma vision-language backbone
    paired with a smaller Gemma "action expert" that generates action chunks via flow matching.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 1):
            Number of environment steps of observation to pass to the policy (the current step and
            additional steps going back).
        input_features (`dict[str, PolicyFeature] | None`, *optional*):
            Mapping from input feature name to its `PolicyFeature` (type and shape). Inferred from the
            dataset when left empty.
        output_features (`dict[str, PolicyFeature] | None`, *optional*):
            Mapping from output feature name to its `PolicyFeature` (type and shape). Inferred from the
            dataset when left empty.
        device (`str | None`, *optional*):
            Device to run the model on, e.g. `"cuda"`, `"cuda:0"`, `"cpu"`, or `"mps"`. Auto-detected when
            `None`.
        use_amp (`bool`, *optional*, defaults to `False`):
            Whether to use Automatic Mixed Precision for training and evaluation.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether the policy is trained with PEFT (parameter-efficient fine-tuning) adapters.
        push_to_hub (`bool`, *optional*, defaults to `True`):
            Whether to push the trained policy to the Hugging Face Hub.
        repo_id (`str | None`, *optional*):
            Repository ID to push the trained policy to on the Hub.
        private (`bool | None`, *optional*):
            Whether to create the Hub repository as private.
        tags (`list[str] | None`, *optional*):
            Tags to attach to the policy's Hub repository.
        license (`str | None`, *optional*):
            License identifier to attach to the policy's Hub repository.
        pretrained_path (`Path | None`, *optional*):
            Repo ID on the Hub or local directory to load pretrained weights from. The policy is
            initialized from scratch when `None`.
        pretrained_revision (`str | None`, *optional*):
            Hub revision (commit hash, branch, or tag) to pin the pretrained model version.
        paligemma_variant (`str`, *optional*, defaults to `"gemma_2b"`):
            Which PaliGemma backbone variant to use for the vision-language encoder. Must be
            `"gemma_2b"` or `"gemma_300m"`.
        action_expert_variant (`str`, *optional*, defaults to `"gemma_300m"`):
            Which Gemma variant to use for the action expert network. Must be `"gemma_2b"` or
            `"gemma_300m"`.
        dtype (`str`, *optional*, defaults to `"float32"`):
            Model computation dtype. Must be `"bfloat16"` or `"float32"`.
        chunk_size (`int`, *optional*, defaults to 50):
            Number of action steps predicted per model invocation (called "action_horizon" in openpi).
        n_action_steps (`int`, *optional*, defaults to 50):
            Number of predicted action steps actually executed in the environment before predicting a new
            chunk. Must not exceed `chunk_size`.
        max_state_dim (`int`, *optional*, defaults to 32):
            Dimension the observation state vector is zero-padded to when shorter.
        max_action_dim (`int`, *optional*, defaults to 32):
            Dimension the action vector is zero-padded to when shorter.
        num_inference_steps (`int`, *optional*, defaults to 10):
            Number of flow-matching denoising steps performed at inference time.
        time_sampling_beta_alpha (`float`, *optional*, defaults to 1.5):
            Alpha shape parameter of the Beta distribution the flow-matching timestep is sampled from
            during training.
        time_sampling_beta_beta (`float`, *optional*, defaults to 1.0):
            Beta shape parameter of the Beta distribution the flow-matching timestep is sampled from
            during training.
        time_sampling_scale (`float`, *optional*, defaults to 0.999):
            Scale applied to the sampled Beta timestep before `time_sampling_offset` is added.
        time_sampling_offset (`float`, *optional*, defaults to 0.001):
            Offset added to the scaled Beta timestep sample.
        min_period (`float`, *optional*, defaults to 0.004):
            Minimum period of the sinusoidal positional encoding used to embed the flow-matching timestep.
        max_period (`float`, *optional*, defaults to 4.0):
            Maximum period of the sinusoidal positional encoding used to embed the flow-matching timestep.
        use_relative_actions (`bool`, *optional*, defaults to `False`):
            Whether to convert absolute actions to relative (relative to the current state) before feeding
            them to the model.
        relative_exclude_joints (`list[str]`, *optional*):
            Joint names to keep absolute (excluded from the relative conversion) when
            `use_relative_actions` is enabled. An empty list means every dimension is made relative.
        action_feature_names (`list[str] | None`, *optional*):
            Names of the action dimensions, in order. Populated at runtime from dataset metadata by
            `make_policy`.
        rtc_config (`RTCConfig | None`, *optional*):
            Real-Time Chunking configuration. `None` disables RTC inference.
        image_resolution (`tuple[int, int]`, *optional*, defaults to `(224, 224)`):
            Target `(height, width)` images are resized (with padding) to before being fed to the vision
            encoder.
        empty_cameras (`int`, *optional*, defaults to 0):
            Number of empty (zero-padded) camera views to add, for models trained with more cameras than
            are available at inference/training time.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Mapping from feature type (`"VISUAL"`, `"STATE"`, `"ACTION"`) to the `NormalizationMode` used
            for it.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to enable gradient checkpointing to reduce memory usage during training.
        compile_model (`bool`, *optional*, defaults to `False`):
            Whether to compile the model with `torch.compile`.
        compile_mode (`str`, *optional*, defaults to `"max-autotune"`):
            The `torch.compile` mode to use when `compile_model` is enabled.
        freeze_vision_encoder (`bool`, *optional*, defaults to `False`):
            Whether to freeze the vision encoder's weights during training.
        train_expert_only (`bool`, *optional*, defaults to `False`):
            Whether to freeze the entire VLM and train only the action expert and its projections.
        optimizer_lr (`float`, *optional*, defaults to 2.5e-05):
            Peak learning rate for the AdamW optimizer preset.
        optimizer_betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.95)`):
            AdamW `(beta1, beta2)` coefficients.
        optimizer_eps (`float`, *optional*, defaults to 1e-08):
            AdamW epsilon term for numerical stability.
        optimizer_weight_decay (`float`, *optional*, defaults to 0.01):
            AdamW weight decay coefficient.
        optimizer_grad_clip_norm (`float`, *optional*, defaults to 1.0):
            Maximum gradient norm for clipping.
        scheduler_warmup_steps (`int`, *optional*, defaults to 1000):
            Number of warmup steps for the cosine-decay-with-warmup learning rate scheduler.
        scheduler_decay_steps (`int`, *optional*, defaults to 30000):
            Number of decay steps for the learning rate scheduler. Auto-scales down when the total number
            of training steps is smaller.
        scheduler_decay_lr (`float`, *optional*, defaults to 2.5e-06):
            Learning rate the scheduler decays to at the end of `scheduler_decay_steps`.
        tokenizer_max_length (`int`, *optional*, defaults to 48):
            Maximum token length for the language tokenizer.
    """

    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10  # Number of denoising steps during inference
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW``
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48  # see openpi `__post_init__`

    def __post_init__(self):
        """Resolve `device` (see [`~configs.PreTrainedConfig.__post_init__`]), then validate this config. Validates the PaliGemma backbone configuration."""
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        """See [`~configs.PreTrainedConfig.get_optimizer_preset`]."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        """See [`~configs.PreTrainedConfig.get_scheduler_preset`]."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        """See [`~configs.PreTrainedConfig.observation_delta_indices`]."""
        return None

    @property
    def action_delta_indices(self) -> list:
        """See [`~configs.PreTrainedConfig.action_delta_indices`]."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """See [`~configs.PreTrainedConfig.reward_delta_indices`]."""
        return None
