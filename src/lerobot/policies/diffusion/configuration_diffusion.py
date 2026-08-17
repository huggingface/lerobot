#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("diffusion")
@dataclass
class DiffusionConfig(PreTrainedConfig):
    """Configuration class for DiffusionPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 2):
            Number of environment steps of observation to pass to the policy (the current step and
            additional steps going back).
        input_features (`dict[str, PolicyFeature] | None`, *optional*):
            Mapping from input feature name to its `PolicyFeature` (type and shape). Populated
            automatically from the dataset when not explicitly provided.
        output_features (`dict[str, PolicyFeature] | None`, *optional*):
            Mapping from output feature name to its `PolicyFeature` (type and shape). Populated
            automatically from the dataset when not explicitly provided.
        device (`str | None`, *optional*):
            Device the policy runs on, e.g. `"cuda"`, `"cuda:0"`, `"cpu"`, or `"mps"`. Falls back to the
            best available device if unset or unavailable.
        use_amp (`bool`, *optional*, defaults to `False`):
            Whether to use Automatic Mixed Precision for training and evaluation.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether this policy is trained with PEFT (parameter-efficient fine-tuning) adapters.
        push_to_hub (`bool`, *optional*, defaults to `True`):
            Whether to push the trained policy to the Hugging Face Hub after training.
        repo_id (`str | None`, *optional*):
            Hugging Face Hub repository id to push the policy to, when `push_to_hub` is enabled.
        private (`bool | None`, *optional*):
            Whether to create/push the Hub repository as private.
        tags (`list[str] | None`, *optional*):
            Tags to attach to the policy's Hub model card.
        license (`str | None`, *optional*):
            License identifier to add to the policy's Hub model card.
        pretrained_path (`Path | None`, *optional*):
            Path or Hub repo id of pretrained weights to initialize the policy from. If `None`, the policy
            is initialized from scratch.
        pretrained_revision (`str | None`, *optional*):
            Hub revision (branch, tag, or commit hash) pinning the pretrained model version.
        horizon (`int`, *optional*, defaults to 64):
            Diffusion model action prediction size as detailed in `DiffusionPolicy.select_action`.
        n_action_steps (`int`, *optional*, defaults to 32):
            The number of action steps to run in the environment for one invocation of the policy. See
            `DiffusionPolicy.select_action` for more details.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Maps a feature type name (e.g. `"STATE"`, `"VISUAL"`) to the `NormalizationMode` to apply to
            it. Defaults to mean/std normalization for visual features and min/max normalization for
            state and action features.
        drop_n_last_frames (`int`, *optional*, defaults to 7):
            Number of frames dropped from the end of each episode when sampling training windows, which
            avoids excessive padding. Should track `horizon - n_action_steps - n_obs_steps + 1`.
        vision_backbone (`str`, *optional*, defaults to `"resnet18"`):
            Name of the torchvision resnet backbone to use for encoding images.
        resize_shape (`tuple[int, int] | None`, *optional*):
            (H, W) shape to resize images to as a preprocessing step for the vision backbone. `None`
            disables resizing, so the original image resolution is used.
        crop_ratio (`float`, *optional*, defaults to 1.0):
            Ratio in (0, 1] used to derive the crop size from `resize_shape` (`crop_h =
            int(resize_shape[0] * crop_ratio)`, likewise for width). Set to 1.0 to disable cropping. Only
            takes effect when `resize_shape` is not `None`.
        crop_shape (`tuple[int, int] | None`, *optional*):
            (H, W) shape to crop images to. Computed automatically when `resize_shape` is set and
            `crop_ratio` < 1.0. Can also be set directly for legacy configs that use crop-only (without
            resize). `None`, with no derivation applying, means no cropping.
        crop_is_random (`bool`, *optional*, defaults to `True`):
            Whether the crop should be random at training time (it's always a center crop in eval mode).
        pretrained_backbone_weights (`str | None`, *optional*, defaults to `"ResNet18_Weights.IMAGENET1K_V1"`):
            Pretrained weights from torchvision to initialize the backbone. `None` means no pretrained
            weights.
        use_group_norm (`bool`, *optional*, defaults to `False`):
            Whether to replace batch normalization with group normalization in the backbone. The group
            sizes are set to be about 16 (`feature_dim // 16`).
        spatial_softmax_num_keypoints (`int`, *optional*, defaults to 32):
            Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoder_per_camera (`bool`, *optional*, defaults to `True`):
            Whether to use a separate RGB encoder for each camera view.
        down_dims (`tuple[int, ...]`, *optional*, defaults to `(512, 1024, 2048)`):
            Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet. You
            may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size (`int`, *optional*, defaults to 5):
            The convolutional kernel size of the diffusion modeling Unet.
        n_groups (`int`, *optional*, defaults to 8):
            Number of groups used in the group norm of the Unet's convolutional blocks.
        diffusion_step_embed_dim (`int`, *optional*, defaults to 128):
            The Unet is conditioned on the diffusion timestep via a small non-linear network. This is the
            output dimension of that network, i.e. the embedding dimension.
        use_film_scale_modulation (`bool`, *optional*, defaults to `True`):
            FiLM (https://huggingface.co/papers/1709.07871) is used for the Unet conditioning. Bias
            modulation is used by default, while this parameter indicates whether to also use scale
            modulation.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to checkpoint the Unet residual blocks during training. This reduces activation memory
            at the cost of recomputing those blocks during the backward pass.
        noise_scheduler_type (`str`, *optional*, defaults to `"DDPM"`):
            Name of the noise scheduler to use. Supported options: `"DDPM"`, `"DDIM"`.
        num_train_timesteps (`int`, *optional*, defaults to 100):
            Number of diffusion steps for the forward diffusion schedule.
        beta_schedule (`str`, *optional*, defaults to `"squaredcos_cap_v2"`):
            Name of the diffusion beta schedule as per `DDPMScheduler` from Hugging Face diffusers.
        beta_start (`float`, *optional*, defaults to 0.0001):
            Beta value for the first forward-diffusion step.
        beta_end (`float`, *optional*, defaults to 0.02):
            Beta value for the last forward-diffusion step.
        prediction_type (`str`, *optional*, defaults to `"epsilon"`):
            The type of prediction that the diffusion modeling Unet makes. Choose from `"epsilon"` or
            `"sample"`. These have equivalent outcomes from a latent variable modeling perspective, but
            `"epsilon"` has been shown to work better in many deep neural network settings.
        clip_sample (`bool`, *optional*, defaults to `True`):
            Whether to clip the sample to `[-clip_sample_range, +clip_sample_range]` for each denoising
            step at inference time. This requires the action space to be normalized to fit within that
            range.
        clip_sample_range (`float`, *optional*, defaults to 1.0):
            The magnitude of the clipping range described above.
        num_inference_steps (`int | None`, *optional*):
            Number of reverse diffusion steps to use at inference time (steps are evenly spaced). If not
            provided, defaults to the same value as `num_train_timesteps`.
        compile_model (`bool`, *optional*, defaults to `False`):
            Whether to compile the Unet with `torch.compile`.
        compile_mode (`str`, *optional*, defaults to `"reduce-overhead"`):
            `torch.compile` mode to use when `compile_model` is enabled.
        do_mask_loss_for_padding (`bool`, *optional*, defaults to `False`):
            Whether to mask the loss when there are copy-padded actions. See `LeRobotDataset` and
            `load_previous_and_future_frames` for more information. This defaults to `False` as the
            original Diffusion Policy implementation does the same.
        optimizer_lr (`float`, *optional*, defaults to 0.0001):
            Learning rate for the Adam optimizer preset.
        optimizer_betas (`tuple`, *optional*, defaults to `(0.95, 0.999)`):
            Adam optimizer's beta coefficients.
        optimizer_eps (`float`, *optional*, defaults to 1e-08):
            Adam optimizer's epsilon for numerical stability.
        optimizer_weight_decay (`float`, *optional*, defaults to 1e-06):
            Weight decay for the Adam optimizer preset.
        scheduler_name (`str`, *optional*, defaults to `"cosine"`):
            Name of the LR scheduler preset to use.
        scheduler_warmup_steps (`int`, *optional*, defaults to 500):
            Number of warmup steps for the LR scheduler preset.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    gradient_checkpointing: bool = False
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Optimization
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        """Resolve `device` (see [`~configs.PreTrainedConfig.__post_init__`]), then validate this config. Validates image/state feature presence and normalization-mode compatibility with the configured vision backbone."""
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                # Explicitly disable cropping for resize+ratio path when crop_ratio == 1.0.
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        """See [`~configs.PreTrainedConfig.get_optimizer_preset`]."""
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        """See [`~configs.PreTrainedConfig.get_scheduler_preset`]."""
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        """See [`~configs.PreTrainedConfig.validate_features`]."""
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        """See [`~configs.PreTrainedConfig.observation_delta_indices`]."""
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        """See [`~configs.PreTrainedConfig.action_delta_indices`]."""
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        """See [`~configs.PreTrainedConfig.reward_delta_indices`]."""
        return None
