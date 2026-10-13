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

# Architectures for the ViT / ConvNeXt vision backbones. Values are kwargs for `transformers.Dinov2Config` (or
# `transformers.Dinov2WithRegistersConfig` if `num_register_tokens` is set) / `transformers.ConvNextConfig` and are
# used to build randomly initialized backbones. When loading pretrained weights, the checkpoint's own config is used
# instead, and its core architecture fields are checked against these entries.
VIT_ARCHITECTURES: dict[str, dict] = {
    # DINOv2 ViTs (https://huggingface.co/facebook/dinov2-small etc.).
    "vit_small_patch14": {
        "patch_size": 14, "image_size": 518, "hidden_size": 384, "num_hidden_layers": 12,
        "num_attention_heads": 6, "use_swiglu_ffn": False,
    },
    "vit_base_patch14": {
        "patch_size": 14, "image_size": 518, "hidden_size": 768, "num_hidden_layers": 12,
        "num_attention_heads": 12, "use_swiglu_ffn": False,
    },
    "vit_large_patch14": {
        "patch_size": 14, "image_size": 518, "hidden_size": 1024, "num_hidden_layers": 24,
        "num_attention_heads": 16, "use_swiglu_ffn": False,
    },
    "vit_giant_patch14": {
        "patch_size": 14, "image_size": 518, "hidden_size": 1536, "num_hidden_layers": 40,
        "num_attention_heads": 24, "use_swiglu_ffn": True,
    },
    # DINOv2 ViTs with 4 register tokens (https://huggingface.co/facebook/dinov2-with-registers-small etc.).
    "vit_small_patch14_reg4": {
        "patch_size": 14, "image_size": 518, "hidden_size": 384, "num_hidden_layers": 12,
        "num_attention_heads": 6, "use_swiglu_ffn": False, "num_register_tokens": 4,
    },
    "vit_base_patch14_reg4": {
        "patch_size": 14, "image_size": 518, "hidden_size": 768, "num_hidden_layers": 12,
        "num_attention_heads": 12, "use_swiglu_ffn": False, "num_register_tokens": 4,
    },
    "vit_large_patch14_reg4": {
        "patch_size": 14, "image_size": 518, "hidden_size": 1024, "num_hidden_layers": 24,
        "num_attention_heads": 16, "use_swiglu_ffn": False, "num_register_tokens": 4,
    },
    "vit_giant_patch14_reg4": {
        "patch_size": 14, "image_size": 518, "hidden_size": 1536, "num_hidden_layers": 40,
        "num_attention_heads": 24, "use_swiglu_ffn": True, "num_register_tokens": 4,
    },
}  # fmt: skip
CONVNEXT_ARCHITECTURES: dict[str, dict] = {
    "convnext_tiny": {"hidden_sizes": [96, 192, 384, 768], "depths": [3, 3, 9, 3]},
    "convnext_small": {"hidden_sizes": [96, 192, 384, 768], "depths": [3, 3, 27, 3]},
    "convnext_base": {"hidden_sizes": [128, 256, 512, 1024], "depths": [3, 3, 27, 3]},
    "convnext_large": {"hidden_sizes": [192, 384, 768, 1536], "depths": [3, 3, 27, 3]},
}
# Pretrained checkpoints on the Hugging Face Hub for each architecture (all Apache 2.0 licensed), for reference /
# error messages. ViTs: self-supervised DINOv2. ConvNeXts: ImageNet-1k supervised.
DEFAULT_PRETRAINED_BACKBONE_WEIGHTS: dict[str, str] = {
    "vit_small_patch14": "facebook/dinov2-small",
    "vit_base_patch14": "facebook/dinov2-base",
    "vit_large_patch14": "facebook/dinov2-large",
    "vit_giant_patch14": "facebook/dinov2-giant",
    "vit_small_patch14_reg4": "facebook/dinov2-with-registers-small",
    "vit_base_patch14_reg4": "facebook/dinov2-with-registers-base",
    "vit_large_patch14_reg4": "facebook/dinov2-with-registers-large",
    "vit_giant_patch14_reg4": "facebook/dinov2-with-registers-giant",
    "convnext_tiny": "facebook/convnext-tiny-224",
    "convnext_small": "facebook/convnext-small-224",
    "convnext_base": "facebook/convnext-base-224",
    "convnext_large": "facebook/convnext-large-224",
}


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
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Diffusion model action prediction size as detailed in `DiffusionPolicy.select_action`.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            See `DiffusionPolicy.select_action` for more details.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the backbone architecture to use for encoding images. One of:
            - a torchvision ResNet (e.g. "resnet18"),
            - a DINOv2 ViT from `VIT_ARCHITECTURES` (e.g. "vit_small_patch14", "vit_base_patch14_reg4"),
            - a ConvNeXt from `CONVNEXT_ARCHITECTURES` (e.g. "convnext_tiny").
            For ViT backbones, the final-layer patch tokens are reshaped into a (C, H/14, W/14) feature map. For
            ConvNeXt backbones, the final-stage (C, H/32, W/32) feature map is used. Either is used in place of the
            ResNet feature map before SpatialSoftmax pooling.
        resize_shape: (H, W) shape to resize images to as a preprocessing step for the vision
            backbone. If None, no resizing is done and the original image resolution is used.
        crop_ratio: Ratio in (0, 1] used to derive the crop size from resize_shape
            (crop_h = int(resize_shape[0] * crop_ratio), likewise for width).
            Set to 1.0 to disable cropping. Only takes effect when resize_shape is not None.
        crop_shape: (H, W) shape to crop images to. When resize_shape is set and crop_ratio < 1.0,
            this is computed automatically. Can also be set directly for legacy configs that use
            crop-only (without resize). If None and no derivation applies, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center
            crop in eval mode).
        pretrained_backbone_weights: Pretrained weights to initialize the backbone with. `None` means no
            pretrained weights (random init). For ResNet backbones: a torchvision weights enum name
            (e.g. "ResNet18_Weights.IMAGENET1K_V1"). For ViT / ConvNeXt backbones: a Hugging Face checkpoint id or
            local path matching the architecture (e.g. "facebook/dinov2-small" for "vit_small_patch14" or
            "facebook/convnext-tiny-224" for "convnext_tiny"; see `DEFAULT_PRETRAINED_BACKBONE_WEIGHTS`).
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16). Only applies to ResNet
            backbones.
        freeze_backbone: Whether to freeze the vision backbone's parameters (no gradient updates).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoder_per_camera: Whether to use a separate RGB encoder for each camera view.
        down_dims: Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
            You may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size: The convolutional kernel size of the diffusion modeling Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
        diffusion_step_embed_dim: The Unet is conditioned on the diffusion timestep via a small non-linear
            network. This is the output dimension of that network, i.e., the embedding dimension.
        use_film_scale_modulation: FiLM (https://huggingface.co/papers/1709.07871) is used for the Unet conditioning.
            Bias modulation is used be default, while this parameter indicates whether to also use scale
            modulation.
        gradient_checkpointing: Whether to checkpoint the Unet residual blocks during training. This reduces
            activation memory at the cost of recomputing those blocks during the backward pass.
        noise_scheduler_type: Name of the noise scheduler to use. Supported options: ["DDPM", "DDIM"].
        num_train_timesteps: Number of diffusion steps for the forward diffusion schedule.
        beta_schedule: Name of the diffusion beta schedule as per DDPMScheduler from Hugging Face diffusers.
        beta_start: Beta value for the first forward-diffusion step.
        beta_end: Beta value for the last forward-diffusion step.
        prediction_type: The type of prediction that the diffusion modeling Unet makes. Choose from "epsilon"
            or "sample". These have equivalent outcomes from a latent variable modeling perspective, but
            "epsilon" has been shown to work better in many deep neural network settings.
        clip_sample: Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
            denoising step at inference time. WARNING: you will need to make sure your action-space is
            normalized to fit within this range.
        clip_sample_range: The magnitude of the clipping range as described above.
        num_inference_steps: Number of reverse diffusion steps to use at inference time (steps are evenly
            spaced). If not provided, this defaults to be the same as `num_train_timesteps`.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
            `LeRobotDataset` and `load_previous_and_future_frames` for more information. Note, this defaults
            to False as the original Diffusion Policy implementation does the same.
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
    freeze_backbone: bool = False
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
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not (self.is_resnet_backbone or self.is_vit_backbone or self.is_convnext_backbone):
            raise ValueError(
                "`vision_backbone` must be a torchvision ResNet variant (e.g. 'resnet18') or one of "
                f"{list(VIT_ARCHITECTURES) + list(CONVNEXT_ARCHITECTURES)}. Got {self.vision_backbone}."
            )
        if not self.is_resnet_backbone:
            if self.use_group_norm:
                raise ValueError("`use_group_norm` is only supported for ResNet backbones.")
            if self.pretrained_backbone_weights is not None and "_Weights." in self.pretrained_backbone_weights:
                raise ValueError(
                    f"`pretrained_backbone_weights={self.pretrained_backbone_weights}` looks like torchvision "
                    f"ResNet weights, which are incompatible with `vision_backbone={self.vision_backbone}`. Use a "
                    "matching Hugging Face checkpoint (e.g. "
                    f"'{DEFAULT_PRETRAINED_BACKBONE_WEIGHTS[self.vision_backbone]}') or `None` for random init."
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
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    @property
    def is_resnet_backbone(self) -> bool:
        return self.vision_backbone.startswith("resnet")

    @property
    def is_vit_backbone(self) -> bool:
        return self.vision_backbone in VIT_ARCHITECTURES

    @property
    def is_convnext_backbone(self) -> bool:
        return self.vision_backbone in CONVNEXT_ARCHITECTURES

    def validate_features(self) -> None:
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
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
