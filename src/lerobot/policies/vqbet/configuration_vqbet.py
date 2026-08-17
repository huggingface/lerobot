#!/usr/bin/env python

# Copyright 2024 Seungjae Lee and Yibin Wang and Haritheja Etukuru
# and H. Jin Kim and Nur Muhammad Mahi Shafiullah and Lerrel Pinto
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
from lerobot.optim import AdamConfig, VQBeTSchedulerConfig


@PreTrainedConfig.register_subclass("vqbet")
@dataclass
class VQBeTConfig(PreTrainedConfig):
    """Configuration class for VQ-BeT.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - At least one key starting with "observation.image is required as an input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 5):
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
        n_action_pred_token (`int`, *optional*, defaults to 3):
            Total number of current token and future tokens that VQ-BeT predicts.
        action_chunk_size (`int`, *optional*, defaults to 5):
            Action chunk size of each action prediction token.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Maps a feature type name (e.g. `"STATE"`, `"VISUAL"`) to the `NormalizationMode` to apply to
            it. Defaults to identity normalization for visual features and min/max normalization for
            state and action features.
        vision_backbone (`str`, *optional*, defaults to `"resnet18"`):
            Name of the torchvision resnet backbone to use for encoding images.
        crop_shape (`tuple[int, int] | None`, *optional*, defaults to `(84, 84)`):
            (H, W) shape to crop images to as a preprocessing step for the vision backbone. Must fit
            within the image size. `None` means no cropping is done.
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
        n_vqvae_training_steps (`int`, *optional*, defaults to 20000):
            Number of optimization steps for training the Residual VQ.
        vqvae_n_embed (`int`, *optional*, defaults to 16):
            Number of embedding vectors in the RVQ dictionary (each layer).
        vqvae_embedding_dim (`int`, *optional*, defaults to 256):
            Dimension of each embedding vector in the RVQ dictionary.
        vqvae_enc_hidden_dim (`int`, *optional*, defaults to 128):
            Size of hidden dimensions of the encoder/decoder part of the Residual VQ-VAE.
        gpt_block_size (`int`, *optional*, defaults to 500):
            Max block size of minGPT (should be larger than the number of input tokens).
        gpt_input_dim (`int`, *optional*, defaults to 512):
            Size of input of GPT. This is also used as the dimension of observation features.
        gpt_output_dim (`int`, *optional*, defaults to 512):
            Size of output dimension of GPT. This is also used as an input dimension of the offset / bin
            prediction headers.
        gpt_n_layer (`int`, *optional*, defaults to 8):
            Number of layers of GPT.
        gpt_n_head (`int`, *optional*, defaults to 8):
            Number of heads of GPT.
        gpt_hidden_dim (`int`, *optional*, defaults to 512):
            Size of hidden dimensions of GPT.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate for GPT.
        offset_loss_weight (`float`, *optional*, defaults to 10000.0):
            A constant that is multiplied to the offset loss.
        primary_code_loss_weight (`float`, *optional*, defaults to 5.0):
            A constant that is multiplied to the primary code prediction loss.
        secondary_code_loss_weight (`float`, *optional*, defaults to 0.5):
            A constant that is multiplied to the secondary code prediction loss.
        bet_softmax_temperature (`float`, *optional*, defaults to 0.1):
            Sampling temperature of code for rollout with VQ-BeT.
        sequentially_select (`bool`, *optional*, defaults to `False`):
            Whether to select the primary / secondary code sequentially (pick the primary code, then
            select the secondary code), or at the same time.
        optimizer_lr (`float`, *optional*, defaults to 0.0001):
            Learning rate for the Adam optimizer preset (GPT and other non-VQ-VAE parameters).
        optimizer_betas (`tuple`, *optional*, defaults to `(0.95, 0.999)`):
            Adam optimizer's beta coefficients.
        optimizer_eps (`float`, *optional*, defaults to 1e-08):
            Adam optimizer's epsilon for numerical stability.
        optimizer_weight_decay (`float`, *optional*, defaults to 1e-06):
            Weight decay for the Adam optimizer preset.
        optimizer_vqvae_lr (`float`, *optional*, defaults to 0.001):
            Learning rate for the VQ-VAE's own Adam optimizer preset.
        optimizer_vqvae_weight_decay (`float`, *optional*, defaults to 0.0001):
            Weight decay for the VQ-VAE's own Adam optimizer preset.
        scheduler_warmup_steps (`int`, *optional*, defaults to 500):
            Number of warmup steps for the LR scheduler preset.
    """

    # Inputs / output structure.
    n_obs_steps: int = 5
    n_action_pred_token: int = 3
    action_chunk_size: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    # VQ-VAE
    n_vqvae_training_steps: int = 20000
    vqvae_n_embed: int = 16
    vqvae_embedding_dim: int = 256
    vqvae_enc_hidden_dim: int = 128
    # VQ-BeT
    gpt_block_size: int = 500
    gpt_input_dim: int = 512
    gpt_output_dim: int = 512
    gpt_n_layer: int = 8
    gpt_n_head: int = 8
    gpt_hidden_dim: int = 512
    dropout: float = 0.1
    offset_loss_weight: float = 10000.0
    primary_code_loss_weight: float = 5.0
    secondary_code_loss_weight: float = 0.5
    bet_softmax_temperature: float = 0.1
    sequentially_select: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    optimizer_vqvae_lr: float = 1e-3
    optimizer_vqvae_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        """Resolve `device` (see [`~configs.PreTrainedConfig.__post_init__`]), then validate this config. Validates the VQ-VAE and action-chunking configuration."""
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        """See [`~configs.PreTrainedConfig.get_optimizer_preset`]."""
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> VQBeTSchedulerConfig:
        """See [`~configs.PreTrainedConfig.get_scheduler_preset`]."""
        return VQBeTSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_vqvae_training_steps=self.n_vqvae_training_steps,
        )

    def validate_features(self) -> None:
        """See [`~configs.PreTrainedConfig.validate_features`]."""
        # Note: this check was previously performed inside VQBeTRgbEncoder in the form of
        # assert len(image_keys) == 1
        if not len(self.image_features) == 1:
            raise ValueError("You must provide only one image among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape.
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
        return list(range(1 - self.n_obs_steps, self.n_action_pred_token + self.action_chunk_size - 1))

    @property
    def reward_delta_indices(self) -> None:
        """See [`~configs.PreTrainedConfig.reward_delta_indices`]."""
        return None
