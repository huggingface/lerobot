#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 1):
            Number of environment steps of observation to pass to the policy (the current step and
            additional steps going back). ACT only supports a value of 1; anything else raises in
            `__post_init__`.
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
        chunk_size (`int`, *optional*, defaults to 100):
            The size of the action prediction "chunks" in units of environment steps.
        n_action_steps (`int`, *optional*, defaults to 100):
            The number of action steps to run in the environment for one invocation of the policy. This
            should be no greater than `chunk_size`. For example, if the chunk size is 100, you may set this
            to 50: the model predicts 100 steps worth of actions, runs 50 in the environment, and throws
            the other 50 out.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Maps a feature type name (e.g. `"STATE"`, `"VISUAL"`) to the `NormalizationMode` to apply to
            it. Defaults to mean/std normalization for visual, state, and action features.
        vision_backbone (`str`, *optional*, defaults to `"resnet18"`):
            Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights (`str | None`, *optional*, defaults to `"ResNet18_Weights.IMAGENET1K_V1"`):
            Pretrained weights from torchvision to initialize the backbone. `None` means no pretrained
            weights.
        replace_final_stride_with_dilation (`int`, *optional*, defaults to `False`):
            Whether to replace the ResNet's final 2x2 stride with a dilated convolution.
        pre_norm (`bool`, *optional*, defaults to `False`):
            Whether to use "pre-norm" in the transformer blocks.
        dim_model (`int`, *optional*, defaults to 512):
            The transformer blocks' main hidden dimension.
        n_heads (`int`, *optional*, defaults to 8):
            The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward (`int`, *optional*, defaults to 3200):
            The dimension to expand the transformer's hidden dimension to in the feed-forward layers.
        feedforward_activation (`str`, *optional*, defaults to `"relu"`):
            The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers (`int`, *optional*, defaults to 4):
            The number of transformer layers to use for the transformer encoder.
        n_decoder_layers (`int`, *optional*, defaults to 1):
            The number of transformer layers to use for the transformer decoder.
        use_vae (`bool`, *optional*, defaults to `True`):
            Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim (`int`, *optional*, defaults to 32):
            The VAE's latent dimension.
        n_vae_encoder_layers (`int`, *optional*, defaults to 4):
            The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff (`float | None`, *optional*):
            Coefficient for the exponential weighting scheme to apply for temporal ensembling. `None` (the
            default) means temporal ensembling is not used. `n_action_steps` must be 1 when using this
            feature, as inference needs to happen at every step to form an ensemble. For more information
            on how ensembling works, see `ACTTemporalEnsembler`.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout to use in the transformer layers (see code for details).
        kl_weight (`float`, *optional*, defaults to 10.0):
            The weight to use for the KL-divergence component of the loss if the variational objective is
            enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
        optimizer_lr (`float`, *optional*, defaults to 1e-05):
            Learning rate for the AdamW optimizer preset.
        optimizer_weight_decay (`float`, *optional*, defaults to 0.0001):
            Weight decay for the AdamW optimizer preset.
        optimizer_lr_backbone (`float`, *optional*, defaults to 1e-05):
            Learning rate for the vision backbone's parameters in the AdamW optimizer preset.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        """Resolve `device` (see [`~configs.PreTrainedConfig.__post_init__`]), then validate this config. Validates `vision_backbone`, `temporal_ensemble_coeff`/`n_action_steps`, `n_action_steps`/`chunk_size`, and `n_obs_steps`."""
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        """See [`~configs.PreTrainedConfig.get_optimizer_preset`]."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        """See [`~configs.PreTrainedConfig.get_scheduler_preset`]."""
        return None

    def validate_features(self) -> None:
        """See [`~configs.PreTrainedConfig.validate_features`]."""
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

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
