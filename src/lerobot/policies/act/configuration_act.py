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
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
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
            ResNet feature map.
        pretrained_backbone_weights: Pretrained weights to initialize the backbone with. `None` means no
            pretrained weights (random init). For ResNet backbones: a torchvision weights enum name
            (e.g. "ResNet18_Weights.IMAGENET1K_V1"). For ViT / ConvNeXt backbones: a Hugging Face checkpoint id or
            local path matching the architecture (e.g. "facebook/dinov2-small" for "vit_small_patch14" or
            "facebook/convnext-tiny-224" for "convnext_tiny"; see `DEFAULT_PRETRAINED_BACKBONE_WEIGHTS`).
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution. Only applies to ResNet backbones.
        freeze_backbone: Whether to freeze the vision backbone's parameters (no gradient updates).
        backbone_resize_shape: Optional (height, width) to resize images to before feeding them to the vision
            backbone. Useful for ViT backbones, whose token count scales with (H/14) * (W/14). Both dimensions
            should be multiples of the backbone stride (14 for ViT, 32 for ConvNeXt and ResNet). `None` means no
            resizing.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
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
    freeze_backbone: bool = False
    backbone_resize_shape: tuple[int, int] | None = None
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
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not (self.is_resnet_backbone or self.is_vit_backbone or self.is_convnext_backbone):
            raise ValueError(
                "`vision_backbone` must be a torchvision ResNet variant (e.g. 'resnet18') or one of "
                f"{list(VIT_ARCHITECTURES) + list(CONVNEXT_ARCHITECTURES)}. Got {self.vision_backbone}."
            )
        if not self.is_resnet_backbone:
            if self.replace_final_stride_with_dilation:
                raise ValueError("`replace_final_stride_with_dilation` is only supported for ResNet backbones.")
            if self.pretrained_backbone_weights is not None and "_Weights." in self.pretrained_backbone_weights:
                raise ValueError(
                    f"`pretrained_backbone_weights={self.pretrained_backbone_weights}` looks like torchvision "
                    f"ResNet weights, which are incompatible with `vision_backbone={self.vision_backbone}`. Use a "
                    "matching Hugging Face checkpoint (e.g. "
                    f"'{DEFAULT_PRETRAINED_BACKBONE_WEIGHTS[self.vision_backbone]}') or `None` for random init."
                )
        if self.backbone_resize_shape is not None and len(self.backbone_resize_shape) != 2:
            raise ValueError(
                f"`backbone_resize_shape` must be (height, width). Got {self.backbone_resize_shape}."
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
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

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
