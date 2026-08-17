#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import CosineDecayWithWarmupSchedulerConfig, XVLAAdamWConfig
from lerobot.utils.constants import OBS_IMAGES

# Conditional import for type checking and lazy loading
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import Florence2Config
else:
    Florence2Config = None


def _translate_vision_config(vision_config: dict[str, Any]) -> dict[str, Any]:
    """Translate a vision config from the vendored Florence-2 format to the native format.

    Translates from the original Microsoft remote-code Florence-2 format (used by existing XVLA
    checkpoints) to the native ``transformers`` format. Configs already in the native format pass
    through unchanged.
    """
    vision = dict(vision_config)
    model_type = vision.pop("model_type", None)
    if model_type not in (None, "davit", "florence_vision"):
        raise ValueError(f"Unsupported Florence-2 vision backbone: {model_type!r}")
    vision.pop("enable_checkpoint", None)

    image_pos_embed = vision.pop("image_pos_embed", None)
    if image_pos_embed is not None:
        if image_pos_embed.get("type") != "learned_abs_2d":
            raise ValueError(f"Unsupported image_pos_embed type: {image_pos_embed.get('type')!r}")
        vision["max_position_embeddings"] = image_pos_embed["max_pos_embeddings"]

    visual_temporal_embedding = vision.pop("visual_temporal_embedding", None)
    if visual_temporal_embedding is not None:
        if visual_temporal_embedding.get("type") != "COSINE":
            raise ValueError(
                f"Unsupported visual_temporal_embedding type: {visual_temporal_embedding.get('type')!r}"
            )
        vision["max_temporal_embeddings"] = visual_temporal_embedding["max_temporal_embeddings"]

    image_feature_source = vision.pop("image_feature_source", None)
    if image_feature_source is not None and list(image_feature_source) != [
        "spatial_avg_pool",
        "temporal_avg_pool",
    ]:
        # the native Florence2MultiModalProjector hardcodes this feature combination
        raise ValueError(f"Unsupported image_feature_source: {image_feature_source!r}")

    if "dim_embed" in vision:
        vision["embed_dim"] = vision.pop("dim_embed")
    return vision


@PreTrainedConfig.register_subclass("xvla")
@dataclass
class XVLAConfig(PreTrainedConfig):
    """Configuration class for the XVLA (Extended Vision-Language-Action) policy.

    Lets the policy plug into the LeRobot training stack. The config mirrors the knobs exposed in the
    original XVLA repository but also declares the input/output feature contract required by LeRobot.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 1):
            Number of environment steps of observation to pass to the policy. Unused by this policy,
            which always consumes the current-step observation only (`observation_delta_indices` is
            `None`).
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
            Path or Hub repo id of pretrained weights to initialize the policy from. If `None`, the
            policy is initialized from scratch.
        pretrained_revision (`str | None`, *optional*):
            Hub revision (branch, tag, or commit hash) pinning the pretrained model version.
        chunk_size (`int`, *optional*, defaults to 32):
            The size of the action prediction chunk, in units of environment steps.
        n_action_steps (`int`, *optional*, defaults to 32):
            The number of action steps to run in the environment for one invocation of the policy. Must
            be no greater than `chunk_size`.
        dtype (`str`, *optional*, defaults to `"float32"`):
            Torch dtype (`"bfloat16"` or `"float32"`) the model's parameters and inputs are cast to.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Maps a feature type name (e.g. `"STATE"`, `"VISUAL"`) to the `NormalizationMode` to apply to
            it. Defaults to identity normalization for all feature types: images are already normalized
            by the ImageNet processor step, and state/action normalization is handled internally by the
            action space.
        florence_config (`dict[str, Any]`, *optional*):
            Florence-2 vision-language backbone configuration, containing `vision_config` and
            `text_config`. Accepted in either the native `transformers` format or the original
            Microsoft remote-code format used by existing XVLA checkpoints; see `get_florence_config`.
        tokenizer_name (`str`, *optional*, defaults to `"facebook/bart-large"`):
            Name or path of the tokenizer used to tokenize the language instruction.
        tokenizer_max_length (`int`, *optional*, defaults to 64):
            Maximum token length for the tokenized instruction.
        tokenizer_padding_side (`str`, *optional*, defaults to `"right"`):
            Padding side used by the tokenizer.
        pad_language_to (`str`, *optional*, defaults to `"max_length"`):
            Padding strategy passed to the tokenizer processor step.
        hidden_size (`int`, *optional*, defaults to 1024):
            Hidden dimension of the soft-prompted policy transformer head.
        depth (`int`, *optional*, defaults to 24):
            Number of transformer layers in the policy transformer head.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in the policy transformer head.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Feed-forward expansion ratio in the policy transformer head.
        num_domains (`int`, *optional*, defaults to 30):
            Number of embodiment domains supported by the domain-conditioned soft prompts.
        len_soft_prompts (`int`, *optional*, defaults to 32):
            Number of learned soft-prompt tokens per domain.
        dim_time (`int`, *optional*, defaults to 32):
            Embedding dimension for the flow-matching timestep.
        max_len_seq (`int`, *optional*, defaults to 512):
            Maximum sequence length supported by the policy transformer head's positional embeddings.
        use_hetero_proj (`bool`, *optional*, defaults to `False`):
            Whether to use domain-specific (heterogeneous) input/output projections in the policy
            transformer head.
        action_mode (`str`, *optional*, defaults to `"ee6d"`):
            Name of the action-space representation used by `build_action_space` (e.g. end-effector pose
            with 6D rotation), which determines the model's action dimensionality and pre/post-processing.
        num_denoising_steps (`int`, *optional*, defaults to 10):
            Number of flow-matching integration steps used to generate an action chunk at inference time.
        use_proprio (`bool`, *optional*, defaults to `True`):
            Whether to feed a proprioceptive robot state input to the model. Requires a state feature in
            `input_features` when enabled.
        max_state_dim (`int`, *optional*, defaults to 32):
            Dimension the proprioceptive state vector is padded (or truncated) to.
        max_action_dim (`int`, *optional*, defaults to 20):
            Maximum action dimension used for padding when `action_mode` is `"auto"`.
        domain_feature_key (`str | None`, *optional*):
            Batch key providing a per-sample domain id. Falls back to a `"domain_id"` batch key, then to
            an all-zeros domain id, when unset or absent from the batch.
        resize_imgs_with_padding (`tuple[int, int] | None`, *optional*):
            Target `(height, width)` to resize and pad input images to. `None` keeps the original
            resolution.
        num_image_views (`int | None`, *optional*):
            Total number of camera views the model expects, including padding views. `None` (the
            default) derives it from the number of image features plus `empty_cameras`; when set
            explicitly, the larger of the two is used.
        empty_cameras (`int`, *optional*, defaults to 0):
            Number of synthetic all-zero camera views added as placeholder input features, e.g. to match
            a pretrained model's expected view count.
        freeze_vision_encoder (`bool`, *optional*, defaults to `False`):
            Whether to freeze the Florence-2 vision encoder's parameters during training.
        freeze_language_encoder (`bool`, *optional*, defaults to `False`):
            Whether to freeze the Florence-2 language encoder's parameters during training.
        train_policy_transformer (`bool`, *optional*, defaults to `True`):
            Whether the policy transformer head's parameters (other than the soft prompts) are
            trainable.
        train_soft_prompts (`bool`, *optional*, defaults to `True`):
            Whether the domain soft-prompt parameters are trainable.
        optimizer_lr (`float`, *optional*, defaults to 0.0001):
            Base learning rate for the XVLA AdamW optimizer preset.
        optimizer_betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.99)`):
            Adam beta coefficients for the XVLA AdamW optimizer preset.
        optimizer_eps (`float`, *optional*, defaults to 1e-08):
            Epsilon for the XVLA AdamW optimizer preset.
        optimizer_weight_decay (`float`, *optional*, defaults to 0.0):
            Weight decay for the XVLA AdamW optimizer preset.
        optimizer_grad_clip_norm (`float`, *optional*, defaults to 10.0):
            Gradient-clipping norm for the XVLA AdamW optimizer preset.
        optimizer_soft_prompt_lr_scale (`float`, *optional*, defaults to 1.0):
            Scale factor applied to `optimizer_lr` for the soft-prompt parameters.
        optimizer_soft_prompt_warmup_lr_scale (`float | None`, *optional*):
            Starting scale factor for an optional soft-prompt learning-rate warmup, e.g. `0.01`. `None`
            disables the warmup.
        scheduler_warmup_steps (`int`, *optional*, defaults to 1000):
            Number of warmup steps for the cosine-decay-with-warmup learning rate scheduler preset.
        scheduler_decay_steps (`int`, *optional*, defaults to 30000):
            Number of decay steps for the cosine-decay-with-warmup learning rate scheduler preset.
        scheduler_decay_lr (`float`, *optional*, defaults to 2.5e-06):
            Final learning rate at the end of decay, for the cosine-decay-with-warmup scheduler preset.
    """

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 32
    dtype: str = "float32"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Florence2 backbone and tokenizer configuration
    florence_config: dict[str, Any] = field(default_factory=dict)
    tokenizer_name: str = "facebook/bart-large"
    tokenizer_max_length: int = 64
    tokenizer_padding_side: str = "right"
    pad_language_to: str = "max_length"

    # Transformer head
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_domains: int = 30
    len_soft_prompts: int = 32
    dim_time: int = 32
    max_len_seq: int = 512
    use_hetero_proj: bool = False

    # Action & proprioception
    action_mode: str = "ee6d"
    num_denoising_steps: int = 10
    use_proprio: bool = True
    max_state_dim: int = 32
    max_action_dim: int = 20
    domain_feature_key: str | None = None

    # Vision preprocessing
    resize_imgs_with_padding: tuple[int, int] | None = None
    num_image_views: int | None = None
    empty_cameras: int = 0

    # Freezing options for VLM components.
    # By default, VLM encoders are frozen and only policy transformer + soft prompts train.
    freeze_vision_encoder: bool = False
    freeze_language_encoder: bool = False
    train_policy_transformer: bool = True
    train_soft_prompts: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.99)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 10.0
    optimizer_soft_prompt_lr_scale: float = 1.0
    optimizer_soft_prompt_warmup_lr_scale: float | None = None

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        """Resolve `device` (see [`~configs.PreTrainedConfig.__post_init__`]), then validate this config. Validates the vendored Florence-2 backbone configuration."""
        super().__post_init__()

        if self.chunk_size <= 0:
            raise ValueError("`chunk_size` must be strictly positive.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must be <= `chunk_size` ({self.chunk_size})."
            )
        if self.num_image_views is not None and self.num_image_views <= 0:
            raise ValueError("`num_image_views` must be > 0 when specified.")
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        self._florence_config_obj: Florence2Config | None = None

    def get_florence_config(self) -> Florence2Config:
        """Build (and cache) the native ``transformers`` Florence-2 config that backs the VLM.

        ``florence_config`` may be given either in the native ``transformers`` format or in the
        original Microsoft remote-code format stored by existing XVLA checkpoints (e.g. with
        ``dim_embed`` / ``image_pos_embed`` in the vision config); the latter is translated
        field-by-field to the native format.
        """
        if self._florence_config_obj is None:
            config_dict = dict(self.florence_config)
            if config_dict.get("vision_config") is None:
                raise ValueError("vision_config is required")
            if config_dict.get("text_config") is None:
                raise ValueError("text_config is required")

            vision_config = _translate_vision_config(config_dict["vision_config"])
            text_config = dict(config_dict["text_config"])
            if text_config.get("model_type", "florence2_language") == "florence2_language":
                # The MS remote-code language config is BART, field for field.
                text_config["model_type"] = "bart"

            kwargs = {
                key: config_dict[key]
                for key in (
                    "pad_token_id",
                    "bos_token_id",
                    "eos_token_id",
                    "image_token_id",
                    "is_encoder_decoder",
                    "tie_word_embeddings",
                )
                if key in config_dict
            }
            self._florence_config_obj = Florence2Config(
                vision_config=vision_config, text_config=text_config, **kwargs
            )
        return self._florence_config_obj

    def validate_features(self) -> None:
        """See [`~configs.PreTrainedConfig.validate_features`]."""
        if not self.image_features:
            raise ValueError("XVLA requires at least one visual feature in the inputs.")
        if self.use_proprio and self.robot_state_feature is None:
            raise ValueError("`use_proprio=True` requires a proprioceptive state feature.")
        if self.num_image_views is None:
            self.num_image_views = len(self.image_features) + self.empty_cameras
        else:
            self.num_image_views = max(self.num_image_views, len(self.image_features) + self.empty_cameras)

        if self.empty_cameras > 0:
            height, width = (480, 640)
            if self.resize_imgs_with_padding is not None:
                height, width = self.resize_imgs_with_padding
            for idx in range(self.empty_cameras):
                key = f"{OBS_IMAGES}.empty_camera_{idx}"
                if key not in self.input_features:
                    self.input_features[key] = PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, height, width),
                    )

    def get_optimizer_preset(self) -> XVLAAdamWConfig:
        """Return the XVLA-specific optimizer with differential learning rates.

        This optimizer applies:
        - 1/10 LR for VLM parameters (stable optimization)
        - Full LR for transformer/action head
        - Configurable LR for soft-prompts (with optional warm-up)
        """
        return XVLAAdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            soft_prompt_lr_scale=self.optimizer_soft_prompt_lr_scale,
            soft_prompt_warmup_lr_scale=self.optimizer_soft_prompt_warmup_lr_scale,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """See [`~configs.PreTrainedConfig.get_scheduler_preset`]."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """See [`~configs.PreTrainedConfig.observation_delta_indices`]."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """See [`~configs.PreTrainedConfig.action_delta_indices`]."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list[int] | None:
        """See [`~configs.PreTrainedConfig.reward_delta_indices`]."""
        return None
