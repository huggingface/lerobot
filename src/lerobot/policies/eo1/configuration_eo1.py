#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )
else:
    Qwen2_5_VLConfig = None
    Qwen2_5_VLTextConfig = None
    Qwen2_5_VLVisionConfig = None


@PreTrainedConfig.register_subclass("eo1")
@dataclass
class EO1Config(PreTrainedConfig):
    """Configuration for native EO1 policy integration in LeRobot.

    EO1 wraps a Qwen2.5-VL vision-language backbone with a flow-matching action head: the backbone attends
    over interleaved vision/language/state/action tokens, and the head denoises an action chunk from noise
    via Euler integration.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 1):
            Number of environment steps of observation to pass to the policy.
        input_features (`dict[str, PolicyFeature]`, *optional*):
            Input feature specification, keyed by feature name. Left empty to infer from the dataset.
        output_features (`dict[str, PolicyFeature]`, *optional*):
            Output feature specification, keyed by feature name. Left empty to infer from the dataset.
        device (`str`, *optional*):
            Torch device to run the policy on, e.g. `"cuda"` or `"cpu"`. Auto-selected when unset or
            unavailable.
        use_amp (`bool`, *optional*, defaults to `False`):
            Whether to use Automatic Mixed Precision for training and evaluation.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether this policy is trained with PEFT adapters.
        push_to_hub (`bool`, *optional*, defaults to `True`):
            Whether to push the trained policy to the Hugging Face Hub.
        repo_id (`str`, *optional*):
            Hub repository id to push the policy to.
        private (`bool`, *optional*):
            Whether the pushed Hub repository is private.
        tags (`list[str]`, *optional*):
            Tags to attach to the policy on the Hub.
        license (`str`, *optional*):
            License identifier for the policy on the Hub.
        pretrained_path (`Path`, *optional*):
            Repo id or local directory of pretrained weights saved with `save_pretrained`. Left unset to
            initialize from scratch.
        pretrained_revision (`str`, *optional*):
            Hub revision to pin when loading `pretrained_path`.
        vlm_base (`str`, *optional*, defaults to `"Qwen/Qwen2.5-VL-3B-Instruct"`):
            Hugging Face Hub id of the Qwen2.5-VL backbone used to initialize the vision-language model.
        vlm_config (`dict`, *optional*):
            Serialized Qwen2.5-VL backbone config. Populated automatically from `vlm_base` in
            `__post_init__` when left unset.
        image_min_pixels (`int`, *optional*, defaults to 50176):
            Minimum number of pixels the vision processor resizes an image down to.
        image_max_pixels (`int`, *optional*, defaults to 100352):
            Maximum number of pixels the vision processor resizes an image up to.
        use_fast_processor (`bool`, *optional*, defaults to `False`):
            Whether to use the Hugging Face "fast" image processor.
        chunk_size (`int`, *optional*, defaults to 8):
            Number of actions predicted per flow-matching sampling call.
        n_action_steps (`int`, *optional*, defaults to 8):
            Number of actions from a predicted chunk that are actually executed before re-querying the
            policy. Must not exceed `chunk_size`.
        max_state_dim (`int`, *optional*, defaults to 32):
            Padded dimensionality of the state vector fed to the flow-matching head.
        max_action_dim (`int`, *optional*, defaults to 32):
            Padded dimensionality of the action vector fed to the flow-matching head.
        num_denoise_steps (`int`, *optional*, defaults to 10):
            Number of Euler integration steps used to sample an action chunk.
        num_action_layers (`int`, *optional*, defaults to 2):
            Number of linear layers in the action output projector MLP.
        action_act (`str`, *optional*, defaults to `"linear"`):
            Activation used between the action output projector's layers.
        time_sampling_beta_alpha (`float`, *optional*, defaults to 1.5):
            Alpha parameter of the Beta distribution used to sample the flow-matching timestep during
            training.
        time_sampling_beta_beta (`float`, *optional*, defaults to 1.0):
            Beta parameter of the same Beta distribution.
        time_sampling_scale (`float`, *optional*, defaults to 0.999):
            Scale applied to the sampled Beta timestep.
        time_sampling_offset (`float`, *optional*, defaults to 0.001):
            Offset added to the scaled Beta timestep.
        min_period (`float`, *optional*, defaults to 0.004):
            Minimum period of the sinusoidal timestep embedding.
        max_period (`float`, *optional*, defaults to 4.0):
            Maximum period of the sinusoidal timestep embedding.
        supervise_padding_action_dims (`bool`, *optional*, defaults to `True`):
            Whether the flow-matching loss also supervises the padded action dimensions that lie beyond
            the dataset's real action size.
        supervise_padding_actions (`bool`, *optional*, defaults to `True`):
            Whether the flow-matching loss also supervises padded action timesteps. Padded timesteps are
            marked by `action_is_pad`.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Dtype requested for the Qwen backbone. `"auto"` follows the backbone checkpoint's default
            dtype (bf16 for Qwen2.5-VL); the flow-matching head always keeps its own parameters in fp32
            regardless. Other supported values are `"bfloat16"` and `"float32"`.
        force_fp32_autocast (`bool`, *optional*, defaults to `True`):
            Whether to disable autocast around the flow-matching head so its projections run in fp32 even
            when the backbone runs under bf16 autocast.
        attn_implementation (`str`, *optional*):
            Attention backend requested for the Qwen backbone, e.g. `"sdpa"` or `"flash_attention_2"`.
            Left unset to use the backbone's default.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to enable gradient checkpointing on the Qwen backbone to reduce memory usage.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Maps each `FeatureType` to the `NormalizationMode` used to normalize/unnormalize it.
        optimizer_lr (`float`, *optional*, defaults to 0.0001):
            Peak learning rate used to build the default `AdamWConfig` optimizer preset.
        optimizer_betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam beta coefficients for the default optimizer preset.
        optimizer_eps (`float`, *optional*, defaults to 1e-08):
            Adam epsilon for the default optimizer preset.
        optimizer_weight_decay (`float`, *optional*, defaults to 0.1):
            Weight decay for the default optimizer preset.
        optimizer_grad_clip_norm (`float`, *optional*, defaults to 1.0):
            Gradient-norm clipping threshold for the default optimizer preset.
        scheduler_warmup_steps (`int`, *optional*, defaults to 900):
            Number of warmup steps for the default cosine-decay-with-warmup scheduler preset.
        scheduler_decay_steps (`int`, *optional*, defaults to 30000):
            Number of decay steps for the default scheduler preset.
        scheduler_decay_lr (`float`, *optional*, defaults to 0.0):
            Learning rate reached at the end of the default scheduler's decay.
    """

    vlm_base: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_config: dict | None = None

    # Vision processor settings.
    image_min_pixels: int | None = 64 * 28 * 28
    image_max_pixels: int | None = 128 * 28 * 28
    use_fast_processor: bool = False

    # Execution and action horizon.
    n_obs_steps: int = 1
    chunk_size: int = 8
    n_action_steps: int = 8

    # State/action padding to match EO1 flow head dimensionality.
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching sampling.
    num_denoise_steps: int = 10
    num_action_layers: int = 2
    action_act: str = "linear"
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    supervise_padding_action_dims: bool = True
    supervise_padding_actions: bool = True

    # Policy-level dtype request for the Qwen backbone.
    # - "auto": follow the backbone config/checkpoint default dtype. For Qwen2.5-VL this resolves to bf16.
    #           The EO1 flow-matching head still keeps its own parameters in fp32.
    # - "bfloat16": force the backbone to initialize/load in bf16 regardless of the saved config default.
    # - "float32": force the backbone to initialize/load in fp32 for maximum numerical conservatism.
    dtype: str = "auto"  # Options: "auto", "bfloat16", "float32"
    force_fp32_autocast: bool = True

    # Optional attention backend request passed through to the Qwen backbone.
    # Common values: None, "eager", "sdpa", "flash_attention_2".
    attn_implementation: str | None = None

    # Training settings.
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Optimizer settings aligned with EO1/experiments/2_libero/train.sh and EO1 TrainPipelineConfig defaults.
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.1
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings aligned with EO1 train.sh: cosine schedule with warmup_ratio=0.03.
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 900  # 0.03 * 30_000 long-run steps
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self):
        """Resolve `device` (see [`~configs.PreTrainedConfig.__post_init__`]), then validate this config. Validates the VLM backbone/tokenizer configuration."""
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        # Populate the serialized backbone config only when the caller did not provide one.
        if self.vlm_config is None:
            require_package("transformers", extra="eo1")
            self.vlm_config = Qwen2_5_VLConfig.from_pretrained(self.vlm_base).to_dict()

    @property
    def vlm_backbone_config(self) -> Qwen2_5_VLConfig:
        """Build the Qwen2.5-VL backbone config from `vlm_config`, applying `attn_implementation` if set."""
        require_package("transformers", extra="eo1")
        config_dict = deepcopy(self.vlm_config)
        if self.attn_implementation is not None:
            config_dict["attn_implementation"] = self.attn_implementation
        return Qwen2_5_VLConfig(**config_dict)

    @property
    def text_config(self) -> Qwen2_5_VLTextConfig:
        """The text-tower sub-config of `vlm_backbone_config`."""
        return self.vlm_backbone_config.text_config

    @property
    def vision_config(self) -> Qwen2_5_VLVisionConfig:
        """The vision-tower sub-config of `vlm_backbone_config`."""
        return self.vlm_backbone_config.vision_config

    def validate_features(self) -> None:
        """Validate and set up EO1 input and output features."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "EO1 policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
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
    def action_delta_indices(self) -> list[int]:
        """See [`~configs.PreTrainedConfig.action_delta_indices`]."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """See [`~configs.PreTrainedConfig.reward_delta_indices`]."""
        return None
