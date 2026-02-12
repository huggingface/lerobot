#!/usr/bin/env python

# Copyright 2026 S-Lab and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("dynamicvla")
@dataclass
class DynamicVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32
    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (384, 384)
    # Add empty images. Used by dynamicvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0
    # Converts the joint and gripper values from the standard Aloha space to the space
    # used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False
    # Converts joint dimensions to deltas with respect to the current state before
    # passing to the model. Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False
    # Use delta action prediction (relative to the current robot state)
    use_delta_action: bool = True
    # Streaming inference
    enable_streaming: bool = False
    # Multi-timestep fusion ("conv", "attn", "flat")
    temporal_fusion: str = "attn"
    # Tokenizer
    tokenizer_max_length: int = 48
    # Decoding
    num_steps: int = 10
    # Attention utils
    use_cache: bool = True
    # Finetuning settings
    freeze_vision_model: bool = True
    freeze_connector: bool = True
    freeze_text_model: bool = True
    train_state_proj: bool = True
    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6
    # Select the VLM backbone.
    attention_mode: str = "cross_attn"
    prefix_length: int = -1
    pad_language_to: str = "longest"  # "max_length"
    # Less or equal to 0 is the default where the action expert has the same number of
    # layers of VLM. Otherwise the expert have less layers.
    num_expert_layers: int = -1
    num_expert_skip_layers: int = 0
    # VLM settings
    vlm_model_name: str = "HuggingFaceTB/SmolLM2-360M"
    num_vlm_layers: int = 16
    # SmolVLM Settings
    smolvlm_patch_size: int = 16
    smolvlm_attention_heads: int = 12
    smolvlm_hidden_size: int = 768
    smolvlm_intermediate_size: int = 3072
    # FastVLM Settings
    fastvlm_inference_mode: bool = True
    # Interleave SA layers each self_attn_every_n_layers
    self_attn_every_n_layers: int = 2
    # The action expert hidden size (wrt to the VLM)
    expert_width_multiplier: float = 0.75
    # sensitivity range for the timestep used in sine-cosine positional encoding
    min_period: float = 4e-3
    max_period: float = 4.0
    # Delta timestamps (prevent errors loading pretrained weights)
    delta_timestamps: dict[str, list[int]] | None = None

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                "The chunk size is the upper bound for the number of action steps per"
                f" model invocation. Got {self.n_action_steps} for `n_action_steps` and"
                f" {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by dynamicvla for aloha real"
                " models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

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
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
