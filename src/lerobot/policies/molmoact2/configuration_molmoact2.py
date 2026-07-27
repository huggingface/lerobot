# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import (
    AdamWConfig,
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
    OptimizerConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("molmoact2")
@dataclass
class MolmoAct2Config(PreTrainedConfig):
    """MolmoAct2 policy backed by the converted HF checkpoint implementation."""

    checkpoint_path: str = "allenai/MolmoAct2"
    checkpoint_revision: str | None = None
    checkpoint_force_download: bool = False

    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30

    action_mode: str = "both"
    inference_action_mode: str | None = None
    discrete_action_tokenizer: str = "allenai/MolmoAct2-FAST-Tokenizer"
    discrete_generation_max_steps: int | None = None
    norm_tag: str | None = None

    setup_type: str = ""
    control_mode: str = ""
    image_keys: list[str] = field(default_factory=list)
    normalize_language: bool = True
    add_setup_tokens: bool = True
    add_control_tokens: bool = True
    normalize_gripper: bool = False
    num_state_tokens: int = 256
    # Leave unset for the default MolmoAct2 sequence budget inferred from the fixed
    # image/prompt/state/action token layout. Override only for unusual long prompts.
    max_sequence_length: int | None = None

    # Fixed by released MolmoAct2 checkpoints. We validate this at model load.
    expected_max_action_dim: int = 32

    # Flow-matching training knobs copied from the original MolmoAct2 training path.
    num_flow_timesteps: int = 8
    flow_matching_cutoff: float = 1.0
    flow_matching_time_offset: float = 0.001
    flow_matching_time_scale: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5
    num_inference_steps: int | None = None
    mask_action_dim_padding: bool = True
    enable_inference_cuda_graph: bool = True
    # MolmoAct2-local eval option. When enabled, stochastic continuous action
    # generation uses a rollout-local generator derived from eval_seed.
    per_episode_seed: bool = False
    eval_seed: int | None = None
    rtc_config: RTCConfig | None = None

    # Joint frame transform for cross-calibration compatibility.
    # Some MolmoAct2 checkpoints were trained on data using a different joint
    # convention than the current LeRobot calibration. Set both to apply a
    # sign/offset correction at runtime (state before model, action after).
    # See: https://huggingface.co/docs/lerobot/backwardcomp
    # Default is None (no transform). Both must be set together.
    joint_signs: list[float] | None = None
    joint_offsets: list[float] | None = None

    # Default is full finetuning with gradients from the action expert flowing into the VLM.
    enable_lora_vlm: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    enable_lora_action_expert: bool = False
    enable_knowledge_insulation: bool = False
    freeze_embedding: bool = True
    train_action_expert_only: bool = False
    gradient_checkpointing: bool = False

    model_dtype: str = "bfloat16"
    softmax_auxiliary_loss: bool = True
    softmax_auxiliary_loss_scale: float = 1e-4
    discrete_loss_token_weighting: str = "root_subsegments_root_tokens"

    optimizer_lr: float = 1e-5
    optimizer_vit_lr: float = 5e-6
    optimizer_connector_lr: float = 5e-6
    optimizer_action_expert_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-6
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 200
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 1e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    dataset_feature_names: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.joint_signs is None) != (self.joint_offsets is None):
            raise ValueError("joint_signs and joint_offsets must both be set or both be None.")
        if self.joint_signs is not None and len(self.joint_signs) != len(self.joint_offsets):
            raise ValueError("joint_signs and joint_offsets must have the same length.")
        if self.action_mode not in {"continuous", "discrete", "both"}:
            raise ValueError(
                f"Unsupported action_mode={self.action_mode!r}. "
                "Expected one of {'continuous', 'discrete', 'both'}."
            )
        if self.inference_action_mode not in {None, "continuous", "discrete"}:
            raise ValueError(
                f"Unsupported inference_action_mode={self.inference_action_mode!r}. "
                "Expected one of {None, 'continuous', 'discrete'}."
            )
        if self.inference_action_mode == "continuous" and self.action_mode == "discrete":
            raise ValueError("MolmoAct2 action_mode='discrete' cannot run continuous inference.")
        if self.inference_action_mode == "discrete" and self.action_mode == "continuous":
            raise ValueError("MolmoAct2 action_mode='continuous' cannot run discrete inference.")
        if self.train_action_expert_only and self.action_mode != "continuous":
            raise ValueError("MolmoAct2 train_action_expert_only requires action_mode='continuous'.")
        if self.train_action_expert_only and self.enable_lora_vlm:
            raise ValueError("MolmoAct2 train_action_expert_only is incompatible with enable_lora_vlm.")
        if self.enable_lora_action_expert and not self.enable_lora_vlm:
            raise ValueError("MolmoAct2 enable_lora_action_expert requires enable_lora_vlm.")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}.")
        if self.n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {self.n_action_steps}.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})."
            )
        if self.expected_max_action_dim != 32:
            raise ValueError("MolmoAct2 released checkpoints use expected_max_action_dim=32.")
        if self.model_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError(
                f"Unsupported model_dtype={self.model_dtype!r}. Expected 'float32', 'bfloat16', or 'float16'."
            )
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}.")
        if self.lora_alpha < 1:
            raise ValueError(f"lora_alpha must be >= 1, got {self.lora_alpha}.")
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError(f"lora_dropout must be in [0, 1], got {self.lora_dropout}.")
        if self.lora_bias not in {"none", "all", "lora_only"}:
            raise ValueError(
                f"Unsupported lora_bias={self.lora_bias!r}. Expected one of 'none', 'all', or 'lora_only'."
            )
        if self.discrete_loss_token_weighting not in {
            "none",
            "token",
            "root_tokens",
            "root_subsegments",
            "root_subsegments_root_tokens",
        }:
            raise ValueError(
                f"Unsupported discrete_loss_token_weighting={self.discrete_loss_token_weighting!r}."
            )
        if self.discrete_generation_max_steps is not None and self.discrete_generation_max_steps < 1:
            raise ValueError(
                f"discrete_generation_max_steps must be >= 1 or None, got {self.discrete_generation_max_steps}."
            )
        if self.max_sequence_length is not None and self.max_sequence_length < 1:
            raise ValueError(f"max_sequence_length must be >= 1 or None, got {self.max_sequence_length}.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def set_dataset_feature_metadata(self, features: dict[str, Any]) -> None:
        self.dataset_feature_names = {}
        for key in (ACTION, OBS_STATE):
            feature = features.get(key) if isinstance(features, dict) else None
            if isinstance(feature, dict) and feature.get("names") is not None:
                self.dataset_feature_names[key] = feature["names"]

    def validate_features(self) -> None:
        """Validate and set up MolmoAct2 input and output features."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "MolmoAct2 policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(0,),
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.expected_max_action_dim,),
            )
            self.output_features[ACTION] = action_feature
