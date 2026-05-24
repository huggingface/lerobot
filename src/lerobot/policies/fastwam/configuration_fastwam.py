# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from lerobot.configs import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
    PreTrainedConfig,
)
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_STATE

WAN22_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"


def _default_video_dit_config(action_dim: int) -> dict[str, Any]:
    return {
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1.0e-6,
        "seperated_timestep": True,
        "use_gradient_checkpointing": False,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": action_dim,
        "action_group_causal_mask_mode": "group_diagonal",
    }


def _default_action_dit_config(action_dim: int) -> dict[str, Any]:
    return {
        "action_dim": action_dim,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1.0e-6,
        "use_gradient_checkpointing": False,
    }


def _coerce_enum(enum_cls: type, value: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (TypeError, ValueError):
        return getattr(enum_cls, str(value), value)


def _coerce_policy_features(features: dict[str, Any] | None) -> dict[str, PolicyFeature] | None:
    if features is None:
        return None
    coerced = {}
    for name, feature in features.items():
        if isinstance(feature, PolicyFeature):
            coerced[name] = feature
            continue
        coerced[name] = PolicyFeature(
            type=_coerce_enum(FeatureType, feature["type"]),
            shape=tuple(feature["shape"]),
        )
    return coerced


def _coerce_normalization_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: _coerce_enum(NormalizationMode, value) for key, value in mapping.items()}


def _is_local_model_id(value: str) -> bool:
    path = Path(value).expanduser()
    return path.is_absolute() or value.startswith(("./", "../", "~")) or path.exists()


def _validate_wan_model_id(value: str, field_name: str) -> str:
    if value == WAN22_MODEL_ID or _is_local_model_id(value):
        return value
    raise ValueError(f"`{field_name}` must be `{WAN22_MODEL_ID}` or an explicit local path, got `{value}`.")


def _coerce_pretrained_tokenizer_model_id(payload: dict[str, Any]) -> None:
    tokenizer_model_id = payload.get("tokenizer_model_id")
    if tokenizer_model_id is None:
        return
    if tokenizer_model_id == WAN22_MODEL_ID or _is_local_model_id(tokenizer_model_id):
        return
    payload["tokenizer_model_id"] = WAN22_MODEL_ID


@PreTrainedConfig.register_subclass("fastwam")
@dataclass
class FastWAMConfig(PreTrainedConfig):
    """Configuration for the FastWAM LeRobot policy.

    Args:
        action_dim (int): Number of scalar action channels per timestep.
        proprio_dim (int | None): Number of proprioception channels used as an
            extra text-context token. `None` disables proprio conditioning.
        action_horizon (int): Number of actions predicted by one policy call.
        num_video_frames (int): Number of video frames used by FastWAM rollout.
        image_size (tuple[int, int]): Concatenated image size as `(height, width)`.
        context_len (int): Maximum text embedding token length.
        video_dit_config (dict[str, Any] | None): Wan video expert config.
        action_dit_config (dict[str, Any] | None): Action expert config.
        invert_dimensions (list[int]): Action dimensions to multiply by -1
            during postprocessing. Supports negative indices.
    """

    n_obs_steps: int = 1
    action_dim: int = 7
    proprio_dim: int | None = 8
    action_horizon: int = 32
    n_action_steps: int = 32
    num_video_frames: int = 33
    image_size: tuple[int, int] = (224, 448)
    context_len: int = 128
    model_id: str = WAN22_MODEL_ID
    tokenizer_model_id: str = WAN22_MODEL_ID
    tokenizer_max_len: int = 128
    load_text_encoder: bool = True
    mot_checkpoint_mixed_attn: bool = False
    torch_dtype: str = "bfloat16"
    prompt_template: str = (
        "A video recorded from a robot's point of view executing the following instruction: {task}"
    )
    num_inference_steps: int = 10
    inference_seed: int | None = 42
    rand_device: str = "cpu"
    text_cfg_scale: float = 1.0
    negative_prompt: str = ""
    sigma_shift: float | None = None
    tiled: bool = False
    toggle_action_dimensions: list[int] = field(default_factory=lambda: [-1])
    video_scheduler: dict[str, float | int] = field(
        default_factory=lambda: {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}
    )
    action_scheduler: dict[str, float | int] = field(
        default_factory=lambda: {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}
    )
    loss: dict[str, float] = field(default_factory=lambda: {"lambda_video": 1.0, "lambda_action": 1.0})
    video_dit_config: dict[str, Any] | None = None
    action_dit_config: dict[str, Any] | None = None
    invert_dimensions: list[int] = field(default_factory=list)
    normalization_mapping: dict[str, Any] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    input_features: dict[str, PolicyFeature] | None = None
    output_features: dict[str, PolicyFeature] | None = None
    optimizer_lr: float = 1.0e-4
    optimizer_weight_decay: float = 1.0e-2

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        self.image_size = tuple(self.image_size)
        self.model_id = _validate_wan_model_id(self.model_id, "model_id")
        self.tokenizer_model_id = _validate_wan_model_id(self.tokenizer_model_id, "tokenizer_model_id")
        self.input_features = _coerce_policy_features(self.input_features)
        self.output_features = _coerce_policy_features(self.output_features)
        self.invert_dimensions = [int(dim) for dim in self.invert_dimensions]
        self.toggle_action_dimensions = [int(dim) for dim in self.toggle_action_dimensions]
        self.normalization_mapping = _coerce_normalization_mapping(self.normalization_mapping)
        self.video_dit_config = self.video_dit_config or _default_video_dit_config(self.action_dim)
        self.action_dit_config = self.action_dit_config or _default_action_dit_config(self.action_dim)
        self.input_features = self.input_features or self._default_input_features()
        self.output_features = self.output_features or self._default_output_features()
        self.validate_features()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str | Path, **_: Any) -> FastWAMConfig:
        config_path = Path(pretrained_name_or_path) / "config.json"
        with open(config_path, encoding="utf-8") as f:
            payload = json.load(f)
        payload.pop("type", None)
        known_fields = {field.name for field in fields(cls)}
        payload = {key: value for key, value in payload.items() if key in known_fields}
        _coerce_pretrained_tokenizer_model_id(payload)
        return cls(**payload)

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if self.action_dim <= 0:
            raise ValueError(f"`action_dim` must be positive, got {self.action_dim}.")
        if self.action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {self.action_horizon}.")
        if self.n_action_steps > self.action_horizon:
            raise ValueError("`n_action_steps` cannot exceed `action_horizon`.")
        if self.num_video_frames % 4 != 1:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {self.num_video_frames}.")
        if not self.image_features:
            raise ValueError("FastWAM requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("FastWAM requires `action` in output_features.")
        action_shape = tuple(self.action_feature.shape)
        if action_shape != (self.action_dim,):
            raise ValueError(
                f"FastWAM action feature shape must be ({self.action_dim},), got {action_shape}."
            )
        if self.proprio_dim is not None:
            state_feature = self.robot_state_feature
            if state_feature is None:
                raise ValueError("FastWAM requires `observation.state` when `proprio_dim` is set.")
            state_shape = tuple(state_feature.shape)
            if state_shape != (self.proprio_dim,):
                raise ValueError(
                    f"FastWAM state feature shape must be ({self.proprio_dim},), got {state_shape}."
                )
        self._validate_image_feature_shapes()

    def _validate_image_feature_shapes(self) -> None:
        height, width = self.image_size
        image_width_sum = 0
        for name, feature in self.image_features.items():
            shape = tuple(feature.shape)
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"FastWAM image feature `{name}` must have shape (3, H, W), got {shape}.")
            if shape[1] != height:
                raise ValueError(f"FastWAM image feature `{name}` height must be {height}, got {shape[1]}.")
            image_width_sum += shape[2]
        if image_width_sum != width:
            raise ValueError(f"FastWAM image feature widths must sum to {width}, got {image_width_sum}.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def _default_input_features(self) -> dict[str, PolicyFeature]:
        height, width = self.image_size
        features = {
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
        }
        if self.proprio_dim is not None:
            features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.proprio_dim,))
        return features

    def _default_output_features(self) -> dict[str, PolicyFeature]:
        return {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))}
