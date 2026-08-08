#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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
from lerobot.configs.recipe import MessageTurn, TrainingRecipe
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE

_UMI_CAMERA_KEYS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)


def _hy_vla_default_recipe() -> TrainingRecipe:
    """Hy-VLA's subtask wording: the bare goal.

    Upstream Hy-Embodied-0.5-VLA trains on raw instruction strings with no added
    prompt sentence — the Hunyuan chat template's role tokens carry the framing —
    so the recipe's user turn is the task alone.
    """
    return TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(
                role="assistant",
                content="${subtask}",
                stream="high_level",
                target=True,
                if_present="subtask",
            ),
        ]
    )


@PreTrainedConfig.register_subclass("hy_vla")
@dataclass
class HyVLAConfig(PreTrainedConfig):
    """LeRobot configuration for Tencent Hy-Embodied-0.5-VLA.

    ``chunk_size`` is the number of action-expert tokens. It is 50 for the
    UMI checkpoint and 40 for the released RoboTwin checkpoint (20 relative
    tokens followed by 20 absolute tokens).
    """

    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    model_action_dim: int = 20

    resize_imgs_with_padding: tuple[int, int] = (224, 224)
    empty_cameras: int = 0
    tokenizer_max_length: int = 64
    task_suffix: str = "<｜hy_Assistant｜>"
    # Hy-VLA's language contract; see `_hy_vla_default_recipe`.
    recipe: TrainingRecipe | dict | None = field(default_factory=lambda: _hy_vla_default_recipe())

    # Joint text + flow supervision. The VLM tower keeps its tied vocabulary head,
    # so assistant tokens can be supervised alongside the flow objective. Text is
    # weighted well below flow: it exists to keep the language head from drifting
    # during an action fine-tune, not to compete with control.
    flow_loss_weight: float = 1.0
    text_loss_weight: float = 0.01
    text_max_new_tokens: int = 64

    proj_width: int = 1024
    num_steps: int = 10
    use_cache: bool = True
    attention_implementation: str = "eager"

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_state_proj: bool = True

    optimizer_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 160_000
    scheduler_decay_lr: float = 5e-6

    vis_attn: bool = False
    vlm_model_path: str = "tencent/HY-Embodied-0.5"
    vlm_config_dict: dict[str, Any] | None = None

    use_video_encoder: bool = False
    spacetime_layer_stride: int = 4
    past_drop_layer: int | None = None
    max_num_frames: int = 18
    visual_segment_isolation: bool = False
    img_history_size: int = 1
    img_history_interval: int = 1
    # Number of physical actions served before sampling a new chunk. ``None``
    # means the complete physical horizon. The official RoboTwin wrapper uses 7.
    execution_horizon: int | None = None

    # Geometry is explicit. No mobile-base dimensions are accepted here.
    embodiment: str = "umi_dual_arm"
    action_representation: str = "relative"
    action_decode_mode: str = "relative"
    relative_absolute_blend_weight: float = 0.5
    relative_convention: str = "first_frame"
    native_quaternion_order: str = "xyzw"
    coordinate_transform: str = "identity"
    convert_gripper: bool = False
    zero_variance_epsilon: float = 1e-8

    author_repository_revision: str | None = None
    source_checkpoint: str | None = None
    source_checkpoint_revision: str | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps != self.chunk_size:
            raise ValueError(
                "Hy-VLA checkpoints use all flow tokens at inference; "
                f"n_action_steps ({self.n_action_steps}) must equal chunk_size ({self.chunk_size})."
            )
        if self.model_action_dim != 20:
            raise ValueError(f"{self.embodiment} requires model_action_dim=20, got {self.model_action_dim}.")
        if self.max_state_dim < 20 or self.max_action_dim < 20:
            raise ValueError("Hy-VLA requires at least 20 state/action channels before padding.")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.flow_loss_weight < 0 or self.text_loss_weight < 0:
            raise ValueError("Hy-VLA loss weights must be non-negative.")
        if self.flow_loss_weight == 0 and self.text_loss_weight == 0:
            raise ValueError("At least one Hy-VLA training loss must be enabled.")
        if self.text_max_new_tokens < 1:
            raise ValueError("text_max_new_tokens must be at least 1.")
        if self.embodiment not in {"umi_dual_arm", "robotwin_dual_arm"}:
            raise ValueError(
                "Hy-VLA supports only the released UMI and RoboTwin dual-arm embodiments; "
                f"got {self.embodiment!r}."
            )
        if self.action_representation not in {"relative", "relative_absolute"}:
            raise ValueError(
                "action_representation must be 'relative' or 'relative_absolute', "
                f"got {self.action_representation!r}"
            )
        if self.action_representation == "relative_absolute" and self.chunk_size % 2:
            raise ValueError("relative_absolute checkpoints require an even model chunk_size.")
        if self.img_history_size < 1 or self.img_history_interval < 1:
            raise ValueError("img_history_size and img_history_interval must be positive.")
        if not self.use_video_encoder and self.img_history_size != 1:
            raise ValueError("img_history_size must be 1 when use_video_encoder is disabled.")
        if self.execution_horizon is None:
            self.execution_horizon = self.physical_action_horizon
        if not 1 <= self.execution_horizon <= self.physical_action_horizon:
            raise ValueError(
                "execution_horizon must be within the physical action horizon; "
                f"got {self.execution_horizon} for horizon {self.physical_action_horizon}."
            )
        if self.action_decode_mode not in {"relative", "absolute", "blend"}:
            raise ValueError("action_decode_mode must be 'relative', 'absolute', or 'blend'.")
        if self.action_representation == "relative" and self.action_decode_mode != "relative":
            raise ValueError("A relative-only checkpoint can only use action_decode_mode='relative'.")
        if not 0 <= self.relative_absolute_blend_weight <= 1:
            raise ValueError("relative_absolute_blend_weight must be in [0, 1].")
        if self.relative_convention not in {"first_frame", "current"}:
            raise ValueError("relative_convention must be 'first_frame' or 'current'.")
        if self.native_quaternion_order not in {"xyzw", "wxyz"}:
            raise ValueError("native_quaternion_order must be 'xyzw' or 'wxyz'.")
        if self.coordinate_transform not in {"identity", "robotwin_to_umi"}:
            raise ValueError("coordinate_transform must be 'identity' or 'robotwin_to_umi'.")
        self.validate_features()

    @property
    def physical_action_horizon(self) -> int:
        return self.chunk_size // 2 if self.action_representation == "relative_absolute" else self.chunk_size

    def validate_features(self) -> None:
        if self.attention_implementation != "eager":
            raise ValueError("Hy-VLA only supports attention_implementation='eager'.")
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}
        if not self.input_features:
            self.input_features = {
                _UMI_CAMERA_KEYS[0]: PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                _UMI_CAMERA_KEYS[1]: PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                _UMI_CAMERA_KEYS[2]: PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                OBS_STATE: PolicyFeature(FeatureType.STATE, (16,)),
            }
        if not self.output_features:
            self.output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (16,))}
        state = self.input_features.get(OBS_STATE)
        action = self.output_features.get(ACTION)
        expected_state_dims = {16, 20, 32}
        if state is not None and state.shape[-1] not in expected_state_dims:
            raise ValueError(
                f"Hy-VLA {self.embodiment} state must have one of {sorted(expected_state_dims)} dimensions; "
                f"got {state.shape}."
            )
        expected_action_dims = {16, 20}
        if action is not None and action.shape[-1] not in expected_action_dims:
            raise ValueError(
                f"Hy-VLA {self.embodiment} action must have one of {sorted(expected_action_dims)} dimensions; "
                f"got {action.shape}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        # MEM history is assembled by the dataset/runtime adapter, not delta indices.
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.physical_action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None


__all__ = ["HyVLAConfig"]
