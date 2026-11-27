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

import copy
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import (
    LRSchedulerConfig,
    MolmoAct2AdamWConfig,
    MolmoAct2CosineWithWarmupSchedulerConfig,
    OptimizerConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("molmoact2")
@dataclass
class MolmoAct2Config(PreTrainedConfig):
    """MolmoAct2 policy: transformers' native Molmo2 VLM with the MolmoAct2 flow-matching action expert.

    The weights come from LeRobot checkpoints (`--policy.path`), converted from the released MolmoAct2 checkpoints.
    """

    # Upstream MolmoAct2 repo (or local directory) whose tokenizer files the processors read, optionally pinned as
    # `repo@revision`. The weights come from LeRobot checkpoints (`--policy.path`), and the image processor settings
    # from their saved preprocessor.
    tokenizer_name: str = "allenai/MolmoAct2"

    # Upstream `action_horizon` (`max_action_horizon` in the HF config).
    chunk_size: int = 30
    n_action_steps: int = 30

    # The released checkpoints were trained with "both" (the `action_mode` of their config.json), the default of
    # AI2's LeRobot fork: the flow-matching loss plus the FAST action-token cross-entropy and z-loss. Upstream's
    # native LeRobot launcher (train_lerobot.py --action_format) defaults to "continuous", flow matching alone.
    action_mode: str = "both"
    inference_action_mode: str = "continuous"
    # FAST tokenizer of the discrete action tokens, optionally pinned as `repo@revision`.
    action_tokenizer_name: str = "allenai/MolmoAct2-FAST-Tokenizer"
    norm_tag: str | None = None
    # Optional standalone norm_stats.json, or a Hub repo id or local directory
    # holding one. This lets a generic base model use the exact embodiment
    # statistics published with a downstream checkpoint without changing which
    # model weights are loaded. None reads the base checkpoint's
    # (allenai/MolmoAct2).
    norm_stats_path: str | None = None

    setup_type: str = ""
    control_mode: str = ""
    image_keys: list[str] = field(default_factory=list)
    # Whether gripper dims are included in state/action quantile normalization. Upstream sets it per embodiment
    # (True for SO-100/101, False otherwise). None, a LeRobot addition, normalizes them only when their stats
    # leave [-1, 1]. A mask in the stats (from norm_tag) decides, unless this is True.
    normalize_gripper: bool | None = None
    # Camera-order augmentation during training (upstream `--random_camera_order`): "none", once per "episode",
    # or for "all" samples. Upstream seeds the per-episode order from the repo, data seed, episode and camera
    # keys; this port seeds it from the episode index.
    random_camera_order: str = "none"

    # Fixed by released MolmoAct2 checkpoints.
    max_action_dim: int = 32

    # Flow-matching training knobs copied from the original MolmoAct2 training path. AI2's LeRobot fork and
    # upstream's DROID recipe use 8 flow timesteps per example; upstream's launcher defaults to 1.
    num_flow_timesteps: int = 8
    flow_matching_time_offset: float = 0.001
    flow_matching_time_scale: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5
    # Euler steps of continuous action generation (upstream `flow_matching_num_steps`).
    num_inference_steps: int = 10
    rtc_config: RTCConfig | None = None

    # Joint units and frame transform for cross-calibration compatibility.
    # Some MolmoAct2 checkpoints were trained on data using other units or a
    # different joint convention than LeRobot robots (degrees, grippers from 0
    # to 100). Set them to map the leading state and action dims to the
    # checkpoint's frame at runtime, `joint_scales * x + joint_offsets` (state
    # before model, action after; `JointAffineProcessorStep`). A negative scale
    # flips a joint. See: https://huggingface.co/docs/lerobot/backwardcomp
    # Default is None (no transform). Offsets default to zeros.
    joint_scales: list[float] | None = None
    joint_offsets: list[float] | None = None

    # Freeze the vision tower (upstream `ft_vit=false`).
    freeze_vision_encoder: bool = False
    # Train only the action expert (upstream `ft_vlm=false` with `ft_embedding=none`).
    train_expert_only: bool = False
    # Freeze the base vocabulary of the token embedding; the 128 extra image-token rows still train
    # (upstream `ft_embedding="lm_head"`).
    freeze_embedding: bool = True
    gradient_checkpointing: bool = False
    # Compile continuous action sampling with torch.compile.
    compile_model: bool = False

    # Pi0.5-style precision switch controlling parameter storage and autocast.
    # ``bfloat16`` stores the LLM in bf16 while the vision tower, connector, the
    # norms and the full action expert stay fp32; operator compute follows bf16
    # autocast. ``float32`` keeps both the full model and compute in fp32.
    dtype: torch.dtype | None = torch.bfloat16
    # Official fine-tuning from the released ``allenai/MolmoAct2`` HF base
    # explicitly applies unmasked residual dropout 0.1 and disables the
    # response-only variant (lerobot_utils/hf.py: residual_dropout=0.1,
    # response_residual_dropout=0.0).
    residual_dropout: float = 0.1

    # Per-group learning rates of upstream fine-tuning (`{llm,vit,connector,action_expert}_learning_rate`).
    # With LoRA, the LLM, ViT and connector groups use 5e-5, as in upstream's LoRA recipe.
    optimizer_lr: float = 1e-5
    optimizer_vit_lr: float = 5e-6
    optimizer_connector_lr: float = 5e-6
    optimizer_action_expert_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-6
    optimizer_weight_decay: float = 0.0
    # Applied separately to the LLM, ViT, connector and action expert groups (upstream `max_grad_norm`).
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 200
    # Step where the cosine decay ends. None ends it at the last training step, as upstream ends it at
    # `max_duration`.
    scheduler_decay_steps: int | None = None
    # Final learning rate as a fraction of each group's peak (upstream `alpha_f`).
    scheduler_alpha_f: float = 0.1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # Redeclared without `None`: `validate_features` guarantees them, so the policy code needs no guards.
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_mode not in {"continuous", "discrete", "both"}:
            raise ValueError(
                f"action_mode must be 'continuous', 'discrete' or 'both', got {self.action_mode!r}."
            )
        if self.inference_action_mode not in {"continuous", "discrete"}:
            raise ValueError(
                f"inference_action_mode must be 'continuous' or 'discrete', got {self.inference_action_mode!r}."
            )
        if self.action_mode not in {self.inference_action_mode, "both"}:
            raise ValueError(
                f"action_mode={self.action_mode!r} cannot run {self.inference_action_mode} inference."
            )
        if self.train_expert_only and self.action_mode != "continuous":
            raise ValueError("train_expert_only freezes the VLM, so it requires action_mode='continuous'.")
        if self.random_camera_order not in {"none", "episode", "all"}:
            raise ValueError(
                f"random_camera_order must be 'none', 'episode' or 'all', got {self.random_camera_order!r}."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})."
            )
        if self.joint_offsets is not None and self.joint_scales is None:
            raise ValueError("joint_offsets requires joint_scales.")
        if self.norm_stats_path is not None and not self.norm_tag:
            raise ValueError("norm_stats_path requires norm_tag.")
        if self.dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError(f"dtype must be torch.float32 or torch.bfloat16, got {self.dtype!r}.")

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save without the norm-tag fields: their stats are already in the saved processors."""
        # A shallow copy, not `dataclasses.replace`: re-running `__post_init__` would resolve an unset device.
        config = copy.copy(self)
        config.norm_tag = config.norm_stats_path = None
        PreTrainedConfig._save_pretrained(config, save_directory)

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
        return MolmoAct2AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            group_grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return MolmoAct2CosineWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            alpha_f=self.scheduler_alpha_f,
            num_decay_steps=self.scheduler_decay_steps,
        )

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
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature
