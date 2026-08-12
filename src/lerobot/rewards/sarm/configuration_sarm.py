# Copyright 2025 Qianzhong Chen, Justin Yu, Mac Schwager, Pieter Abbeel, Yide Shentu, Philipp Wu
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

"""SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation.

Paper: https://arxiv.org/abs/2509.25358
"""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@RewardModelConfig.register_subclass("sarm")
@dataclass
class SARMConfig(RewardModelConfig):
    """Configuration class for SARM (Stage-Aware Reward Modeling).

    Supports three annotation modes:

    1. single_stage (default): No annotations needed. Uses the episode's task description
       as a single stage covering the entire episode.

    2. dense_only: Uses dense (fine-grained) annotations from VLM, with an auto-generated
       single sparse "task" stage covering the full episode. The dense head learns detailed
       subtask progression while sparse provides overall task completion.

    3. dual: Full dual-head mode with both sparse (high-level) and dense (fine-grained)
       annotations from VLM. Both heads are trained on their respective annotations.

    The annotation_mode determines how sparse_temporal_proportions and dense_temporal_proportions
    are loaded/generated during model initialization.

    Args:
        input_features (`dict`, *optional*):
            Populated by `__post_init__` from `image_key`/`state_key`; not meant to be set directly.
        output_features (`dict`, *optional*):
            Populated by `__post_init__` based on `annotation_mode`; not meant to be set directly.
        device (`str | None`, *optional*):
            Torch device to run the model on.
        pretrained_path (`str | None`, *optional*):
            Local directory or Hugging Face Hub repo id to load pretrained weights from.
        pretrained_revision (`str | None`, *optional*):
            Optional Hub revision (commit hash, branch, or tag) to pin the pretrained model version.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push this model to the Hugging Face Hub.
        repo_id (`str | None`, *optional*):
            Hub repository id to push to when `push_to_hub` is `True`.
        license (`str | None`, *optional*):
            License tag for the Hub model card.
        tags (`list[str] | None`, *optional*):
            Hub model card tags.
        private (`bool | None`, *optional*):
            Whether the pushed Hub repository is private.
        annotation_mode (`str`, *optional*, defaults to `"single_stage"`):
            One of `"single_stage"`, `"dense_only"`, or `"dual"`.
        n_obs_steps (`int`, *optional*, defaults to 8):
            Number of observation history steps.
        frame_gap (`int`, *optional*, defaults to 30):
            Frame gap between sampled frames (at 30 fps = 1 second).
        max_rewind_steps (`int`, *optional*, defaults to 4):
            Maximum rewind steps for temporal augmentation. Total frames = `1 + n_obs_steps +
            max_rewind_steps`; during training with rewind the sequence is `[obs_frames] +
            [rewind_frames]`, during inference it is `[obs_frames]` only.
        image_dim (`int`, *optional*, defaults to 512):
            Image embedding dimension.
        text_dim (`int`, *optional*, defaults to 512):
            Text embedding dimension.
        hidden_dim (`int`, *optional*, defaults to 768):
            Transformer hidden dimension.
        num_heads (`int`, *optional*, defaults to 12):
            Number of transformer attention heads.
        num_layers (`int`, *optional*, defaults to 8):
            Number of transformer layers.
        max_state_dim (`int`, *optional*, defaults to 32):
            Dimension the proprioceptive state vector is padded to.
        drop_n_last_frames (`int`, *optional*, defaults to 1):
            Number of trailing frames dropped from the loss.
        batch_size (`int`, *optional*, defaults to 64):
            Training batch size.
        clip_batch_size (`int`, *optional*, defaults to 64):
            Batch size used for CLIP text/image encoding.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
        stage_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight for the stage classification loss when using subtask annotations.
        rewind_probability (`float`, *optional*, defaults to 0.8):
            Probability of applying rewind temporal augmentation during training.
        language_perturbation_probability (`float`, *optional*, defaults to 0.2):
            Probability of perturbing the language instruction during training.
        num_sparse_stages (`int`, *optional*, defaults to 1):
            Number of high-level (sparse) stages. Overwritten by `__post_init__` for
            `annotation_mode` `"single_stage"`/`"dense_only"`.
        sparse_subtask_names (`list | None`, *optional*):
            Names of the high-level stages.
        sparse_temporal_proportions (`list | None`, *optional*):
            Fraction of the episode each high-level stage covers.
        num_dense_stages (`int | None`, *optional*):
            Number of fine-grained (dense) stages, used when `annotation_mode` is `"dense_only"` or
            `"dual"`.
        dense_subtask_names (`list | None`, *optional*):
            Names of the fine-grained stages.
        dense_temporal_proportions (`list | None`, *optional*):
            Fraction of the episode each fine-grained stage covers.
        pretrained_model_path (`str | None`, *optional*):
            Path to a pretrained SARM checkpoint.
        image_key (`str`, *optional*, defaults to `"observation.images.top"`):
            Observation key of the camera view fed to the model.
        state_key (`str`, *optional*, defaults to `"observation.state"`):
            Observation key of the proprioceptive state fed to the model.
        normalization_mapping (`dict`, *optional*):
            Maps feature types to their normalization mode.
    """

    annotation_mode: str = "single_stage"
    n_obs_steps: int = 8
    frame_gap: int = 30
    max_rewind_steps: int = 4

    # Architecture params
    image_dim: int = 512
    text_dim: int = 512
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 8
    max_state_dim: int = 32
    drop_n_last_frames: int = 1
    batch_size: int = 64
    clip_batch_size: int = 64
    dropout: float = 0.1
    stage_loss_weight: float = 1.0

    rewind_probability: float = 0.8
    language_perturbation_probability: float = 0.2

    # Sparse annotations (high-level stages)
    num_sparse_stages: int = 1
    sparse_subtask_names: list | None = None
    sparse_temporal_proportions: list | None = None

    # Dense annotations (fine-grained stages)
    num_dense_stages: int | None = None
    dense_subtask_names: list | None = None
    dense_temporal_proportions: list | None = None

    pretrained_model_path: str | None = None
    device: str | None = None
    image_key: str = OBS_IMAGES + ".top"
    state_key: str = OBS_STATE

    # Populated by the processor (video_features, state_features, text_features)
    input_features: dict = field(default_factory=lambda: {})

    # Output features (updated in __post_init__)
    output_features: dict = field(
        default_factory=lambda: {
            "stage": PolicyFeature(shape=(9, 5), type=FeatureType.REWARD),
            "progress": PolicyFeature(shape=(9, 1), type=FeatureType.REWARD),
        }
    )

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "LANGUAGE": NormalizationMode.IDENTITY,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        """Resolve `annotation_mode`-dependent defaults and (re)populate input/output features.

        Raises:
            ValueError: If `annotation_mode` is invalid, `max_rewind_steps >= n_obs_steps`,
                `num_sparse_stages < 1`, or `num_dense_stages` is set below 2 in a dual-head mode.
        """
        super().__post_init__()
        if self.annotation_mode not in ["single_stage", "dense_only", "dual"]:
            raise ValueError(
                f"annotation_mode must be 'single_stage', 'dense_only', or 'dual', got {self.annotation_mode}"
            )

        if self.annotation_mode == "single_stage":
            # Use task description as stage name, full episode as one stage
            self.num_sparse_stages = 1
            self.sparse_subtask_names = ["task"]
            self.sparse_temporal_proportions = [1.0]
            self.num_dense_stages = None
            self.dense_subtask_names = None
            self.dense_temporal_proportions = None

        elif self.annotation_mode == "dense_only":
            self.num_sparse_stages = 1
            self.sparse_subtask_names = ["task"]
            self.sparse_temporal_proportions = [1.0]

        self.input_features = {}
        self.output_features = {}

        if self.image_key:
            self.input_features[self.image_key] = PolicyFeature(shape=(480, 640, 3), type=FeatureType.VISUAL)

        self.input_features[self.state_key] = PolicyFeature(
            shape=(self.max_state_dim,),
            type=FeatureType.STATE,
        )

        # Update output features based on annotation_mode
        if self.annotation_mode in ["dense_only", "dual"]:
            self.output_features["sparse_stage"] = PolicyFeature(
                shape=(self.num_frames, self.num_sparse_stages), type=FeatureType.REWARD
            )
            self.output_features["sparse_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), type=FeatureType.REWARD
            )
            dense_stages = self.num_dense_stages or self.num_sparse_stages
            self.output_features["dense_stage"] = PolicyFeature(
                shape=(self.num_frames, dense_stages), type=FeatureType.REWARD
            )
            self.output_features["dense_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), type=FeatureType.REWARD
            )
        else:
            self.output_features["sparse_stage"] = PolicyFeature(
                shape=(self.num_frames, self.num_sparse_stages), type=FeatureType.REWARD
            )
            self.output_features["sparse_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), type=FeatureType.REWARD
            )

        if self.max_rewind_steps >= self.n_obs_steps:
            raise ValueError(
                f"max_rewind_steps ({self.max_rewind_steps}) must be less than n_obs_steps ({self.n_obs_steps})"
            )
        if self.num_sparse_stages < 1:
            raise ValueError(f"num_sparse_stages must be at least 1, got {self.num_sparse_stages}")
        if (
            self.annotation_mode in ["dense_only", "dual"]
            and self.num_dense_stages is not None
            and self.num_dense_stages < 2
        ):
            raise ValueError(f"num_dense_stages must be at least 2, got {self.num_dense_stages}")

    def get_optimizer_preset(self) -> AdamWConfig:
        """Get default optimizer configuration for SARM training."""
        return AdamWConfig(
            lr=5e-5,
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """Get default learning rate scheduler configuration."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=5e-5,
            decay_lr=5e-6,
            num_warmup_steps=500,
            num_decay_steps=50000,
        )

    def validate_features(self) -> None:
        """No-op: feature validity is enforced in `__post_init__` instead."""

    @property
    def uses_dual_heads(self) -> bool:
        """Whether the model uses dual heads (dense_only or dual annotation modes)."""
        return self.annotation_mode in ["dense_only", "dual"]

    @property
    def num_frames(self) -> int:
        """Total number of frames in sequence.

        For training: 1 + n_obs_steps + max_rewind_steps
        The sequence is: [obs_frames (n_obs_steps + 1)] + [rewind_frames (max_rewind_steps)]
        """
        return 1 + self.n_obs_steps + self.max_rewind_steps

    @property
    def max_length(self) -> int:
        """Alias for `num_frames`."""
        return self.num_frames

    @property
    def observation_delta_indices(self) -> list[int]:
        """Bidirectional frame sampling centered on target frame.

        Example with n_obs_steps=8, gap=30:
        Before: [-120, -90, -60, -30]  (4 frames)
        Current: [0]                   (1 frame)
        After:  [30, 60, 90, 120]      (4 frames)
        Total: 9 frames
        """
        half_steps = self.n_obs_steps // 2

        past_deltas = [-self.frame_gap * i for i in range(half_steps, 0, -1)]
        future_deltas = [self.frame_gap * i for i in range(1, half_steps + 1)]
        obs_deltas = past_deltas + [0] + future_deltas

        # Rewind placeholders
        rewind_deltas = [-self.frame_gap * (i + 1) for i in range(self.max_rewind_steps)]

        return obs_deltas + rewind_deltas

    @property
    def action_delta_indices(self) -> None:
        """SARM is a reward model, not an action policy."""
        return None

    @property
    def reward_delta_indices(self) -> None:
        """`None`: SARM does not consume past rewards."""
        return None
