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

"""Configuration for a bounded IMPRINT policy.

IMPRINT is "Memory Without Curation" (arXiv:2610.14406). The reference
implementation is Apache-2.0 at https://github.com/amazon-far/imprint.

The defaults below are a CPU-sized policy. They are not the paper's RMBench
widths. Paper values are noted on each field. Existing LeRobot policies are
unchanged.
"""

from dataclasses import dataclass, field
from typing import Literal

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

EncoderName = Literal["recurrent", "full_attention"]
TrainingStage = Literal["memory", "joint"]


@PreTrainedConfig.register_subclass("imprint")
@dataclass
class ImprintConfig(PreTrainedConfig):
    """Bounded IMPRINT memory policy.

    ``training_stage="joint"`` is the deployable policy: flow-matching action
    loss with memory tokens read inside the action expert. ``training_stage="memory"``
    is stage 1: reconstruct past streams and estimate episode progress. The paper
    discards the stage-1 decoder before joint adaptation.

    CLI: ``--policy.type=imprint``.
    """

    # Observation window ending at the current step. Paper histories are much longer.
    history_length: int = 4
    # Action horizon. Paper RMBench uses 50 and RoboCasa uses 20.
    chunk_size: int = 16
    n_action_steps: int = 16
    # Frames inside one finalized visual chunk. Paper ``temporal_factor`` is 4.
    chunk_frames: int = 4
    # Spatial tokens per camera from the bounded image stem. Paper DINOv3 uses 64
    # tokens per view. Paper Wan chunks use num_views * pool_h * pool_w tokens.
    tokens_per_view: int = 1

    encoder: str = "recurrent"
    training_stage: str = "joint"
    # Turn off to train the action expert with no memory modulation.
    use_memory: bool = True

    # Width of memory tokens. Paper modulation path uses 1024.
    memory_dim: int = 64
    # Learned queries. Paper ablations peak at 256.
    num_latents: int = 8
    # Attention heads for the perceiver, recent fusion, decoder, and action expert.
    num_heads: int = 4
    # Full-attention perceiver depth. Paper ``num_cross_blocks`` is 2.
    num_cross_blocks: int = 2
    # Gated-delta stack. Paper: 2 layers, 8 heads, head dim 128, conv kernel 4.
    delta_num_layers: int = 1
    delta_num_heads: int = 4
    delta_head_dim: int = 16
    delta_conv_kernel: int = 4
    delta_ffn_inner: int = 128
    delta_decay_bias: float = -4.0
    delta_beta_bias: float = -1.0
    # Recent-window fusion blocks after the recurrent read. Paper default is 1.
    recent_fusion_blocks: int = 1
    # Time embedding normalizes the chunk index by this cap. Paper uses 384.
    max_chunks: int = 384

    # Bounded visual stem width. The paper uses frozen Wan 2.2 latents (48-d)
    # for chunks and frozen DINOv3 features (768-d) for recent frames. This
    # policy does not load those models. chunk_feature_dim and recent_feature_dim
    # are linear projections of the stem. They are not extra batch keys.
    visual_dim: int = 32
    chunk_feature_dim: int = 32
    recent_feature_dim: int = 32
    # Language embedding width. Paper Pi0.5 path uses 2048. None skips FiLM.
    lang_dim: int = 32
    tokenizer_hidden: int = 64

    # Stage-1 decoder. Paper samples 16 queries, 192 align patches, align dim 768.
    num_decoder_blocks: int = 2
    align_dim: int = 32
    num_query_samples: int = 4
    max_query_steps: int = 2048
    decoder_ffn_inner: int = 128

    # Action expert. Paper stage 2 finetunes the Pi0.5 flow-matching expert.
    expert_layers: int = 1
    num_inference_steps: int = 10
    # Pi0.5 flow-matching time sample, which is the native stage-2 objective.
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # LeRobot optimizer preset for this small policy. Paper stage 1 uses peak
    # 1e-4. Paper stage 2 uses peak 2.5e-5 on the VLA. Neither is launched here.
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_obs_steps = self.history_length
        if self.encoder not in ("recurrent", "full_attention"):
            raise ValueError(f"encoder must be 'recurrent' or 'full_attention', got {self.encoder!r}.")
        if self.training_stage not in ("memory", "joint"):
            raise ValueError(f"training_stage must be 'memory' or 'joint', got {self.training_stage!r}.")
        if self.history_length < 1:
            raise ValueError(f"history_length must be positive, got {self.history_length}.")
        if self.chunk_frames < 1:
            raise ValueError(f"chunk_frames must be positive, got {self.chunk_frames}.")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}.")
        if self.n_action_steps < 1 or self.n_action_steps > self.chunk_size:
            raise ValueError(
                "n_action_steps must be in 1..chunk_size, "
                f"got {self.n_action_steps} and chunk_size {self.chunk_size}."
            )
        if self.tokens_per_view < 1:
            raise ValueError(f"tokens_per_view must be positive, got {self.tokens_per_view}.")
        if self.memory_dim < 2 or self.memory_dim % 2 != 0:
            raise ValueError(f"memory_dim must be a positive even integer, got {self.memory_dim}.")
        if self.num_heads < 1 or self.memory_dim % self.num_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must divide memory_dim ({self.memory_dim}).")
        if self.delta_num_heads * self.delta_head_dim != self.memory_dim:
            raise ValueError(
                "delta_num_heads * delta_head_dim must equal memory_dim, got "
                f"{self.delta_num_heads} * {self.delta_head_dim} != {self.memory_dim}."
            )
        if self.delta_num_layers < 1:
            raise ValueError(f"delta_num_layers must be positive, got {self.delta_num_layers}.")
        if self.delta_conv_kernel < 1:
            raise ValueError(f"delta_conv_kernel must be positive, got {self.delta_conv_kernel}.")
        if self.num_latents < 1:
            raise ValueError(f"num_latents must be positive, got {self.num_latents}.")
        if self.num_cross_blocks < 1:
            raise ValueError(f"num_cross_blocks must be positive, got {self.num_cross_blocks}.")
        if self.recent_fusion_blocks < 1:
            raise ValueError(f"recent_fusion_blocks must be positive, got {self.recent_fusion_blocks}.")
        if self.max_chunks < 1:
            raise ValueError(f"max_chunks must be positive, got {self.max_chunks}.")
        if self.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be positive, got {self.num_inference_steps}.")
        if self.expert_layers < 1:
            raise ValueError(f"expert_layers must be positive, got {self.expert_layers}.")
        if self.max_query_steps < 1:
            raise ValueError(f"max_query_steps must be positive, got {self.max_query_steps}.")
        if self.num_query_samples < 1:
            raise ValueError(f"num_query_samples must be positive, got {self.num_query_samples}.")
        for name, value in (
            ("visual_dim", self.visual_dim),
            ("chunk_feature_dim", self.chunk_feature_dim),
            ("recent_feature_dim", self.recent_feature_dim),
            ("lang_dim", self.lang_dim),
            ("tokenizer_hidden", self.tokenizer_hidden),
            ("align_dim", self.align_dim),
            ("decoder_ffn_inner", self.decoder_ffn_inner),
            ("delta_ffn_inner", self.delta_ffn_inner),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("IMPRINT requires at least one visual input feature.")
        if self.robot_state_feature is None:
            raise ValueError("IMPRINT requires observation.state.")
        if self.action_feature is None:
            raise ValueError("IMPRINT requires an action output feature.")
        shapes = {tuple(feature.shape) for feature in self.image_features.values()}
        if len(shapes) != 1:
            raise ValueError(f"IMPRINT cameras must share one shape, got {sorted(shapes)}.")
        channels, height, width = next(iter(shapes))
        if len(next(iter(shapes))) != 3 or min(channels, height, width) < 1:
            raise ValueError(f"Visual features must have shape (C, H, W), got {next(iter(shapes))}.")
        _vector_dim(self.robot_state_feature.shape, name="observation.state")
        _vector_dim(self.action_feature.shape, name="action")

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.history_length, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        # Joint batches carry causal history and the future chunk in one action tensor.
        # Memory batches carry only the causal history. Index 0 is the current frame.
        if self.training_stage == "memory":
            return list(range(-self.history_length, 0))
        return list(range(-self.history_length, self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


def _vector_dim(shape: tuple[int, ...], *, name: str) -> int:
    if len(shape) == 1 and shape[0] > 0:
        return int(shape[0])
    raise ValueError(f"{name} must have a 1D shape (D,), got {shape}.")
