# Copyright 2026 Ahad Jawaid and The HuggingFace Inc. team. All rights reserved.
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
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("nat_vla")
@dataclass
class NATVLAConfig(PreTrainedConfig):
    """vlm: Hub repository or folder of the pretrained NAT VLA, whose weights a new policy starts from and whose
    vlm/ subfolder holds the VLM's config, tokenizer and image processor. vlm_config: the VLM's architecture,
    read from vlm/ when the policy is first built and kept in its checkpoints. action_tokenizer: Hub repository
    or folder of the NAT tokenizer. vlm_revision, action_tokenizer_revision: the Hub commits these are downloaded
    from; change them together with the repositories. rtc_steps: actions of a chunk that run while the next chunk
    is generated, which then starts with their tokens (0 turns this off)."""

    vlm: str = "ahad-j/nat-vla-450m"
    vlm_revision: str | None = "a5c2e13d45832050e364265962f6ad536460e955"
    action_tokenizer: str = "ahad-j/nat-tokenizer-h32-l4-c6400"
    action_tokenizer_revision: str | None = "daee61dc36d74129d7bf3e823871d3e8b3c5fa85"
    setup: str = ""
    control_mode: str = ""
    n_obs_steps: int = 1
    dtype: torch.dtype | None = torch.bfloat16
    chunk_size: int = 32
    n_action_steps: int = 32
    rtc_steps: int = 0
    state_bins: int = 256
    z_loss_scale: float = 1e-4
    vlm_config: dict[str, Any] | None = None
    action_tokenizer_config: dict[str, Any] | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    optimizer_lr: float = 5e-5
    optimizer_vision_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-6
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 200
    scheduler_decay_steps: int = 50_000
    scheduler_decay_lr: float = 5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.normalization_mapping.get("ACTION") != NormalizationMode.MIN_MAX:
            raise ValueError(
                "The NAT tokenizer encodes actions in [-1, 1], so ACTION needs MIN_MAX normalization"
            )
        if not 0 < self.n_action_steps <= self.chunk_size:
            raise ValueError(f"n_action_steps must be between 1 and chunk_size ({self.chunk_size})")
        if self.rtc_steps and not self.rtc_steps <= self.n_action_steps <= self.chunk_size - self.rtc_steps:
            raise ValueError(
                "rtc_steps must be at most n_action_steps, and n_action_steps + rtc_steps at most chunk_size"
            )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("NAT VLA needs at least one camera image.")
        if self.action_feature is None:
            raise ValueError("NAT VLA needs an 'action' output feature.")

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
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


def downloaded(repository: str, patterns: list[str], revision: str | None) -> Path:
    """A local directory, or the files of a Hub repository at revision matching patterns, downloaded."""
    folder = Path(repository)
    if folder.is_dir():
        return folder
    return Path(snapshot_download(repository, revision=revision, allow_patterns=patterns))
