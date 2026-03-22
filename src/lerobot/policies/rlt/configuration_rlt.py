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
"""RLT (RL Token) policy configuration.

Reference: "RL Token: Bootstrapping Online RL with Vision-Language-Action Models"
(Xu et al., Physical Intelligence, 2026)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.sac.configuration_sac import ActorLearnerConfig, ConcurrencyConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


@dataclass
class RLTokenConfig:
    """Configuration for the RL-token encoder/decoder transformer."""

    input_dim: int = 2048
    rl_token_dim: int = 2048
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    num_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.0


@dataclass
class RLTActorConfig:
    """Configuration for the lightweight RL actor MLP."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    std: float = 0.1


@dataclass
class RLTCriticConfig:
    """Configuration for the RLT critic MLP."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])


@PreTrainedConfig.register_subclass("rlt")
@dataclass
class RLTConfig(PreTrainedConfig):
    """Configuration for the RLT (RL Token) policy.

    RLT adds an RL-token encoder/decoder to a frozen VLA backbone, then trains
    a lightweight actor-critic head using the RL token as state representation.
    The frozen VLA also provides reference action chunks that the actor refines.
    """

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            OBS_IMAGE: {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            OBS_STATE: {"min": [0.0], "max": [1.0]},
            ACTION: {"min": [0.0], "max": [1.0]},
        }
    )

    # ── Device ──
    device: str = "cuda"
    storage_device: str = "cpu"

    # ── VLA backbone ──
    vla_checkpoint: str | None = None

    # ── RL-token ──
    rl_token: RLTokenConfig = field(default_factory=RLTokenConfig)

    # ── Actor / Critic heads ──
    actor: RLTActorConfig = field(default_factory=RLTActorConfig)
    critic: RLTCriticConfig = field(default_factory=RLTCriticConfig)

    # ── Action chunks ──
    chunk_size: int = 10
    vla_chunk_size: int = 50

    # ── Training parameters ──
    online_steps: int = 50000
    offline_steps: int = 5000
    online_buffer_capacity: int = 100000
    offline_buffer_capacity: int = 100000
    online_step_before_learning: int = 500
    warmup_steps: int = 500
    async_prefetch: bool = False

    # ── Algorithm hyperparameters ──
    utd_ratio: int = 5
    policy_update_freq: int = 2
    discount: float = 0.99
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    rl_token_lr: float = 1e-4
    tau: float = 0.005
    clip_grad_norm: float = 10.0
    num_critics: int = 2
    bc_reg_coeff: float = 0.1
    ref_dropout: float = 0.5
    chunk_stride: int = 2
    vla_finetune_weight: float = 0.0

    # ── Distributed ──
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    def __post_init__(self):
        super().__post_init__()

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        if ACTION not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
