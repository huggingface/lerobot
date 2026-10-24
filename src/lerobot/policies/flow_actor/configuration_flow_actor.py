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

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.optim import AdamConfig, LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("flow_actor")
@dataclass
class FlowActorConfig(PreTrainedConfig):
    """State-only flow actor used by QF3.

    The velocity field is an MLP. ``select_action`` integrates it with explicit Euler
    steps. This is the from-scratch actor in arXiv:2610.08789, not Pi0 and not a
    Gaussian SAC actor. Image observations are rejected.

    Defaults for width, time embedding, and Euler steps follow appendix table
    ``tab:qf3-arch``. QF3 training hyperparameters live on ``QF3AlgorithmConfig``.
    """

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Two hidden layers, width 256. Table tab:qf3-arch, MuJoCo column.
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    # Appendix sinusoidal embedding dimension.
    time_dim: int = 64
    # Appendix Euler step count. Rollouts and the critic bootstrap should use the same count.
    euler_steps: int = 5

    def get_optimizer_preset(self) -> AdamConfig:
        # RL training builds its own Adam from QF3AlgorithmConfig. This preset only
        # satisfies PreTrainedConfig.
        return AdamConfig(lr=3e-4, weight_decay=0.0)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        if not self.input_features or not self.output_features:
            raise ValueError("input_features and output_features must be set before use.")
        state = self.input_features.get(OBS_STATE)
        if state is None or state.type is not FeatureType.STATE:
            raise ValueError("flow_actor requires observation.state and no other observation.")
        if any(feature.type is FeatureType.VISUAL for feature in self.input_features.values()):
            raise ValueError("flow_actor is state-only. Image observations are out of scope.")
        if len(self.input_features) != 1:
            raise ValueError("flow_actor accepts only observation.state.")
        action = self.output_features.get(ACTION)
        if action is None or action.type is not FeatureType.ACTION:
            raise ValueError("flow_actor requires an action output feature.")
        if self.time_dim < 2 or self.time_dim % 2 != 0:
            raise ValueError(f"time_dim must be even and at least 2, got {self.time_dim}.")
        if self.euler_steps < 1:
            raise ValueError(f"euler_steps must be positive, got {self.euler_steps}.")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one width.")

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int] | None:
        return None

    @property
    def reward_delta_indices(self) -> list[int] | None:
        return None
