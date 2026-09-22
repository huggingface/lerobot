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

"""Top-level pipeline config for distributed RL training (actor / learner)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.envs import HILSerlRobotEnvConfig
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig

from .algorithms.configs import RLAlgorithmConfig
from .algorithms.factory import make_algorithm_config
from .algorithms.sac import SACAlgorithmConfig  # noqa: F401


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    # NOTE: In RL, we don't need an offline dataset
    # TODO: Make `TrainPipelineConfig.dataset` optional
    dataset: DatasetConfig | None = None  # type: ignore[assignment] # because the parent class has made it's type non-optional

    # Algorithm config.
    algorithm: RLAlgorithmConfig | None = None

    # Data mixer strategy name. Currently supports "online_offline".
    mixer: str = "online_offline"
    # Fraction sampled from online replay when using OnlineOfflineMixer.
    online_ratio: float = 0.5

    def validate(self) -> None:
        super().validate()

        if self.algorithm is None:
            self.algorithm = make_algorithm_config("sac")

        if getattr(self.algorithm, "policy_config", None) is None:
            self.algorithm.policy_config = self.policy


def get_gaussian_actor_config(cfg: TrainRLServerPipelineConfig) -> GaussianActorConfig:
    """Return ``cfg.policy`` as the HIL-SERL policy config the actor and learner run."""
    if not isinstance(cfg.policy, GaussianActorConfig):
        raise TypeError(f"policy config must be a GaussianActorConfig, got {type(cfg.policy).__name__}")
    return cfg.policy


def get_hilserl_env_config(cfg: TrainRLServerPipelineConfig) -> HILSerlRobotEnvConfig:
    """Return ``cfg.env`` as the robot env config ``make_robot_env`` and ``make_processors`` build from."""
    if not isinstance(cfg.env, HILSerlRobotEnvConfig):
        raise TypeError(f"env config must be a HILSerlRobotEnvConfig, got {type(cfg.env).__name__}")
    return cfg.env


def require_not_none[T](value: T | None, what: str) -> T:
    """Return ``value``, raising ``ValueError`` if a required config or transition field is None."""
    if value is None:
        raise ValueError(f"{what} is required but is None")
    return value


def get_log_file(cfg: TrainRLServerPipelineConfig, name: str) -> Path:
    """Return ``<output_dir>/logs/<name>.log``, creating the directory if needed."""
    log_dir = Path(require_not_none(cfg.output_dir, "cfg.output_dir")) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{name}.log"
