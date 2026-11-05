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

"""Reinforcement learning modules.

Distributed actor / learner entry points (``actor``, ``learner``,
``learner_service``) require ``pip install 'lerobot[hilserl]'``. Algorithms,
buffer, data sources and trainer are gRPC-free and usable standalone.
"""

from .algorithms.base import RLAlgorithm as RLAlgorithm
from .algorithms.configs import RLAlgorithmConfig as RLAlgorithmConfig, TrainingStats as TrainingStats
from .algorithms.factory import (
    make_algorithm as make_algorithm,
    make_algorithm_config as make_algorithm_config,
)
from .algorithms.sac.configuration_sac import SACAlgorithmConfig as SACAlgorithmConfig
from .algorithms.sac.sac_algorithm import SACAlgorithm
from .buffer import ReplayBuffer as ReplayBuffer
from .data_sources import DataMixer as DataMixer, OnlineOfflineMixer as OnlineOfflineMixer
from .trainer import RLTrainer as RLTrainer

# NOTE: gym_manipulator is intentionally NOT re-exported here to avoid pulling the camera, robot
# and teleoperator stacks.
# Import from submodules: ``from lerobot.rl.gym_manipulator import make_robot_env``

__all__ = [
    "RLAlgorithm",
    "RLAlgorithmConfig",
    "TrainingStats",
    "make_algorithm",
    "make_algorithm_config",
    "SACAlgorithm",
    "SACAlgorithmConfig",
    "RLTrainer",
    "ReplayBuffer",
    "DataMixer",
    "OnlineOfflineMixer",
]
