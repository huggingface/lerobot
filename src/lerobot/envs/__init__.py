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

"""Envs: what a LeRobot command runs, selected with ``--env.type`` (``--robot.env.type`` for a sim robot).

An ``EnvConfig`` holds an env's command-line options. Not every env is a simulator: ``gym_manipulator`` drives
a real robot, and EnvHub's native batches keep the provider's own vector environment. A simulated env runs a
sim (``lerobot.sim``) in this process, or on a server (``lerobot.env_server``) when it has an ``endpoint``.
"""

# NOTE: gymnasium is currently a core dependency but is a candidate for moving to an
# optional extra in the future. When that transition happens, uncomment the guard below
# and update the extra name to the one that will contain gymnasium.
# from lerobot.utils.import_utils import require_package
# require_package("gymnasium", extra="<update_extra>", import_name="gymnasium")

from .configs import AlohaEnv, EnvConfig, EnvServerConfig, HILSerlRobotEnvConfig, HubEnvConfig, PushtEnv
from .factory import make_env, make_env_config, make_env_pre_post_processors
from .utils import check_env_attributes_and_types, close_envs, env_to_policy_features, preprocess_observation

__all__ = [
    "AlohaEnv",
    "EnvConfig",
    "EnvServerConfig",
    "HILSerlRobotEnvConfig",
    "HubEnvConfig",
    "PushtEnv",
    "check_env_attributes_and_types",
    "close_envs",
    "env_to_policy_features",
    "make_env",
    "make_env_config",
    "make_env_pre_post_processors",
    "preprocess_observation",
]
