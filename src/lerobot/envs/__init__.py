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

# NOTE: gymnasium is currently a core dependency but is a candidate for moving to an
# optional extra in the future. When that transition happens, uncomment the guard below
# and update the extra name to the one that will contain gymnasium.
# from lerobot.utils.import_utils import require_package
# require_package("gymnasium", extra="<update_extra>", import_name="gymnasium")

from .configs import (
    AlohaEnv,
    EnvConfig,
    G1EndEffector,
    HILSerlRobotEnvConfig,
    HubEnvConfig,
    PushtEnv,
    UnitreeG1MujocoEnv,
)
from .factory import make_env, make_env_config, make_env_pre_post_processors
from .utils import (
    NEW_ROLLOUT_OPTION,
    check_env_attributes_and_types,
    close_envs,
    env_to_policy_features,
    preprocess_observation,
)

# TODO: rename the homonymic classes so their names are unambiguous:
# configs.LiberoEnv and libero.LiberoEnv, configs.MetaworldEnv and metaworld.MetaworldEnv,
# configs.RoboCasaEnv and robocasa.RoboCasaEnv, configs.VLABenchEnv and vlabench.VLABenchEnv,
# configs.RewardClassifierConfig and rewards.classifier.configuration_classifier.RewardClassifierConfig.

__all__ = [
    "AlohaEnv",
    "EnvConfig",
    "G1EndEffector",
    "HILSerlRobotEnvConfig",
    "HubEnvConfig",
    "NEW_ROLLOUT_OPTION",
    "PushtEnv",
    "UnitreeG1MujocoEnv",
    "check_env_attributes_and_types",
    "close_envs",
    "env_to_policy_features",
    "make_env",
    "make_env_config",
    "make_env_pre_post_processors",
    "preprocess_observation",
]
