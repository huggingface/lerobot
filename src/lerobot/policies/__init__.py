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

from lerobot.utils.action_interpolator import ActionInterpolator as ActionInterpolator

from .act.configuration_act import ACTConfig as ACTConfig
from .diffusion.configuration_diffusion import DiffusionConfig as DiffusionConfig
from .eo1.configuration_eo1 import EO1Config as EO1Config
from .factory import get_policy_class, make_policy, make_policy_config, make_pre_post_processors
from .gaussian_actor.configuration_gaussian_actor import GaussianActorConfig as GaussianActorConfig
from .groot.configuration_groot import GrootConfig as GrootConfig
from .multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig as MultiTaskDiTConfig
from .pi0.configuration_pi0 import PI0Config as PI0Config
from .pi0_fast.configuration_pi0_fast import PI0FastConfig as PI0FastConfig
from .pi05.configuration_pi05 import PI05Config as PI05Config
from .pretrained import PreTrainedPolicy as PreTrainedPolicy
from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig
from .tdmpc.configuration_tdmpc import TDMPCConfig as TDMPCConfig
from .utils import make_robot_action, prepare_observation_for_inference
from .vqbet.configuration_vqbet import VQBeTConfig as VQBeTConfig
from .wall_x.configuration_wall_x import WallXConfig as WallXConfig
from .xvla.configuration_xvla import XVLAConfig as XVLAConfig

# NOTE: Policy modeling classes (e.g., GaussianActorPolicy) are intentionally NOT re-exported here.
# They have heavy optional dependencies and are loaded lazily via get_policy_class().
# Import directly: ``from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy``

__all__ = [
    # Configuration classes
    "ACTConfig",
    "DiffusionConfig",
    "EO1Config",
    "GaussianActorConfig",
    "GrootConfig",
    "MultiTaskDiTConfig",
    "PI0Config",
    "PI0FastConfig",
    "PI05Config",
    "SmolVLAConfig",
    "TDMPCConfig",
    "VQBeTConfig",
    "WallXConfig",
    "XVLAConfig",
    # Base class
    "PreTrainedPolicy",
    # RTC utilities
    "ActionInterpolator",
    # Utility functions
    "make_robot_action",
    "prepare_observation_for_inference",
    # Factory functions
    "get_policy_class",
    "make_policy",
    "make_policy_config",
    "make_pre_post_processors",
]
