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

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import lazy_exports

# These import torch, and each config imports its policy package with the model, so each is imported
# the first time it is used.
if TYPE_CHECKING:
    from lerobot.utils.action_interpolator import ActionInterpolator as ActionInterpolator

    from .act.configuration_act import ACTConfig as ACTConfig
    from .diffusion.configuration_diffusion import DiffusionConfig as DiffusionConfig
    from .eo1.configuration_eo1 import EO1Config as EO1Config
    from .evo1.configuration_evo1 import Evo1Config as Evo1Config
    from .factory import get_policy_class, make_policy, make_policy_config, make_pre_post_processors
    from .fastwam.configuration_fastwam import FastWAMConfig as FastWAMConfig
    from .flux3.configuration_flux3 import Flux3Config as Flux3Config
    from .gaussian_actor.configuration_gaussian_actor import GaussianActorConfig as GaussianActorConfig
    from .groot.configuration_groot import GrootConfig as GrootConfig
    from .lawam.configuration_lawam import LaWAMConfig as LaWAMConfig
    from .lingbot_va.configuration_lingbot_va import LingBotVAConfig as LingBotVAConfig
    from .molmoact2.configuration_molmoact2 import MolmoAct2Config as MolmoAct2Config
    from .multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig as MultiTaskDiTConfig
    from .pi0.configuration_pi0 import PI0Config as PI0Config
    from .pi0_fast.configuration_pi0_fast import PI0FastConfig as PI0FastConfig
    from .pi05.configuration_pi05 import PI05Config as PI05Config
    from .pretrained import PreTrainedPolicy as PreTrainedPolicy
    from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig
    from .tdmpc.configuration_tdmpc import TDMPCConfig as TDMPCConfig
    from .utils import make_robot_action, prepare_observation_for_inference
    from .vla_jepa.configuration_vla_jepa import VLAJEPAConfig as VLAJEPAConfig
    from .vqbet.configuration_vqbet import VQBeTConfig as VQBeTConfig
    from .wall_x.configuration_wall_x import WallXConfig as WallXConfig
    from .xvla.configuration_xvla import XVLAConfig as XVLAConfig
else:
    __getattr__, __dir__ = lazy_exports(
        __name__,
        {
            "ActionInterpolator": "lerobot.utils.action_interpolator.ActionInterpolator",
            "ACTConfig": ".act.configuration_act.ACTConfig",
            "DiffusionConfig": ".diffusion.configuration_diffusion.DiffusionConfig",
            "EO1Config": ".eo1.configuration_eo1.EO1Config",
            "Evo1Config": ".evo1.configuration_evo1.Evo1Config",
            "get_policy_class": ".factory.get_policy_class",
            "make_policy": ".factory.make_policy",
            "make_policy_config": ".factory.make_policy_config",
            "make_pre_post_processors": ".factory.make_pre_post_processors",
            "FastWAMConfig": ".fastwam.configuration_fastwam.FastWAMConfig",
            "Flux3Config": ".flux3.configuration_flux3.Flux3Config",
            "GaussianActorConfig": ".gaussian_actor.configuration_gaussian_actor.GaussianActorConfig",
            "GrootConfig": ".groot.configuration_groot.GrootConfig",
            "LaWAMConfig": ".lawam.configuration_lawam.LaWAMConfig",
            "LingBotVAConfig": ".lingbot_va.configuration_lingbot_va.LingBotVAConfig",
            "MolmoAct2Config": ".molmoact2.configuration_molmoact2.MolmoAct2Config",
            "MultiTaskDiTConfig": ".multi_task_dit.configuration_multi_task_dit.MultiTaskDiTConfig",
            "PI0Config": ".pi0.configuration_pi0.PI0Config",
            "PI0FastConfig": ".pi0_fast.configuration_pi0_fast.PI0FastConfig",
            "PI05Config": ".pi05.configuration_pi05.PI05Config",
            "PreTrainedPolicy": ".pretrained.PreTrainedPolicy",
            "SmolVLAConfig": ".smolvla.configuration_smolvla.SmolVLAConfig",
            "TDMPCConfig": ".tdmpc.configuration_tdmpc.TDMPCConfig",
            "make_robot_action": ".utils.make_robot_action",
            "prepare_observation_for_inference": ".utils.prepare_observation_for_inference",
            "VLAJEPAConfig": ".vla_jepa.configuration_vla_jepa.VLAJEPAConfig",
            "VQBeTConfig": ".vqbet.configuration_vqbet.VQBeTConfig",
            "WallXConfig": ".wall_x.configuration_wall_x.WallXConfig",
            "XVLAConfig": ".xvla.configuration_xvla.XVLAConfig",
        },
    )

# NOTE: Policy modeling classes (e.g., GaussianActorPolicy) are intentionally NOT re-exported here.
# They have heavy optional dependencies and are loaded lazily via get_policy_class().
# Import directly: ``from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy``

__all__ = [
    # Configuration classes
    "ACTConfig",
    "DiffusionConfig",
    "EO1Config",
    "FastWAMConfig",
    "Flux3Config",
    "GaussianActorConfig",
    "Evo1Config",
    "GrootConfig",
    "LaWAMConfig",
    "LingBotVAConfig",
    "MolmoAct2Config",
    "MultiTaskDiTConfig",
    "PI0Config",
    "PI0FastConfig",
    "PI05Config",
    "SmolVLAConfig",
    "TDMPCConfig",
    "VLAJEPAConfig",
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
