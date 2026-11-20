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
from lerobot.utils.feature_spec import FeatureSpec

from .act.configuration_act import ACTConfig as ACTConfig
from .chunk import ChunkPolicySpec, ExecutionMode
def _safe_import(module_path: str, class_name: str):
    try:
        import importlib
        mod = importlib.import_module(module_path, package=__package__)
        return getattr(mod, class_name)
    except Exception:
        return None

Cosmos3EdgeConfig = _safe_import(".cosmos3.configuration_cosmos3", "Cosmos3EdgeConfig")
Cosmos3NanoConfig = _safe_import(".cosmos3.configuration_cosmos3", "Cosmos3NanoConfig")
DiffusionConfig = _safe_import(".diffusion.configuration_diffusion", "DiffusionConfig")
DM05Config = _safe_import(".dm05.configuration_dm05", "DM05Config")
EO1Config = _safe_import(".eo1.configuration_eo1", "EO1Config")
Evo1Config = _safe_import(".evo1.configuration_evo1", "Evo1Config")
FastWAMConfig = _safe_import(".fastwam.configuration_fastwam", "FastWAMConfig")
FineARTVLAConfig = _safe_import(".fineart_vla.configuration_fineart_vla", "FineARTVLAConfig")
Flux3Config = _safe_import(".flux3.configuration_flux3", "Flux3Config")
G05Config = _safe_import(".g05.configuration_g05", "G05Config")
GaussianActorConfig = _safe_import(".gaussian_actor.configuration_gaussian_actor", "GaussianActorConfig")
GrootConfig = _safe_import(".groot.configuration_groot", "GrootConfig")
LaWAMConfig = _safe_import(".lawam.configuration_lawam", "LaWAMConfig")
LingBotVAConfig = _safe_import(".lingbot_va.configuration_lingbot_va", "LingBotVAConfig")
LingbotVLAV2Config = _safe_import(".lingbot_vla_v2.configuration_lingbot_vla_v2", "LingbotVLAV2Config")
LongWAMConfig = _safe_import(".longwam.configuration_longwam", "LongWAMConfig")
MolmoAct2Config = _safe_import(".molmoact2.configuration_molmoact2", "MolmoAct2Config")
MultiTaskDiTConfig = _safe_import(".multi_task_dit.configuration_multi_task_dit", "MultiTaskDiTConfig")
PI0Config = _safe_import(".pi0.configuration_pi0", "PI0Config")
PI0FastConfig = _safe_import(".pi0_fast.configuration_pi0_fast", "PI0FastConfig")
PI05Config = _safe_import(".pi05.configuration_pi05", "PI05Config")
PraxisConfig = _safe_import(".praxis.configuration_praxis", "PraxisConfig")
SmolVLAConfig = _safe_import(".smolvla.configuration_smolvla", "SmolVLAConfig")
TDMPCConfig = _safe_import(".tdmpc.configuration_tdmpc", "TDMPCConfig")
VLAJEPAConfig = _safe_import(".vla_jepa.configuration_vla_jepa", "VLAJEPAConfig")

from .factory import get_policy_class, make_policy, make_policy_config, make_pre_post_processors
from .pretrained import PreTrainedPolicy as PreTrainedPolicy
from .rtc import (
    ActionQueue,
    QueueSnapshot,
    RTCConfig,
    reanchor_relative_rtc_prefix,
    validate_trained_rtc_horizon,
)
from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig
from .tdmpc.configuration_tdmpc import TDMPCConfig as TDMPCConfig
from .utils import make_robot_action, prepare_observation_for_inference
from .vla_jepa.configuration_vla_jepa import VLAJEPAConfig as VLAJEPAConfig
VQBeTConfig = _safe_import(".vqbet.configuration_vqbet", "VQBeTConfig")
WallOSS05Config = _safe_import(".wall_oss_05.configuration_wall_oss_05", "WallOSS05Config")
WallXConfig = _safe_import(".wall_x.configuration_wall_x", "WallXConfig")
XVLAConfig = _safe_import(".xvla.configuration_xvla", "XVLAConfig")

# NOTE: Policy modeling classes (e.g., GaussianActorPolicy) are intentionally NOT re-exported here.
# They have heavy optional dependencies and are loaded lazily via get_policy_class().
# Import directly: ``from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy``

__all__ = [
    # Configuration classes
    "ACTConfig",
    "Cosmos3EdgeConfig",
    "Cosmos3NanoConfig",
    "DiffusionConfig",
    "DM05Config",
    "EO1Config",
    "FastWAMConfig",
    "FineARTVLAConfig",
    "Flux3Config",
    "GaussianActorConfig",
    "G05Config",
    "Evo1Config",
    "GrootConfig",
    "LaWAMConfig",
    "LingBotVAConfig",
    "LingbotVLAV2Config",
    "LongWAMConfig",
    "MolmoAct2Config",
    "MultiTaskDiTConfig",
    "PI0Config",
    "PI0FastConfig",
    "PI05Config",
    "PraxisConfig",
    "SmolVLAConfig",
    "TDMPCConfig",
    "VLAJEPAConfig",
    "VQBeTConfig",
    "WallXConfig",
    "WallOSS05Config",
    "XVLAConfig",
    # Chunk execution contracts
    "ChunkPolicySpec",
    "ExecutionMode",
    "FeatureSpec",
    # Base class
    "PreTrainedPolicy",
    # RTC utilities
    "ActionInterpolator",
    "ActionQueue",
    "QueueSnapshot",
    "RTCConfig",
    "reanchor_relative_rtc_prefix",
    "validate_trained_rtc_horizon",
    # Utility functions
    "make_robot_action",
    "prepare_observation_for_inference",
    # Factory functions
    "get_policy_class",
    "make_policy",
    "make_policy_config",
    "make_pre_post_processors",
]
