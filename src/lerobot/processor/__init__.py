#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.types import (
    EnvAction,
    EnvTransition,
    PolicyAction,
    RobotAction,
    RobotObservation,
    TransitionKey,
)

from .batch_processor import AddBatchDimensionProcessorStep
from .converters import (
    batch_to_transition,
    create_transition,
    from_tensor_to_numpy,
    identity_transition,
    observation_to_transition,
    policy_action_to_transition,
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_batch,
    transition_to_observation,
    transition_to_policy_action,
    transition_to_robot_action,
)
from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .device_processor import DeviceProcessorStep
from .env_processor import IsaaclabArenaProcessorStep, LiberoProcessorStep
from .factory import (
    make_default_processors,
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
    make_default_teleop_action_processor,
)
from .gym_action_processor import (
    Numpy2TorchActionProcessorStep,
    Torch2NumpyActionProcessorStep,
)
from .hil_processor import (
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    GripperPenaltyProcessorStep,
    GymHILAdapterProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    RewardClassifierProcessorStep,
    TimeLimitProcessorStep,
)
from .newline_task_processor import NewLineTaskProcessorStep
from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep, hotswap_stats
from .observation_processor import VanillaObservationProcessorStep
from .pipeline import (
    ActionProcessorStep,
    ComplementaryDataProcessorStep,
    DataProcessorPipeline,
    DoneProcessorStep,
    IdentityProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorKwargs,
    ProcessorStep,
    ProcessorStepRegistry,
    RewardProcessorStep,
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    TruncatedProcessorStep,
)
from .policy_robot_bridge import (
    PolicyActionToRobotActionProcessorStep,
    RobotActionToPolicyActionProcessorStep,
)
from .relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    to_absolute_actions,
    to_relative_actions,
)
from .rename_processor import RenameObservationsProcessorStep, rename_stats
from .tokenizer_processor import ActionTokenizerProcessorStep, TokenizerProcessorStep

__all__ = [
    "ActionProcessorStep",
    "AddTeleopActionAsComplimentaryDataStep",
    "AddTeleopEventsAsInfoStep",
    "ComplementaryDataProcessorStep",
    "batch_to_transition",
    "create_transition",
    "from_tensor_to_numpy",
    "identity_transition",
    "observation_to_transition",
    "policy_action_to_transition",
    "robot_action_observation_to_transition",
    "robot_action_to_transition",
    "transition_to_observation",
    "transition_to_policy_action",
    "transition_to_robot_action",
    "DeviceProcessorStep",
    "DoneProcessorStep",
    "EnvAction",
    "EnvTransition",
    "GymHILAdapterProcessorStep",
    "GripperPenaltyProcessorStep",
    "hotswap_stats",
    "IdentityProcessorStep",
    "ImageCropResizeProcessorStep",
    "InfoProcessorStep",
    "InterventionActionProcessorStep",
    "make_default_processors",
    "make_default_teleop_action_processor",
    "make_default_robot_action_processor",
    "make_default_robot_observation_processor",
    "AbsoluteActionsProcessorStep",
    "RelativeActionsProcessorStep",
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "NewLineTaskProcessorStep",
    "NormalizerProcessorStep",
    "Numpy2TorchActionProcessorStep",
    "ObservationProcessorStep",
    "PolicyAction",
    "PolicyActionProcessorStep",
    "PolicyProcessorPipeline",
    "ProcessorKwargs",
    "ProcessorStep",
    "ProcessorStepRegistry",
    "RobotAction",
    "RobotActionProcessorStep",
    "RobotObservation",
    "rename_stats",
    "RenameObservationsProcessorStep",
    "RewardClassifierProcessorStep",
    "RewardProcessorStep",
    "DataProcessorPipeline",
    "IsaaclabArenaProcessorStep",
    "LiberoProcessorStep",
    "TimeLimitProcessorStep",
    "AddBatchDimensionProcessorStep",
    "RobotProcessorPipeline",
    "TokenizerProcessorStep",
    "ActionTokenizerProcessorStep",
    "Torch2NumpyActionProcessorStep",
    "RobotActionToPolicyActionProcessorStep",
    "PolicyActionToRobotActionProcessorStep",
    "transition_to_batch",
    "TransitionKey",
    "TruncatedProcessorStep",
    "to_absolute_actions",
    "to_relative_actions",
    "UnnormalizerProcessorStep",
    "VanillaObservationProcessorStep",
]
