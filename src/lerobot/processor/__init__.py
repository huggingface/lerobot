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

# Lazy module: submodules are imported only when their names are first accessed.
# This avoids pulling heavy dependencies into lightweight hardware workflows.

_SUBMODULE_ATTRS = {
    "lerobot.types": [
        "EnvAction",
        "EnvTransition",
        "PolicyAction",
        "TransitionKey",
    ],
    ".types": [
        "RobotAction",
        "RobotObservation",
    ],
    ".batch_processor": [
        "AddBatchDimensionProcessorStep",
    ],
    ".converters": [
        "batch_to_transition",
        "create_transition",
        "from_tensor_to_numpy",
        "identity_transition",
        "observation_to_transition",
        "policy_action_to_transition",
        "robot_action_observation_to_transition",
        "robot_action_to_transition",
        "transition_to_batch",
        "transition_to_observation",
        "transition_to_policy_action",
        "transition_to_robot_action",
    ],
    ".delta_action_processor": [
        "MapDeltaActionToRobotActionStep",
        "MapTensorToDeltaActionDictStep",
    ],
    ".device_processor": [
        "DeviceProcessorStep",
    ],
    ".env_processor": [
        "IsaaclabArenaProcessorStep",
        "LiberoProcessorStep",
    ],
    ".factory": [
        "make_default_processors",
        "make_default_robot_action_processor",
        "make_default_robot_observation_processor",
        "make_default_teleop_action_processor",
    ],
    ".gym_action_processor": [
        "Numpy2TorchActionProcessorStep",
        "Torch2NumpyActionProcessorStep",
    ],
    ".hil_processor": [
        "AddTeleopActionAsComplimentaryDataStep",
        "AddTeleopEventsAsInfoStep",
        "GripperPenaltyProcessorStep",
        "GymHILAdapterProcessorStep",
        "ImageCropResizeProcessorStep",
        "InterventionActionProcessorStep",
        "RewardClassifierProcessorStep",
        "TimeLimitProcessorStep",
    ],
    ".newline_task_processor": [
        "NewLineTaskProcessorStep",
    ],
    ".normalize_processor": [
        "NormalizerProcessorStep",
        "UnnormalizerProcessorStep",
        "hotswap_stats",
    ],
    ".observation_processor": [
        "VanillaObservationProcessorStep",
    ],
    ".pipeline": [
        "ActionProcessorStep",
        "ComplementaryDataProcessorStep",
        "DataProcessorPipeline",
        "DoneProcessorStep",
        "IdentityProcessorStep",
        "InfoProcessorStep",
        "ObservationProcessorStep",
        "PolicyActionProcessorStep",
        "PolicyProcessorPipeline",
        "ProcessorKwargs",
        "ProcessorStep",
        "ProcessorStepRegistry",
        "RewardProcessorStep",
        "RobotActionProcessorStep",
        "RobotProcessorPipeline",
        "TruncatedProcessorStep",
    ],
    ".policy_robot_bridge": [
        "PolicyActionToRobotActionProcessorStep",
        "RobotActionToPolicyActionProcessorStep",
    ],
    ".relative_action_processor": [
        "AbsoluteActionsProcessorStep",
        "RelativeActionsProcessorStep",
        "to_absolute_actions",
        "to_relative_actions",
    ],
    ".rename_processor": [
        "RenameObservationsProcessorStep",
        "rename_stats",
    ],
    ".tokenizer_processor": [
        "ActionTokenizerProcessorStep",
        "TokenizerProcessorStep",
    ],
}

_ATTR_TO_MODULE: dict[str, str] = {}
for _mod, _attrs in _SUBMODULE_ATTRS.items():
    for _attr in _attrs:
        _ATTR_TO_MODULE[_attr] = _mod


def __getattr__(name: str):
    if name in _ATTR_TO_MODULE:
        import importlib

        mod = importlib.import_module(_ATTR_TO_MODULE[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_ATTR_TO_MODULE.keys())
