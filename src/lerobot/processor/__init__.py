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

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import lazy_exports

# Many of these import torch, so each is imported the first time it is used.
if TYPE_CHECKING:
    from lerobot.lerobot_types import (
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
        DefaultPolicyProcessorSteps,
        load_pretrained_policy_processors,
        make_default_policy_processor_steps,
        make_default_pre_post_processors,
        make_default_processors,
        make_default_robot_action_processor,
        make_default_robot_observation_processor,
        make_default_teleop_action_processor,
        make_policy_processor_pipelines,
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
        bind_relative_anchor,
        to_absolute_actions,
        to_relative_actions,
    )
    from .rename_processor import RenameObservationsProcessorStep, rename_stats
    from .render_messages_processor import RenderRuntimeMessagesStep, RenderTrainingMessagesStep
    from .tokenizer_processor import ActionTokenizerProcessorStep, TokenizerProcessorStep
else:
    __getattr__, __dir__ = lazy_exports(
        __name__,
        {
            "EnvAction": "lerobot.lerobot_types.EnvAction",
            "EnvTransition": "lerobot.lerobot_types.EnvTransition",
            "PolicyAction": "lerobot.lerobot_types.PolicyAction",
            "RobotAction": "lerobot.lerobot_types.RobotAction",
            "RobotObservation": "lerobot.lerobot_types.RobotObservation",
            "TransitionKey": "lerobot.lerobot_types.TransitionKey",
            "AddBatchDimensionProcessorStep": ".batch_processor.AddBatchDimensionProcessorStep",
            "batch_to_transition": ".converters.batch_to_transition",
            "create_transition": ".converters.create_transition",
            "from_tensor_to_numpy": ".converters.from_tensor_to_numpy",
            "identity_transition": ".converters.identity_transition",
            "observation_to_transition": ".converters.observation_to_transition",
            "policy_action_to_transition": ".converters.policy_action_to_transition",
            "robot_action_observation_to_transition": ".converters.robot_action_observation_to_transition",
            "robot_action_to_transition": ".converters.robot_action_to_transition",
            "transition_to_batch": ".converters.transition_to_batch",
            "transition_to_observation": ".converters.transition_to_observation",
            "transition_to_policy_action": ".converters.transition_to_policy_action",
            "transition_to_robot_action": ".converters.transition_to_robot_action",
            "MapDeltaActionToRobotActionStep": ".delta_action_processor.MapDeltaActionToRobotActionStep",
            "MapTensorToDeltaActionDictStep": ".delta_action_processor.MapTensorToDeltaActionDictStep",
            "DeviceProcessorStep": ".device_processor.DeviceProcessorStep",
            "IsaaclabArenaProcessorStep": ".env_processor.IsaaclabArenaProcessorStep",
            "LiberoProcessorStep": ".env_processor.LiberoProcessorStep",
            "DefaultPolicyProcessorSteps": ".factory.DefaultPolicyProcessorSteps",
            "load_pretrained_policy_processors": ".factory.load_pretrained_policy_processors",
            "make_default_policy_processor_steps": ".factory.make_default_policy_processor_steps",
            "make_default_pre_post_processors": ".factory.make_default_pre_post_processors",
            "make_default_processors": ".factory.make_default_processors",
            "make_default_robot_action_processor": ".factory.make_default_robot_action_processor",
            "make_default_robot_observation_processor": ".factory.make_default_robot_observation_processor",
            "make_default_teleop_action_processor": ".factory.make_default_teleop_action_processor",
            "make_policy_processor_pipelines": ".factory.make_policy_processor_pipelines",
            "Numpy2TorchActionProcessorStep": ".gym_action_processor.Numpy2TorchActionProcessorStep",
            "Torch2NumpyActionProcessorStep": ".gym_action_processor.Torch2NumpyActionProcessorStep",
            "AddTeleopActionAsComplimentaryDataStep": ".hil_processor.AddTeleopActionAsComplimentaryDataStep",
            "AddTeleopEventsAsInfoStep": ".hil_processor.AddTeleopEventsAsInfoStep",
            "GripperPenaltyProcessorStep": ".hil_processor.GripperPenaltyProcessorStep",
            "GymHILAdapterProcessorStep": ".hil_processor.GymHILAdapterProcessorStep",
            "ImageCropResizeProcessorStep": ".hil_processor.ImageCropResizeProcessorStep",
            "InterventionActionProcessorStep": ".hil_processor.InterventionActionProcessorStep",
            "RewardClassifierProcessorStep": ".hil_processor.RewardClassifierProcessorStep",
            "TimeLimitProcessorStep": ".hil_processor.TimeLimitProcessorStep",
            "NewLineTaskProcessorStep": ".newline_task_processor.NewLineTaskProcessorStep",
            "NormalizerProcessorStep": ".normalize_processor.NormalizerProcessorStep",
            "UnnormalizerProcessorStep": ".normalize_processor.UnnormalizerProcessorStep",
            "hotswap_stats": ".normalize_processor.hotswap_stats",
            "VanillaObservationProcessorStep": ".observation_processor.VanillaObservationProcessorStep",
            "ActionProcessorStep": ".pipeline.ActionProcessorStep",
            "ComplementaryDataProcessorStep": ".pipeline.ComplementaryDataProcessorStep",
            "DataProcessorPipeline": ".pipeline.DataProcessorPipeline",
            "DoneProcessorStep": ".pipeline.DoneProcessorStep",
            "IdentityProcessorStep": ".pipeline.IdentityProcessorStep",
            "InfoProcessorStep": ".pipeline.InfoProcessorStep",
            "ObservationProcessorStep": ".pipeline.ObservationProcessorStep",
            "PolicyActionProcessorStep": ".pipeline.PolicyActionProcessorStep",
            "PolicyProcessorPipeline": ".pipeline.PolicyProcessorPipeline",
            "ProcessorKwargs": ".pipeline.ProcessorKwargs",
            "ProcessorStep": ".pipeline.ProcessorStep",
            "ProcessorStepRegistry": ".pipeline.ProcessorStepRegistry",
            "RewardProcessorStep": ".pipeline.RewardProcessorStep",
            "RobotActionProcessorStep": ".pipeline.RobotActionProcessorStep",
            "RobotProcessorPipeline": ".pipeline.RobotProcessorPipeline",
            "TruncatedProcessorStep": ".pipeline.TruncatedProcessorStep",
            "PolicyActionToRobotActionProcessorStep": ".policy_robot_bridge.PolicyActionToRobotActionProcessorStep",
            "RobotActionToPolicyActionProcessorStep": ".policy_robot_bridge.RobotActionToPolicyActionProcessorStep",
            "AbsoluteActionsProcessorStep": ".relative_action_processor.AbsoluteActionsProcessorStep",
            "RelativeActionsProcessorStep": ".relative_action_processor.RelativeActionsProcessorStep",
            "bind_relative_anchor": ".relative_action_processor.bind_relative_anchor",
            "to_absolute_actions": ".relative_action_processor.to_absolute_actions",
            "to_relative_actions": ".relative_action_processor.to_relative_actions",
            "RenameObservationsProcessorStep": ".rename_processor.RenameObservationsProcessorStep",
            "rename_stats": ".rename_processor.rename_stats",
            "RenderRuntimeMessagesStep": ".render_messages_processor.RenderRuntimeMessagesStep",
            "RenderTrainingMessagesStep": ".render_messages_processor.RenderTrainingMessagesStep",
            "ActionTokenizerProcessorStep": ".tokenizer_processor.ActionTokenizerProcessorStep",
            "TokenizerProcessorStep": ".tokenizer_processor.TokenizerProcessorStep",
        },
    )

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
    "DefaultPolicyProcessorSteps",
    "load_pretrained_policy_processors",
    "make_default_policy_processor_steps",
    "make_default_pre_post_processors",
    "make_default_processors",
    "make_default_teleop_action_processor",
    "make_default_robot_action_processor",
    "make_default_robot_observation_processor",
    "make_policy_processor_pipelines",
    "AbsoluteActionsProcessorStep",
    "RelativeActionsProcessorStep",
    "bind_relative_anchor",
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
    "RenderRuntimeMessagesStep",
    "RenderTrainingMessagesStep",
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
