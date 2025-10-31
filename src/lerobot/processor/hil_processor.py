#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812
from lerobot.model.kinematics import RobotKinematics

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.rotation import Rotation

from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import (
    ComplementaryDataProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TruncatedProcessorStep,
)

GRIPPER_KEY = "gripper"
DISCRETE_PENALTY_KEY = "discrete_penalty"
TELEOP_ACTION_KEY = "teleop_action"


@runtime_checkable
class HasTeleopEvents(Protocol):
    """
    Minimal protocol for objects that provide teleoperation events.

    This protocol defines the `get_teleop_events()` method, allowing processor
    steps to interact with teleoperators that support event-based controls
    (like episode termination or success flagging) without needing to know the
    teleoperator's specific class.
    """

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the teleoperator.

        Returns:
            A dictionary containing control events such as:
            - `is_intervention`: bool - Whether the human is currently intervening.
            - `terminate_episode`: bool - Whether to terminate the current episode.
            - `success`: bool - Whether the episode was successful.
            - `rerecord_episode`: bool - Whether to rerecord the episode.
        """
        ...


# Type variable constrained to Teleoperator subclasses that also implement events
TeleopWithEvents = TypeVar("TeleopWithEvents", bound=Teleoperator)


def _check_teleop_with_events(teleop: Teleoperator) -> None:
    """
    Runtime check that a teleoperator implements the `HasTeleopEvents` protocol.

    Args:
        teleop: The teleoperator instance to check.

    Raises:
        TypeError: If the teleoperator does not have a `get_teleop_events` method.
    """
    if not isinstance(teleop, HasTeleopEvents):
        raise TypeError(
            f"Teleoperator {type(teleop).__name__} must implement get_teleop_events() method. "
            f"Compatible teleoperators: GamepadTeleop, KeyboardEndEffectorTeleop"
        )


@ProcessorStepRegistry.register("add_teleop_action_as_complementary_data")
@dataclass
class AddTeleopActionAsComplimentaryDataStep(ComplementaryDataProcessorStep):
    """
    Adds the raw action from a teleoperator to the transition's complementary data.

    This is useful for human-in-the-loop scenarios where the human's input needs to
    be available to downstream processors, for example, to override a policy's action
    during an intervention.

    Attributes:
        teleop_device: The teleoperator instance to get the action from.
    """

    teleop_device: Teleoperator

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        Retrieves the teleoperator's action and adds it to the complementary data.

        Args:
            complementary_data: The incoming complementary data dictionary.

        Returns:
            A new dictionary with the teleoperator action added under the
            `teleop_action` key.
        """
        new_complementary_data = dict(complementary_data)
        new_complementary_data[TELEOP_ACTION_KEY] = self.teleop_device.get_action()
        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("add_teleop_action_as_info")
@dataclass
class AddTeleopEventsAsInfoStep(InfoProcessorStep):
    """
    Adds teleoperator control events (e.g., terminate, success) to the transition's info.

    This step extracts control events from teleoperators that support event-based
    interaction, making these signals available to other parts of the system.

    Attributes:
        teleop_device: An instance of a teleoperator that implements the
                       `HasTeleopEvents` protocol.
    """

    teleop_device: TeleopWithEvents
    _debug_frame_count: int = 0
    _last_space_state: bool = False

    def __post_init__(self):
        """Validates that the provided teleoperator supports events after initialization."""
        _check_teleop_with_events(self.teleop_device)

    def info(self, info: dict) -> dict:
        """
        Retrieves teleoperator events and updates the info dictionary.

        Args:
            info: The incoming info dictionary.

        Returns:
            A new dictionary including the teleoperator events.
        """
        self._debug_frame_count += 1

        teleop_events = self.teleop_device.get_teleop_events()

        if self._debug_frame_count % 30 == 30:  # Disable now
            print(
                f"\n=== DEEP DEBUG TELEOP EVENTS (Frame {self._debug_frame_count}) ==="
            )
            print(f"1. Raw teleop_events: {teleop_events}")
            print("2. Teleop device details:")
            print(f"   Type: {type(self.teleop_device)}")
            print(f"   Module: {self.teleop_device.__class__.__module__}")

            print("3. Space key detection (multiple methods):")

            if hasattr(self.teleop_device, "is_space_pressed"):
                space_pressed = self.teleop_device.is_space_pressed()
                print(f"   is_space_pressed(): {space_pressed}")
            else:
                print("   is_space_pressed(): Method not available")

            if hasattr(self.teleop_device, "is_intervention_triggered"):
                intervention_triggered = self.teleop_device.is_intervention_triggered()
                print(f"   is_intervention_triggered(): {intervention_triggered}")
            else:
                print("   is_intervention_triggered(): Method not available")

            if hasattr(self.teleop_device, "get_state"):
                state = self.teleop_device.get_state()
                print(f"   get_state(): {state}")
            else:
                print("   get_state(): Method not available")

            if hasattr(self.teleop_device, "buttons"):
                print(f"   buttons: {self.teleop_device.buttons}")
            else:
                print("   buttons: Attribute not available")

            if hasattr(self.teleop_device, "key_events"):
                print(f"   key_events: {self.teleop_device.key_events}")
            else:
                print("   key_events: Attribute not available")

            print("=== END DEEP DEBUG ===\n")

        new_info = dict(info)
        new_info.update(teleop_events)

        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("image_crop_resize_processor")
@dataclass
class ImageCropResizeProcessorStep(ObservationProcessorStep):
    """
    Crops and/or resizes image observations.

    This step iterates through all image keys in an observation dictionary and applies
    the specified transformations. It handles device placement, moving tensors to the
    CPU if necessary for operations not supported on certain accelerators like MPS.

    Attributes:
        crop_params_dict: A dictionary mapping image keys to cropping parameters
                          (top, left, height, width).
        resize_size: A tuple (height, width) to resize all images to.
    """

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None

    def observation(self, observation: dict) -> dict:
        """
        Applies cropping and resizing to all images in the observation dictionary.

        Args:
            observation: The observation dictionary, potentially containing image tensors.

        Returns:
            A new observation dictionary with transformed images.
        """
        if self.resize_size is None and not self.crop_params_dict:
            return observation

        new_observation = dict(observation)

        # Process all image keys in the observation
        for key in observation:
            if "image" not in key:
                continue

            image = observation[key]
            device = image.device
            # NOTE (maractingi): No mps kernel for crop and resize, so we need to move to cpu
            if device.type == "mps":
                image = image.cpu()
            # Crop if crop params are provided for this key
            if self.crop_params_dict is not None and key in self.crop_params_dict:
                crop_params = self.crop_params_dict[key]
                image = F.crop(image, *crop_params)
            if self.resize_size is not None:
                image = F.resize(image, self.resize_size)
                image = image.clamp(0.0, 1.0)
            new_observation[key] = image.to(device)

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary with the crop parameters and resize dimensions.
        """
        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the image feature shapes in the policy features dictionary if resizing is applied.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary with new image shapes.
        """
        if self.resize_size is None:
            return features
        for key in features[PipelineFeatureType.OBSERVATION]:
            if "image" in key:
                nb_channel = features[PipelineFeatureType.OBSERVATION][key].shape[0]
                features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                    type=features[PipelineFeatureType.OBSERVATION][key].type,
                    shape=(nb_channel, *self.resize_size),
                )
        return features


@dataclass
@ProcessorStepRegistry.register("time_limit_processor")
class TimeLimitProcessorStep(TruncatedProcessorStep):
    """
    Tracks episode steps and enforces a time limit by truncating the episode.

    Attributes:
        max_episode_steps: The maximum number of steps allowed per episode.
        current_step: The current step count for the active episode.
    """

    max_episode_steps: int
    current_step: int = 0

    def truncated(self, truncated: bool) -> bool:
        """
        Increments the step counter and sets the truncated flag if the time limit is reached.

        Args:
            truncated: The incoming truncated flag.

        Returns:
            True if the episode step limit is reached, otherwise the incoming value.
        """
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        # TODO (steven): missing an else truncated = False?
        return truncated

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the `max_episode_steps`.
        """
        return {
            "max_episode_steps": self.max_episode_steps,
        }

    def reset(self) -> None:
        """Resets the step counter, typically called at the start of a new episode."""
        self.current_step = 0

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("gripper_penalty_processor")
class GripperPenaltyProcessorStep(ComplementaryDataProcessorStep):
    """
    Applies a penalty for inefficient gripper usage.

    This step penalizes actions that attempt to close an already closed gripper or
    open an already open one, based on position thresholds.

    Attributes:
        penalty: The negative reward value to apply.
        max_gripper_pos: The maximum position value for the gripper, used for normalization.
    """

    penalty: float = -0.01
    max_gripper_pos: float = 30.0

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        Calculates the gripper penalty and adds it to the complementary data.

        Args:
            complementary_data: The incoming complementary data, which should contain
                                raw joint positions.

        Returns:
            A new complementary data dictionary with the `discrete_penalty` key added.
        """
        action = self.transition.get(TransitionKey.ACTION)

        raw_joint_positions = complementary_data.get("raw_joint_positions")
        if raw_joint_positions is None:
            return complementary_data

        current_gripper_pos = raw_joint_positions.get(GRIPPER_KEY, None)
        if current_gripper_pos is None:
            return complementary_data

        # Gripper action is a PolicyAction at this stage
        gripper_action = action[-1].item()
        gripper_action_normalized = gripper_action / self.max_gripper_pos

        # Normalize gripper state and action
        gripper_state_normalized = current_gripper_pos / self.max_gripper_pos

        # Calculate penalty boolean as in original
        gripper_penalty_bool = (gripper_state_normalized < 0.5 and gripper_action_normalized > 0.5) or (
            gripper_state_normalized > 0.75 and gripper_action_normalized < 0.5
        )

        gripper_penalty = self.penalty * int(gripper_penalty_bool)

        # Create new complementary data with penalty info
        new_complementary_data = dict(complementary_data)
        new_complementary_data[DISCRETE_PENALTY_KEY] = gripper_penalty

        return new_complementary_data

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the penalty value and max gripper position.
        """
        return {
            "penalty": self.penalty,
            "max_gripper_pos": self.max_gripper_pos,
        }

    def reset(self) -> None:
        """Resets the processor's internal state."""
        pass

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("intervention_action_processor")
class InterventionActionProcessorStep(ProcessorStep):
    """
    Handles human intervention, overriding policy actions and managing episode termination.

    When an intervention is detected (via teleoperator events in the `info` dict),
    this step replaces the policy's action with the human's teleoperated action.
    It also processes signals to terminate the episode or flag success.

    Attributes:
        use_gripper: Whether to include the gripper in the teleoperated action.
        terminate_on_success: If True, automatically sets the `done` flag when a
                              `success` event is received.
    """

    use_gripper: bool = False
    terminate_on_success: bool = True
    _debug_frame_count: int = 0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Processes the transition to handle interventions.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition, potentially with an overridden action, updated
            reward, and termination status.
        """
        self._debug_frame_count += 1
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        # Get intervention signals from complementary data
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        if self._debug_frame_count % 30 == 1:  # fps=30
            print(
                f"\n=== DEBUG INTERVENTION PROCESSOR (Frame {self._debug_frame_count}) ==="
            )
            print(
                f"Input action type: {type(action)}, shape: {getattr(action, 'shape', 'No shape')}"
            )
            print(f"Info keys: {list(info.keys())}")
            print(f"Complementary data keys: {list(complementary_data.keys())}")
            print(f"Teleop action type: {type(teleop_action)}, value: {teleop_action}")
            print("Intervention signals:")
            print(f"  IS_INTERVENTION: {is_intervention}")
            print(f"  TERMINATE_EPISODE: {terminate_episode}")
            print(f"  SUCCESS: {success}")
            print(f"  RERECORD_EPISODE: {rerecord_episode}")

        new_transition = transition.copy()

        # Override action if intervention is active
        if is_intervention and teleop_action is not None:
            if isinstance(teleop_action, dict):
                # Convert teleop_action dict to tensor format
                action_list = [
                    teleop_action.get("delta_x", 0.0),
                    teleop_action.get("delta_y", 0.0),
                    teleop_action.get("delta_z", 0.0),
                ]
                if self.use_gripper:
                    action_list.append(teleop_action.get(GRIPPER_KEY, 1.0))

                if self._debug_frame_count % 30 == 1:
                    print(f"Converting teleop dict to list: {action_list}")

            elif isinstance(teleop_action, np.ndarray):
                action_list = teleop_action.tolist()
                if self._debug_frame_count % 30 == 1:
                    print(f"Converting teleop numpy array to list: {action_list}")
            else:
                action_list = teleop_action
                if self._debug_frame_count % 30 == 1:
                    print(f"Using teleop action as-is: {action_list}")

            teleop_action_tensor = torch.tensor(
                action_list, dtype=action.dtype, device=action.device
            )

            if self._debug_frame_count % 30 == 1:
                print("ACTION OVERRIDE:")
                print(f"  Original policy action: {action}")
                print(f"  New teleop action: {teleop_action_tensor}")
                print(f"  Action device: {teleop_action_tensor.device}")
                print(f"  Action dtype: {teleop_action_tensor.dtype}")

            new_transition[TransitionKey.ACTION] = teleop_action_tensor

        # Handle episode termination
        original_done = transition.get(TransitionKey.DONE, False)
        new_done = bool(terminate_episode) or (self.terminate_on_success and success)
        new_transition[TransitionKey.DONE] = new_done
        new_transition[TransitionKey.REWARD] = float(success)

        if self._debug_frame_count % 30 == 1 and (original_done != new_done or success):
            print("TERMINATION STATUS:")
            print(f"  Original done: {original_done}")
            print(
                f"  New done: {new_done} (terminate_episode: {terminate_episode}, success: {success})"
            )
            print(f"  Reward set to: {float(success)}")

        # Update info with intervention metadata
        info = new_transition.get(TransitionKey.INFO, {})
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        # Update complementary data with teleop action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        # 最终状态检查
        if self._debug_frame_count % 30 == 1:
            final_action = new_transition.get(TransitionKey.ACTION)
            print("FINAL TRANSITION STATE:")
            print(f"  Final action: {final_action}")
            print(f"  Final action type: {type(final_action)}")
            print(f"  Final done: {new_transition.get(TransitionKey.DONE)}")
            print(f"  Final reward: {new_transition.get(TransitionKey.REWARD)}")
            print(f"  Info[IS_INTERVENTION]: {info.get(TeleopEvents.IS_INTERVENTION)}")
            print("=== END DEBUG ===\n")

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "use_gripper": self.use_gripper,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("leader_arm_intervention")
class LeaderArmInterventionProcessorStep(ProcessorStep):
    """
    Leader arm intervention with manual position synchronization.
    User manually moves leader arm to match follower position before teleoperation.
    Now supports 7-dimensional action: [x, y, z, rx, ry, rz, gripper] using kinematics
    """

    use_gripper: bool = False
    terminate_on_success: bool = True

    def __init__(
        self,
        use_gripper: bool = False,
        terminate_on_success: bool = True,
        sync_tolerances: dict | None = None,
        kinematics_solver: RobotKinematics | None = None,
        motor_names: list[str] | None = None,
    ):
        self.use_gripper = use_gripper
        self.terminate_on_success = terminate_on_success
        self.kinematics_solver = kinematics_solver
        self.motor_names = motor_names or []

        # Position tolerance in degrees
        self.sync_tolerances = sync_tolerances or {
            "shoulder_pan.pos": 5.0,
            "shoulder_lift.pos": 10.0,
            "elbow_flex.pos": 10.0,
            "wrist_flex.pos": 10.0,
            "wrist_roll.pos": 10.0,
            "gripper.pos": 10.0,
        }

        # Movement detection thresholds for sync completion
        self.position_threshold = 0.002  # 2mm in meters
        self.orientation_threshold = 0.01  # Small rotation threshold in radians
        self.gripper_threshold = 1.0  # 1 degree for gripper

        self._debug_frame_count = 0
        self._last_intervention_state = False
        self._is_position_synced = False
        self._follower_reference_positions = None
        self._leader_base_positions = None
        self._leader_base_ee_pose = None  # Store base EE pose for delta calculation
        self._sync_start_time = None
        self._last_leader_ee_pose = (
            None  # Track previous EE pose for movement detection
        )
        self._stable_frames_count = 0  # Count consecutive stable frames
        self._stable_frames_required = (
            10  # Require 10 stable frames for sync completion
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        self._debug_frame_count += 1
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        # Get intervention signals and teleoperation data
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        # IMPORTANT: For LeaderArmInterventionProcessorStep, teleop_action contains
        # joint positions (including gripper) in degrees, NOT delta actions.
        # This is different from other teleop devices like gamepad/keyboard which output
        # direct delta actions [x, y, z, gripper].
        #
        # Leader Arm teleop_action format:
        # {
        #   "shoulder_pan.pos": 45.0,
        #   "shoulder_lift.pos": -30.0,
        #   "elbow_flex.pos": 60.0,
        #   "wrist_flex.pos": -15.0,
        #   "wrist_roll.pos": 20.0,
        #   "gripper.pos": 50.0
        # }
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()

        # Detect intervention state transitions
        intervention_started = is_intervention and not self._last_intervention_state
        intervention_ended = not is_intervention and self._last_intervention_state

        # Handle intervention start - initialize manual position synchronization
        if intervention_started:
            print(
                "LEADER ARM INTERVENTION: Intervention started - initializing manual position synchronization"
            )
            self._initialize_manual_sync(transition)

        # Handle intervention end - reset synchronization state
        if intervention_ended:
            print("LEADER ARM INTERVENTION: Intervention ended")
            self._reset_sync_state()

        self._last_intervention_state = is_intervention

        # Process teleoperation during intervention
        if is_intervention and teleop_action is not None:
            if isinstance(teleop_action, dict) and any(
                ".pos" in key for key in teleop_action.keys()
            ):

                # Check if position synchronization is complete
                if not self._is_position_synced:
                    # Still in synchronization phase - check if user moved leader to match follower
                    sync_complete = self._check_manual_sync_complete(
                        teleop_action, transition
                    )
                    if sync_complete:
                        print(
                            "🎯 LEADER ARM INTERVENTION: Position synchronization COMPLETE!"
                        )
                        print(
                            "🎯 LEADER ARM INTERVENTION: Ready for teleoperation - you can now control the follower"
                        )
                        self._is_position_synced = True
                        self._leader_base_positions = teleop_action.copy()
                        # Calculate base EE pose using forward kinematics
                        if self.kinematics_solver is not None:
                            self._leader_base_ee_pose = (
                                self._compute_ee_pose_from_joints(teleop_action)
                            )
                        self._last_leader_ee_pose = (
                            self._leader_base_ee_pose
                        )  # Initialize tracking

                        # IMPORTANT: Immediately generate an action to move follower to exact leader position
                        # This eliminates the tolerance difference between arms
                        action_list = self._convert_to_exact_sync_action(
                            teleop_action, transition
                        )
                        action_tensor = torch.tensor(
                            action_list, dtype=action.dtype, device=action.device
                        )
                        new_transition[TransitionKey.ACTION] = action_tensor

                        # Update complementary data
                        complementary_data = new_transition.get(
                            TransitionKey.COMPLEMENTARY_DATA, {}
                        )
                        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(
                            TransitionKey.ACTION
                        )
                        new_transition[TransitionKey.COMPLEMENTARY_DATA] = (
                            complementary_data
                        )

                        print(
                            f"🎯 LEADER ARM INTERVENTION: Sending exact sync action to eliminate tolerance difference"
                        )
                    else:
                        # Still synchronizing - return zero action to wait for completion
                        zero_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + (
                            [1.0] if self.use_gripper else [0.0]
                        )
                        zero_tensor = torch.tensor(
                            zero_action, dtype=action.dtype, device=action.device
                        )
                        new_transition[TransitionKey.ACTION] = zero_tensor

                        # Also update complementary data with the tensor action
                        complementary_data = new_transition.get(
                            TransitionKey.COMPLEMENTARY_DATA, {}
                        )
                        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(
                            TransitionKey.ACTION
                        )
                        new_transition[TransitionKey.COMPLEMENTARY_DATA] = (
                            complementary_data
                        )
                else:
                    # 🚨 对比关节运动模式
                    if self._debug_frame_count % 20 == 1:
                        print(f"JOINT MOVEMENT COMPARISON:")
                        print(f"LEADER JOINTS: {[teleop_action.get(f'{name}.pos', 0.0) for name in self.motor_names]}")

                        # 获取follower当前关节位置
                        observation = transition.get(TransitionKey.OBSERVATION, {})
                        follower_joint_positions = {}
                        for motor_name in self.motor_names:
                            for key_suffix in [".pos", "_pos"]:
                                key = motor_name + key_suffix
                                if key in observation:
                                    follower_joint_positions[motor_name] = observation[key]
                                    break

                        # 对比每个关节的变化
                        for motor_name in self.motor_names:
                            leader_current = teleop_action.get(f"{motor_name}.pos", 0.0)
                            leader_base = self._leader_base_positions.get(f"{motor_name}.pos", 0.0)
                            leader_change = leader_current - leader_base

                            follower_current = follower_joint_positions.get(motor_name, 0.0)
                            # 需要记录follower的base位置
                            if not hasattr(self, '_follower_base_positions'):
                                self._follower_base_positions = follower_joint_positions.copy()
                            follower_base = self._follower_base_positions.get(motor_name, 0.0)
                            follower_change = follower_current - follower_base

                            print(f"  {motor_name}: LeaderΔ={leader_change:6.1f}°, FollowerΔ={follower_change:6.1f}°")

                    # Position sync complete - convert to 7D action commands using kinematics
                    action_list = self._convert_to_7d_action_with_kinematics(teleop_action)

                    # 🚨 判断是否发送delta action
                    delta_magnitude = np.linalg.norm(action_list[:3])  # 位置变化幅度
                    rot_magnitude = np.linalg.norm(action_list[3:6])   # 旋转变化幅度

                    if delta_magnitude > 0.005 or rot_magnitude > 0.01:  # 阈值判断
                        # 发送delta action
                        teleop_action_tensor = torch.tensor(
                            action_list, dtype=action.dtype, device=action.device
                        )

                        if self._debug_frame_count % 30 == 1:
                            print(
                                f"LEADER ARM INTERVENTION: Sending 7D command - "
                                f"pos[{teleop_action_tensor[0]:6.3f}, {teleop_action_tensor[1]:6.3f}, {teleop_action_tensor[2]:6.3f}] "
                                f"rot[{teleop_action_tensor[3]:6.3f}, {teleop_action_tensor[4]:6.3f}, {teleop_action_tensor[5]:6.3f}] "
                                f"gripper[{teleop_action_tensor[6]:6.3f}]"
                            )

                        new_transition[TransitionKey.ACTION] = teleop_action_tensor

                        # 🚨 关键：发送action后立即更新base到当前位置
                        self._leader_base_positions = teleop_action.copy()
                        if self.kinematics_solver is not None:
                            self._leader_base_ee_pose = self._compute_ee_pose_from_joints(teleop_action)

                        if self._debug_frame_count % 30 == 1:
                            print(f"🔄 Base updated after sending delta action")

                    else:
                        # Delta太小，发送零action，不更新base
                        zero_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + (
                            [1.0] if self.use_gripper else [0.0]
                        )
                        zero_tensor = torch.tensor(
                            zero_action, dtype=action.dtype, device=action.device
                        )
                        new_transition[TransitionKey.ACTION] = zero_tensor

                        if self._debug_frame_count % 60 == 1:
                            print(f"⏹️  Delta too small, sending zero action")

        else:
            # Non-intervention mode: output zero 7D action [x, y, z, rx, ry, rz, gripper]
            # This ensures consistent 7D output format regardless of intervention state
            zero_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + (
                [1.0] if self.use_gripper else [0.0]
            )
            zero_tensor = torch.tensor(
                zero_action, dtype=action.dtype, device=action.device
            )
            new_transition[TransitionKey.ACTION] = zero_tensor

        # Handle episode termination and update metadata
        new_transition[TransitionKey.DONE] = bool(terminate_episode) or (
            self.terminate_on_success and success
        )
        new_transition[TransitionKey.REWARD] = float(success)

        # Update info dictionary with intervention metadata
        info = new_transition.get(TransitionKey.INFO, {})
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        # Update complementary data with current action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        action_tensor = new_transition[TransitionKey.ACTION]
        return new_transition

    def _convert_to_exact_sync_action(
        self, leader_positions: dict, transition: EnvTransition
    ) -> list:
        """
        Generate a ONE-TIME correction action to move follower to leader's current position.
        This is only called once when synchronization completes.

        Computes: delta = leader_current * inv(follower_current)
        This moves follower from its current position to leader's current position.
        """
        if self.kinematics_solver is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + (
                [1.0] if self.use_gripper else [0.0]
            )

        # Get current follower EE pose from observation
        observation = transition.get(TransitionKey.OBSERVATION, {})

        # Compute leader's current EE pose
        leader_current_pose = self._compute_ee_pose_from_joints(leader_positions)

        # Get follower's current EE pose
        follower_joint_positions = []
        follower_joint_positions_dict = {}
        for motor_name in self.motor_names:
            joint_key = motor_name + ".pos"
            if joint_key in observation:
                follower_joint_positions.append(observation[joint_key])
                follower_joint_positions_dict[motor_name] = observation[joint_key]
            else:
                raise ValueError(
                    f"Cannot find follower joint position for {motor_name}"
                )

        if len(follower_joint_positions) != len(self.motor_names):
            raise ValueError(
                f"Missing follower joint positions. Expected {len(self.motor_names)}, got {len(follower_joint_positions)}"
            )
        follower_current_pose = self.kinematics_solver.forward_kinematics(
            np.array(follower_joint_positions)
        )

        # 对比每个关节
        for motor_name in self.motor_names:
            leader_joint = leader_positions.get(f"{motor_name}.pos", 0.0)
            follower_joint = follower_joint_positions_dict.get(motor_name, 0.0)
            error = abs(leader_joint - follower_joint)
            tolerance = self.sync_tolerances.get(f"{motor_name}.pos", 10.0)

            status = "✅" if error <= tolerance else "❌"
            print(f"  {motor_name}: {status} Leader={leader_joint:6.1f}°, Follower={follower_joint:6.1f}°, Error={error:5.1f}°, Tol={tolerance}°")

        # Calculate correction transformation:
        # We want: follower_future = leader_current
        # So: delta = leader_current * inv(follower_current)
        follower_current_inv = np.linalg.inv(follower_current_pose)
        correction_pose = leader_current_pose @ follower_current_inv

        # Convert to 7D action
        correction_action = self._pose_to_7d_action(correction_pose)

        # Handle gripper correction
        if self.use_gripper:
            leader_gripper = leader_positions["gripper.pos"]
            follower_gripper = observation["gripper.pos"]
            gripper_delta = (leader_gripper - follower_gripper) * 0.1  # 小的velocity
            correction_action.append(gripper_delta)
        else:
            correction_action.append(0.0)

        print(f"🎯 EXACT SYNC CORRECTION: Moving follower to leader position")
        print(f"  Follower current: {follower_current_pose}")
        print(f"  Leader current: {leader_current_pose[:3, 3]}")
        print(f"  Correction action: {[f'{x:.3f}' for x in correction_action]}")

        return correction_action

    def _compute_ee_pose_from_joints(self, joint_positions: dict) -> np.ndarray:
        if self.kinematics_solver is None:
            return np.eye(4)

        joint_array = []
        joint_info = []
        for motor_name in self.motor_names:
            joint_key = motor_name + ".pos"
            if joint_key in joint_positions:
                joint_array.append(joint_positions[joint_key])
                joint_info.append(f"{motor_name}={joint_positions[joint_key]:.1f}°")
            else:
                joint_array.append(0.0)
                joint_info.append(f"{motor_name}=0.0°")

        ee_pose = self.kinematics_solver.forward_kinematics(np.array(joint_array))

        # 验证FK一致性：相同的inputs应该产生相同的outputs
        if hasattr(self, '_last_fk_joints') and hasattr(self, '_last_fk_pose'):
            if np.allclose(joint_array, self._last_fk_joints, atol=0.01):
                if not np.allclose(ee_pose, self._last_fk_pose, atol=0.001):
                    print(f"FK INCONSISTENCY: Same joints but different EE pose!")
                    print(f"  Joints: {joint_array}")
                    print(f"  Old pose: {self._last_fk_pose[:3, 3]}")
                    print(f"  New pose: {ee_pose[:3, 3]}")

        self._last_fk_joints = joint_array
        self._last_fk_pose = ee_pose.copy()

        if self._debug_frame_count % 40 == 1:
            print(f"FK JOINT MAPPING:")
            print(f"  Joints: {', '.join(joint_info)}")
            print(f"  EE Position: {ee_pose[:3, 3]}")
        return ee_pose

    def _pose_to_7d_action(self, pose: np.ndarray) -> list:
        """
        Convert 4x4 transformation matrix to 7D action [x, y, z, rx, ry, rz].
        Uses Euler angles for rotation representation.
        """
        # Extract position
        position = pose[:3, 3]

        # Extract rotation matrix and convert to Euler angles (ZYX convention)
        rotation_matrix = pose[:3, :3]

        # Convert to Euler angles
        try:
            # Using scipy for robust rotation conversion
            from scipy.spatial.transform import Rotation

            rot = Rotation.from_matrix(rotation_matrix)
            euler_angles = rot.as_euler("zyx", degrees=False)  # Return in radians

            # Reorder to [rx, ry, rz] if needed, or keep as [rz, ry, rx] depending on your convention
            # Here I'm using [rx, ry, rz] corresponding to roll, pitch, yaw
            rx, ry, rz = euler_angles[2], euler_angles[1], euler_angles[0]

        except ImportError:
            # Fallback to simple calculation if scipy not available
            # This is a simplified conversion - consider using a proper library
            rx = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            ry = np.arctan2(
                -rotation_matrix[2, 0],
                np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2),
            )
            rz = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        return [position[0], position[1], position[2], rx, ry, rz]

    def _convert_to_7d_action_with_kinematics(
        self, current_leader_positions: dict
    ) -> list:
        """
        Generate CONTINUOUS following action based on leader's movement from base position.

        Computes: delta = leader_current * inv(leader_base)
        This makes follower move relative to the synchronized base position.
        """
        # 计算当前pose
        leader_current_pose = self._compute_ee_pose_from_joints(
            current_leader_positions
        )

        # Debug: 检查base和current的差异
        if self._debug_frame_count % 30 == 1:
            base_pos = self._leader_base_ee_pose[:3, 3]
            current_pos = leader_current_pose[:3, 3]
            pos_error = np.linalg.norm(current_pos - base_pos)
            print(f"DELTA DEBUG: pos_error={pos_error:.4f}m")
            if pos_error > 0.001:
                print(f"  Base:    {base_pos}")
                print(f"  Current: {current_pos}")

        # 计算delta
        base_pose_inv = np.linalg.inv(self._leader_base_ee_pose)
        delta_pose = leader_current_pose @ base_pose_inv

        delta_action = self._pose_to_7d_action(delta_pose)

        # Handle gripper
        if self.use_gripper:
            current_gripper = current_leader_positions["gripper.pos"]
            base_gripper = self._leader_base_positions["gripper.pos"]
            gripper_delta = (current_gripper - base_gripper) * 0.1
            normalized_gripper = 1.0 + gripper_delta
            normalized_gripper = max(0.0, min(2.0, normalized_gripper))
            delta_action.append(gripper_delta)
        else:
            delta_action.append(0.0)

        return delta_action

    def _initialize_manual_sync(self, transition: EnvTransition):
        """
        Initialize manual position synchronization.
        Display instructions for user to manually move leader arm.
        """
        # Get follower's current joint positions from observation
        self._follower_reference_positions = self._get_follower_joint_positions(
            transition
        )

        if self._follower_reference_positions:
            print("\n" + "=" * 70)
            print(
                "🎯 LEADER ARM INTERVENTION: MANUAL POSITION SYNCHRONIZATION REQUIRED"
            )
            print("=" * 70)
            print("Follower arm current positions:")
            for joint, pos in self._follower_reference_positions.items():
                print(f"  {joint}: {pos:7.2f}°")
            print("\nINSTRUCTIONS:")
            print(
                "1. Manually move the LEADER arm to match the FOLLOWER positions above"
            )
            print(
                "2. Keep the leader arm STEADY for a moment when positions are matched"
            )
            print(
                "3. System will detect when leader arm is stable and synchronization is complete"
            )
            print("4. Joint-specific tolerances:")
            for joint, tolerance in self.sync_tolerances.items():
                print(f"   - {joint}: ±{tolerance}°")
            print(
                "5. Movement thresholds: 2mm position, 0.01rad orientation, 1° gripper"
            )
            print("6. You will see 'Position synchronization COMPLETE' when ready")
            print(
                "7. New action format: [x, y, z, rx, ry, rz, gripper] using kinematics"
            )
            print("=" * 70 + "\n")

            # Initialize synchronization state
            self._is_position_synced = False
            self._leader_base_positions = None
            self._leader_base_ee_pose = None
            self._last_leader_ee_pose = None
            self._stable_frames_count = 0
            self._sync_start_time = time.time()
        else:
            print("LEADER ARM INTERVENTION: WARNING - Could not get follower positions")
            self._is_position_synced = True  # Skip synchronization

    def _reset_sync_state(self):
        """Reset synchronization state when intervention ends."""
        self._is_position_synced = False
        self._follower_reference_positions = None
        self._leader_base_positions = None
        self._leader_base_ee_pose = None
        self._last_leader_ee_pose = None
        self._stable_frames_count = 0
        self._sync_start_time = None

    def _get_follower_joint_positions(self, transition: EnvTransition) -> dict:
        """
        Extract follower robot joint positions from observation.
        """
        observation = transition.get(TransitionKey.OBSERVATION, {})

        follower_positions = {}
        for joint_name in [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]:
            if joint_name in observation:
                follower_positions[joint_name] = observation[joint_name]

        return follower_positions

    def _check_manual_sync_complete(
        self, current_leader_positions: dict, transition: EnvTransition
    ) -> bool:
        """
        Check synchronization with enhanced movement detection using EE pose.
        """
        if self._follower_reference_positions is None:
            return True

        # Check synchronization timeout
        if self._sync_start_time and (time.time() - self._sync_start_time) > 120:
            print("⏰ LEADER ARM INTERVENTION: Synchronization timeout")
            return True

        # Calculate current leader EE pose
        current_leader_ee_pose = self._compute_ee_pose_from_joints(
            current_leader_positions
        )

        # Check if leader arm is stable (not moving significantly)
        is_stable = self._check_leader_stability(
            current_leader_ee_pose, current_leader_positions
        )

        # Check joint-based synchronization
        joints_synced = self._check_joint_based_sync(current_leader_positions)

        # Sync complete only when joints are synced AND leader is stable
        sync_complete = joints_synced and is_stable

        if self._debug_frame_count % 10 == 1:
            print(
                f"SYNC STATUS: joints_synced={joints_synced}, stable={is_stable}, stable_frames={self._stable_frames_count}/{self._stable_frames_required}"
            )

        return sync_complete

    def _check_leader_stability(
        self, current_ee_pose: np.ndarray, current_leader_positions: dict
    ) -> bool:
        """
        Check if leader arm is stable by comparing EE pose changes.
        Returns True if leader arm has been stable for required consecutive frames.
        """
        if self._last_leader_ee_pose is None:
            self._last_leader_ee_pose = current_ee_pose
            return False

        # Calculate position change
        current_pos = current_ee_pose[:3, 3]
        last_pos = self._last_leader_ee_pose[:3, 3]
        pos_change = np.linalg.norm(current_pos - last_pos)

        # Calculate orientation change (using rotation matrix difference)
        current_rot = current_ee_pose[:3, :3]
        last_rot = self._last_leader_ee_pose[:3, :3]
        rot_change = np.linalg.norm(current_rot - last_rot)

        # Calculate gripper change
        current_gripper = current_leader_positions.get("gripper.pos", 0.0)
        last_gripper = (
            self._leader_base_positions.get("gripper.pos", 0.0)
            if self._leader_base_positions
            else current_gripper
        )
        gripper_change = abs(current_gripper - last_gripper)

        # Check if all changes are below thresholds
        is_stable = (
            pos_change < self.position_threshold
            and rot_change < self.orientation_threshold
            and gripper_change < self.gripper_threshold
        )

        if is_stable:
            self._stable_frames_count += 1
        else:
            self._stable_frames_count = 0
            if self._debug_frame_count % 10 == 1:
                print(
                    f"LEADER MOVING: pos_change={pos_change:.4f}m, rot_change={rot_change:.4f}, gripper_change={gripper_change:.1f}°"
                )

        # Update last pose for next comparison
        self._last_leader_ee_pose = current_ee_pose

        return self._stable_frames_count >= self._stable_frames_required

    def _check_joint_based_sync(self, current_leader_positions: dict) -> bool:
        """Check joint-based synchronization with tolerance."""
        if self._follower_reference_positions is None:
            return True

        all_within_tolerance = True

        for joint_name, ref_pos in self._follower_reference_positions.items():
            if joint_name in current_leader_positions:
                current_pos = current_leader_positions[joint_name]
                error = abs(current_pos - ref_pos)
                tolerance = self.sync_tolerances.get(joint_name, 10.0)
                if error > tolerance:
                    all_within_tolerance = False
                    if self._debug_frame_count % 30 == 1:
                        print(
                            f"JOINT SYNC: {joint_name} error={error:.1f}° > tolerance={tolerance}°"
                        )

        return all_within_tolerance

    def get_config(self) -> dict[str, Any]:
        return {
            "use_gripper": self.use_gripper,
            "terminate_on_success": self.terminate_on_success,
            "sync_tolerances": self.sync_tolerances,
            "position_threshold": self.position_threshold,
            "orientation_threshold": self.orientation_threshold,
            "gripper_threshold": self.gripper_threshold,
            "stable_frames_required": self._stable_frames_required,
            "has_kinematics_solver": self.kinematics_solver is not None,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("reward_classifier_processor")
class RewardClassifierProcessorStep(ProcessorStep):
    """
    Applies a pretrained reward classifier to image observations to predict success.

    This step uses a model to determine if the current state is successful, updating
    the reward and potentially terminating the episode.

    Attributes:
        pretrained_path: Path to the pretrained reward classifier model.
        device: The device to run the classifier on.
        success_threshold: The probability threshold to consider a prediction as successful.
        success_reward: The reward value to assign on success.
        terminate_on_success: If True, terminates the episode upon successful classification.
        reward_classifier: The loaded classifier model instance.
    """

    pretrained_path: str | None = None
    device: str = "cpu"
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True

    reward_classifier: Any = None

    def __post_init__(self):
        """Initializes the reward classifier model after the dataclass is created."""
        if self.pretrained_path is not None:
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

            self.reward_classifier = Classifier.from_pretrained(self.pretrained_path)
            self.reward_classifier.to(self.device)
            self.reward_classifier.eval()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Processes a transition, applying the reward classifier to its image observations.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition with an updated reward and done flag based on the
            classifier's prediction.
        """
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.reward_classifier is None:
            return new_transition

        # Extract images from observation
        images = {key: value for key, value in observation.items() if "image" in key}

        if not images:
            return new_transition

        # Run reward classifier
        start_time = time.perf_counter()
        with torch.inference_mode():
            success = self.reward_classifier.predict_reward(images, threshold=self.success_threshold)

        classifier_frequency = 1 / (time.perf_counter() - start_time)

        # Calculate reward and termination
        reward = new_transition.get(TransitionKey.REWARD, 0.0)
        terminated = new_transition.get(TransitionKey.DONE, False)

        if math.isclose(success, 1, abs_tol=1e-2):
            reward = self.success_reward
            if self.terminate_on_success:
                terminated = True

        # Update transition
        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        # Update info with classifier frequency
        info = new_transition.get(TransitionKey.INFO, {})
        info["reward_classifier_frequency"] = classifier_frequency
        new_transition[TransitionKey.INFO] = info

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "device": self.device,
            "success_threshold": self.success_threshold,
            "success_reward": self.success_reward,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
