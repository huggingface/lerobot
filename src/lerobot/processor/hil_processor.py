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

def _is_valid_policy_action(policy_action):
    """Check if policy-predicted action is valid"""
    if not isinstance(policy_action, torch.Tensor):
        return False

    if policy_action.shape[-1] != 6:  # Should be 6D action
        return False

    # Check for all zeros or abnormal values
    if torch.all(policy_action == 0):
        return False

    # Check for NaN or inf
    if torch.isnan(policy_action).any() or torch.isinf(policy_action).any():
        return False

    return True

@ProcessorStepRegistry.register("leader_arm_intervention")
class LeaderArmInterventionProcessorStep(ProcessorStep):
    """
    Leader arm intervention with direct joint position control.
    User moves leader arm and follower directly copies joint positions.
    Still computes and saves delta action for dataset.
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

        # Movement detection thresholds
        self.position_threshold = 0.002
        self.orientation_threshold = 0.01
        self.gripper_threshold = 1.0

        self._debug_frame_count = 0
        self._last_intervention_state = False
        self._is_position_synced = False
        self._follower_reference_positions = None
        self._leader_base_positions = None
        self._leader_base_ee_pose = None
        self._sync_start_time = None
        self._last_leader_ee_pose = None
        self._stable_frames_count = 0
        self._stable_frames_required = 10

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        start_time = time.perf_counter()

        self._debug_frame_count += 1
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        teleop_action = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()

        # Detect intervention state transitions
        intervention_started = is_intervention and not self._last_intervention_state
        intervention_ended = not is_intervention and self._last_intervention_state

        if intervention_started:
            print(
                "LEADER ARM INTERVENTION: Intervention started - initializing manual position synchronization"
            )
            self._initialize_manual_sync(transition)

        if intervention_ended:
            print("LEADER ARM INTERVENTION: Intervention ended")
            self._reset_sync_state()

        self._last_intervention_state = is_intervention

        # Process teleoperation during intervention
        if is_intervention and teleop_action is not None:
            if isinstance(teleop_action, dict) and any(
                ".pos" in key for key in teleop_action.keys()
            ):
                if not self._is_position_synced:
                    sync_complete = self._check_manual_sync_complete(
                        teleop_action, transition
                    )
                    if sync_complete:
                        print(
                            "LEADER ARM INTERVENTION: Position synchronization COMPLETE!"
                        )
                        print("LEADER ARM INTERVENTION: Ready for teleoperation")
                        self._is_position_synced = True
                        self._leader_base_positions = teleop_action.copy()
                        if self.kinematics_solver is not None:
                            self._leader_base_ee_pose = (
                                self._compute_ee_pose_from_joints(teleop_action)
                            )
                        self._last_leader_ee_pose = self._leader_base_ee_pose

                        # Store leader joint positions for direct control
                        leader_joint_positions = self._extract_leader_joint_positions(
                            teleop_action
                        )
                        complementary_data["leader_joint_positions"] = (
                            leader_joint_positions
                        )

                        # Compute and store delta action for dataset
                        delta_action = self._compute_delta_action(teleop_action)
                        complementary_data["teleop_action"] = torch.tensor(
                            delta_action, dtype=action.dtype, device=action.device
                        )

                        # Send current positions to maintain pose during sync transition
                        observation = transition.get(TransitionKey.OBSERVATION, {})
                        current_joint_positions = []
                        for motor_name in self.motor_names:
                            joint_key = f"{motor_name}.pos"
                            if joint_key in observation:
                                current_joint_positions.append(observation[joint_key])
                            else:
                                current_joint_positions.append(0.0)

                        current_positions_tensor = torch.tensor(
                            current_joint_positions,
                            dtype=action.dtype,
                            device=action.device,
                        )
                        new_transition[TransitionKey.ACTION] = current_positions_tensor

                    else:
                        # Still syncing: maintain current position but compute delta for recording
                        observation = transition.get(TransitionKey.OBSERVATION, {})

                        # Compute delta action for dataset (even though we're not moving)
                        delta_action = self._compute_delta_action(teleop_action)
                        complementary_data["teleop_action"] = torch.tensor(
                            delta_action, dtype=action.dtype, device=action.device
                        )

                        # Send current positions to maintain pose
                        current_joint_positions = []
                        for motor_name in self.motor_names:
                            joint_key = f"{motor_name}.pos"
                            if joint_key in self._follower_reference_positions:
                                current_joint_positions.append(self._follower_reference_positions[joint_key])
                            else:
                                current_joint_positions.append(0.0)

                        current_positions_tensor = torch.tensor(
                            current_joint_positions,
                            dtype=action.dtype,
                            device=action.device,
                        )
                        new_transition[TransitionKey.ACTION] = current_positions_tensor
                else:
                    # Position sync complete - compute EE delta action first
                    delta_action = self._compute_delta_action(teleop_action)

                    # Store delta action for dataset (always)
                    complementary_data["teleop_action"] = torch.tensor(
                        delta_action, dtype=action.dtype, device=action.device
                    )

                    # Check EE delta magnitude to decide if we should move follower
                    delta_magnitude = np.linalg.norm(delta_action[:3])
                    rot_magnitude = np.linalg.norm(delta_action[3:6])

                    if delta_magnitude > 0.005 or rot_magnitude > 0.01:
                        # Movement detected: store leader joint positions for direct control
                        leader_joint_positions = self._extract_leader_joint_positions(teleop_action)
                        complementary_data["leader_joint_positions"] = leader_joint_positions
                        print(f"Movement detected: pos={delta_magnitude:.4f}, rot={rot_magnitude:.4f}")
                    else:
                        # No movement: clear leader joint positions
                        complementary_data["leader_joint_positions"] = None

                    # Send current joint positions to maintain pose (not EE delta action)
                    observation = transition.get(TransitionKey.OBSERVATION, {})
                    current_joint_positions = []
                    for motor_name in self.motor_names:
                        joint_key = f"{motor_name}.pos"
                        if joint_key in observation:
                            current_joint_positions.append(observation[joint_key])
                        else:
                            current_joint_positions.append(0.0)

                    current_positions_tensor = torch.tensor(
                        current_joint_positions, dtype=action.dtype, device=action.device
                    )
                    new_transition[TransitionKey.ACTION] = current_positions_tensor

                    # Update base for delta calculation
                    self._leader_base_positions = teleop_action.copy()
                    if self.kinematics_solver is not None:
                        self._leader_base_ee_pose = self._compute_ee_pose_from_joints(
                            teleop_action
                        )
        else:
            policy_action = transition.get(TransitionKey.ACTION)
            is_valid_policy_action = _is_valid_policy_action(policy_action)
            if False:
                # Collecting data from scratch
                # No intervention: maintain current position
                observation = transition.get(TransitionKey.OBSERVATION, {})

                # Get current joint positions from observation
                current_joint_positions = []
                for motor_name in self.motor_names:
                    joint_key = f"{motor_name}.pos"
                    if joint_key in observation:
                        current_joint_positions.append(observation[joint_key])
                    else:
                        current_joint_positions.append(0.0)

                # Send current positions to maintain pose
                current_positions_tensor = torch.tensor(
                    current_joint_positions, dtype=action.dtype, device=action.device
                )
                new_transition[TransitionKey.ACTION] = current_positions_tensor
                # print(f"DEBUG: Maintaining current position: {[f'{x:.1f}' for x in current_joint_positions]}")

        # Handle episode termination
        new_transition[TransitionKey.DONE] = bool(terminate_episode) or (
            self.terminate_on_success and success
        )
        new_transition[TransitionKey.REWARD] = float(success)

        # Update info dictionary
        info = new_transition.get(TransitionKey.INFO, {})
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        # Update complementary data
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def _extract_leader_joint_positions(self, leader_positions: dict) -> list:
        """Extract leader joint positions in the correct order for motor_names"""
        joint_positions = []
        for motor_name in self.motor_names:
            joint_key = f"{motor_name}.pos"
            if joint_key in leader_positions:
                joint_positions.append(leader_positions[joint_key])
            else:
                joint_positions.append(0.0)

        return joint_positions

    def _compute_delta_action(self, current_leader_positions: dict) -> list:
        """Compute delta action for dataset recording"""
        if self._leader_base_ee_pose is None or self.kinematics_solver is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + (
                [0.0] if self.use_gripper else [0.0]
            )

        leader_current_pose = self._compute_ee_pose_from_joints(
            current_leader_positions
        )
        base_pose_inv = np.linalg.inv(self._leader_base_ee_pose)
        delta_pose = leader_current_pose @ base_pose_inv

        delta_action = self._pose_to_7d_action(delta_pose)

        if self.use_gripper:
            current_gripper = current_leader_positions["gripper.pos"]
            base_gripper = self._leader_base_positions["gripper.pos"]
            gripper_delta = (current_gripper - base_gripper) * 0.1
            delta_action.append(gripper_delta)
        else:
            delta_action.append(0.0)

        return delta_action

    def _compute_ee_pose_from_joints(self, joint_positions: dict) -> np.ndarray:
        if self.kinematics_solver is None:
            return np.eye(4)

        joint_array = []
        for motor_name in self.motor_names:
            joint_key = motor_name + ".pos"
            if joint_key in joint_positions:
                joint_array.append(joint_positions[joint_key])
            else:
                joint_array.append(0.0)

        ee_pose = self.kinematics_solver.forward_kinematics(np.array(joint_array))
        return ee_pose

    def _pose_to_7d_action(self, pose: np.ndarray) -> list:
        """Convert 4x4 transform to 7D action [x, y, z, rx, ry, rz]"""
        position = pose[:3, 3]
        rotation_matrix = pose[:3, :3]

        try:
            from scipy.spatial.transform import Rotation

            rot = Rotation.from_matrix(rotation_matrix)
            euler_angles = rot.as_euler("zyx", degrees=False)
            rx, ry, rz = euler_angles[2], euler_angles[1], euler_angles[0]
        except ImportError:
            rx = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            ry = np.arctan2(
                -rotation_matrix[2, 0],
                np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2),
            )
            rz = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        return [position[0], position[1], position[2], rx, ry, rz]

    def _initialize_manual_sync(self, transition: EnvTransition):
        """Initialize manual position synchronization."""
        self._follower_reference_positions = self._get_follower_joint_positions(
            transition
        )

        if self._follower_reference_positions:
            print("\n" + "=" * 80)
            print("LEADER ARM INTERVENTION: MANUAL POSITION SYNCHRONIZATION REQUIRED")
            print("=" * 80)
            print("Follower arm current positions:")
            for joint, pos in self._follower_reference_positions.items():
                tolerance = self.sync_tolerances.get(joint, 10.0)
                print(f"  {joint}: {pos:7.2f}°  (Tolerance: ±{tolerance}°)")

            print("\nINSTRUCTIONS:")
            print("1. Manually move LEADER arm to match FOLLOWER positions above")
            print("2. Keep leader arm STEADY when green checkmarks appear")
            print("3. System will auto-detect when synchronization is complete")
            print("4. Real-time joint differences will be displayed below")
            print("=" * 80 + "\n")

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
        """Extract follower robot joint positions from observation."""
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
        Display joint position differences for user guidance.
        """
        if self._follower_reference_positions is None:
            return True

        # Check synchronization timeout
        if self._sync_start_time and (time.time() - self._sync_start_time) > 120:
            print("LEADER ARM INTERVENTION: Synchronization timeout")
            return True

        # Calculate current leader EE pose
        current_leader_ee_pose = self._compute_ee_pose_from_joints(
            current_leader_positions
        )

        # Check if leader arm is stable (not moving significantly)
        is_stable = self._check_leader_stability(
            current_leader_ee_pose, current_leader_positions
        )

        # Check joint-based synchronization and display differences
        joints_synced = self._check_joint_based_sync_with_display(current_leader_positions)

        # Sync complete only when joints are synced AND leader is stable
        sync_complete = joints_synced and is_stable

        if self._debug_frame_count % 10 == 1:
            print(
                f"SYNC STATUS: joints_synced={joints_synced}, stable={is_stable}, stable_frames={self._stable_frames_count}/{self._stable_frames_required}"
            )

        return sync_complete

    def _check_joint_based_sync_with_display(self, current_leader_positions: dict) -> bool:
        """Check joint-based synchronization with tolerance and display differences."""
        if self._follower_reference_positions is None:
            return True

        all_within_tolerance = True

        # Display header every 30 frames
        if self._debug_frame_count % 60 == 1:
            print("\n" + "="*80)
            print("JOINT SYNCHRONIZATION STATUS")
            print("="*80)
            print(f"{'Joint':<20} {'Follower':<10} {'Leader':<10} {'Diff':<10} {'Tolerance':<10} {'Status':<10}")
            print("-"*80)

        for joint_name, ref_pos in self._follower_reference_positions.items():
            if joint_name in current_leader_positions:
                current_pos = current_leader_positions[joint_name]
                error = abs(current_pos - ref_pos)
                tolerance = self.sync_tolerances.get(joint_name, 10.0)
                within_tolerance = error <= tolerance

                if not within_tolerance:
                    all_within_tolerance = False

                # Display status every frame for user feedback
                status = "✓ OK" if within_tolerance else "✗ NEED MOVE"
                print(f"{joint_name:<20} {ref_pos:>9.1f}° {current_pos:>9.1f}° {error:>9.1f}° {tolerance:>9.1f}° {status:>10}")

        # Display summary every 30 frames
        if self._debug_frame_count % 60 == 1:
            print("="*80)
            if all_within_tolerance:
                print("ALL JOINTS WITHIN TOLERANCE - Keep arm steady to complete sync")
            else:
                print("ADJUST LEADER ARM to match follower positions above")
            print("="*80)

        return all_within_tolerance

    def _check_leader_stability(
        self, current_ee_pose: np.ndarray, current_leader_positions: dict
    ) -> bool:
        """Check if leader arm is stable."""
        if self._last_leader_ee_pose is None:
            self._last_leader_ee_pose = current_ee_pose
            return False

        current_pos = current_ee_pose[:3, 3]
        last_pos = self._last_leader_ee_pose[:3, 3]
        pos_change = np.linalg.norm(current_pos - last_pos)

        current_rot = current_ee_pose[:3, :3]
        last_rot = self._last_leader_ee_pose[:3, :3]
        rot_change = np.linalg.norm(current_rot - last_rot)

        current_gripper = current_leader_positions.get("gripper.pos", 0.0)
        last_gripper = (
            self._leader_base_positions.get("gripper.pos", 0.0)
            if self._leader_base_positions
            else current_gripper
        )
        gripper_change = abs(current_gripper - last_gripper)

        is_stable = (
            pos_change < self.position_threshold
            and rot_change < self.orientation_threshold
            and gripper_change < self.gripper_threshold
        )

        if is_stable:
            self._stable_frames_count += 1
        else:
            self._stable_frames_count = 0

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
