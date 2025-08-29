import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.configs.types import PolicyFeature
from lerobot.constants import ACTION
from lerobot.processor.pipeline import (
    ComplementaryDataProcessor,
    EnvTransition,
    InfoProcessor,
    ObservationProcessor,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
    TruncatedProcessor,
)
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

GRIPPER_KEY = "gripper"
DISCRETE_PENALTY_KEY = "discrete_penalty"
TELEOP_ACTION_KEY = "teleop_action"


@ProcessorStepRegistry.register("add_teleop_action_as_complementary_data")
@dataclass
class AddTeleopActionAsComplimentaryData(ComplementaryDataProcessor):
    """Add teleoperator action to transition complementary data."""

    teleop_device: Teleoperator

    def complementary_data(self, complementary_data: dict) -> dict:
        new_complementary_data = dict(complementary_data)
        new_complementary_data[TELEOP_ACTION_KEY] = self.teleop_device.get_action()
        return new_complementary_data


@ProcessorStepRegistry.register("add_teleop_action_as_info")
@dataclass
class AddTeleopEventsAsInfo(InfoProcessor):
    """Add teleoperator control events to transition info."""

    teleop_device: Teleoperator

    def info(self, info: dict) -> dict:
        new_info = dict(info)
        teleop_events = getattr(self.teleop_device, "get_teleop_events", lambda: {})()
        new_info.update(teleop_events)
        return new_info


@ProcessorStepRegistry.register("image_crop_resize_processor")
@dataclass
class ImageCropResizeProcessor(ObservationProcessor):
    """Crop and resize image observations."""

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None

    def observation(self, observation: dict) -> dict:
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
        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
        }

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        if self.resize_size is None:
            return features
        for key in features:
            if "image" in key:
                features[key] = PolicyFeature(type=features[key].type, shape=self.resize_size)
        return features


@dataclass
@ProcessorStepRegistry.register("time_limit_processor")
class TimeLimitProcessor(TruncatedProcessor):
    """Track episode steps and enforce time limits."""

    max_episode_steps: int
    current_step: int = 0

    def truncated(self, truncated):
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        # TODO (steven): missing an else truncated = False?
        return truncated

    def get_config(self) -> dict[str, Any]:
        return {
            "max_episode_steps": self.max_episode_steps,
        }

    def reset(self) -> None:
        self.current_step = 0


@dataclass
@ProcessorStepRegistry.register("gripper_penalty_processor")
class GripperPenaltyProcessor(ComplementaryDataProcessor):
    """Apply penalty for inappropriate gripper usage."""

    penalty: float = -0.01
    max_gripper_pos: float = 30.0

    def complementary_data(self, complementary_data):
        """Calculate gripper penalty and add to complementary data."""
        action = self.transition.get(TransitionKey.ACTION)

        current_gripper_pos = complementary_data.get("raw_joint_positions", None).get(GRIPPER_KEY, None)
        if current_gripper_pos is None:
            return complementary_data

        gripper_action = action[f"{ACTION}.{GRIPPER_KEY}.pos"]
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
        return {
            "penalty": self.penalty,
            "max_gripper_pos": self.max_gripper_pos,
        }

    def reset(self) -> None:
        """Reset the processor state."""
        self.last_gripper_state = None


@dataclass
@ProcessorStepRegistry.register("intervention_action_processor")
class InterventionActionProcessor(ProcessorStep):
    """Handle human intervention actions and episode termination."""

    use_gripper: bool = False
    terminate_on_success: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        # Get intervention signals from complementary data
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()

        # Override action if intervention is active
        if is_intervention and teleop_action is not None:
            if isinstance(teleop_action, dict):
                # Convert teleop_action dict to tensor format
                action_list = [
                    teleop_action.get(f"{ACTION}.delta_x", 0.0),
                    teleop_action.get(f"{ACTION}.delta_y", 0.0),
                    teleop_action.get(f"{ACTION}.delta_z", 0.0),
                ]
                if self.use_gripper:
                    action_list.append(teleop_action.get(GRIPPER_KEY, 1.0))
            elif isinstance(teleop_action, np.ndarray):
                action_list = teleop_action.tolist()
            else:
                action_list = teleop_action

            teleop_action_tensor = torch.tensor(action_list, dtype=action.dtype, device=action.device)
            new_transition[TransitionKey.ACTION] = teleop_action_tensor

        # Handle episode termination
        new_transition[TransitionKey.DONE] = bool(terminate_episode) or (
            self.terminate_on_success and success
        )
        new_transition[TransitionKey.REWARD] = float(success)

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

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "use_gripper": self.use_gripper,
            "terminate_on_success": self.terminate_on_success,
        }


@dataclass
@ProcessorStepRegistry.register("reward_classifier_processor")
class RewardClassifierProcessor(ProcessorStep):
    """Apply reward classification to image observations."""

    pretrained_path: str | None = None
    device: str = "cpu"
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True

    reward_classifier: Any = None

    def __post_init__(self):
        """Initialize the reward classifier after dataclass initialization."""
        if self.pretrained_path is not None:
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

            self.reward_classifier = Classifier.from_pretrained(self.pretrained_path)
            self.reward_classifier.to(self.device)
            self.reward_classifier.eval()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.reward_classifier is None:
            return transition

        # Extract images from observation
        images = {key: value for key, value in observation.items() if "image" in key}

        if not images:
            return transition

        # Run reward classifier
        start_time = time.perf_counter()
        with torch.inference_mode():
            success = self.reward_classifier.predict_reward(images, threshold=self.success_threshold)

        classifier_frequency = 1 / (time.perf_counter() - start_time)

        # Calculate reward and termination
        reward = transition.get(TransitionKey.REWARD, 0.0)
        terminated = transition.get(TransitionKey.DONE, False)

        if math.isclose(success, 1, abs_tol=1e-2):
            reward = self.success_reward
            if self.terminate_on_success:
                terminated = True

        # Update transition
        new_transition = transition.copy()
        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        # Update info with classifier frequency
        info = new_transition.get(TransitionKey.INFO, {})
        info["reward_classifier_frequency"] = classifier_frequency
        new_transition[TransitionKey.INFO] = info

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "success_threshold": self.success_threshold,
            "success_reward": self.success_reward,
            "terminate_on_success": self.terminate_on_success,
        }
