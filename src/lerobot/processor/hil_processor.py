from dataclasses import dataclass
from typing import Any

import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.configs.types import PolicyFeature
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionKey


@dataclass
@ProcessorStepRegistry.register("image_crop_resize_processor")
class ImageCropResizeProcessor:
    """Crop and resize image observations."""

    crop_params_dict: dict[str, tuple[int, int, int, int]]
    resize_size: tuple[int, int] = (128, 128)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition

        if self.resize_size is None and not self.crop_params_dict:
            return transition

        new_observation = dict(observation)

        # Process all image keys in the observation
        for key in observation:
            if "image" not in key:
                continue

            image = observation[key]
            device = image.device
            if device.type == "mps":
                image = image.cpu()
            # Crop if crop params are provided for this key
            if key in self.crop_params_dict:
                crop_params = self.crop_params_dict[key]
                image = F.crop(image, *crop_params)
            # Always resize
            image = F.resize(image, self.resize_size)
            image = image.clamp(0.0, 1.0)
            new_observation[key] = image.to(device)

        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


@dataclass
@ProcessorStepRegistry.register("time_limit_processor")
class TimeLimitProcessor:
    """Track episode time and enforce time limits."""

    max_episode_steps: int
    current_step: int = 0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        truncated = transition.get(TransitionKey.TRUNCATED)
        if truncated is None:
            return transition

        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        new_transition = transition.copy()
        new_transition[TransitionKey.TRUNCATED] = truncated
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "max_episode_steps": self.max_episode_steps,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        self.current_step = 0

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


@dataclass
@ProcessorStepRegistry.register("gripper_penalty_processor")
class GripperPenaltyProcessor:
    penalty: float = -0.01
    max_gripper_pos: float = 30.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Calculate gripper penalty and add to complementary data."""
        action = transition.get(TransitionKey.ACTION)
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)

        if complementary_data is None or action is None:
            return transition

        current_gripper_pos = complementary_data.get("raw_joint_positions", None)[-1]
        if current_gripper_pos is None:
            return transition

        gripper_action = action[-1].item()
        gripper_action_normalized = gripper_action / self.max_gripper_pos

        # Normalize gripper state and action
        gripper_state_normalized = current_gripper_pos / self.max_gripper_pos
        gripper_action_normalized = gripper_action - 1.0

        # Calculate penalty boolean as in original
        gripper_penalty_bool = (gripper_state_normalized < 0.5 and gripper_action_normalized > 0.5) or (
            gripper_state_normalized > 0.75 and gripper_action_normalized < 0.5
        )

        gripper_penalty = self.penalty * int(gripper_penalty_bool)

        # Add penalty information to complementary data
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        # Create new complementary data with penalty info
        new_complementary_data = dict(complementary_data)
        new_complementary_data["discrete_penalty"] = gripper_penalty

        # Create new transition with updated complementary data
        new_transition = transition.copy()
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "penalty": self.penalty,
            "max_gripper_pos": self.max_gripper_pos,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        """Reset the processor state."""
        self.last_gripper_state = None

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


@dataclass
@ProcessorStepRegistry.register("intervention_action_processor")
class InterventionActionProcessor:
    """Handle action intervention based on signals in the transition.

    This processor checks for intervention signals in the transition's complementary data
    and overrides agent actions when intervention is active.
    """

    use_gripper: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        # Get intervention signals from complementary data
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get("teleop_action", {})
        is_intervention = complementary_data.get("is_intervention", False)
        terminate_episode = complementary_data.get("terminate_episode", False)
        success = complementary_data.get("success", False)
        rerecord_episode = complementary_data.get("rerecord_episode", False)

        new_transition = transition.copy()

        # Override action if intervention is active
        if is_intervention and teleop_action:
            # Convert teleop_action dict to tensor format
            action_list = [
                teleop_action.get("delta_x", 0.0),
                teleop_action.get("delta_y", 0.0),
                teleop_action.get("delta_z", 0.0),
            ]
            if self.use_gripper:
                action_list.append(teleop_action.get("gripper", 1.0))

            teleop_action_tensor = torch.tensor(action_list, dtype=action.dtype, device=action.device)
            new_transition[TransitionKey.ACTION] = teleop_action_tensor

        # Handle episode termination
        new_transition[TransitionKey.DONE] = bool(terminate_episode)
        new_transition[TransitionKey.REWARD] = float(success)

        # Update info with intervention metadata
        info = new_transition.get(TransitionKey.INFO, {})
        info["is_intervention"] = is_intervention
        info["rerecord_episode"] = rerecord_episode
        info["next.success"] = success if terminate_episode else info.get("next.success", False)
        new_transition[TransitionKey.INFO] = info
        new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] = new_transition[
            TransitionKey.ACTION
        ]

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "use_gripper": self.use_gripper,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features
