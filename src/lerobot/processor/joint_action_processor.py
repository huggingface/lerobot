import numpy as np
import torch

from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

from .core import PolicyAction, RobotAction, TransitionKey, EnvTransition
from .pipeline import (
    ActionProcessorStep,
    ProcessorStepRegistry,
    RobotActionProcessorStep,
    ProcessorStep,
)


@ProcessorStepRegistry.register("direct_joint_control")
@dataclass
class DirectJointControlStep(ProcessorStep):
    """Process direct joint control commands from leader arm."""

    motor_names: list[str] = field(default_factory=list)
    use_gripper: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        leader_joint_positions = complementary_data.get("leader_joint_positions")

        if leader_joint_positions is not None:
            # Create robot action from leader joint positions
            robot_action = {}

            # Handle arm joints
            for i, motor_name in enumerate(self.motor_names):
                if i < len(leader_joint_positions):
                    if isinstance(leader_joint_positions, torch.Tensor):
                        robot_action[f"{motor_name}.pos"] = leader_joint_positions[
                            i
                        ].item()
                    else:
                        robot_action[f"{motor_name}.pos"] = float(
                            leader_joint_positions[i]
                        )

            # Handle gripper if used
            if self.use_gripper:
                gripper_index = len(self.motor_names)
                if (
                    isinstance(leader_joint_positions, (list, tuple))
                    and len(leader_joint_positions) > gripper_index
                ):
                    if isinstance(leader_joint_positions, torch.Tensor):
                        robot_action["gripper.pos"] = leader_joint_positions[
                            gripper_index
                        ].item()
                    else:
                        robot_action["gripper.pos"] = float(
                            leader_joint_positions[gripper_index]
                        )

            # Store the robot action
            complementary_data["robot_action"] = robot_action
            transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
            print(f"Created robot_action: {robot_action}")
        else:
            # print("No leader_joint_positions found")
            policy_action = transition.get(TransitionKey.ACTION)
            # if policy_action is not None:
            #     print(
            #         f"DEBUG: DirectJointControl - Using policy action: {policy_action.cpu().numpy()}"
            #     )
            # else:
            #     print("DEBUG: DirectJointControl - No policy action found")
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # This step doesn't change the feature definitions
        # It only processes complementary data, so we return features as-is
        return features


@ProcessorStepRegistry.register("joint_bounds_and_safety")
@dataclass
class JointBoundsAndSafetyStep(ActionProcessorStep):
    """Apply joint bounds and safety checks for direct joint control."""

    joint_bounds: dict = field(default_factory=dict)

    def action(self, action: RobotAction) -> RobotAction:
        if not isinstance(action, dict):
            return action

        # Apply joint bounds if specified
        bounded_action = action.copy()
        for joint_name, action_value in action.items():
            if joint_name in self.joint_bounds:
                bounds = self.joint_bounds[joint_name]
                min_bound = bounds.get("min", -180.0)
                max_bound = bounds.get("max", 180.0)
                bounded_action[joint_name] = np.clip(action_value, min_bound, max_bound)
            elif ".pos" in joint_name:
                # Default safety bounds for joint positions
                bounded_action[joint_name] = np.clip(action_value, -175.0, 175.0)

        return bounded_action

    def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # This step doesn't change the feature definitions
        return features
