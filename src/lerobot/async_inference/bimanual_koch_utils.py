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

"""Shared utilities for bimanual Koch robot control with FK/IK."""

import torch

from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import RobotProcessorPipeline

# Default initial/home position for bimanual Koch robot (in EE space)
# Format: [left_x, left_y, left_z, left_wx, left_wy, left_wz, left_gripper,
#          right_x, right_y, right_z, right_wx, right_wy, right_wz, right_gripper]
INITIAL_EE_POSE = torch.tensor([
    -1.0168e-01,  # left_x
    1.1525e-03,   # left_y
    9.4441e-02,   # left_z
    -7.4215e-01,  # left_wx
    -1.2467e00,   # left_wy
    -5.7231e-01,  # left_wz
    4.9067e01,    # left_gripper
    -8.8268e-02,  # right_x
    4.3833e-03,   # right_y
    9.6670e-02,   # right_z
    -6.4383e-01,  # right_wx
    -1.2725e00,   # right_wy
    -5.1562e-01,  # right_wz
    5.0397e01,    # right_gripper
])


def action_tensor_to_dict(action_tensor: torch.Tensor, action_features: list[str]) -> dict[str, float]:
    """Convert action tensor to dictionary.

    Args:
        action_tensor: Action values as tensor
        action_features: List of action feature names in order

    Returns:
        Dictionary mapping feature names to values
    """
    return {key: action_tensor[i].item() for i, key in enumerate(action_features)}


def action_dict_to_tensor(action_dict: dict[str, float], action_features: list[str]) -> torch.Tensor:
    """Convert action dictionary to tensor.

    Args:
        action_dict: Dictionary mapping feature names to values
        action_features: List of action feature names in order

    Returns:
        Action values as tensor
    """
    return torch.tensor([action_dict[key] for key in action_features])


def generate_linear_trajectory(
    start: torch.Tensor, target: torch.Tensor, num_steps: int
) -> torch.Tensor:
    """Generate a linearly interpolated trajectory from start to target.

    Args:
        start: Starting pose tensor (num_dims,)
        target: Target pose tensor (num_dims,)
        num_steps: Number of interpolation steps

    Returns:
        Interpolated trajectory tensor (num_steps, num_dims)
    """
    num_dims = start.shape[0]

    # Create interpolation weights from 0 to 1
    t_vals = torch.linspace(
        0.0, 1.0, steps=num_steps, dtype=start.dtype, device=start.device
    ).unsqueeze(1)  # (num_steps, 1)

    # Expand start and target to (num_steps, num_dims)
    start_expanded = start.unsqueeze(0).expand(num_steps, num_dims)
    target_expanded = (
        target.to(start.dtype).to(start.device).unsqueeze(0).expand(num_steps, num_dims)
    )

    # Linear interpolation: (1 - t) * start + t * target
    interpolated = (1.0 - t_vals) * start_expanded + t_vals * target_expanded

    return interpolated


def get_bimanual_action_features(robot, teleop_processor: RobotProcessorPipeline) -> list[str]:
    """Derive action features for bimanual robot using pipeline aggregation.

    Args:
        robot: Robot instance with action_features attribute
        teleop_processor: Teleop action processor pipeline

    Returns:
        List of action feature names in correct order
    """
    action_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_processor,
        initial_features=create_initial_features(action=robot.action_features),
        use_videos=True,
    )["action"]["names"]
    return action_features


def compute_current_ee(
    raw_observation: dict,
    teleop_processor: RobotProcessorPipeline,
    action_features: list[str],
) -> torch.Tensor:
    """Compute current end effector pose using forward kinematics.

    Args:
        raw_observation: Raw robot observation (joint angles + camera images)
        teleop_processor: Teleop action processor for FK computation
        action_features: List of action feature names in order

    Returns:
        Current EE pose tensor [left_x, left_y, left_z, left_wx, left_wy, left_wz, left_gripper,
                                right_x, right_y, right_z, right_wx, right_wy, right_wz, right_gripper]
    """
    # Use FK processor to compute EE pose from joint angles
    ee_action_dict = teleop_processor((raw_observation, raw_observation))

    # Convert to tensor in the correct order
    return action_dict_to_tensor(ee_action_dict, action_features)
