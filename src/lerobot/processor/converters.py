# !/usr/bin/env python

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

from __future__ import annotations

from collections.abc import Sequence
from functools import singledispatch
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import ACTION, DONE, OBS_PREFIX, REWARD, TRUNCATED

from .core import EnvTransition, PolicyAction, RobotAction, RobotObservation, TransitionKey


@singledispatch
def to_tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Convert various data types to PyTorch tensors with configurable options.

    This is a unified tensor conversion function using single dispatch to handle
    different input types appropriately.

    Args:
        value: Input value to convert (tensor, array, scalar, sequence, etc.).
        dtype: Target tensor dtype. If None, preserves original dtype.
        device: Target device for the tensor.

    Returns:
        A PyTorch tensor.

    Raises:
        TypeError: If the input type is not supported.
    """
    raise TypeError(f"Unsupported type for tensor conversion: {type(value)}")


@to_tensor.register(torch.Tensor)
def _(value: torch.Tensor, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """Handle conversion for existing PyTorch tensors."""
    if dtype is not None:
        value = value.to(dtype=dtype)
    if device is not None:
        value = value.to(device=device)
    return value


@to_tensor.register(np.ndarray)
def _(
    value: np.ndarray,
    *,
    dtype=torch.float32,
    device=None,
    **kwargs,
) -> torch.Tensor:
    """Handle conversion for numpy arrays."""
    # Check for numpy scalars (0-dimensional arrays) and treat them as scalars.
    if value.ndim == 0:
        # Numpy scalars should be converted to 0-dimensional tensors.
        scalar_value = value.item()
        return torch.tensor(scalar_value, dtype=dtype, device=device)

    # Create tensor from numpy array.
    tensor = torch.from_numpy(value)

    # Apply dtype and device conversion if specified.
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)

    return tensor


@to_tensor.register(int)
@to_tensor.register(float)
@to_tensor.register(np.integer)
@to_tensor.register(np.floating)
def _(value, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """Handle conversion for scalar values including numpy scalars."""
    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(list)
@to_tensor.register(tuple)
def _(value: Sequence, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """Handle conversion for sequences (lists, tuples)."""
    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(dict)
def _(value: dict, *, device=None, **kwargs) -> dict:
    """Handle conversion for dictionaries by recursively converting their values to tensors."""
    if not value:
        return {}

    result = {}
    for key, sub_value in value.items():
        if sub_value is None:
            continue

        if isinstance(sub_value, dict):
            # Recursively process nested dictionaries.
            result[key] = to_tensor(
                sub_value,
                device=device,
                **kwargs,
            )
            continue

        # Convert individual values to tensors.
        result[key] = to_tensor(
            sub_value,
            device=device,
            **kwargs,
        )
    return result


def from_tensor_to_numpy(x: torch.Tensor | Any) -> np.ndarray | float | int | Any:
    """
    Convert a PyTorch tensor to a numpy array or scalar if applicable.

    If the input is not a tensor, it is returned unchanged.

    Args:
        x: The input, which can be a tensor or any other type.

    Returns:
        A numpy array, a scalar, or the original input.
    """
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().numpy()
    return x


def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:
    """
    Extract complementary data from a batch dictionary.

    This includes padding flags, task description, and indices.

    Args:
        batch: The batch dictionary.

    Returns:
        A dictionary with the extracted complementary data.
    """
    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    task_key = {"task": batch["task"]} if "task" in batch else {}
    index_key = {"index": batch["index"]} if "index" in batch else {}
    task_index_key = {"task_index": batch["task_index"]} if "task_index" in batch else {}

    return {**pad_keys, **task_key, **index_key, **task_index_key}


def create_transition(
    observation: dict[str, Any] | None = None,
    action: PolicyAction | RobotAction | None = None,
    reward: float = 0.0,
    done: bool = False,
    truncated: bool = False,
    info: dict[str, Any] | None = None,
    complementary_data: dict[str, Any] | None = None,
) -> EnvTransition:
    """
    Create an `EnvTransition` dictionary with sensible defaults.

    Args:
        observation: Observation dictionary.
        action: Action dictionary.
        reward: Scalar reward value.
        done: Episode termination flag.
        truncated: Episode truncation flag.
        info: Additional info dictionary.
        complementary_data: Complementary data dictionary.

    Returns:
        A complete `EnvTransition` dictionary.
    """
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info if info is not None else {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data is not None else {},
    }


def robot_action_observation_to_transition(
    action_observation: tuple[RobotAction, RobotObservation],
) -> EnvTransition:
    """
    Convert a raw robot action and observation dictionary into a standardized `EnvTransition`.

    Args:
        action: The raw action dictionary from a teleoperation device or controller.
        observation: The raw observation dictionary from the environment.

    Returns:
        An `EnvTransition` containing the formatted observation.
    """
    if not isinstance(action_observation, tuple):
        raise ValueError("action_observation should be a tuple type with an action and observation")

    action, observation = action_observation

    if action is not None and not isinstance(action, dict):
        raise ValueError(f"Action should be a RobotAction type got {type(action)}")

    if observation is not None and not isinstance(observation, dict):
        raise ValueError(f"Observation should be a RobotObservation type got {type(observation)}")

    return create_transition(action=action, observation=observation)


def robot_action_to_transition(action: RobotAction) -> EnvTransition:
    """
    Convert a raw robot action dictionary into a standardized `EnvTransition`.

    Args:
        action: The raw action dictionary from a teleoperation device or controller.

    Returns:
        An `EnvTransition` containing the formatted action.
    """
    if not isinstance(action, dict):
        raise ValueError(f"Action should be a RobotAction type got {type(action)}")
    return create_transition(action=action)


def observation_to_transition(observation: RobotObservation) -> EnvTransition:
    """
    Convert a raw robot observation dictionary into a standardized `EnvTransition`.

    Args:
        observation: The raw observation dictionary from the environment.

    Returns:
        An `EnvTransition` containing the formatted observation.
    """
    if not isinstance(observation, dict):
        raise ValueError(f"Observation should be a RobotObservation type got {type(observation)}")
    return create_transition(observation=observation)


def transition_to_robot_action(transition: EnvTransition) -> RobotAction:
    """
    Extract a raw robot action dictionary for a robot from an `EnvTransition`.

    This function searches for keys in the format "action.*.pos" or "action.*.vel"
    and converts them into a flat dictionary suitable for sending to a robot controller.

    Args:
        transition: The `EnvTransition` containing the action.

    Returns:
        A dictionary representing the raw robot action.
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    action = transition.get(TransitionKey.ACTION)
    if not isinstance(action, dict):
        raise ValueError(f"Action should be a RobotAction type (dict) got {type(action)}")
    return transition.get(TransitionKey.ACTION)


def transition_to_policy_action(transition: EnvTransition) -> PolicyAction:
    """
    Convert an `EnvTransition` to a `PolicyAction`.
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    action = transition.get(TransitionKey.ACTION)
    if not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
    return action


def transition_to_observation(transition: EnvTransition) -> RobotObservation:
    """
    Convert an `EnvTransition` to a `RobotObservation`.
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    observation = transition.get(TransitionKey.OBSERVATION)
    if not isinstance(observation, dict):
        raise ValueError(f"Observation should be a RobotObservation (dict) type got {type(observation)}")
    return observation


def policy_action_to_transition(action: PolicyAction) -> EnvTransition:
    """
    Convert a `PolicyAction` to an `EnvTransition`.
    """
    if not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
    return create_transition(action=action)


def batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """
    Convert a batch dictionary from a dataset/dataloader into an `EnvTransition`.

    This function maps recognized keys from a batch to the `EnvTransition` structure,
    filling in missing keys with sensible defaults.

    Args:
        batch: A batch dictionary.

    Returns:
        An `EnvTransition` dictionary.

    Raises:
        ValueError: If the input is not a dictionary.
    """

    # Validate input type.
    if not isinstance(batch, dict):
        raise ValueError(f"EnvTransition must be a dictionary. Got {type(batch).__name__}")

    action = batch.get(ACTION)
    if action is not None and not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

    # Extract observation and complementary data keys.
    observation_keys = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
    complementary_data = _extract_complementary_data(batch)

    return create_transition(
        observation=observation_keys if observation_keys else None,
        action=batch.get(ACTION),
        reward=batch.get(REWARD, 0.0),
        done=batch.get(DONE, False),
        truncated=batch.get(TRUNCATED, False),
        info=batch.get("info", {}),
        complementary_data=complementary_data if complementary_data else None,
    )


def transition_to_batch(transition: EnvTransition) -> dict[str, Any]:
    """
    Convert an `EnvTransition` back to the canonical batch format used in LeRobot.

    This is the inverse of `batch_to_transition`.

    Args:
        transition: The `EnvTransition` to convert.

    Returns:
        A batch dictionary with canonical LeRobot field names.
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    batch = {
        ACTION: transition.get(TransitionKey.ACTION),
        REWARD: transition.get(TransitionKey.REWARD, 0.0),
        DONE: transition.get(TransitionKey.DONE, False),
        TRUNCATED: transition.get(TransitionKey.TRUNCATED, False),
        "info": transition.get(TransitionKey.INFO, {}),
    }

    # Add complementary data.
    comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
    if comp_data:
        batch.update(comp_data)

    # Flatten observation dictionary.
    observation = transition.get(TransitionKey.OBSERVATION)
    if isinstance(observation, dict):
        batch.update(observation)

    return batch


def identity_transition(transition: EnvTransition) -> EnvTransition:
    """
    An identity function for transitions, returning the input unchanged.

    Useful as a default or placeholder in processing pipelines.

    Args:
        tr: An `EnvTransition`.

    Returns:
        The same `EnvTransition`.
    """
    return transition
