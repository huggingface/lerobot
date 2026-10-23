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

from typing import TYPE_CHECKING, Any

from lerobot.lerobot_types import EnvAction, EnvTransition, RobotAction, RobotObservation, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    DONE,
    INFO,
    MESSAGES_RENDERED,
    OBS_PREFIX,
    QUERY_KIND,
    QUERY_TEXT,
    REWARD,
    TRUNCATED,
)
from lerobot.utils.import_utils import lazy_exports

# These import torch, so each is imported the first time it is used.
if TYPE_CHECKING:
    import torch

    from lerobot.lerobot_types import PolicyAction

    # The aliases mark these as re-exports, which ruff keeps.
    from .tensor_converters import from_tensor_to_numpy as from_tensor_to_numpy, to_tensor as to_tensor
else:
    __getattr__, __dir__ = lazy_exports(
        __name__,
        {
            "PolicyAction": "lerobot.lerobot_types.PolicyAction",
            "from_tensor_to_numpy": ".tensor_converters.from_tensor_to_numpy",
            "to_tensor": ".tensor_converters.to_tensor",
        },
    )


_COMPLEMENTARY_KEYS = (
    "task",
    "index",
    "task_index",
    "episode_index",
    "frame_index",
    "timestamp",
    "language_persistent",
    "language_events",
    MESSAGES_RENDERED,
    "message_streams",
    "target_message_indices",
    # Text-generation request keys: carried into complementary_data so a prompt-formatting
    # processor step can read the kind and rewrite QUERY_TEXT.
    QUERY_KIND,
    QUERY_TEXT,
)


def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:
    """Extract complementary data from a batch dictionary.

    Includes padding flags (any key containing ``_is_pad``) plus the fixed
    set of metadata / language keys defined in ``_COMPLEMENTARY_KEYS`` —
    each only when present in ``batch``.
    """
    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    extras = {k: batch[k] for k in _COMPLEMENTARY_KEYS if k in batch}
    return {**pad_keys, **extras}


def create_transition(
    observation: RobotObservation | None = None,
    action: PolicyAction | RobotAction | EnvAction | None = None,
    reward: float | torch.Tensor = 0.0,
    done: bool | torch.Tensor = False,
    truncated: bool | torch.Tensor = False,
    info: dict[str, Any] | None = None,
    complementary_data: dict[str, Any] | None = None,
) -> EnvTransition:
    """
    Create an `EnvTransition` dictionary with sensible defaults.

    Args:
        observation: Observation dictionary.
        action: Policy, robot or environment action.
        reward: Reward value, a scalar or a tensor (e.g. for a batch).
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
    return action


def transition_to_policy_action(transition: EnvTransition) -> PolicyAction:
    """
    Convert an `EnvTransition` to a `PolicyAction`.
    """
    import torch

    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    action = transition.get(TransitionKey.ACTION)
    if not isinstance(action, torch.Tensor):
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
    import torch

    if not isinstance(action, torch.Tensor):
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
    import torch

    # Validate input type.
    if not isinstance(batch, dict):
        raise ValueError(f"EnvTransition must be a dictionary. Got {type(batch).__name__}")

    action = batch.get(ACTION)
    if action is not None and not isinstance(action, torch.Tensor):
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
        INFO: transition.get(TransitionKey.INFO, {}),
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
