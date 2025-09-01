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

from collections.abc import Iterable, Sequence
from copy import deepcopy
from functools import singledispatch
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD, TRUNCATED

from .pipeline import EnvTransition, TransitionKey


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
        value: Input value to convert (tensor, array, scalar, sequence, etc.)
        dtype: Target tensor dtype. If None, preserves original dtype.
        device: Target device for the tensor.

    Returns:
        PyTorch tensor.

    Raises:
        TypeError: If the input type is not supported.
    """
    raise TypeError(f"Unsupported type for tensor conversion: {type(value)}")


@to_tensor.register(torch.Tensor)
def _(value: torch.Tensor, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """Handle existing PyTorch tensors."""
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
    """Handle numpy arrays."""
    # Check for numpy scalars (0-dimensional arrays) and treat them as scalars
    if value.ndim == 0:
        # Numpy scalars should be converted to 0-dimensional tensors
        scalar_value = value.item()
        return torch.tensor(scalar_value, dtype=dtype, device=device)

    # Create tensor from numpy array (torch.from_numpy handles contiguity automatically)
    tensor = torch.from_numpy(value)

    # Apply dtype conversion if specified
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
    """Handle scalar values including numpy scalars."""
    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(list)
@to_tensor.register(tuple)
def _(value: Sequence, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """Handle sequences (lists, tuples)."""
    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(dict)
def _(value: dict, *, device=None, **kwargs) -> dict:
    """Handle dictionaries by recursively converting values to tensors."""
    if not value:
        return {}

    result = {}
    for key, sub_value in value.items():
        if sub_value is None:
            continue

        if isinstance(sub_value, dict):
            # Recursively process nested dictionaries
            result[key] = to_tensor(
                sub_value,
                device=device,
                **kwargs,
            )
            continue

        # Convert individual values to tensors
        result[key] = to_tensor(
            sub_value,
            device=device,
            **kwargs,
        )
    return result


def _from_tensor(x: Any):
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().numpy()
    return x


def _is_image(arr: Any) -> bool:
    return isinstance(arr, np.ndarray) and arr.dtype == np.uint8 and arr.ndim == 3


def _split_obs_to_state_and_images(obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state, images = {}, {}
    for k, v in obs.items():
        if "image" in k.lower() or _is_image(v):
            images[k] = v
        else:
            state[k] = v
    return state, images


def make_obs_act_transition(
    *, obs: dict[str, Any] | None = None, act: dict[str, Any] | None = None
) -> EnvTransition:
    return {
        TransitionKey.OBSERVATION: {} if obs is None else obs,
        TransitionKey.ACTION: {} if act is None else act,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {},
        TransitionKey.REWARD: None,
        TransitionKey.DONE: None,
        TransitionKey.TRUNCATED: None,
    }


def to_transition_teleop_action(action: dict[str, Any]) -> EnvTransition:
    """
    Convert a raw teleop action dict into an EnvTransition under the ACTION TransitionKey.
    """
    act_dict: dict[str, Any] = {}
    for k, v in action.items():
        # Check if the value is a type that should not be converted to a tensor.
        if isinstance(v, (Rotation, dict)):
            act_dict[f"{ACTION}.{k}"] = v
            continue

        arr = np.array(v) if np.isscalar(v) else v
        act_dict[f"{ACTION}.{k}"] = to_tensor(arr)

    return make_obs_act_transition(act=act_dict)


# TODO(Adil, Pepijn): Overtime we can maybe add these converters to pipeline.py itself
def to_transition_robot_observation(observation: dict[str, Any]) -> EnvTransition:
    """
    Convert a raw robot observation dict into an EnvTransition under the OBSERVATION TransitionKey.
    """
    state, images = _split_obs_to_state_and_images(observation)

    obs_dict: dict[str, Any] = {}
    for k, v in state.items():
        arr = np.array(v) if np.isscalar(v) else v
        obs_dict[f"{OBS_STATE}.{k}"] = to_tensor(arr)

    for cam, img in images.items():
        obs_dict[f"{OBS_IMAGES}.{cam}"] = img

    return make_obs_act_transition(obs=obs_dict)


def to_output_robot_action(transition: EnvTransition) -> dict[str, Any]:
    """
    Converts a EnvTransition under the ACTION TransitionKey to a dict with keys ending in '.pos' for raw robot actions.
    """
    out: dict[str, Any] = {}
    action_dict = transition.get(TransitionKey.ACTION) or {}

    if action_dict is None:
        return out

    for k, v in action_dict.items():
        if isinstance(k, str) and k.startswith(f"{ACTION}.") and k.endswith((".pos", ".vel")):
            out_key = k[len(f"{ACTION}.") :]  # Strip the 'action.' prefix.
            out[out_key] = float(v)

    return out


def to_dataset_frame(
    transitions_or_transition: EnvTransition | Iterable[EnvTransition], features: dict[str, dict]
) -> dict[str, any]:
    """
    Converts a single EnvTransition or an iterable of them into a flat,
    dataset-friendly dictionary for training or evaluation, according to
    the provided `features` spec.

    Args:
        transitions_or_transition: Either a single EnvTransition dict
            or an iterable of them (which will be merged).
        features (dict[str, dict]):
            A feature specification dictionary:
              - 'action': dict with 'names': list of action feature names
              - 'observation.state': dict with 'names': list of state feature names
              - keys starting with 'observation.images.' are passed through

    Returns:
        batch (dict[str, any]): Flat dictionary containing:
          - numpy arrays for "observation.state" and "action"
          - any image tensors defined in features
          - next.{reward,done,truncated}
          - info dict
          - *_is_pad flags and task from complementary_data
    """
    action_names = features.get(ACTION, {}).get("names", [])
    obs_state_names = features.get(OBS_STATE, {}).get("names", [])
    image_keys = [k for k in features if k.startswith(OBS_IMAGES)]

    def _merge(base: EnvTransition, other: EnvTransition) -> EnvTransition:
        out = deepcopy(base)
        for key in (
            TransitionKey.OBSERVATION,
            TransitionKey.ACTION,
            TransitionKey.INFO,
            TransitionKey.COMPLEMENTARY_DATA,
        ):
            if other.get(key):
                out.setdefault(key, {}).update(deepcopy(other[key]))
        for k in (TransitionKey.REWARD, TransitionKey.DONE, TransitionKey.TRUNCATED):
            if k in other:
                out[k] = other[k]
        return out

    def _ensure_transition(obj) -> EnvTransition:
        # single transition
        if isinstance(obj, dict) and any(isinstance(k, TransitionKey) for k in obj):
            return obj
        # iterable of transitions
        if isinstance(obj, Iterable):
            items = list(obj)
            if not items:
                return {}
            acc = items[0]
            for t in items[1:]:
                acc = _merge(acc, t)
            return acc
        raise TypeError("Expected EnvTransition or iterable of them")

    tr = _ensure_transition(transitions_or_transition)
    obs = tr.get(TransitionKey.OBSERVATION, {}) or {}
    act = tr.get(TransitionKey.ACTION, {}) or {}
    batch: dict[str, any] = {}

    # Images passthrough
    for k in image_keys:
        if k in obs:
            batch[k] = obs[k]

    # Observation.state vector
    if obs_state_names:
        vals = [_from_tensor(obs.get(f"{OBS_STATE}.{n}", 0.0)) for n in obs_state_names]
        batch[OBS_STATE] = np.asarray(vals, dtype=np.float32)

    # Action vector
    if action_names:
        vals = [_from_tensor(act.get(f"{ACTION}.{n}", 0.0)) for n in action_names]
        batch[ACTION] = np.asarray(vals, dtype=np.float32)

    if tr.get(TransitionKey.REWARD) is not None:
        batch[REWARD] = _from_tensor(tr[TransitionKey.REWARD])
    if tr.get(TransitionKey.DONE) is not None:
        batch[DONE] = _from_tensor(tr[TransitionKey.DONE])
    if tr.get(TransitionKey.TRUNCATED) is not None:
        batch[TRUNCATED] = _from_tensor(tr[TransitionKey.TRUNCATED])

    # Complementary data flags and task
    comp = tr.get(TransitionKey.COMPLEMENTARY_DATA) or {}
    if comp:
        # pad flags
        for k, v in comp.items():
            if k.endswith("_is_pad"):
                batch[k] = v
        # task label
        if comp.get("task") is not None:
            batch["task"] = comp["task"]

    return batch


def _default_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:  # noqa: D401
    """Convert a *batch* dict coming from Learobot replay/dataset code into an
    ``EnvTransition`` dictionary.

    The function maps well known keys to the EnvTransition structure. Missing keys are
    filled with sane defaults (``None`` or ``0.0``/``False``).

    Keys recognised (case-sensitive):

    * "observation.*" (keys starting with "observation." are grouped into observation dict)
    * "action"
    * "next.reward"
    * "next.done"
    * "next.truncated"
    * "info"

    Additional keys are ignored so that existing dataloaders can carry extra
    metadata without breaking the processor.
    """

    # Extract observation keys
    observation_keys = {k: v for k, v in batch.items() if k.startswith("observation.")}
    observation = observation_keys if observation_keys else None

    # Extract padding, task, index, and task_index keys for complementary data
    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    task_key = {"task": batch["task"]} if "task" in batch else {}
    index_key = {"index": batch["index"]} if "index" in batch else {}
    task_index_key = {"task_index": batch["task_index"]} if "task_index" in batch else {}
    complementary_data = (
        {**pad_keys, **task_key, **index_key, **task_index_key}
        if pad_keys or task_key or index_key or task_index_key
        else {}
    )

    transition: EnvTransition = {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: batch.get("action"),
        TransitionKey.REWARD: batch.get("next.reward", 0.0),
        TransitionKey.DONE: batch.get("next.done", False),
        TransitionKey.TRUNCATED: batch.get("next.truncated", False),
        TransitionKey.INFO: batch.get("info", {}),
        TransitionKey.COMPLEMENTARY_DATA: complementary_data,
    }
    return transition


def _default_transition_to_batch(transition: EnvTransition) -> dict[str, Any]:  # noqa: D401
    """Inverse of :pyfunc:`_default_batch_to_transition`. Returns a dict with
    the canonical field names used throughout *LeRobot*.
    """

    batch = {
        "action": transition.get(TransitionKey.ACTION),
        "next.reward": transition.get(TransitionKey.REWARD, 0.0),
        "next.done": transition.get(TransitionKey.DONE, False),
        "next.truncated": transition.get(TransitionKey.TRUNCATED, False),
        "info": transition.get(TransitionKey.INFO, {}),
    }

    # Add padding, task, index, and task_index data from complementary_data
    complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
    if complementary_data:
        pad_data = {k: v for k, v in complementary_data.items() if "_is_pad" in k}
        batch.update(pad_data)

        if "task" in complementary_data:
            batch["task"] = complementary_data["task"]

        if "index" in complementary_data:
            batch["index"] = complementary_data["index"]

        if "task_index" in complementary_data:
            batch["task_index"] = complementary_data["task_index"]

    # Handle observation - flatten dict to observation.* keys if it's a dict
    observation = transition.get(TransitionKey.OBSERVATION)
    if isinstance(observation, dict):
        batch.update(observation)

    return batch
