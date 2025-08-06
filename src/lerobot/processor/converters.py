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

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from .pipeline import EnvTransition, TransitionKey


def _to_tensor(x: Any):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        # Keep images (uint8 HWC) and python objects as-is
        if x.dtype == np.uint8 or x.dtype == np.object_:
            return x
        # Scalars/arrays to float32 tensor
        return torch.as_tensor(x, dtype=torch.float32)
    # Anything else to float32 tensor
    return torch.as_tensor(x, dtype=torch.float32)


def _from_tensor(x: Any):
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().numpy()
    return x


def _is_image(arr: Any) -> bool:
    return isinstance(arr, np.ndarray) and arr.dtype == np.uint8 and arr.ndim == 3


def _split_obs_to_state_and_images(obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state, images = {}, {}
    for k, v in obs.items():
        if _is_image(v):
            images[k] = v
        else:
            state[k] = v
    return state, images


def make_obs_act_transition(*, obs: dict | None = None, act: dict | None = None) -> EnvTransition:
    return {
        TransitionKey.OBSERVATION: {} if obs is None else obs,
        TransitionKey.ACTION: {} if act is None else act,
    }


def to_transition_teleop_action(action: dict[str, Any]) -> EnvTransition:
    """
    Convert a raw teleop action dict into an EnvTransition under the ACTION TransitionKey.
    """
    act_dict: dict[str, Any] = {}
    for k, v in action.items():
        arr = np.array(v) if np.isscalar(v) else v
        act_dict[f"action.{k}"] = _to_tensor(arr)

    return make_obs_act_transition(act=act_dict)


def to_transition_robot_observation(observation: dict[str, Any]) -> EnvTransition:
    """
    Convert a raw robot observation dict into an EnvTransition under the OBSERVATION TransitionKey.
    """
    state, images = _split_obs_to_state_and_images(observation)

    obs_dict: dict[str, Any] = {}
    for k, v in state.items():
        arr = np.array(v) if np.isscalar(v) else v
        obs_dict[f"observation.state.{k}"] = _to_tensor(arr)

    for cam, img in images.items():
        obs_dict[f"observation.images.{cam}"] = img

    return make_obs_act_transition(obs=obs_dict)


def to_output_robot_action(transition: EnvTransition) -> dict[str, Any]:
    """
    Converts a EnvTransition under the ACTION TransitionKey to a dict with keys ending in '.pos' for raw robot actions.
    """
    out: dict[str, Any] = {}
    action_dict = transition.get(TransitionKey.ACTION) or {}

    for k, v in action_dict.items():
        if isinstance(k, str) and k.startswith("action.") and k.endswith(".pos"):
            out_key = k[len("action.") :]  # Strip the 'action.' prefix.
            out[out_key] = float(v)

    return out


def to_dataset_frame(features: dict[str, dict]) -> dict[str, any]:
    """
    Converts a dictionary of transitions (or a single transition) into a flat,
    dataset-friendly dictionary for training or evaluation.

    Args:
        features (dict[str, dict]):
            A feature specification dictionary.
            It can include:
              - 'action': dict with 'names' (ordered list of action feature names).
              - 'observation.state': dict with 'names' (ordered list of observation state feature names).
              - Any keys starting with 'observation.images.' will be treated as image features.

    Returns:
        to_output (Callable):
            A function that takes an `EnvTransition` or iterable of `EnvTransition`s and returns
            a flat dictionary with preprocessed state, action, reward, done, and any extra metadata.
    """
    # Ordered names for vectors
    action_names = (features.get("action", {}) or {}).get("names", [])
    obs_state_names = (features.get("observation.state", {}) or {}).get("names", [])

    # All image keys that should be copied through if present in the observation
    image_keys = [k for k, ft in features.items() if k.startswith("observation.images.")]

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
        # reward/done/truncated: last writer wins
        for k in (TransitionKey.REWARD, TransitionKey.DONE, TransitionKey.TRUNCATED):
            if k in other:
                out[k] = other[k]
        return out

    def _ensure_transition(obj) -> EnvTransition:
        # Accept either a single transition or a list/tuple of them
        if isinstance(obj, dict) and any(isinstance(k, TransitionKey) for k in obj.keys()):
            return obj
        if isinstance(obj, Iterable):
            it = list(obj)
            if not it:
                return {}
            acc = it[0]
            for t in it[1:]:
                acc = _merge(acc, t)
            return acc
        raise TypeError("to_output expected an EnvTransition or an iterable of EnvTransitions")

    def to_output(transitions_or_transition) -> dict[str, any]:
        tr = _ensure_transition(transitions_or_transition)
        obs = tr.get(TransitionKey.OBSERVATION) or {}
        act = tr.get(TransitionKey.ACTION) or {}

        batch: dict[str, any] = {}

        # Images passthrough (only the ones declared in features)
        for k in image_keys:
            if k in obs:
                batch[k] = obs[k]

        # observation.state vector according to feature order
        if obs_state_names:
            vals = []
            for name in obs_state_names:
                key = f"observation.state.{name}"
                vals.append(_from_tensor(obs.get(key, 0.0)))
            batch["observation.state"] = np.asarray(vals, dtype=np.float32)

        # action vector according to feature order
        if action_names:
            vals = []
            for name in action_names:
                key = f"action.{name}"
                vals.append(_from_tensor(act.get(key, 0.0)))
            batch["action"] = np.asarray(vals, dtype=np.float32)

        # reward/done/truncated/info
        if TransitionKey.REWARD in tr:
            batch["next.reward"] = _from_tensor(tr[TransitionKey.REWARD])
        if TransitionKey.DONE in tr:
            batch["next.done"] = _from_tensor(tr[TransitionKey.DONE])
        if TransitionKey.TRUNCATED in tr:
            batch["next.truncated"] = _from_tensor(tr[TransitionKey.TRUNCATED])
        if TransitionKey.INFO in tr:
            batch["info"] = tr[TransitionKey.INFO] or {}

        # complementary data: keep *_is_pad and task
        comp = tr.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        for k, v in comp.items():
            if k.endswith("_is_pad"):
                batch[k] = v
        if "task" in comp:
            batch["task"] = comp["task"]

        return batch

    return to_output
