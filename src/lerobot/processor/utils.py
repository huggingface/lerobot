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
        # keep images (uint8 HWC) and python objects as-is
        if x.dtype == np.uint8 or x.dtype == np.object_:
            return x
        return torch.from_numpy(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return torch.tensor(x, dtype=torch.float32)
    return x


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


def make_transition(*, obs: dict | None = None, act: dict | None = None) -> EnvTransition:
    return {
        TransitionKey.OBSERVATION: {} if obs is None else obs,
        TransitionKey.ACTION: {} if act is None else act,
    }


def to_transition_teleop_action(action: dict[str, Any]) -> EnvTransition:
    """
    Convert a raw teleop action dict (provided under 'teleop_action') into an EnvTransition:
        ACTION -> {f"action.{k}": tensor(v)}
    """
    act_dict: dict[str, Any] = {}
    for k, v in action.items():
        arr = np.array(v) if np.isscalar(v) else v
        act_dict[f"action.{k}"] = _to_tensor(arr)

    return make_transition(act=act_dict)


def to_transition_robot_observation(observation: dict[str, Any]) -> EnvTransition:
    """
    Convert a raw robot observation dict (provided under 'robot_observation') into an EnvTransition:
        OBSERVATION.state  -> scalars/tensors
        OBSERVATION.images -> pass-through uint8 HWC images
    """
    state, images = _split_obs_to_state_and_images(observation)

    obs_dict: dict[str, Any] = {}
    for k, v in state.items():
        arr = np.array(v) if np.isscalar(v) else v
        obs_dict[f"observation.state.{k}"] = _to_tensor(arr)

    for cam, img in images.items():
        obs_dict[f"observation.images.{cam}"] = img  # keep raw uint8 HWC

    return make_transition(obs=obs_dict)


def to_output_robot_action(transition: EnvTransition) -> dict[str, Any]:
    """
    Strip 'action.' so the result can be passed straight into Robot.send_action().
    """
    out: dict[str, Any] = {}
    for k, v in (transition.get(TransitionKey.ACTION) or {}).items():
        if isinstance(k, str) and k.startswith("action."):
            from lerobot.processor.utils import _from_tensor

            out[k[len("action.") :]] = _from_tensor(v)
    return out


def make_to_output_dataset(features: dict[str, dict]):
    """
    Build a to_output(...) function that returns a dataset-ready frame dict.
    The function can be called with:
      - a single EnvTransition, or
      - an iterable of EnvTransitions (which will be merged).
    The packing order of vectors is taken from features['...']['names'].
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
