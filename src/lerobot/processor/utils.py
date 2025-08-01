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

from copy import deepcopy
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import DatasetFeatureType

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


def prepare_teleop_action_pipeline(teleop_action: dict) -> EnvTransition:
    act_dict = {}
    for k, v in teleop_action.items():
        arr = np.array(v) if np.isscalar(v) else v
        act_dict[f"action.{k}"] = _to_tensor(arr)
    return make_transition(act=act_dict)


def prepare_robot_observation_pipeline(robot_obs: dict) -> EnvTransition:
    state, images = _split_obs_to_state_and_images(robot_obs)
    obs_dict = {}
    for k, v in state.items():
        arr = np.array(v) if np.isscalar(v) else v
        obs_dict[f"observation.state.{k}"] = _to_tensor(arr)
    for cam, img in images.items():
        obs_dict[f"observation.images.{cam}"] = img
    return make_transition(obs=obs_dict)


def pipeline_to_robot_action(transition: EnvTransition) -> dict[str, Any]:
    """Strip 'action.' for Robot.send_action"""
    out = {}
    for k, v in (transition.get(TransitionKey.ACTION) or {}).items():
        if isinstance(k, str) and k.startswith("action."):
            out[k[len("action.") :]] = _from_tensor(v)
    return out


def transition_to_dataset_batch(
    transition: EnvTransition,
    action_type: DatasetFeatureType | list[DatasetFeatureType] = DatasetFeatureType.JOINT,
) -> dict[str, Any]:
    """
    Build a dataset-ready frame.

    - observation.images.* : pass-through
    - observation.state    : 6-dim EE [x,y,z,wx,wy,wz] (if present)
    - action               : depends on action_type (EE, JOINT, or BOTH -> EE then JOINT)
    - keep next.*, info, *_is_pad, task
    """
    batch: dict[str, Any] = {}

    obs = transition.get(TransitionKey.OBSERVATION) or {}
    act = transition.get(TransitionKey.ACTION) or {}

    def _g(d: dict, key: str, default=0.0):
        val = d.get(key, default)
        return _from_tensor(val)

    # images passthrough
    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(k, str) and k.startswith("observation.images."):
                batch[k] = v

    # pack observation.state (EE 6D if available)
    if isinstance(obs, dict):
        ee_obs_keys = [
            "observation.state.ee.x",
            "observation.state.ee.y",
            "observation.state.ee.z",
            "observation.state.ee.wx",
            "observation.state.ee.wy",
            "observation.state.ee.wz",
        ]
        if all(k in obs for k in ee_obs_keys):
            batch["observation.state"] = np.asarray([_g(obs, k) for k in ee_obs_keys], dtype=np.float32)

    # pack action according to action_type
    action_types = [action_type] if isinstance(action_type, DatasetFeatureType) else list(action_type)

    ee_act_keys = [
        "action.ee.x",
        "action.ee.y",
        "action.ee.z",
        "action.ee.wx",
        "action.ee.wy",
        "action.ee.wz",
    ]
    joint_act_keys = sorted(
        k
        for k in act.keys()
        if isinstance(k, str)
        and k.startswith("action.")
        and k.endswith(".pos")
        and not k.startswith("action.ee.")
    )

    ordered_keys: list[str] = []
    if DatasetFeatureType.EE in action_types:
        ordered_keys.extend(ee_act_keys)
    if DatasetFeatureType.JOINT in action_types:
        ordered_keys.extend(joint_act_keys)

    if ordered_keys:
        batch["action"] = np.asarray([_g(act, k, 0.0) for k in ordered_keys], dtype=np.float32)

    # reward/done/truncated/info
    if TransitionKey.REWARD in transition:
        batch["next.reward"] = _from_tensor(transition[TransitionKey.REWARD])
    if TransitionKey.DONE in transition:
        batch["next.done"] = _from_tensor(transition[TransitionKey.DONE])
    if TransitionKey.TRUNCATED in transition:
        batch["next.truncated"] = _from_tensor(transition[TransitionKey.TRUNCATED])
    if TransitionKey.INFO in transition:
        batch["info"] = transition[TransitionKey.INFO] or {}

    # complementary data: keep *_is_pad and task
    comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
    for k, v in comp.items():
        if k.endswith("_is_pad"):
            batch[k] = v
    if "task" in comp:
        batch["task"] = comp["task"]

    return batch


def merge_transitions(base: EnvTransition, *others: EnvTransition) -> EnvTransition:
    out = deepcopy(base)
    for tr in others:
        if tr.get(TransitionKey.OBSERVATION):
            out.setdefault(TransitionKey.OBSERVATION, {}).update(deepcopy(tr[TransitionKey.OBSERVATION]))
        if tr.get(TransitionKey.ACTION):
            out.setdefault(TransitionKey.ACTION, {}).update(deepcopy(tr[TransitionKey.ACTION]))
        if tr.get(TransitionKey.INFO):
            out.setdefault(TransitionKey.INFO, {}).update(deepcopy(tr[TransitionKey.INFO]))
        if tr.get(TransitionKey.COMPLEMENTARY_DATA):
            out.setdefault(TransitionKey.COMPLEMENTARY_DATA, {}).update(
                deepcopy(tr[TransitionKey.COMPLEMENTARY_DATA])
            )
        # reward/done/truncated: last writer wins
        for k in (TransitionKey.REWARD, TransitionKey.DONE, TransitionKey.TRUNCATED):
            if k in tr:
                out[k] = tr[k]
    return out
