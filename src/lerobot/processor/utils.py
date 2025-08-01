# src/lerobot/processor/utils.py

from __future__ import annotations

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


def transition_to_dataset_batch(transition: EnvTransition) -> dict[str, Any]:
    """
    Flatten EnvTransition into a dataset-ready frame:
      - observation.* are copied as-is
      - action.* remain action.* (no double 'action.action.')
      - next.*, info and selected complementary_data
    """
    batch: dict[str, Any] = {}

    # observation.*
    observation = transition.get(TransitionKey.OBSERVATION) or {}
    if isinstance(observation, dict):
        batch.update(observation)

    # action.* (keep 'action.' if already present)
    action = transition.get(TransitionKey.ACTION) or {}
    for k, v in action.items():
        key = k if isinstance(k, str) and k.startswith("action.") else f"action.{k}"
        batch[key] = _from_tensor(v)

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
    out = {**base}
    for tr in others:
        if tr.get(TransitionKey.OBSERVATION):
            out.setdefault(TransitionKey.OBSERVATION, {}).update(tr[TransitionKey.OBSERVATION])
        if tr.get(TransitionKey.ACTION):
            out.setdefault(TransitionKey.ACTION, {}).update(tr[TransitionKey.ACTION])
        if tr.get(TransitionKey.INFO):
            out.setdefault(TransitionKey.INFO, {}).update(tr[TransitionKey.INFO])
        if tr.get(TransitionKey.COMPLEMENTARY_DATA):
            out.setdefault(TransitionKey.COMPLEMENTARY_DATA, {}).update(tr[TransitionKey.COMPLEMENTARY_DATA])
        # reward/done/truncated: last writer wins
        for k in (TransitionKey.REWARD, TransitionKey.DONE, TransitionKey.TRUNCATED):
            if k in tr:
                out[k] = tr[k]
    return out
