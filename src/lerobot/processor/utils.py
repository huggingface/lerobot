from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lerobot.robots.robot import Robot
from lerobot.teleoperators.teleoperator import Teleoperator

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


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if np.isscalar(x):
        return np.array(x)
    return np.asarray(x)


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


def robot_observation_to_pipeline(robot: Robot) -> EnvTransition:
    """
    Reads the robot observation and converts it into an EnvTransition.
    observation.* keys are populated; action is empty.
    Numbers/float arrays -> torch.Tensor; images (uint8 HWC) stay as np.ndarray.
    """
    obs = robot.get_observation()
    state, images = _split_obs_to_state_and_images(obs)

    obs_dict: dict[str, Any] = {}
    for k, v in state.items():
        obs_dict[f"observation.state.{k}"] = _to_tensor(np.array(v) if np.isscalar(v) else v)
    for cam, img in images.items():
        obs_dict[f"observation.images.{cam}"] = img

    transition: EnvTransition = {
        TransitionKey.OBSERVATION: obs_dict,
        TransitionKey.ACTION: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            "robot_action_keys": list(robot.action_features.keys()),
            "robot_observation_state_keys": list(state.keys()),
            "robot_observation_image_keys": list(images.keys()),
        },
    }
    return transition


def pipeline_to_robot_action(transition: EnvTransition, robot: Robot | None = None) -> dict[str, Any]:
    """
    Extract a robot action dict from EnvTransition ACTION, keeping only keys
    that the robot supports, and converting tensors to Python scalars/ndarrays.
    Expected ACTION keys (unscoped): e.g. "shoulder_pan.pos", "gripper.pos", ...
    """
    act = transition.get(TransitionKey.ACTION) or {}
    if not isinstance(act, dict):
        return {}
    allowed = set(robot.action_features.keys() if robot is not None else act.keys())
    return {k: _from_tensor(v) for k, v in act.items() if k in allowed}


def teleop_action_to_pipeline(teleop: Teleoperator) -> EnvTransition:
    """
    Convert raw teleop readings into ACTION keys.
    Convention: INSIDE EnvTransition, ACTION keys are **unscoped** (no 'action.' prefix).
    """
    raw = teleop.get_action()  # e.g. {"phone.pos": np.ndarray, "phone.rot": Rotation, ...}
    act_dict: dict[str, Any] = {}
    for k, v in raw.items():
        val = np.array(v) if np.isscalar(v) else v
        act_dict[k] = _to_tensor(val)
    transition: EnvTransition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.ACTION: act_dict,
        TransitionKey.COMPLEMENTARY_DATA: {
            "teleop_action_keys": list(raw.keys()),
            "teleop_feedback_keys": list(teleop.feedback_features.keys()) if teleop.feedback_features else [],
        },
    }
    return transition


def transition_to_dataset_batch(transition: EnvTransition) -> dict[str, Any]:
    """
    Flatten an EnvTransition into a dataset-ready frame:
      - Copy observation.* as-is from OBSERVATION dict
      - For ACTION unscoped keys -> 'action.<key>'
      - next.*, info, and complementary padding/task
    """
    batch: dict[str, Any] = {}

    # observation.*
    observation = transition.get(TransitionKey.OBSERVATION) or {}
    if isinstance(observation, dict):
        batch.update(observation)

    # action.*
    action = transition.get(TransitionKey.ACTION) or {}
    for k, v in action.items():
        batch[f"action.{k}"] = _from_tensor(v)

    # reward/done/truncated/info
    if TransitionKey.REWARD in transition:
        batch["next.reward"] = _from_tensor(transition[TransitionKey.REWARD])
    if TransitionKey.DONE in transition:
        batch["next.done"] = _from_tensor(transition[TransitionKey.DONE])
    if TransitionKey.TRUNCATED in transition:
        batch["next.truncated"] = _from_tensor(transition[TransitionKey.TRUNCATED])
    if TransitionKey.INFO in transition:
        batch["info"] = transition[TransitionKey.INFO] or {}

    # complementary data
    comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
    for k, v in comp.items():
        if k.endswith("_is_pad"):
            batch[k] = v
    if "task" in comp:
        batch["task"] = comp["task"]

    return batch
