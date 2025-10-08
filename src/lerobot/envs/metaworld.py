#!/usr/bin/env python

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
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import metaworld
import metaworld.policies as policies
import numpy as np
from gymnasium import spaces

data = {
    "TASK_DESCRIPTIONS": {
        "assembly-v3": "Pick up a nut and place it onto a peg",
        "basketball-v3": "Dunk the basketball into the basket",
        "bin-picking-v3": "Grasp the puck from one bin and place it into another bin",
        "box-close-v3": "Grasp the cover and close the box with it",
        "button-press-topdown-v3": "Press a button from the top",
        "button-press-topdown-wall-v3": "Bypass a wall and press a button from the top",
        "button-press-v3": "Press a button",
        "button-press-wall-v3": "Bypass a wall and press a button",
        "coffee-button-v3": "Push a button on the coffee machine",
        "coffee-pull-v3": "Pull a mug from a coffee machine",
        "coffee-push-v3": "Push a mug under a coffee machine",
        "dial-turn-v3": "Rotate a dial 180 degrees",
        "disassemble-v3": "Pick a nut out of a peg",
        "door-close-v3": "Close a door with a revolving joint",
        "door-lock-v3": "Lock the door by rotating the lock clockwise",
        "door-open-v3": "Open a door with a revolving joint",
        "door-unlock-v3": "Unlock the door by rotating the lock counter-clockwise",
        "hand-insert-v3": "Insert the gripper into a hole",
        "drawer-close-v3": "Push and close a drawer",
        "drawer-open-v3": "Open a drawer",
        "faucet-open-v3": "Rotate the faucet counter-clockwise",
        "faucet-close-v3": "Rotate the faucet clockwise",
        "hammer-v3": "Hammer a screw on the wall",
        "handle-press-side-v3": "Press a handle down sideways",
        "handle-press-v3": "Press a handle down",
        "handle-pull-side-v3": "Pull a handle up sideways",
        "handle-pull-v3": "Pull a handle up",
        "lever-pull-v3": "Pull a lever down 90 degrees",
        "peg-insert-side-v3": "Insert a peg sideways",
        "pick-place-wall-v3": "Pick a puck, bypass a wall and place the puck",
        "pick-out-of-hole-v3": "Pick up a puck from a hole",
        "reach-v3": "Reach a goal position",
        "push-back-v3": "Push the puck to a goal",
        "push-v3": "Push the puck to a goal",
        "pick-place-v3": "Pick and place a puck to a goal",
        "plate-slide-v3": "Slide a plate into a cabinet",
        "plate-slide-side-v3": "Slide a plate into a cabinet sideways",
        "plate-slide-back-v3": "Get a plate from the cabinet",
        "plate-slide-back-side-v3": "Get a plate from the cabinet sideways",
        "peg-unplug-side-v3": "Unplug a peg sideways",
        "soccer-v3": "Kick a soccer into the goal",
        "stick-push-v3": "Grasp a stick and push a box using the stick",
        "stick-pull-v3": "Grasp a stick and pull a box with the stick",
        "push-wall-v3": "Bypass a wall and push a puck to a goal",
        "reach-wall-v3": "Bypass a wall and reach a goal",
        "shelf-place-v3": "Pick and place a puck onto a shelf",
        "sweep-into-v3": "Sweep a puck into a hole",
        "sweep-v3": "Sweep a puck off the table",
        "window-open-v3": "Push and open a window",
        "window-close-v3": "Push and close a window",
    },
    "TASK_NAME_TO_ID": {
        "assembly-v3": 0,
        "basketball-v3": 1,
        "bin-picking-v3": 2,
        "box-close-v3": 3,
        "button-press-topdown-v3": 4,
        "button-press-topdown-wall-v3": 5,
        "button-press-v3": 6,
        "button-press-wall-v3": 7,
        "coffee-button-v3": 8,
        "coffee-pull-v3": 9,
        "coffee-push-v3": 10,
        "dial-turn-v3": 11,
        "disassemble-v3": 12,
        "door-close-v3": 13,
        "door-lock-v3": 14,
        "door-open-v3": 15,
        "door-unlock-v3": 16,
        "drawer-close-v3": 17,
        "drawer-open-v3": 18,
        "faucet-close-v3": 19,
        "faucet-open-v3": 20,
        "hammer-v3": 21,
        "hand-insert-v3": 22,
        "handle-press-side-v3": 23,
        "handle-press-v3": 24,
        "handle-pull-side-v3": 25,
        "handle-pull-v3": 26,
        "lever-pull-v3": 27,
        "peg-insert-side-v3": 28,
        "peg-unplug-side-v3": 29,
        "pick-out-of-hole-v3": 30,
        "pick-place-v3": 31,
        "pick-place-wall-v3": 32,
        "plate-slide-back-side-v3": 33,
        "plate-slide-back-v3": 34,
        "plate-slide-side-v3": 35,
        "plate-slide-v3": 36,
        "push-back-v3": 37,
        "push-v3": 38,
        "push-wall-v3": 39,
        "reach-v3": 40,
        "reach-wall-v3": 41,
        "shelf-place-v3": 42,
        "soccer-v3": 43,
        "stick-pull-v3": 44,
        "stick-push-v3": 45,
        "sweep-into-v3": 46,
        "sweep-v3": 47,
        "window-open-v3": 48,
        "window-close-v3": 49,
    },
    "DIFFICULTY_TO_TASKS": {
        "easy": [
            "button-press-v3",
            "button-press-topdown-v3",
            "button-press-topdown-wall-v3",
            "button-press-wall-v3",
            "coffee-button-v3",
            "dial-turn-v3",
            "door-close-v3",
            "door-lock-v3",
            "door-open-v3",
            "door-unlock-v3",
            "drawer-close-v3",
            "drawer-open-v3",
            "faucet-close-v3",
            "faucet-open-v3",
            "handle-press-v3",
            "handle-press-side-v3",
            "handle-pull-v3",
            "handle-pull-side-v3",
            "lever-pull-v3",
            "plate-slide-v3",
            "plate-slide-back-v3",
            "plate-slide-back-side-v3",
            "plate-slide-side-v3",
            "reach-v3",
            "reach-wall-v3",
            "window-close-v3",
            "window-open-v3",
            "peg-unplug-side-v3",
        ],
        "medium": [
            "basketball-v3",
            "bin-picking-v3",
            "box-close-v3",
            "coffee-pull-v3",
            "coffee-push-v3",
            "hammer-v3",
            "peg-insert-side-v3",
            "push-wall-v3",
            "soccer-v3",
            "sweep-v3",
            "sweep-into-v3",
        ],
        "hard": [
            "assembly-v3",
            "hand-insert-v3",
            "pick-out-of-hole-v3",
            "pick-place-v3",
            "push-v3",
            "push-back-v3",
        ],
        "very_hard": [
            "shelf-place-v3",
            "disassemble-v3",
            "stick-pull-v3",
            "stick-push-v3",
            "pick-place-wall-v3",
        ],
    },
    "TASK_POLICY_MAPPING": {
        "assembly-v3": "SawyerAssemblyV3Policy",
        "basketball-v3": "SawyerBasketballV3Policy",
        "bin-picking-v3": "SawyerBinPickingV3Policy",
        "box-close-v3": "SawyerBoxCloseV3Policy",
        "button-press-topdown-v3": "SawyerButtonPressTopdownV3Policy",
        "button-press-topdown-wall-v3": "SawyerButtonPressTopdownWallV3Policy",
        "button-press-v3": "SawyerButtonPressV3Policy",
        "button-press-wall-v3": "SawyerButtonPressWallV3Policy",
        "coffee-button-v3": "SawyerCoffeeButtonV3Policy",
        "coffee-pull-v3": "SawyerCoffeePullV3Policy",
        "coffee-push-v3": "SawyerCoffeePushV3Policy",
        "dial-turn-v3": "SawyerDialTurnV3Policy",
        "disassemble-v3": "SawyerDisassembleV3Policy",
        "door-close-v3": "SawyerDoorCloseV3Policy",
        "door-lock-v3": "SawyerDoorLockV3Policy",
        "door-open-v3": "SawyerDoorOpenV3Policy",
        "door-unlock-v3": "SawyerDoorUnlockV3Policy",
        "drawer-close-v3": "SawyerDrawerCloseV3Policy",
        "drawer-open-v3": "SawyerDrawerOpenV3Policy",
        "faucet-close-v3": "SawyerFaucetCloseV3Policy",
        "faucet-open-v3": "SawyerFaucetOpenV3Policy",
        "hammer-v3": "SawyerHammerV3Policy",
        "hand-insert-v3": "SawyerHandInsertV3Policy",
        "handle-press-side-v3": "SawyerHandlePressSideV3Policy",
        "handle-press-v3": "SawyerHandlePressV3Policy",
        "handle-pull-side-v3": "SawyerHandlePullSideV3Policy",
        "handle-pull-v3": "SawyerHandlePullV3Policy",
        "lever-pull-v3": "SawyerLeverPullV3Policy",
        "peg-insert-side-v3": "SawyerPegInsertionSideV3Policy",
        "peg-unplug-side-v3": "SawyerPegUnplugSideV3Policy",
        "pick-out-of-hole-v3": "SawyerPickOutOfHoleV3Policy",
        "pick-place-v3": "SawyerPickPlaceV3Policy",
        "pick-place-wall-v3": "SawyerPickPlaceWallV3Policy",
        "plate-slide-back-side-v3": "SawyerPlateSlideBackSideV3Policy",
        "plate-slide-back-v3": "SawyerPlateSlideBackV3Policy",
        "plate-slide-side-v3": "SawyerPlateSlideSideV3Policy",
        "plate-slide-v3": "SawyerPlateSlideV3Policy",
        "push-back-v3": "SawyerPushBackV3Policy",
        "push-v3": "SawyerPushV3Policy",
        "push-wall-v3": "SawyerPushWallV3Policy",
        "reach-v3": "SawyerReachV3Policy",
        "reach-wall-v3": "SawyerReachWallV3Policy",
        "shelf-place-v3": "SawyerShelfPlaceV3Policy",
        "soccer-v3": "SawyerSoccerV3Policy",
        "stick-pull-v3": "SawyerStickPullV3Policy",
        "stick-push-v3": "SawyerStickPushV3Policy",
        "sweep-into-v3": "SawyerSweepIntoV3Policy",
        "sweep-v3": "SawyerSweepV3Policy",
        "window-open-v3": "SawyerWindowOpenV3Policy",
        "window-close-v3": "SawyerWindowCloseV3Policy",
    },
}
# extract and type-check top-level dicts
task_descriptions_obj = data.get("TASK_DESCRIPTIONS")
if not isinstance(task_descriptions_obj, dict):
    raise TypeError("Expected TASK_DESCRIPTIONS to be a dict[str, str]")

TASK_DESCRIPTIONS: dict[str, str] = task_descriptions_obj

task_name_to_id_obj = data.get("TASK_NAME_TO_ID")
if not isinstance(task_name_to_id_obj, dict):
    raise TypeError("Expected TASK_NAME_TO_ID to be a dict[str, int]")

TASK_NAME_TO_ID: dict[str, int] = task_name_to_id_obj

# difficulty -> tasks mapping
difficulty_to_tasks = data.get("DIFFICULTY_TO_TASKS")
if not isinstance(difficulty_to_tasks, dict):
    raise TypeError("Expected 'DIFFICULTY_TO_TASKS' to be a dict[str, list[str]]")
DIFFICULTY_TO_TASKS: dict[str, list[str]] = difficulty_to_tasks

# convert policy strings -> actual policy classes
task_policy_mapping = data.get("TASK_POLICY_MAPPING")
if not isinstance(task_policy_mapping, dict):
    raise TypeError("Expected 'TASK_POLICY_MAPPING' to be a dict[str, str]")
TASK_POLICY_MAPPING: dict[str, Any] = {
    task_name: getattr(policies, policy_class_name)
    for task_name, policy_class_name in task_policy_mapping.items()
}
ACTION_DIM = 4
OBS_DIM = 4


class MetaworldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task,
        camera_name="corner2",
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=480,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task.replace("metaworld-", "")
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.camera_name = camera_name

        self._env = self._make_envs_task(self.task)
        self._max_episode_steps = self._env.max_path_length
        self.task_description = TASK_DESCRIPTIONS[self.task]

        self.expert_policy = TASK_POLICY_MAPPING[self.task]()

        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(OBS_DIM,),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(ACTION_DIM,), dtype=np.float32)

    def render(self) -> np.ndarray:
        """
        Render the current environment frame.

        Returns:
            np.ndarray: The rendered RGB image from the environment.
        """
        image = self._env.render()
        if self.camera_name == "corner2":
            # Images from this camera are flipped — correct them
            image = np.flip(image, (0, 1))
        return image

    def _make_envs_task(self, env_name: str):
        mt1 = metaworld.MT1(env_name, seed=42)
        env = mt1.train_classes[env_name](render_mode="rgb_array", camera_name=self.camera_name)
        env.set_task(mt1.train_tasks[0])
        if self.camera_name == "corner2":
            env.model.cam_pos[2] = [
                0.75,
                0.075,
                0.7,
            ]  # corner2 position, similar to https://arxiv.org/pdf/2206.14244
        env.reset()
        env._freeze_rand_vec = False  # otherwise no randomization
        return env

    def _format_raw_obs(self, raw_obs: np.ndarray, env=None) -> dict[str, Any]:
        image = None
        if env is not None:
            image = env.render()
            if self.camera_name == "corner2":
                # NOTE: The "corner2" camera in MetaWorld environments outputs images with both axes inverted.
                image = np.flip(image, (0, 1))
        agent_pos = raw_obs[:4]
        if self.obs_type == "state":
            raise NotImplementedError()

        elif self.obs_type in ("pixels", "pixels_agent_pos"):
            assert image is not None, (
                "Expected `image` to be rendered before constructing pixel-based observations. "
                "This likely means `env.render()` returned None or the environment was not provided."
            )

            if self.obs_type == "pixels":
                obs = {"pixels": image.copy()}

            else:  # pixels_agent_pos
                obs = {
                    "pixels": image.copy(),
                    "agent_pos": agent_pos,
                }
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")
        return obs

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (Optional[int]): Random seed for environment initialization.

        Returns:
            observation (Dict[str, Any]): The initial formatted observation.
            info (Dict[str, Any]): Additional info about the reset state.
        """
        super().reset(seed=seed)

        raw_obs, info = self._env.reset(seed=seed)

        observation = self._format_raw_obs(raw_obs, env=self._env)

        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Perform one environment step.

        Args:
            action (np.ndarray): The action to execute, must be 1-D with shape (action_dim,).

        Returns:
            observation (Dict[str, Any]): The formatted observation after the step.
            reward (float): The scalar reward for this step.
            terminated (bool): Whether the episode terminated successfully.
            truncated (bool): Whether the episode was truncated due to a time limit.
            info (Dict[str, Any]): Additional environment info.
        """
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_obs, reward, _done, truncated, info = self._env.step(action)

        # Determine whether the task was successful
        is_success = bool(info.get("success", 0))
        terminated = is_success
        info["is_success"] = is_success

        # Format the raw observation into the expected structure
        observation = self._format_raw_obs(raw_obs, env=self._env)

        return observation, reward, terminated, truncated, info

    def close(self):
        self._env.close()


# ---- Main API ----------------------------------------------------------------


def create_metaworld_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized Meta-World environments with a consistent return shape.

    Returns:
        dict[task_group][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single difficulty group (e.g., "easy", "medium", "hard") or a comma-separated list.
        - If a task name is not in DIFFICULTY_TO_TASKS, we treat it as a single custom task.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_groups = [t.strip() for t in task.split(",") if t.strip()]
    if not task_groups:
        raise ValueError("`task` must contain at least one Meta-World task or difficulty group.")

    print(f"Creating Meta-World envs | task_groups={task_groups} | n_envs(per task)={n_envs}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for group in task_groups:
        # if not in difficulty presets, treat it as a single custom task
        tasks = DIFFICULTY_TO_TASKS.get(group, [group])

        for tid, task_name in enumerate(tasks):
            print(f"Building vec env | group={group} | task_id={tid} | task={task_name}")

            # build n_envs factories
            fns = [(lambda tn=task_name: MetaworldEnv(task=tn, **gym_kwargs)) for _ in range(n_envs)]

            out[group][tid] = env_cls(fns)

    # return a plain dict for consistency
    return {group: dict(task_map) for group, task_map in out.items()}
