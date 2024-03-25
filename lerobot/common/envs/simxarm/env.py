import importlib
from collections import deque
from typing import Optional

import einops
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform

from lerobot.common.envs.abstract import AbstractEnv
from lerobot.common.utils import set_seed

MAX_NUM_ACTIONS = 4

_has_gym = importlib.util.find_spec("gymnasium") is not None


class SimxarmEnv(AbstractEnv):
    def __init__(
        self,
        task,
        frame_skip: int = 1,
        from_pixels: bool = False,
        pixels_only: bool = False,
        image_size=None,
        seed=1337,
        device="cpu",
        num_prev_obs=0,
        num_prev_action=0,
    ):
        super().__init__(
            task=task,
            frame_skip=frame_skip,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
            image_size=image_size,
            seed=seed,
            device=device,
            num_prev_obs=num_prev_obs,
            num_prev_action=num_prev_action,
        )

    def _make_env(self):
        # if not _has_simxarm:
        #     raise ImportError("Cannot import simxarm.")
        if not _has_gym:
            raise ImportError("Cannot import gym.")

        import gymnasium

        from lerobot.common.envs.simxarm.simxarm import TASKS

        if self.task not in TASKS:
            raise ValueError(f"Unknown task {self.task}. Must be one of {list(TASKS.keys())}")

        self._env = TASKS[self.task]["env"]()

        num_actions = len(TASKS[self.task]["action_space"])
        self._action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))
        self._action_padding = np.zeros((MAX_NUM_ACTIONS - num_actions), dtype=np.float32)
        if "w" not in TASKS[self.task]["action_space"]:
            self._action_padding[-1] = 1.0

    def render(self, mode="rgb_array", width=384, height=384):
        return self._env.render(mode, width=width, height=height)

    def _format_raw_obs(self, raw_obs):
        if self.from_pixels:
            image = self.render(mode="rgb_array", width=self.image_size, height=self.image_size)
            image = image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            image = torch.tensor(image.copy(), dtype=torch.uint8)

            obs = {"image": image}

            if not self.pixels_only:
                obs["state"] = torch.tensor(self._env.robot_state, dtype=torch.float32)
        else:
            obs = {"state": torch.tensor(raw_obs["observation"], dtype=torch.float32)}

        # obs = TensorDict(obs, batch_size=[])
        return obs

    def _reset(self, tensordict: Optional[TensorDict] = None):
        td = tensordict
        if td is None or td.is_empty():
            raw_obs = self._env.reset()

            obs = self._format_raw_obs(raw_obs)

            if self.num_prev_obs > 0:
                stacked_obs = {}
                if "image" in obs:
                    self._prev_obs_image_queue = deque(
                        [obs["image"]] * (self.num_prev_obs + 1), maxlen=(self.num_prev_obs + 1)
                    )
                    stacked_obs["image"] = torch.stack(list(self._prev_obs_image_queue))
                if "state" in obs:
                    self._prev_obs_state_queue = deque(
                        [obs["state"]] * (self.num_prev_obs + 1), maxlen=(self.num_prev_obs + 1)
                    )
                    stacked_obs["state"] = torch.stack(list(self._prev_obs_state_queue))
                obs = stacked_obs

            td = TensorDict(
                {
                    "observation": TensorDict(obs, batch_size=[]),
                    "done": torch.tensor([False], dtype=torch.bool),
                },
                batch_size=[],
            )
        else:
            raise NotImplementedError()

        return td

    def _step(self, tensordict: TensorDict):
        td = tensordict
        action = td["action"].numpy()
        # step expects shape=(4,) so we pad if necessary
        action = np.concatenate([action, self._action_padding])
        # TODO(rcadene): add info["is_success"] and info["success"] ?
        sum_reward = 0

        if action.ndim == 1:
            action = einops.repeat(action, "c -> t c", t=self.frame_skip)
        else:
            if self.frame_skip > 1:
                raise NotImplementedError()

        num_action_steps = action.shape[0]
        for i in range(num_action_steps):
            raw_obs, reward, done, info = self._env.step(action[i])
            sum_reward += reward

            obs = self._format_raw_obs(raw_obs)

            if self.num_prev_obs > 0:
                stacked_obs = {}
                if "image" in obs:
                    self._prev_obs_image_queue.append(obs["image"])
                    stacked_obs["image"] = torch.stack(list(self._prev_obs_image_queue))
                if "state" in obs:
                    self._prev_obs_state_queue.append(obs["state"])
                    stacked_obs["state"] = torch.stack(list(self._prev_obs_state_queue))
                obs = stacked_obs

        td = TensorDict(
            {
                "observation": self._format_raw_obs(raw_obs),
                "reward": torch.tensor([sum_reward], dtype=torch.float32),
                "done": torch.tensor([done], dtype=torch.bool),
                "success": torch.tensor([info["success"]], dtype=torch.bool),
            },
            batch_size=[],
        )
        return td

    def _make_spec(self):
        obs = {}
        if self.from_pixels:
            image_shape = (3, self.image_size, self.image_size)
            if self.num_prev_obs > 0:
                image_shape = (self.num_prev_obs + 1, *image_shape)

            obs["image"] = BoundedTensorSpec(
                low=0,
                high=255,
                shape=image_shape,
                dtype=torch.uint8,
                device=self.device,
            )
            if not self.pixels_only:
                state_shape = (len(self._env.robot_state),)
                if self.num_prev_obs > 0:
                    state_shape = (self.num_prev_obs + 1, *state_shape)

                obs["state"] = UnboundedContinuousTensorSpec(
                    shape=state_shape,
                    dtype=torch.float32,
                    device=self.device,
                )
        else:
            # TODO(rcadene): add observation_space achieved_goal and desired_goal?
            state_shape = self._env.observation_space["observation"].shape
            if self.num_prev_obs > 0:
                state_shape = (self.num_prev_obs + 1, *state_shape)

            obs["state"] = UnboundedContinuousTensorSpec(
                # TODO:
                shape=state_shape,
                dtype=torch.float32,
                device=self.device,
            )
        self.observation_spec = CompositeSpec({"observation": obs})

        self.action_spec = _gym_to_torchrl_spec_transform(
            self._action_space,
            device=self.device,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32,
            device=self.device,
        )

        self.done_spec = CompositeSpec(
            {
                "done": DiscreteTensorSpec(
                    2,
                    shape=(1,),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "success": DiscreteTensorSpec(
                    2,
                    shape=(1,),
                    dtype=torch.bool,
                    device=self.device,
                ),
            }
        )

    def _set_seed(self, seed: Optional[int]):
        set_seed(seed)
        # self._env.seed(seed)
        # self._env.action_space.seed(seed)
        # self.set_seed(seed)
        self._seed = seed
