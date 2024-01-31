import importlib
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform

from lerobot.common.utils import set_seed

_has_gym = importlib.util.find_spec("gym") is not None
_has_simxarm = importlib.util.find_spec("simxarm") is not None and _has_gym


class SimxarmEnv(EnvBase):

    def __init__(
        self,
        task,
        from_pixels: bool = False,
        pixels_only: bool = False,
        image_size=None,
        seed=1337,
        device="cpu",
    ):
        super().__init__(device=device, batch_size=[])
        self.task = task
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only
        self.image_size = image_size

        if pixels_only:
            assert from_pixels
        if from_pixels:
            assert image_size

        if not _has_simxarm:
            raise ImportError("Cannot import simxarm.")
        if not _has_gym:
            raise ImportError("Cannot import gym.")

        import gym
        from gym.wrappers import TimeLimit
        from simxarm import TASKS

        if self.task not in TASKS:
            raise ValueError(
                f"Unknown task {self.task}. Must be one of {list(TASKS.keys())}"
            )

        self._env = TASKS[self.task]["env"]()
        self._env = TimeLimit(self._env, TASKS[self.task]["episode_length"])

        MAX_NUM_ACTIONS = 4
        num_actions = len(TASKS[self.task]["action_space"])
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))
        self._action_padding = np.zeros(
            (MAX_NUM_ACTIONS - num_actions), dtype=np.float32
        )
        if "w" not in TASKS[self.task]["action_space"]:
            self._action_padding[-1] = 1.0

        self._make_spec()
        self.set_seed(seed)

    def render(self, mode="rgb_array", width=384, height=384):
        return self._env.render(mode, width=width, height=height)

    def _format_raw_obs(self, raw_obs):
        if self.from_pixels:
            camera = self.render(
                mode="rgb_array", width=self.image_size, height=self.image_size
            )
            camera = camera.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            camera = torch.tensor(camera.copy(), dtype=torch.uint8)

            obs = {"camera": camera}

            if not self.pixels_only:
                obs["robot_state"] = torch.tensor(
                    self._env.robot_state, dtype=torch.float32
                )
        else:
            obs = {"state": torch.tensor(raw_obs["observation"], dtype=torch.float32)}

        obs = TensorDict(obs, batch_size=[])
        return obs

    def _reset(self, tensordict: Optional[TensorDict] = None):
        td = tensordict
        if td is None or td.is_empty():
            raw_obs = self._env.reset()

            td = TensorDict(
                {
                    "observation": self._format_raw_obs(raw_obs),
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
        raw_obs, reward, done, info = self._env.step(action)

        td = TensorDict(
            {
                "observation": self._format_raw_obs(raw_obs),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done], dtype=torch.bool),
                "success": torch.tensor([info["success"]], dtype=torch.bool),
            },
            batch_size=[],
        )
        return td

    def _make_spec(self):
        obs = {}
        if self.from_pixels:
            obs["camera"] = BoundedTensorSpec(
                low=0,
                high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=torch.uint8,
                device=self.device,
            )
            if not self.pixels_only:
                obs["robot_state"] = UnboundedContinuousTensorSpec(
                    shape=(len(self._env.robot_state),),
                    dtype=torch.float32,
                    device=self.device,
                )
        else:
            # TODO(rcadene): add observation_space achieved_goal and desired_goal?
            obs["state"] = UnboundedContinuousTensorSpec(
                shape=self._env.observation_space["observation"].shape,
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

        self.done_spec = DiscreteTensorSpec(
            2,
            shape=(1,),
            dtype=torch.bool,
            device=self.device,
        )

        self.success_spec = DiscreteTensorSpec(
            2,
            shape=(1,),
            dtype=torch.bool,
            device=self.device,
        )

    def _set_seed(self, seed: Optional[int]):
        set_seed(seed)
        self._env.seed(seed)
