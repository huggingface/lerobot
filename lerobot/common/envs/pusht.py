import importlib
from typing import Optional

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
_has_diffpolicy = importlib.util.find_spec("diffusion_policy") is not None and _has_gym


class PushtEnv(EnvBase):
    def __init__(
        self,
        frame_skip: int = 1,
        from_pixels: bool = False,
        pixels_only: bool = False,
        image_size=None,
        seed=1337,
        device="cpu",
    ):
        super().__init__(device=device, batch_size=[])
        self.frame_skip = frame_skip
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only
        self.image_size = image_size

        if pixels_only:
            assert from_pixels
        if from_pixels:
            assert image_size

        if not _has_diffpolicy:
            raise ImportError("Cannot import diffusion_policy.")
        if not _has_gym:
            raise ImportError("Cannot import gym.")

        # TODO(rcadene) (PushTEnv is similar to PushTImageEnv, but without the image rendering, it's faster to iterate on)
        # from diffusion_policy.env.pusht.pusht_env import PushTEnv

        if not from_pixels:
            raise NotImplementedError("Use PushTEnv, instead of PushTImageEnv")
        from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

        self._env = PushTImageEnv(render_size=self.image_size)

        self._make_spec()
        self._current_seed = self.set_seed(seed)

    def render(self, mode="rgb_array", width=384, height=384):
        if width != height:
            raise NotImplementedError()
        tmp = self._env.render_size
        self._env.render_size = width
        out = self._env.render(mode)
        self._env.render_size = tmp
        return out

    def _format_raw_obs(self, raw_obs):
        if self.from_pixels:
            obs = {"image": torch.from_numpy(raw_obs["image"])}

            if not self.pixels_only:
                obs["state"] = torch.from_numpy(raw_obs["agent_pos"]).type(torch.float32)
        else:
            # TODO:
            obs = {"state": torch.from_numpy(raw_obs["observation"]).type(torch.float32)}

        obs = TensorDict(obs, batch_size=[])
        return obs

    def _reset(self, tensordict: Optional[TensorDict] = None):
        td = tensordict
        if td is None or td.is_empty():
            # we need to handle seed iteration, since self._env.reset() rely an internal _seed.
            self._current_seed += 1
            self.set_seed(self._current_seed)
            raw_obs = self._env.reset()
            assert self._current_seed == self._env._seed

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
        # remove batch dim
        action = td["action"].squeeze(0).numpy()
        # step expects shape=(4,) so we pad if necessary
        # TODO(rcadene): add info["is_success"] and info["success"] ?
        sum_reward = 0
        for _ in range(self.frame_skip):
            raw_obs, reward, done, info = self._env.step(action)
            sum_reward += reward

        td = TensorDict(
            {
                "observation": self._format_raw_obs(raw_obs),
                "reward": torch.tensor([sum_reward], dtype=torch.float32),
                # succes and done are true when coverage > self.success_threshold in env
                "done": torch.tensor([done], dtype=torch.bool),
                "success": torch.tensor([done], dtype=torch.bool),
            },
            batch_size=[],
        )
        return td

    def _make_spec(self):
        obs = {}
        if self.from_pixels:
            obs["image"] = BoundedTensorSpec(
                low=0,
                high=1,
                shape=(3, self.image_size, self.image_size),
                dtype=torch.float32,
                device=self.device,
            )
            if not self.pixels_only:
                obs["state"] = BoundedTensorSpec(
                    low=0,
                    high=512,
                    shape=self._env.observation_space["agent_pos"].shape,
                    dtype=torch.float32,
                    device=self.device,
                )
        else:
            # TODO(rcadene): add observation_space achieved_goal and desired_goal?
            obs["state"] = UnboundedContinuousTensorSpec(
                # TODO:
                shape=self._env.observation_space["observation"].shape,
                dtype=torch.float32,
                device=self.device,
            )
        self.observation_spec = CompositeSpec({"observation": obs})

        self.action_spec = _gym_to_torchrl_spec_transform(
            self._env.action_space,
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
        self._env.seed(seed)
