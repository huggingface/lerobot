import importlib
import logging
from collections import deque
from typing import Optional

import cv2
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
from lerobot.common.utils import set_global_seed

_has_gym = importlib.util.find_spec("gymnasium") is not None


class PushtEnv(AbstractEnv):
    name = "pusht"
    available_tasks = ["pusht"]
    _reset_warning_issued = False

    def __init__(
        self,
        task="pusht",
        frame_skip: int = 1,
        from_pixels: bool = False,
        pixels_only: bool = False,
        image_size=None,
        seed=1337,
        device="cpu",
        num_prev_obs=1,
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
        if not _has_gym:
            raise ImportError("Cannot import gymnasium.")

        # TODO(rcadene) (PushTEnv is similar to PushTImageEnv, but without the image rendering, it's faster to iterate on)
        # from lerobot.common.envs.pusht.pusht_env import PushTEnv

        if not self.from_pixels:
            raise NotImplementedError("Use PushTEnv, instead of PushTImageEnv")
        from lerobot.common.envs.pusht.pusht_image_env import PushTImageEnv

        self._env = PushTImageEnv(render_size=self.image_size)

    def render(self, mode="rgb_array", width=96, height=96, with_marker=True):
        """
        with_marker adds a cursor showing the targeted action for the controller.
        """
        if width != height:
            raise NotImplementedError()
        tmp = self._env.render_size
        if width != self._env.render_size:
            self._env.render_cache = None
            self._env.render_size = width
        out = self._env.render(mode).copy()
        if with_marker and self._env.latest_action is not None:
            action = np.array(self._env.latest_action)
            coord = (action / 512 * self._env.render_size).astype(np.int32)
            marker_size = int(8 / 96 * self._env.render_size)
            thickness = int(1 / 96 * self._env.render_size)
            cv2.drawMarker(
                out,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        self._env.render_size = tmp
        return out

    def _format_raw_obs(self, raw_obs):
        if self.from_pixels:
            image = torch.from_numpy(raw_obs["image"])
            obs = {"image": image}

            if not self.pixels_only:
                obs["state"] = torch.from_numpy(raw_obs["agent_pos"]).type(torch.float32)
        else:
            # TODO:
            obs = {"state": torch.from_numpy(raw_obs["observation"]).type(torch.float32)}

        return obs

    def _reset(self, tensordict: Optional[TensorDict] = None):
        if tensordict is not None and not PushtEnv._reset_warning_issued:
            logging.warning(f"{self.__class__.__name__}._reset ignores the provided tensordict.")
            PushtEnv._reset_warning_issued = True

        # Seed the environment and update the seed to be used for the next reset.
        self._next_seed = self.set_seed(self._next_seed)
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

        return td

    def _step(self, tensordict: TensorDict):
        td = tensordict
        action = td["action"].numpy()
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?

        raw_obs, reward, done, info = self._env.step(action)

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
                "observation": TensorDict(obs, batch_size=[]),
                "reward": torch.tensor([reward], dtype=torch.float32),
                # success and done are true when coverage > self.success_threshold in env
                "done": torch.tensor([done], dtype=torch.bool),
                "success": torch.tensor([done], dtype=torch.bool),
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
                state_shape = self._env.observation_space["agent_pos"].shape
                if self.num_prev_obs > 0:
                    state_shape = (self.num_prev_obs + 1, *state_shape)

                obs["state"] = BoundedTensorSpec(
                    low=0,
                    high=512,
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
        # Set global seed.
        set_global_seed(seed)
        # Set PushTImageEnv seed as it relies on it's own internal _seed attribute.
        self._env.seed(seed)
