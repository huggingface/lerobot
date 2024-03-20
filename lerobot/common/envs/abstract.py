import abc
from collections import deque
from typing import Optional

from tensordict import TensorDict
from torchrl.envs import EnvBase


class AbstractEnv(EnvBase):
    def __init__(
        self,
        task,
        frame_skip: int = 1,
        from_pixels: bool = False,
        pixels_only: bool = False,
        image_size=None,
        seed=1337,
        device="cpu",
        num_prev_obs=1,
        num_prev_action=0,
    ):
        super().__init__(device=device, batch_size=[])
        self.task = task
        self.frame_skip = frame_skip
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only
        self.image_size = image_size
        self.num_prev_obs = num_prev_obs
        self.num_prev_action = num_prev_action

        if pixels_only:
            assert from_pixels
        if from_pixels:
            assert image_size

        self._make_env()
        self._make_spec()
        self._current_seed = self.set_seed(seed)

        if self.num_prev_obs > 0:
            self._prev_obs_image_queue = deque(maxlen=self.num_prev_obs)
            self._prev_obs_state_queue = deque(maxlen=self.num_prev_obs)
        if self.num_prev_action > 0:
            raise NotImplementedError()
            # self._prev_action_queue = deque(maxlen=self.num_prev_action)

    @abc.abstractmethod
    def render(self, mode="rgb_array", width=640, height=480):
        raise NotImplementedError()

    @abc.abstractmethod
    def _reset(self, tensordict: Optional[TensorDict] = None):
        raise NotImplementedError()

    @abc.abstractmethod
    def _step(self, tensordict: TensorDict):
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_env(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_spec(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _set_seed(self, seed: Optional[int]):
        raise NotImplementedError()
