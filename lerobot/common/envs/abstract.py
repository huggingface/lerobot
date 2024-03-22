from collections import deque
from typing import Optional

from tensordict import TensorDict
from torchrl.envs import EnvBase

from lerobot.common.utils import set_seed


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

        # self._next_seed will be used for the next reset. It is recommended that when self.set_seed is called
        # you store the return value in self._next_seed (it will be a new randomly generated seed).
        self._next_seed = seed
        # Don't store the result of this in self._next_seed, as we want to make sure that the first time
        # self._reset is called, we use seed.
        self.set_seed(seed)

        if self.num_prev_obs > 0:
            self._prev_obs_image_queue = deque(maxlen=self.num_prev_obs)
            self._prev_obs_state_queue = deque(maxlen=self.num_prev_obs)
        if self.num_prev_action > 0:
            raise NotImplementedError()
            # self._prev_action_queue = deque(maxlen=self.num_prev_action)

    def render(self, mode="rgb_array", width=640, height=480):
        raise NotImplementedError("Abstract method")

    def _reset(self, tensordict: Optional[TensorDict] = None):
        raise NotImplementedError("Abstract method")

    def _step(self, tensordict: TensorDict):
        raise NotImplementedError("Abstract method")

    def _make_env(self):
        raise NotImplementedError("Abstract method")

    def _make_spec(self):
        raise NotImplementedError("Abstract method")

    def _set_seed(self, seed: Optional[int]):
        set_seed(seed)
