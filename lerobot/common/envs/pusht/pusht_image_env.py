import numpy as np
from gym import spaces

from lerobot.common.envs.pusht.pusht_env import PushTEnv


class PushTImageEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    # Note: legacy defaults to True for compatibility with original
    def __init__(self, legacy=True, block_cog=None, damping=None, render_size=96):
        super().__init__(
            legacy=legacy, block_cog=block_cog, damping=damping, render_size=render_size, render_action=False
        )
        ws = self.window_size
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=1, shape=(3, render_size, render_size), dtype=np.float32),
                "agent_pos": spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float32),
            }
        )
        self.render_cache = None

    def _get_obs(self):
        img = super()._render_frame(mode="rgb_array")

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img, -1, 0)
        obs = {"image": img_obs, "agent_pos": agent_pos}

        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == "rgb_array"

        if self.render_cache is None:
            self._get_obs()

        return self.render_cache
