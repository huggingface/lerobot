import gymnasium as gym
import numpy as np


class DoraEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, model="aloha"):
        ...
        # TODO: add code to connect with dora client here

    def reset(self, seed: int | None = None):
        ...
        # TODO: same as `step` but doesn't take `actions`
        observation = {
            "pixels": {
                "top": ...,
                "bottom": ...,
                "left": ...,
                "right": ...,
            },
            "agent_pos": ...,
            # "agent_vel": ...,  # will be added later
        }
        reward = 0
        terminated = truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self): ...

    def step(self, action: np.ndarray):
        ...
        # TODO: this is the important bit: the data to be return by Dora to the policy.
        observation = {
            "pixels": {
                "top": ...,
                "bottom": ...,
                "left": ...,
                "right": ...,
            },
            "agent_pos": ...,
            # "agent_vel": ...,  # will be added later
        }
        reward = 0
        terminated = truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
        # TODO: If code needs to be run when closing the env (e.g. shutting down Dora client),
        # this is the place to do it. Otherwise this can stay as is.
