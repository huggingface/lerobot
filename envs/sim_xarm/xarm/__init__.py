from collections import OrderedDict, deque

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

from xarm.tasks.base import Base as Base
from xarm.tasks.lift import Lift
from xarm.tasks.peg_in_box import PegInBox
from xarm.tasks.push import Push
from xarm.tasks.reach import Reach

TASKS = OrderedDict(
    (
        (
            "reach",
            {
                "env": Reach,
                "action_space": "xyz",
                "episode_length": 50,
                "description": "Reach a target location with the end effector",
            },
        ),
        (
            "push",
            {
                "env": Push,
                "action_space": "xyz",
                "episode_length": 50,
                "description": "Push a cube to a target location",
            },
        ),
        (
            "peg_in_box",
            {
                "env": PegInBox,
                "action_space": "xyz",
                "episode_length": 50,
                "description": "Insert a peg into a box",
            },
        ),
        (
            "lift",
            {
                "env": Lift,
                "action_space": "xyzw",
                "episode_length": 50,
                "description": "Lift a cube above a height threshold",
            },
        ),
    )
)


class SimXarmWrapper(gym.Wrapper):
    """
    A wrapper for the SimXarm environments. This wrapper is used to
    convert the action and observation spaces to the correct format.
    """

    def __init__(self, env, task, obs_mode, image_size, action_repeat, frame_stack=1, channel_last=False):
        super().__init__(env)
        self._env = env
        self.obs_mode = obs_mode
        self.image_size = image_size
        self.action_repeat = action_repeat
        self.frame_stack = frame_stack
        self._frames = deque([], maxlen=frame_stack)
        self.channel_last = channel_last
        self._max_episode_steps = task["episode_length"] // action_repeat

        image_shape = (
            (image_size, image_size, 3 * frame_stack)
            if channel_last
            else (3 * frame_stack, image_size, image_size)
        )
        if obs_mode == "state":
            self.observation_space = env.observation_space["observation"]
        elif obs_mode == "rgb":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
        elif obs_mode == "all":
            self.observation_space = gym.spaces.Dict(
                state=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                rgb=gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            )
        else:
            raise ValueError(f"Unknown obs_mode {obs_mode}. Must be one of [rgb, all, state]")
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(task["action_space"]),))
        self.action_padding = np.zeros(4 - len(task["action_space"]), dtype=np.float32)
        if "w" not in task["action_space"]:
            self.action_padding[-1] = 1.0

    def _render_obs(self):
        obs = self.render(mode="rgb_array", width=self.image_size, height=self.image_size)
        if not self.channel_last:
            obs = obs.transpose(2, 0, 1)
        return obs.copy()

    def _update_frames(self, reset=False):
        pixels = self._render_obs()
        self._frames.append(pixels)
        if reset:
            for _ in range(1, self.frame_stack):
                self._frames.append(pixels)
        assert len(self._frames) == self.frame_stack

    def transform_obs(self, obs, reset=False):
        if self.obs_mode == "state":
            return obs["observation"]
        elif self.obs_mode == "rgb":
            self._update_frames(reset=reset)
            rgb_obs = np.concatenate(list(self._frames), axis=-1 if self.channel_last else 0)
            return rgb_obs
        elif self.obs_mode == "all":
            self._update_frames(reset=reset)
            rgb_obs = np.concatenate(list(self._frames), axis=-1 if self.channel_last else 0)
            return OrderedDict((("rgb", rgb_obs), ("state", self.robot_state)))
        else:
            raise ValueError(f"Unknown obs_mode {self.obs_mode}. Must be one of [rgb, all, state]")

    def reset(self):
        return self.transform_obs(self._env.reset(), reset=True)

    def step(self, action):
        action = np.concatenate([action, self.action_padding])
        reward = 0.0
        for _ in range(self.action_repeat):
            obs, r, done, info = self._env.step(action)
            reward += r
        return self.transform_obs(obs), reward, done, info

    def render(self, mode="rgb_array", width=384, height=384, **kwargs):
        return self._env.render(mode, width=width, height=height)

    @property
    def state(self):
        return self._env.robot_state


def make(task, obs_mode="state", image_size=84, action_repeat=1, frame_stack=1, channel_last=False, seed=0):
    """
    Create a new environment.
    Args:
            task (str): The task to create an environment for. Must be one of:
                    - 'reach'
                    - 'push'
                    - 'peg-in-box'
                    - 'lift'
            obs_mode (str): The observation mode to use. Must be one of:
                    - 'state': Only state observations
                    - 'rgb': RGB images
                    - 'all': RGB images and state observations
            image_size (int): The size of the image observations
            action_repeat (int): The number of times to repeat the action
            seed (int): The random seed to use
    Returns:
            gym.Env: The environment
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task}. Must be one of {list(TASKS.keys())}")
    env = TASKS[task]["env"]()
    env = TimeLimit(env, TASKS[task]["episode_length"])
    env = SimXarmWrapper(env, TASKS[task], obs_mode, image_size, action_repeat, frame_stack, channel_last)
    env.seed(seed)

    return env
