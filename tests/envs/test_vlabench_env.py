import importlib
from dataclasses import dataclass, field
import os
import json
import gymnasium as gym
import pytest
import numpy as np
from gymnasium.envs.registration import register, registry as gym_registry
from gymnasium.utils.env_checker import check_env

from lerobot.configs.types import PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.envs.factory import make_env, make_env_config
from VLABench.tasks import *
from VLABench.robots import *

def test_create_vlabench_envs():
    # Example for create VLABench episode config
    n_envs = 2
    test_track = "track_1_in_distribution"
    test_task = "select_fruit"
    
    with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/evaluation/tracks", f"{test_track}.json"), "r") as f:
        episode_configs = json.load(f)
    assert test_task in episode_configs.keys(), f"{test_task} not found in {test_track} episode configs"
    target_task_episode_configs = episode_configs[test_task][:n_envs]
    
    cfg = make_env_config(
        "vlabench",
        task=test_task,
        robot="franka",
        env_configs=None,
        episode_configs=target_task_episode_configs,
        render_resolution=(480, 480),
        max_episode_length=500,
    )
    envs = make_env(cfg, n_envs=2)
    task_name = next(iter(envs))
    task_id = next(iter(envs[task_name]))
    env = envs[task_name][task_id]
    obs, info = env.reset()
    env.last_obs = obs
    
    dummy_action = np.array([0.05, 0.0, 0.02, 0, 0, 0, 0]) 
    # the action could be chosen from ['eef', 'joint', 'delta_eef']
    obs, reward, done, truncated, info = env.step(dummy_action, mode="delta_eef")
    for key in obs:
        print(f"obs key: {key}")
    
    env.close()

if __name__ == "__main__":
    test_create_vlabench_envs()