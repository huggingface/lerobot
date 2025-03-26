import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import einops
import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from lerobot.common.envs.configs import EnvConfig, ManiskillEnvConfig
from lerobot.configs import parser
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.common.constants import ACTION, OBS_IMAGE, OBS_ROBOT


def preprocess_maniskill_observation(
    observations: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    # TODO: You have to merge all tensors from agent key and extra key
    # You don't keep sensor param key in the observation
    # And you keep sensor data rgb
    q_pos = observations["agent"]["qpos"]
    q_vel = observations["agent"]["qvel"]
    tcp_pos = observations["extra"]["tcp_pose"]
    img = observations["sensor_data"]["base_camera"]["rgb"]

    _, h, w, c = img.shape
    assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

    # sanity check that images are uint8
    assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

    # convert to channel first of type float32 in range [0,1]
    img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
    img = img.type(torch.float32)
    img /= 255

    state = torch.cat([q_pos, q_vel, tcp_pos], dim=-1)

    return_observations["observation.image"] = img
    return_observations["observation.state"] = state
    return return_observations





class ManiSkillObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, device: torch.device = "cuda"):
        super().__init__(env)
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def observation(self, observation):
        observation = preprocess_maniskill_observation(observation)
        observation = {k: v.to(self.device) for k, v in observation.items()}
        return observation


class ManiSkillCompat(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        new_action_space_shape = env.action_space.shape[-1]
        new_low = np.squeeze(env.action_space.low, axis=0)
        new_high = np.squeeze(env.action_space.high, axis=0)
        self.action_space = gym.spaces.Box(low=new_low, high=new_high, shape=(new_action_space_shape,))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        options = {}
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = reward.item()
        terminated = terminated.item()
        truncated = truncated.item()
        return obs, reward, terminated, truncated, info


class ManiSkillActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Tuple(spaces=(env.action_space, gym.spaces.Discrete(2)))

    def action(self, action):
        action, telop = action
        return action


class ManiSkillMultiplyActionWrapper(gym.Wrapper):
    def __init__(self, env, multiply_factor: float = 1):
        super().__init__(env)
        self.multiply_factor = multiply_factor
        action_space_agent: gym.spaces.Box = env.action_space[0]
        action_space_agent.low = action_space_agent.low * multiply_factor
        action_space_agent.high = action_space_agent.high * multiply_factor
        self.action_space = gym.spaces.Tuple(spaces=(action_space_agent, gym.spaces.Discrete(2)))

    def step(self, action):
        if isinstance(action, tuple):
            action, telop = action
        else:
            telop = 0
        action = action / self.multiply_factor
        obs, reward, terminated, truncated, info = self.env.step((action, telop))
        return obs, reward, terminated, truncated, info


class BatchCompatibleWrapper(gym.ObservationWrapper):
    """Ensures observations are batch-compatible by adding a batch dimension if necessary."""
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key in observation:
            if "image" in key and observation[key].dim() == 3:
                observation[key] = observation[key].unsqueeze(0)
            if "state" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
        return observation


class TimeLimitWrapper(gym.Wrapper):
    """Adds a time limit to the environment based on fps and control_time."""
    def __init__(self, env, control_time_s, fps):
        super().__init__(env)
        self.control_time_s = control_time_s
        self.fps = fps
        self.max_episode_steps = int(self.control_time_s * self.fps)
        self.current_step = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        if self.current_step >= self.max_episode_steps:
            terminated = True
            
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        return super().reset(seed=seed, options=options)


def make_maniskill(
    cfg: ManiskillEnvConfig,
    n_envs: int | None = None,
) -> gym.Env:
    """
    Factory function to create a ManiSkill environment with standard wrappers.

    Args:
        cfg: Configuration for the ManiSkill environment
        n_envs: Number of parallel environments

    Returns:
        A wrapped ManiSkill environment
    """
    env = gym.make(
        cfg.task,
        obs_mode=cfg.obs_type,
        control_mode=cfg.control_mode,
        render_mode=cfg.render_mode,
        sensor_configs={"width": cfg.image_size, "height": cfg.image_size},
        num_envs=n_envs,
    )

    # Add video recording if enabled
    if cfg.video_record.enabled:
        env = RecordEpisode(
            env,
            output_dir=cfg.video_record.record_dir,
            save_trajectory=True,
            trajectory_name=cfg.video_record.trajectory_name,
            save_video=True,
            video_fps=30,
        )
    
    # Add observation and image processing
    env = ManiSkillObservationWrapper(env, device=cfg.device)
    env = ManiSkillVectorEnv(env, ignore_terminations=True, auto_reset=False)
    env._max_episode_steps = env.max_episode_steps = cfg.episode_length
    env.unwrapped.metadata["render_fps"] = cfg.fps
    
    # Add compatibility wrappers
    env = ManiSkillCompat(env)
    env = ManiSkillActionWrapper(env)
    env = ManiSkillMultiplyActionWrapper(env, multiply_factor=0.03)  # Scale actions for better control
    
    return env


@parser.wrap()
def main(cfg: ManiskillEnvConfig):
    """Main function to run the ManiSkill environment."""
    # Create the ManiSkill environment
    env = make_maniskill(cfg, n_envs=1)
    
    # Reset the environment
    obs, info = env.reset()
    
    # Run a simple interaction loop
    sum_reward = 0
    for i in range(100):
        # Sample a random action
        action = env.action_space.sample()
        
        # Step the environment
        start_time = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.perf_counter() - start_time
        sum_reward += reward
        # Log information
        
        # Reset if episode terminated
        if terminated or truncated:
            logging.info(f"Step {i}, reward: {sum_reward}, step time: {step_time}s")
            sum_reward = 0
            obs, info = env.reset()
    
    # Close the environment
    env.close()


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     main()

if __name__ == "__main__":
    import draccus
    config = ManiskillEnvConfig()
    draccus.set_config_type("json")
    draccus.dump(config=config, stream=open(file='run_config.json', mode='w'), )