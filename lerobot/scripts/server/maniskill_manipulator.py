import einops
import numpy as np
import gymnasium as gym
import torch

from omegaconf import DictConfig
from typing import Any
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode


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
        self.action_space = gym.spaces.Box(
            low=new_low, high=new_high, shape=(new_action_space_shape,)
        )

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
        self.action_space = gym.spaces.Tuple(
            spaces=(env.action_space, gym.spaces.Discrete(2))
        )

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
        self.action_space = gym.spaces.Tuple(
            spaces=(action_space_agent, gym.spaces.Discrete(2))
        )

    def step(self, action):
        if isinstance(action, tuple):
            action, telop = action
        else:
            telop = 0
        action = action / self.multiply_factor
        obs, reward, terminated, truncated, info = self.env.step((action, telop))
        return obs, reward, terminated, truncated, info


def make_maniskill(
    cfg: DictConfig,
    n_envs: int | None = None,
) -> gym.Env:
    """
    Factory function to create a ManiSkill environment with standard wrappers.

    Args:
        task: Name of the ManiSkill task
        obs_mode: Observation mode (rgb, rgbd, etc)
        control_mode: Control mode for the robot
        render_mode: Rendering mode
        sensor_configs: Camera sensor configurations
        n_envs: Number of parallel environments

    Returns:
        A wrapped ManiSkill environment
    """

    env = gym.make(
        cfg.env.task,
        obs_mode=cfg.env.obs,
        control_mode=cfg.env.control_mode,
        render_mode=cfg.env.render_mode,
        sensor_configs={"width": cfg.env.image_size, "height": cfg.env.image_size},
        num_envs=n_envs,
    )

    if cfg.env.video_record.enabled:
        env = RecordEpisode(
            env,
            output_dir=cfg.env.video_record.record_dir,
            save_trajectory=True,
            trajectory_name=cfg.env.video_record.trajectory_name,
            save_video=True,
            video_fps=30,
        )
    env = ManiSkillObservationWrapper(env, device=cfg.env.device)
    env = ManiSkillVectorEnv(env, ignore_terminations=True, auto_reset=False)
    env._max_episode_steps = env.max_episode_steps = (
        50  # gym_utils.find_max_episode_steps_value(env)
    )
    env.unwrapped.metadata["render_fps"] = 20
    env = ManiSkillCompat(env)
    env = ManiSkillActionWrapper(env)
    env = ManiSkillMultiplyActionWrapper(env, multiply_factor=1)

    return env


if __name__ == "__main__":
    import argparse
    import hydra

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="lerobot/configs/env/maniskill_example.yaml"
    )
    args = parser.parse_args()

    # Initialize config
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(config_name="env/maniskill_example.yaml")

    env = make_maniskill(
        task=cfg.env.task,
        obs_mode=cfg.env.obs,
        control_mode=cfg.env.control_mode,
        render_mode=cfg.env.render_mode,
        sensor_configs={"width": cfg.env.render_size, "height": cfg.env.render_size},
    )

    print("env done")
    obs, info = env.reset()
    random_action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(random_action)
