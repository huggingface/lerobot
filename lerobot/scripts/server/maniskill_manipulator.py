import einops
import numpy as np
import gymnasium as gym
import torch

"""Make ManiSkill3 gym environment"""
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def preprocess_maniskill_observation(observations: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
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
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return preprocess_maniskill_observation(observation)


class ManiSkillToDeviceWrapper(gym.Wrapper):
    def __init__(self, env, device: torch.device = "cuda"):
        super().__init__(env)
        self.device = device

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = {k: v.to(self.device) for k, v in obs.items()}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = {k: v.to(self.device) for k, v in obs.items()}
        return obs, reward, terminated, truncated, info


class ManiSkillCompat(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

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
    def __init__(self, env, multiply_factor: float = 10):
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


def make_maniskill(
    task: str = "PushCube-v1",
    obs_mode: str = "rgb",
    control_mode: str = "pd_ee_delta_pose",
    render_mode: str = "rgb_array",
    sensor_configs: dict[str, int] | None = None,
    n_envs: int = 1,
    device: torch.device = "cuda",
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
    if sensor_configs is None:
        sensor_configs = {"width": 64, "height": 64}

    env = gym.make(
        task,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        sensor_configs=sensor_configs,
        num_envs=n_envs,
    )
    env = ManiSkillCompat(env)
    env = ManiSkillObservationWrapper(env)
    env = ManiSkillActionWrapper(env)
    env = ManiSkillMultiplyActionWrapper(env)
    env = ManiSkillToDeviceWrapper(env, device=device)
    return env


if __name__ == "__main__":
    import argparse
    import hydra
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="lerobot/configs/env/maniskill_example.yaml")
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
