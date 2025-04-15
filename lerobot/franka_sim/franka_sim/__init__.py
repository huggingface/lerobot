from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gymnasium.envs.registration import register
# from gym.envs.registration import register

register(
    id="PandaPushCube-v0",
    entry_point="franka_sim.envs:PandaPushCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaPushCubeVision-v0",
    entry_point="franka_sim.envs:PandaPushCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
