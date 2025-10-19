from lerobot.envs.gymnasium_robotics import GymRoboticsEnv
import numpy as np

env = GymRoboticsEnv("FetchPickAndPlace-v4")
obs, info = env.reset()
print({k: type(v) for k, v in obs.items()})
print({k: v.shape for k, v in obs["images"].items()})
print("state shape:", obs["state"].shape)
print("goal in obs:", "goal" in obs)
print(env.action_space)
print(env.action_space.shape[0])

done = False
while not done:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print("rollout ok")
env.close()
