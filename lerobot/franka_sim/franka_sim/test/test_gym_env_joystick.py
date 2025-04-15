import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np

from franka_sim import envs
import gymnasium as gym

# import joystick wrapper
from franka_env.envs.wrappers import JoystickIntervention
from franka_env.spacemouse.spacemouse_expert import ControllerType

from franka_sim.utils.viewer_utils import DualMujocoViewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, default="xbox", help="Controller type. xbox|ps5")

    args = parser.parse_args()
    controller_type = ControllerType[args.controller.upper()]

# env = envs.PandaPickCubeGymEnv(render_mode="human", image_obs=True)
env = gym.make("PandaPickCubeVision-v0", render_mode="human", image_obs=True)
env = JoystickIntervention(env, controller_type=controller_type)

env.reset()
m = env.unwrapped.model
d = env.unwrapped.data

# Create the dual viewer
dual_viewer = DualMujocoViewer(env.unwrapped.model, env.unwrapped.data)

# intervene on position control
with dual_viewer as viewer:
    for i in range(100000):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        viewer.sync()
