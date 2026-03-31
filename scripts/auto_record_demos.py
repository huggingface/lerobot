"""Automated demonstration recording for PandaPickCube in MuJoCo simulation.

Records approach-and-grasp demonstrations using the LeRobot dataset format.
These demos serve as offline data for the HIL-SERL RL training pipeline.

Usage (must use mjpython on macOS):
    mjpython scripts/auto_record_demos.py
"""

import logging
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_EPISODES = 10
MAX_STEPS = 100


def main():
    from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv
    from gym_hil.wrappers.factory import wrap_env

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info("Creating PandaPickCube environment with viewer...")
    base_env = PandaPickCubeGymEnv(image_obs=True, render_mode="human")
    env = wrap_env(base_env, use_viewer=True, use_gripper=True, gripper_penalty=-0.02)

    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Observation space: {env.observation_space}")

    # Create LeRobot dataset with video format for images
    repo_id = "LIJianxuanLeo/sim_pick_cube"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        features={
            "observation.images.front": {
                "dtype": "video",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (18,),
                "names": {"motors": [f"motor_{i}" for i in range(18)]},
            },
            "action": {
                "dtype": "float32",
                "shape": (4,),
                "names": {"motors": ["x", "y", "z", "gripper"]},
            },
        },
    )

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        logger.info(f"\n=== Episode {ep+1}/{NUM_EPISODES} ===")

        for step in range(MAX_STEPS):
            tcp = obs["agent_pos"][-3:]

            # Heuristic: approach block, lower, close gripper, lift
            action = np.zeros(4, dtype=np.float32)
            if step < 20:
                # Move toward block center, open gripper, go down
                target_xy = np.array([0.5, 0.0])
                diff_xy = target_xy - tcp[:2]
                action[0] = np.clip(diff_xy[0] / 0.025, -1, 1)
                action[1] = np.clip(diff_xy[1] / 0.025, -1, 1)
                action[2] = -1.0   # down
                action[3] = 2.0    # open gripper
            elif step < 45:
                # Close gripper while staying low
                action[2] = -0.3
                action[3] = 0.0    # close gripper
            else:
                # Lift up with gripper closed
                action[2] = 1.0
                action[3] = 0.0    # keep closed

            next_obs, reward, terminated, truncated, info = env.step(action)

            # Get images from observation
            front_img = obs.get("pixels", {}).get("front", np.zeros((128, 128, 3), dtype=np.uint8))
            wrist_img = obs.get("pixels", {}).get("wrist", np.zeros((128, 128, 3), dtype=np.uint8))

            # Add frame to dataset
            frame = {
                "observation.images.front": torch.from_numpy(front_img),
                "observation.images.wrist": torch.from_numpy(wrist_img),
                "observation.state": torch.from_numpy(obs["agent_pos"].copy()),
                "action": torch.from_numpy(action.copy()),
                "task": "pick_cube",
            }
            dataset.add_frame(frame)

            obs = next_obs
            if terminated or truncated:
                break

            time.sleep(0.01)

        dataset.save_episode()
        logger.info(f"  Episode {ep+1} saved ({step+1} steps)")

    # Finalize
    dataset.writer.close()
    logger.info(f"\nDataset saved to {dataset.root}")
    logger.info(f"Total episodes: {dataset.num_episodes}")
    env.close()


if __name__ == "__main__":
    main()
