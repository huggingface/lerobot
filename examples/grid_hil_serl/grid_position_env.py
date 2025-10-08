#!/usr/bin/env python

"""
Grid Position Prediction Environment for HIL-SERL

This environment provides image observations and grid position labels for training
a position prediction policy using HIL-SERL. The task is simplified to predict
which of the 64 grid cells contains the red cube.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from pathlib import Path
from PIL import Image
import time

# Register the environment with gymnasium
gym.register(
    id="GridPositionPrediction-v0",
    entry_point="examples.grid_hil_serl.grid_position_env:GridPositionLeRobotEnv",
    kwargs={"xml_path": "grid_scene.xml"}
)


class GridPositionEnv(gym.Env):
    """
    Simplified environment for grid position prediction.

    Observations: High-definition RGB images (1920x1080)
    Actions: None (prediction-only task)
    Reward: Binary feedback from human (correct/incorrect prediction)
    """

    def __init__(self, xml_path="grid_scene.xml", render_mode="rgb_array"):
        super().__init__()

        # Load Mujoco model
        self.xml_path = Path(__file__).parent / xml_path
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Setup renderer for observations
        self.renderer = mujoco.Renderer(self.model, height=1080, width=1920)

        # Observation space: High-definition RGB images
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1080, 1920, 3),
            dtype=np.uint8
        )

        # Action space: Discrete grid positions (0-63 for 8x8 grid)
        # We'll use a dummy action space since this is prediction-only
        self.action_space = spaces.Discrete(1)  # Placeholder

        # Environment state
        self.current_position = None
        self.grid_size = 8

        # Initialize to random position
        self._randomize_cube_position()

    def _randomize_cube_position(self):
        """Randomize cube position and return grid coordinates."""
        # Generate random cell indices (0-7 for 8x8 grid)
        x_cell = np.random.randint(0, self.grid_size)
        y_cell = np.random.randint(0, self.grid_size)

        # Convert to physical positions (top-left origin)
        x_pos = (x_cell - self.grid_size // 2) + 0.5
        y_pos = (self.grid_size // 2 - y_cell) - 0.5

        # Set cube position
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.data.qpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"):mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint") + 6] = [x_pos, y_pos, 0.5, 0, 0, 0]
        self.data.qvel[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"):mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint") + 6] = [0, 0, 0, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        # Store current grid position (0-63)
        self.current_position = y_cell * self.grid_size + x_cell

        return self.current_position

    def reset(self, seed=None, options=None):
        """Reset environment to random cube position."""
        super().reset(seed=seed)

        # Randomize cube position
        position = self._randomize_cube_position()

        # Get observation
        obs = self._get_observation()

        return obs, {"grid_position": position}

    def step(self, action):
        """
        Step function for HIL-SERL compatibility.

        In this simplified version, actions are not used for control,
        but for receiving human feedback on predictions.
        """
        # For now, just randomize position on each step
        # In real HIL-SERL, this would be triggered by the learning algorithm
        position = self._randomize_cube_position()
        obs = self._get_observation()

        # Placeholder reward (would be set by human feedback in HIL)
        reward = 0.0
        terminated = False
        truncated = False

        info = {
            "grid_position": position,
            "x_cell": position % self.grid_size,
            "y_cell": position // self.grid_size
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Get current camera observation."""
        self.renderer.update_scene(self.data, camera="grid_camera")
        img = self.renderer.render()
        return img

    def render(self):
        """Render current observation."""
        return self._get_observation()

    def close(self):
        """Clean up resources."""
        # Renderer doesn't have a close method in this version
        pass

    def get_current_position(self):
        """Get current grid position (0-63)."""
        return self.current_position

    def validate_prediction(self, predicted_position):
        """
        Validate a predicted position against current cube position.

        Args:
            predicted_position: Integer 0-63 representing predicted grid cell

        Returns:
            bool: True if prediction is correct
        """
        return predicted_position == self.current_position


# LeRobot-compatible wrapper
class GridPositionLeRobotEnv:
    """
    LeRobot-compatible wrapper for the grid position environment.
    """

    def __init__(self, xml_path="grid_scene.xml"):
        self.env = GridPositionEnv(xml_path)

        # LeRobot observation space format - use 'observation.image' as expected by SAC
        self.observation_space = {
            "observation.image": spaces.Box(
                low=0, high=255, shape=(1080, 1920, 3), dtype=np.uint8
            ),
            "observation.state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )
        }

        # Dummy action space for compatibility
        self.action_space = spaces.Discrete(64)  # 64 possible grid positions

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        # Convert to LeRobot format
        lerobot_obs = {
            "observation.image": obs,
            "observation.state": np.array([info["grid_position"]], dtype=np.float32)
        }

        return lerobot_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert to LeRobot format
        lerobot_obs = {
            "observation.image": obs,
            "observation.state": np.array([info["grid_position"]], dtype=np.float32)
        }

        return lerobot_obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


if __name__ == "__main__":
    # Test the environment
    env = GridPositionEnv()

    print("Testing Grid Position Environment")
    print("=" * 40)

    obs, info = env.reset()
    print(f"Initial position: {info['grid_position']}")
    print(f"Observation shape: {obs.shape}")

    # Test a few steps
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(0)
        pos = info['grid_position']
        x_cell = info['x_cell']
        y_cell = info['y_cell']
        print(f"Step {i+1}: Grid cell ({x_cell}, {y_cell}) = Position {pos}")

        # Save sample image
        img = Image.fromarray(obs)
        img.save(f"sample_position_{pos}.jpg")
        print(f"  Saved image: sample_position_{pos}.jpg")

    env.close()
    print("Environment test completed!")
