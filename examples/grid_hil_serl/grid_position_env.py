#!/usr/bin/env python

"""
Grid Position Prediction Environment for HIL-SERL

This environment provides image observations and grid position labels for training
a position prediction policy using HIL-SERL. Episodes are single-step: the agent
predicts which grid cell contains the cube and immediately receives binary feedback.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from pathlib import Path
from PIL import Image
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
    Actions: Discrete grid index (0-63)
    Reward: Binary feedback from human (correct/incorrect prediction)
    """

    def __init__(self, xml_path="grid_scene.xml", render_mode="rgb_array", show_viewer: bool = True):
        super().__init__()

        # Load Mujoco model
        self.xml_path = Path(__file__).parent / xml_path
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Setup renderer for observations
        self.renderer = mujoco.Renderer(self.model, height=1080, width=1920)
        # Optional interactive viewer for visualization
        self.viewer = None
        self._show_viewer = bool(show_viewer)
        if self._show_viewer:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception:
                self.viewer = None

        # Observation space: High-definition RGB images
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1080, 1920, 3),
            dtype=np.uint8
        )

        self.grid_size = 8

        # Action space: continuous pair in [-1, 1]
        self.action_space = spaces.Box(
            low=np.full(2, -1.0, dtype=np.float32),
            high=np.full(2, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        # Environment state
        self.current_position = None


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
        if self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception:
                pass

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
        """Advance to the next random position (single-step episodes)."""
        position = self._randomize_cube_position()
        obs = self._get_observation()

        # Placeholder reward; the wrapper computes the binary reward.
        reward = 0.0
        terminated = True
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
        try:
            if self.viewer is not None:
                self.viewer.close()
        except Exception:
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
        self.env = GridPositionEnv(xml_path, show_viewer=True)

        # LeRobot observation space format - use 'observation.image' as expected by SAC
        self.observation_space = {
            "observation.image": spaces.Box(
                low=0, high=255, shape=(1080, 1920, 3), dtype=np.uint8
            )
        }

        # Action is predicted grid cell coordinates in [-1, 1]
        self.action_space = spaces.Box(
            low=np.full(2, -1.0, dtype=np.float32),
            high=np.full(2, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        # Convert to LeRobot format
        lerobot_obs = {
            "observation.image": obs
        }

        return lerobot_obs, info

    def step(self, action):
        # Compute reward based on predicted grid index
        gt_position = int(self.env.get_current_position())
        gt_x = gt_position % self.env.grid_size
        gt_y = gt_position // self.env.grid_size

        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size >= 2:
            # Policy emits values in [-1, 1]; rescale to [0, grid_size - 1]
            scaled = (np.clip(action_arr[:2], -1.0, 1.0) + 1.0) * 0.5 * (self.env.grid_size - 1)
            pred_x = int(np.clip(np.rint(scaled[0]), 0, self.env.grid_size - 1))
            pred_y = int(np.clip(np.rint(scaled[1]), 0, self.env.grid_size - 1))
        else:
            pred_x = pred_y = 0

        # Binary reward: 1 for exact match, 0 otherwise
        correct = pred_x == gt_x and pred_y == gt_y
        reward = 1.0 if correct else 0.0

        # Advance environment to next random position
        obs, _, _, _, info = self.env.step([0.0, 0.0])
        terminated = True
        truncated = False

        # Convert to LeRobot format
        lerobot_obs = {
            "observation.image": obs
        }

        # Enrich info with prediction details
        l1_error = abs(pred_x - gt_x) + abs(pred_y - gt_y)

        info = {
            **info,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "gt_index": gt_position,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "pred_index": pred_y * self.env.grid_size + pred_x,
            "correct": correct,
            "error.l1": float(l1_error),
            "reward": reward,
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
