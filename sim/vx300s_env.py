import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from mujoco import MjData, MjModel

class VX300sEnv(gym.Env):
    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        self.model = MjModel.from_xml_path(xml_path)
        self.data = MjData(self.model)
        self.n_joints = 6
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(self.n_joints,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(self.n_joints,), dtype=np.float32
        )
        
        # render_mode を受け取る
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.viewer = mujoco.MjViewer(self.model)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.goal = np.zeros(6, dtype=np.float32)  # 目標状態をゼロに設定（例）

        self.data.qpos[:self.n_joints] = np.random.uniform(-0.1, 0.1, size=self.n_joints)
        self.data.qvel[:self.n_joints] = 0
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:self.n_joints] = action
        mujoco.mj_step(self.model, self.data)

        observation = self._get_obs()
        reward = float(-np.linalg.norm(observation - self.goal))
        terminated = np.linalg.norm(observation - self.goal) < 0.1
        terminated = bool(terminated)
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return self.data.qpos[:self.n_joints].astype(np.float32)

    def render(self):
        if self.render_mode == "human":
            self.viewer.render()  # MjViewerを使ってシミュレーションをレンダリング
