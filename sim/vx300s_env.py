import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces
from mujoco import MjData, MjModel


class VX300sEnv(gym.Env):
    def __init__(self, xml_path: str, render_mode: str | None = None):
        super().__init__()
        self.model = MjModel.from_xml_path(xml_path)
        self.data = MjData(self.model)
        self.n_joints = 6
        self.goal = self._sample_goal()
        self.ee_site_name = "pinch"
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(self.n_joints,), dtype=np.float32
        )
        # 6つのサーボがある
        # 現在の関節位置(1 * 6)
        # 現在の関節速度(1 * 6)
        # エンドエフェクタの現在位置(3)とゴール位置の差分
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6 + 6 + 3,), dtype=np.float32
        )

        # render_mode を受け取る
        self.render_mode = render_mode
        self.viewer = None
        if self.render_mode == "human":
            # MjViewer の代わりに、viewer モジュールを使う
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _sample_goal(self):
        # XYZ 方向に適当な範囲でゴールをサンプリング
        return np.array([0.3, 0.3, 0.3]) + np.random.uniform(-0.1, 0.1, size=3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.goal = self._sample_goal()
        self.data.qpos[: self.n_joints] = np.random.uniform(
            -0.1,
            0.1,
            size=self.n_joints,
        )
        self.data.qvel[: self.n_joints] = 0
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[: self.n_joints] = action
        mujoco.mj_step(self.model, self.data)

        ee_pos = self._get_ee_position()
        distance = np.linalg.norm(ee_pos - self.goal)

        reward = -distance  # ← 近いほど高報酬
        terminated = distance < 0.05
        truncated = False
        info = {"distance": distance}

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def _get_ee_position(self):
        site_id = self.model.site(self.ee_site_name).id
        return self.data.site_xpos[site_id]

    def _get_obs(self):
        qpos = self.data.qpos[: self.n_joints]
        qvel = self.data.qvel[: self.n_joints]
        error = self.goal - self._get_ee_position()
        return np.concatenate([qpos, qvel, error]).astype(np.float32)

    def render(self):
        self.viewer.sync()

    def close(self):
        # 環境を閉じる際にviewerも閉じる
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.close()
        self.viewer = None
