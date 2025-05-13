import random

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces
from mujoco import MjData, MjModel


class VX300sEnv(gym.Env):
    def __init__(self, xml_path: str, render: bool = True) -> None:
        super().__init__()
        self.model = MjModel.from_xml_path(xml_path)
        self.data = MjData(self.model)
        self.n_joints = 6
        self.max_step = 100
        self.step_count = 0
        self.success_dist = 0.03
        self.is_running = True
        self.goal = self._sample_goal()
        self.ee_site_name = "pinch"
        self.action_space = spaces.Box(
            low=-np.deg2rad(2),
            high=np.deg2rad(2),
            shape=(self.n_joints,),
            dtype=np.float32,
        )
        # 6つのサーボがある
        # 現在の関節位置(1 * 6)
        # 現在の関節速度(1 * 6)
        # エンドエフェクタの現在位置(3)とゴール位置の差分
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6 + 6 + 3 + 3,), dtype=np.float32
        )

        self.viewer = None
        if render:
            # MjViewer の代わりに、viewer モジュールを使う
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                key_callback=self.key_callback,
            )

    def key_callback(self, keycode: int) -> None:
        if chr(keycode) == "Q":
            self.is_running = False
            print("Finish!")

    def _sample_goal(self):
        # XYZ 方向に適当な範囲でゴールをサンプリング
        scale = 0.5
        x = 0.4
        y = scale * random.uniform(-1, 1)
        z = scale * random.uniform(0.3, 1)
        return np.array([x, y, z])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.goal = self._sample_goal()

        # ゴール位置に目標オブジェクトを置く
        goal_site_id = int(self.model.site("goal_site").id)
        self.model.site_pos[goal_site_id] = self.goal

        self.data.qpos[: self.n_joints] = np.random.uniform(
            -0.1,
            0.1,
            size=self.n_joints,
        )
        self.data.qvel[: self.n_joints] = 0
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def get_reward(self):
        ee_pos = self._get_ee_position()
        distance = np.linalg.norm(ee_pos - self.goal)
        reward = 1.0 if distance <= self.success_dist else -distance
        return reward, distance

    def step(self, action):
        self.data.ctrl[: self.n_joints] = action
        mujoco.mj_step(self.model, self.data)

        reward, distance = self.get_reward()
        terminated = distance <= self.success_dist
        truncated = self.step_count >= self.max_step
        self.step_count += 1
        info = {"distance": distance}

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def _get_ee_position(self):
        site_id = self.model.site(self.ee_site_name).id
        return self.data.site_xpos[site_id]

    def _get_obs(self):
        qpos = self.data.qpos[: self.n_joints]
        qvel = self.data.qvel[: self.n_joints]
        ee_pos = self._get_ee_position()
        return np.concatenate([qpos, qvel, ee_pos, self.goal]).astype(np.float32)

    def render(self):
        # エンドエフェクタの位置を取得しボールの位置を変える
        pinch_site_id = int(self.model.site("pinch").id)
        ee_site_id = int(self.model.site("ee_site").id)
        pinch_pos = self.data.site_xpos[pinch_site_id]
        self.model.site_pos[ee_site_id] = pinch_pos

        self.viewer.sync()

    def close(self):
        # 環境を閉じる際にviewerも閉じる
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.close()
        self.viewer = None
