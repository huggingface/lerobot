import time
from typing import Any

import numpy as np


class DummyRobot:
    """
    Простой тестовый робот.
    - Генерирует состояние (массив float) и одну/несколько камер (H, W, C, uint8).
    - Принимает действия и возвращает их же (как подтверждение).
    """

    def __init__(
        self,
        *,
        robot_id: str = "dummy",
        num_joints: int = 4,
        cameras: dict[str, tuple[int, int, int]] | None = None,
        action_extra_names: list[str] | None = None,
        fps: int = 30,
    ):
        self.id = robot_id
        self._is_connected = False
        self._is_calibrated = True
        self.fps = fps
        self.dt = 1.0 / max(1, fps)

        self._joint_names = [f"joint_{i}" for i in range(num_joints)]
        self._joint_state = np.zeros((num_joints,), dtype=np.float32)
        self._extra_actions = list(action_extra_names or [])

        self._cameras = cameras or {"front": (240, 320, 3)}

        # описание наблюдений: скаляры (float) + изображения (H, W, C)
        self._observation_features: dict[str, Any] = {n: float for n in self._joint_names}
        for cam, shape in self._cameras.items():
            self._observation_features[cam] = shape

        # порядок действий: сочленения + дополнительные каналы (например, gripper)
        self._action_features = list(self._joint_names) + self._extra_actions

        self._last_obs_ts = time.time()

    @property
    def observation_features(self) -> dict:
        return self._observation_features

    @property
    def action_features(self):
        return self._action_features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        self._is_connected = True
        self._is_calibrated = True

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        self._is_calibrated = True

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        # простая эволюция состояния
        now = time.time()
        dt = max(1e-3, now - self._last_obs_ts)
        self._last_obs_ts = now
        self._joint_state += 0.01 * dt  # линейная дрейфовая динамика

        obs: dict[str, Any] = {}
        for i, name in enumerate(self._joint_names):
            obs[name] = float(self._joint_state[i])

        for cam, (h, w, c) in self._cameras.items():
            # синтетическая картинка (градиент), гарантированно в [0, 255] uint8
            xx = (np.arange(w, dtype=np.uint16) * 255) // max(1, w - 1)
            yy = (np.arange(h, dtype=np.uint16) * 255) // max(1, h - 1)
            xv, yv = np.meshgrid(xx, yy, indexing="xy")
            ch0 = xv.astype(np.uint8)
            ch1 = yv.astype(np.uint8)
            ch2 = ((xv + yv) % 256).astype(np.uint8)
            img = np.stack([ch0, ch1, ch2], axis=-1)
            if c == 1:
                img = img[..., :1]
            obs[cam] = img

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected:
            raise RuntimeError("Robot not connected")
        # обновим состояние как будто действия влияют
        for i, name in enumerate(self._joint_names):
            if name in action:
                self._joint_state[i] += float(action[name]) * self.dt
        # дополнительные действия (например, gripper) игнорируем в динамике dummy
        return action

    def disconnect(self) -> None:
        self._is_connected = False
