"""Gym wrapper + registration for simulator_for_IL_RL's AssemblingEnv (UR10e+2F85, 3-obj nesting).

Adapts the sim's nested obs / 8-D task-space action to a gym-hil-style interface
that lerobot's HIL-SERL processor pipeline expects:

- ``pixels.front``, ``pixels.wrist``: (H, W, 3) uint8, default resized to 128x128
- ``agent_pos``: flat 15-dim float32 = joint_pos(7) + ee_pos(3) + ee_quat(4) + gripper_qpos(1)
- action: Box[-1, 1] shape=(3,) — delta (dx, dy, dz); optional discrete gripper
  (num_discrete_actions=3: no-op/open/close) appended when ``use_gripper=True``

The wrapper maintains an internal ee reference pose (captured on reset) and
advances it per step by ``action_step_size`` clipped to a box; quaternion held
constant (task is vertical pick-and-place). IK is handled inside the sim env.

Registration: ``sim_assembling/AssembleBase-v0`` (``gym.make(...)``).
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from simulator_for_il_rl.env import AssemblingEnv
except ImportError as e:  # pragma: no cover - installation pointer
    raise ImportError(
        "simulator_for_il_rl not installed. Install via `uv pip install -e ../simulator_for_IL_RL`."
    ) from e

logger = logging.getLogger(__name__)


def _resize_uint8(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize (H, W, 3) uint8 image. Uses cv2 if available, else numpy stride-slice."""
    h, w, _ = img.shape
    new_h, new_w = size
    if (h, w) == (new_h, new_w):
        return img
    try:
        import cv2

        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:  # pragma: no cover
        ys = np.linspace(0, h - 1, new_h).astype(np.int64)
        xs = np.linspace(0, w - 1, new_w).astype(np.int64)
        return img[ys][:, xs]


class AssemblingHILAdapter(gym.Wrapper):
    """Adapt AssemblingEnv to gym-hil-style obs/action for HIL-SERL.

    Args:
        env: underlying ``AssemblingEnv`` (instantiated with ``use_task_space=True``).
        image_size: target (H, W) for both cameras.
        cam_front_key: source camera name in env obs (default "cam_front").
        cam_wrist_key: source camera name in env obs (default "cam_gripper").
        action_step_size: max |delta_xyz| in metres per control step.
        use_gripper: append a gripper dim to action.
        num_discrete_actions: number of discrete buckets for gripper (3 = noop/open/close).
        include_yaw_slot: if True, expose action shape (dx, dy, dz, dyaw, gripper) = 5D
            to match lerobot's ps4_joystick ``delta_mode`` teleop (which emits 5 values).
            ``dyaw`` is ignored internally — the task is vertical pick-and-place so the
            wrist orientation is held at the reset quaternion.
    """

    metadata = {"render_fps": 10, "render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        env: AssemblingEnv,
        image_size: tuple[int, int] = (128, 128),
        cam_front_key: str = "cam_front",
        cam_wrist_key: str = "cam_gripper",
        action_step_size: float = 0.025,
        use_gripper: bool = True,
        num_discrete_actions: int = 3,
        include_yaw_slot: bool = True,
    ):
        super().__init__(env)
        assert env.use_task_space, "AssemblingHILAdapter requires use_task_space=True"
        self.image_size = image_size
        self.cam_front_key = cam_front_key
        self.cam_wrist_key = cam_wrist_key
        self.action_step_size = float(action_step_size)
        self.use_gripper = bool(use_gripper)
        self.num_discrete_actions = int(num_discrete_actions) if use_gripper else 0
        self.include_yaw_slot = bool(include_yaw_slot)

        # [dx, dy, dz] + optional [dyaw] + optional [gripper]
        low = [-1.0, -1.0, -1.0]
        high = [1.0, 1.0, 1.0]
        if self.include_yaw_slot:
            low.append(-1.0)
            high.append(1.0)
        if self.use_gripper:
            low.append(0.0)
            high.append(float(max(num_discrete_actions - 1, 1)))
        self.action_space = spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            dtype=np.float32,
        )

        h, w = image_size
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        "front": spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8),
                        "wrist": spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8),
                    }
                ),
                "agent_pos": spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32),
            }
        )

        self._ee_ref_pos: np.ndarray | None = None
        self._ee_ref_quat: np.ndarray | None = None
        self._gripper_cmd: float = 0.0  # [0, 1]; sim expects 1=closed (passes 255*gripper to actuator)

    def _adapt_obs(self, obs: dict) -> dict:
        state = obs["state"]
        joint_pos = np.asarray(state["joint_pos"], dtype=np.float32)  # 7
        ee_pos = np.asarray(state["ee_pos"], dtype=np.float32)  # 3
        ee_quat = np.asarray(state["ee_quat"], dtype=np.float32)  # 4
        # last joint is gripper driver — use it as the gripper proxy.
        gripper_q = np.asarray([joint_pos[-1]], dtype=np.float32)
        agent_pos = np.concatenate([joint_pos, ee_pos, ee_quat, gripper_q]).astype(np.float32)

        imgs = obs.get("images", {})
        front = imgs.get(self.cam_front_key)
        wrist = imgs.get(self.cam_wrist_key)
        if front is None or wrist is None:
            raise KeyError(
                f"AssemblingEnv obs missing camera(s) {self.cam_front_key} / {self.cam_wrist_key}. "
                f"Got: {list(imgs.keys())}. Ensure render_mode='rgb_array' or 'all'."
            )
        front = _resize_uint8(np.asarray(front, dtype=np.uint8), self.image_size)
        wrist = _resize_uint8(np.asarray(wrist, dtype=np.uint8), self.image_size)

        return {
            "pixels": {"front": front, "wrist": wrist},
            "agent_pos": agent_pos,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        ee_pos = np.asarray(obs["state"]["ee_pos"], dtype=np.float64)
        ee_quat = np.asarray(obs["state"]["ee_quat"], dtype=np.float64)
        self._ee_ref_pos = ee_pos.copy()
        self._ee_ref_quat = ee_quat.copy()
        self._gripper_cmd = 0.0
        return self._adapt_obs(obs), info

    def _decode_gripper(self, g: float) -> float:
        """Map discrete gripper action {0,1,2,...} to continuous gripper in [0, 1].

        0 → no-op (hold current). 1 → open (gripper=0). 2 → close (gripper=1).
        Higher discrete indices map linearly to [0, 1].
        """
        if not self.use_gripper:
            return self._gripper_cmd
        idx = int(round(float(g)))
        if idx <= 0:
            return self._gripper_cmd
        if self.num_discrete_actions <= 1:
            return 0.0
        if idx == 1:
            return 0.0
        if idx == 2:
            return 1.0
        return float(idx - 1) / float(self.num_discrete_actions - 1)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).flatten()
        if self._ee_ref_pos is None:
            raise RuntimeError("AssemblingHILAdapter.step called before reset().")

        # Layout: [dx, dy, dz] + (optional [dyaw]) + (optional [gripper]).
        dxyz = np.clip(a[:3], -1.0, 1.0) * self.action_step_size
        self._ee_ref_pos = self._ee_ref_pos + dxyz

        # dyaw is accepted but ignored (quaternion held constant for a vertical pick task).
        gripper_idx = 4 if self.include_yaw_slot else 3
        if self.use_gripper and a.shape[0] > gripper_idx:
            self._gripper_cmd = float(np.clip(self._decode_gripper(a[gripper_idx]), 0.0, 1.0))

        sim_action = np.concatenate(
            [self._ee_ref_pos, self._ee_ref_quat, [self._gripper_cmd]]
        ).astype(np.float32)

        obs, reward, terminated, truncated, info = self.env.step(sim_action)
        return self._adapt_obs(obs), float(reward), bool(terminated), bool(truncated), info


def make_assembling_env(
    xml_path: str = "scene.xml",
    control_hz: float = 20.0,
    sim_timestep: float = 0.002,
    mode: str = "fast",
    max_episode_steps: int = 300,
    render_mode: str = "rgb_array",
    image_size: tuple[int, int] = (128, 128),
    action_step_size: float = 0.025,
    use_gripper: bool = True,
    num_discrete_actions: int = 3,
    include_yaw_slot: bool = False,
    **_ignored: Any,
) -> gym.Env:
    """Factory used by ``gym.make("sim_assembling/AssembleBase-v0", ...)``.

    ``include_yaw_slot=False`` (default) → action = (dx, dy, dz, gripper) 4D,
    matches lerobot's ``gamepad`` teleop in delta_mode.

    ``include_yaw_slot=True`` → action = (dx, dy, dz, dyaw, gripper) 5D,
    matches lerobot's ``ps4_joystick`` teleop in delta_mode (dyaw is ignored).
    """
    base = AssemblingEnv(
        xml_path=xml_path,
        sim_timestep=sim_timestep,
        control_hz=control_hz,
        mode=mode,
        max_episode_steps=max_episode_steps,
        use_task_space=True,
        render_mode=render_mode,
    )
    return AssemblingHILAdapter(
        base,
        image_size=image_size,
        action_step_size=action_step_size,
        use_gripper=use_gripper,
        num_discrete_actions=num_discrete_actions,
        include_yaw_slot=include_yaw_slot,
    )


def _register() -> None:
    try:
        gym.register(
            id="sim_assembling/AssembleBase-v0",
            entry_point="lerobot.envs.sim_assembling:make_assembling_env",
            max_episode_steps=300,
        )
    except gym.error.Error:
        # Already registered (re-import) — harmless.
        pass


_register()
