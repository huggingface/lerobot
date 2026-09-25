#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SO-101 MuJoCo simulation environment.

Wraps TheRobotStudio/SO-ARM100's official MJCF model (the same asset used by SO-101's
real-hardware calibration/kinematics tooling elsewhere in ``lerobot``) into a thin,
generic, joint-space Gymnasium environment: ``reset()``/``step()`` drive the arm's 6
position actuators (5 arm joints + gripper, same order as the SO-101 dataset schema's
``observation.state``/``action`` columns) toward a commanded target and step physics.

This is infrastructure, not a task: there is no reward signal, no goal, no success
criterion, and no scene content beyond the robot and a ground plane. ``reward`` is always
``0.0`` and ``terminated`` is always ``False`` — episodes only end via ``truncated`` at
``episode_length``. Anyone building a concrete task (RL reward shaping, a Cartesian/IK
action space, scripted pick-place, etc.) is expected to wrap or subclass this rather than
have those concerns baked in here, matching the pattern of a raw manipulator ``EnvConfig``
elsewhere in this file (e.g. ``pusht``/``aloha`` before task-specific wrappers).

Known, deliberately-unfixed gap: the MJCF's ``gripper`` joint is a hinge with range
``[-10deg, 100deg]`` (its own native calibration), not LeRobot's separate linear 0=closed /
100=open convention. This mismatch is upstream in TheRobotStudio's own MJCF/URDF (their
README already documents it as unresolved) and is not something this env's action space
can silently paper over without lying about what MuJoCo will actually do with a value in
[0, 100]. Rather than inventing an unvalidated remapping, ``action_space``/``agent_pos``
simply expose each joint's real MJCF range in degrees, gripper included — the same raw,
per-joint-calibrated-degree convention SO-101's own dataset recordings already use (not
the normalized 0-100 gripper convention some other LeRobot tooling assumes). Callers that
need the 0-100 convention should convert at their own boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import requests  # type: ignore[import-untyped]
from gymnasium import spaces

from lerobot.utils.constants import HF_LEROBOT_HOME

from .utils import _LazyAsyncVectorEnv

# Joint/actuator order — matches the MJCF's <actuator> block and the SO-101 dataset's
# observation.state/action column order (5 arm joints + gripper).
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Pin TheRobotStudio/SO-ARM100 to a commit, not ``main``. Two users on the same
# ``lerobot`` commit must get the same physics; an unpinned ``main`` fetch plus a
# never-invalidated cache marker would silently freeze whatever ``main`` happened
# to contain on first run. Latest commit that touched ``Simulation/SO101`` as of
# 2026-08-22: actuator-model update (#141). Bump ``_MJCF_COMMIT`` (and the
# checksums) to pick up an upstream asset fix — the SHA is in the marker name so
# a bump re-syncs automatically.
_MJCF_COMMIT = "aec17bbc256d1a7342d53aaa4950595d4c30b40d"
_MJCF_RAW_BASE = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{_MJCF_COMMIT}/Simulation/SO101"
# ``joints_properties.xml`` is not ``<include>``d by ``scene.xml`` / ``so101_new_calib.xml``
# (those values are already inlined on the ``sts3215`` default class). Do not fetch it.
_MJCF_FILES = ["scene.xml", "so101_new_calib.xml"]
_ASSET_NAMES = [
    "base_motor_holder_so101_v1.stl",
    "base_so101_v2.stl",
    "motor_holder_so101_base_v1.stl",
    "motor_holder_so101_wrist_v1.stl",
    "moving_jaw_so101_v1.stl",
    "rotation_pitch_so101_v1.stl",
    "sts3215_03a_no_horn_v1.stl",
    "sts3215_03a_v1.stl",
    "under_arm_so101_v1.stl",
    "upper_arm_so101_v1.stl",
    "waveshare_mounting_plate_so101_v2.stl",
    "wrist_roll_follower_so101_v1.stl",
    "wrist_roll_pitch_so101_v2.stl",
]
# SHA-256 of each fetched file at ``_MJCF_COMMIT``. Catches a partial/corrupt download
# that a bare marker file would accept. Always fetch the MJCF-side meshes — do not
# copy STLs from ``HF_LEROBOT_HOME/robot-urdfs/so101/assets``; those URDF assets are
# versioned independently upstream (``_v1`` / ``_v2`` names already mix).
_FILE_SHA256 = {
    "scene.xml": "3b79a253a742f55ff0b16682173609229560d97744be472a238ea2e0a6a31ef6",
    "so101_new_calib.xml": "d75253eb568e8a7214db9c631ab7bed4217f608a26f7276ebe9a7636cac82580",
    "assets/base_motor_holder_so101_v1.stl": "8cd2f241037ea377af1191fffe0dd9d9006beea6dcc48543660ed41647072424",
    "assets/base_so101_v2.stl": "bb12b7026575e1f70ccc7240051f9d943553bf34e5128537de6cd86fae33924d",
    "assets/motor_holder_so101_base_v1.stl": "31242ae6fb59d8b15c66617b88ad8e9bded62d57c35d11c0c43a70d2f4caa95b",
    "assets/motor_holder_so101_wrist_v1.stl": "887f92e6013cb64ea3a1ab8675e92da1e0beacfd5e001f972523540545e08011",
    "assets/moving_jaw_so101_v1.stl": "785a9dded2f474bc1d869e0d3dae398a3dcd9c0c345640040472210d2861fa9d",
    "assets/rotation_pitch_so101_v1.stl": "9be900cc2a2bf718102841ef82ef8d2873842427648092c8ed2ca1e2ef4ffa34",
    "assets/sts3215_03a_no_horn_v1.stl": "75ef3781b752e4065891aea855e34dc161a38a549549cd0970cedd07eae6f887",
    "assets/sts3215_03a_v1.stl": "a37c871fb502483ab96c256baf457d36f2e97afc9205313d9c5ab275ef941cd0",
    "assets/under_arm_so101_v1.stl": "d01d1f2de365651dcad9d6669e94ff87ff7652b5bb2d10752a66a456a86dbc71",
    "assets/upper_arm_so101_v1.stl": "475056e03a17e71919b82fd88ab9a0b898ab50164f2a7943652a6b2941bb2d4f",
    "assets/waveshare_mounting_plate_so101_v2.stl": "e197e24005a07d01bbc06a8c42311664eaeda415bf859f68fa247884d0f1a6e9",
    "assets/wrist_roll_follower_so101_v1.stl": "4b17b410a12d64ec39554abc3e8054d8a97384b2dc4a8d95a5ecb2a93670f5f4",
    "assets/wrist_roll_pitch_so101_v2.stl": "6c7ec5525b4d8b9e397a30ab4bb0037156a5d5f38a4adf2c7d943d6c56eda5ae",
}


def _download(url: str, dest: Path, expected_sha256: str) -> None:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {dest.name}: expected {expected_sha256}, got {digest}. "
            "Partial or unexpected download — delete the cache dir and retry, or bump "
            f"_MJCF_COMMIT if the pin moved. url={url}"
        )
    dest.write_bytes(response.content)


def _ensure_so101_mjcf() -> Path:
    """Fetch (once, cached under ``HF_LEROBOT_HOME``) TheRobotStudio's official SO-101 MJCF +
    mesh assets pinned at ``_MJCF_COMMIT``. Returns the directory containing ``scene.xml``."""
    dest_dir = HF_LEROBOT_HOME / "robot-mjcf" / "so101"
    marker = dest_dir / f".sync_complete.{_MJCF_COMMIT}"
    if marker.exists():
        return dest_dir

    assets_dir = dest_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    for fname in _MJCF_FILES:
        _download(f"{_MJCF_RAW_BASE}/{fname}", dest_dir / fname, _FILE_SHA256[fname])

    # Always the MJCF-side meshes at the same pin. Never substitute the separately
    # versioned URDF cache under ``robot-urdfs/so101/assets``.
    for name in _ASSET_NAMES:
        rel = f"assets/{name}"
        _download(f"{_MJCF_RAW_BASE}/{rel}", assets_dir / name, _FILE_SHA256[rel])

    marker.touch()
    return dest_dir


def load_model():
    """Load the SO-101 MJCF (robot + ground plane) and return a fresh ``(model, data)`` pair.

    Deferred ``mujoco`` import — this module is only imported when the ``so101`` env is
    actually requested, keeping ``mujoco`` an optional dependency (``pip install
    lerobot[so101]``) rather than a hard one for everyone using ``lerobot.envs``.
    """
    import mujoco

    mjcf_dir = _ensure_so101_mjcf()
    model = mujoco.MjModel.from_xml_path(str(mjcf_dir / "scene.xml"))
    data = mujoco.MjData(model)
    return model, data


def joint_limits_deg(model) -> dict[str, tuple[float, float]]:
    """Joint limits in degrees, ``MOTOR_NAMES`` order, read directly from the MJCF."""
    import mujoco

    limits = {}
    for name in MOTOR_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[joint_id]
        limits[name] = (float(np.rad2deg(lo)), float(np.rad2deg(hi)))
    return limits


def set_qpos_deg(model, data, joint_pos_deg: np.ndarray) -> None:
    """Teleport the arm directly to ``joint_pos_deg`` (degrees, ``MOTOR_NAMES`` order) — no
    physics stepping. Calls ``mj_forward`` so derived quantities reflect the new pose."""
    import mujoco

    joint_pos_rad = np.deg2rad(np.asarray(joint_pos_deg, dtype=float)[: len(MOTOR_NAMES)])
    for name, val in zip(MOTOR_NAMES, joint_pos_rad, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = val
    mujoco.mj_forward(model, data)


class SO101MujocoEnv(gym.Env):
    """Thin Gymnasium wrapper around the SO-101 MuJoCo model: generic joint-space control,
    no reward/task logic. See module docstring for the gripper-mapping caveat.

    Observation (``obs_type="state"``, default): ``{"agent_pos": Box(6,)}``, the 6 joint
    positions in degrees, ``MOTOR_NAMES`` order.
    Observation (``obs_type="pixels_agent_pos"``): adds ``"pixels": Box(H, W, 3, uint8)``,
    an offscreen render from an explicit robot-framed camera (the MJCF ships no named
    ``<camera>`` and MuJoCo's default free-camera auto-framing breaks against this scene's
    infinite ground plane — see ``render()``).
    Action: ``Box(6,)``, absolute target joint positions in degrees, ``MOTOR_NAMES`` order,
    bounded by each joint's real MJCF range (``joint_limits_deg``).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        obs_type: str = "state",
        render_mode: str | None = "rgb_array",
        episode_length: int = 300,
        fps: int = 30,
        observation_height: int = 480,
        observation_width: int = 480,
        reset_qpos_deg: Sequence[float] | None = None,
    ):
        super().__init__()
        if obs_type not in ("state", "pixels_agent_pos"):
            raise ValueError(f"Unsupported obs_type: {obs_type!r}")
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.episode_length = episode_length
        self.fps = fps
        self.observation_height = observation_height
        self.observation_width = observation_width

        self.model, self.data = load_model()
        limits = joint_limits_deg(self.model)
        low = np.array([limits[name][0] for name in MOTOR_NAMES], dtype=np.float32)
        high = np.array([limits[name][1] for name in MOTOR_NAMES], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        state_space = spaces.Box(low=low, high=high, dtype=np.float32)
        if obs_type == "state":
            self.observation_space = spaces.Dict({"agent_pos": state_space})
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": state_space,
                    "pixels": spaces.Box(
                        low=0, high=255, shape=(observation_height, observation_width, 3), dtype=np.uint8
                    ),
                }
            )

        # MJCF ships no <option timestep=.../> so MuJoCo's 0.002s default applies (500Hz);
        # read it back rather than hardcoding, in case that ever changes upstream.
        self._n_substeps = max(1, round(1.0 / (fps * self.model.opt.timestep)))
        self._elapsed_steps = 0
        self._default_reset_qpos_deg = (
            np.zeros(len(MOTOR_NAMES), dtype=np.float32)
            if reset_qpos_deg is None
            else np.asarray(reset_qpos_deg, dtype=np.float32)
        )

    def _get_obs(self) -> dict[str, np.ndarray]:
        state = np.array(
            [np.rad2deg(self.data.qpos[self.model.jnt_qposadr[i]]) for i in range(len(MOTOR_NAMES))],
            dtype=np.float32,
        )
        obs = {"agent_pos": state}
        if self.obs_type == "pixels_agent_pos":
            obs["pixels"] = self.render()
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        import mujoco

        mujoco.mj_resetData(self.model, self.data)
        reset_qpos_deg = self._default_reset_qpos_deg
        if options is not None and "reset_qpos_deg" in options:
            reset_qpos_deg = np.asarray(options["reset_qpos_deg"], dtype=np.float32)
        set_qpos_deg(self.model, self.data, reset_qpos_deg)
        self._elapsed_steps = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        import mujoco

        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        target_rad = np.deg2rad(action)
        for name, val in zip(MOTOR_NAMES, target_rad, strict=True):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.data.ctrl[actuator_id] = val

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps += 1
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = self._elapsed_steps >= self.episode_length
        return obs, reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        """Offscreen RGB render from an explicit camera framed on the robot's own bounding
        box. Requires a working offscreen rendering backend (EGL/OSMesa/GLFW) — not required
        for ``obs_type="state"``. The MJCF has no named ``<camera>`` and MuJoCo's default
        free-camera auto-framing breaks against this scene's "infinite" ground plane (its
        size falls back to a generic default instead of the robot's real ~0.5m scale)."""
        import mujoco

        with mujoco.Renderer(
            self.model, height=self.observation_height, width=self.observation_width
        ) as renderer:
            cam = mujoco.MjvCamera()
            body_positions = self.data.xpos[1:]  # skip the world body
            lookat = body_positions.mean(axis=0)
            span = (
                float(np.max(np.linalg.norm(body_positions - lookat, axis=1))) if len(body_positions) else 0.3
            )
            cam.lookat[:] = lookat
            cam.distance = max(0.4, span * 3.0)
            cam.azimuth, cam.elevation = 130, -25
            renderer.update_scene(self.data, camera=cam)
            return renderer.render()


def create_so101_envs(
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create ``n_envs`` vectorized SO-101 MuJoCo envs.

    Returns ``{"so101": {0: vec_env}}`` — a single suite/task, matching the shape every
    other ``EnvConfig.create_envs`` override in this package returns, since there is no
    task/suite concept here (see module docstring).
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    fns = [(lambda: SO101MujocoEnv(**gym_kwargs)) for _ in range(n_envs)]

    vec = _LazyAsyncVectorEnv(fns) if env_cls is gym.vector.AsyncVectorEnv else env_cls(fns)
    return {"so101": {0: vec}}
