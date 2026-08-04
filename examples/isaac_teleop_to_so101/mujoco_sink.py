#!/usr/bin/env python

"""MuJoCo output adapter for the Isaac Teleop -> SO-101 example.

The public contract deliberately matches the small part of ``SO101Follower`` used by the
teleoperation loop: observations and actions are ``{motor}.pos`` dictionaries in LeRobot
units (degrees for the arm, 0..100 for the gripper).  No serial modules are imported.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .common import _ensure_so101_urdf

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
MODEL_PATH = Path(__file__).with_name("mujoco") / "scene.xml"


def ensure_mujoco_model() -> Path:
    """Place the small MJCF files beside LeRobot's existing cached SO-101 STL assets."""
    cache_dir = Path(_ensure_so101_urdf()).parent
    for name in ("scene.xml", "so101_new_calib.xml"):
        source = MODEL_PATH.parent / name
        destination = cache_dir / name
        if not destination.exists() or source.read_bytes() != destination.read_bytes():
            shutil.copyfile(source, destination)
    return cache_dir / "scene.xml"


@dataclass(frozen=True)
class GripperMapping:
    """Measured MJCF endpoints corresponding to LeRobot 0=closed and 100=open."""

    closed_rad: float
    open_rad: float

    def to_rad(self, value: float) -> float:
        return self.closed_rad + np.clip(value, 0.0, 100.0) / 100.0 * (self.open_rad - self.closed_rad)

    def from_rad(self, value: float) -> float:
        span = self.open_rad - self.closed_rad
        if abs(span) < 1e-9:
            raise ValueError("gripper endpoints must differ")
        return float(np.clip((value - self.closed_rad) / span * 100.0, 0.0, 100.0))


DEFAULT_GRIPPER_MAPPING = GripperMapping(closed_rad=-0.17453, open_rad=1.74533)


class MuJoCoSO101Sink:
    """Six-position-actuator SO-101 sink with decoupled physics stepping."""

    name = "so101_mujoco"
    id = "so101_mujoco"

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        gripper_mapping: GripperMapping | None = DEFAULT_GRIPPER_MAPPING,
    ) -> None:
        resolved_model = ensure_mujoco_model() if model_path is None else Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(resolved_model))
        self.data = mujoco.MjData(self.model)
        self.gripper_mapping = gripper_mapping
        self._actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in MOTOR_NAMES]
        )
        self._joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in MOTOR_NAMES]
        )
        if np.any(self._actuator_ids < 0) or np.any(self._joint_ids < 0):
            raise ValueError(f"MJCF must contain all motors in order: {MOTOR_NAMES}")
        actual = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i)) for i in self._actuator_ids
        )
        if actual != MOTOR_NAMES:
            raise ValueError(f"unexpected actuator ordering: {actual}")
        self._qpos_addresses = self.model.jnt_qposadr[self._joint_ids]
        self.data.ctrl[:] = np.clip(self.data.qpos[self._qpos_addresses], *self.ctrl_limits)
        mujoco.mj_forward(self.model, self.data)

    @property
    def ctrl_limits(self) -> tuple[np.ndarray, np.ndarray]:
        limits = self.model.actuator_ctrlrange[self._actuator_ids]
        return limits[:, 0], limits[:, 1]

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def get_observation(self) -> dict[str, float]:
        q = self.data.qpos[self._qpos_addresses]
        values = np.rad2deg(q[:5]).tolist()
        if self.gripper_mapping is None:
            # ``None`` remains available for inspecting a different/unverified MJCF.
            values.append(float(q[5]))
        else:
            values.append(self.gripper_mapping.from_rad(float(q[5])))
        return {f"{name}.pos": float(value) for name, value in zip(MOTOR_NAMES, values, strict=True)}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        target = self.data.ctrl[self._actuator_ids].copy()
        for index, name in enumerate(MOTOR_NAMES[:5]):
            key = f"{name}.pos"
            if key in action:
                target[index] = np.deg2rad(float(action[key]))
        if "gripper.pos" in action:
            if self.gripper_mapping is None:
                raise RuntimeError(
                    "gripper mapping is unverified; supply GripperMapping(closed_rad=..., open_rad=...)"
                )
            target[5] = self.gripper_mapping.to_rad(float(action["gripper.pos"]))
        low, high = self.ctrl_limits
        self.data.ctrl[self._actuator_ids] = np.clip(target, low, high)
        return action

    def send_native_radians(self, target: np.ndarray) -> None:
        """Smoke-test hook; bypasses LeRobot conversion, including the unknown jaw mapping."""
        target = np.asarray(target, dtype=float)
        if target.shape != (6,):
            raise ValueError(f"expected six targets, got {target.shape}")
        low, high = self.ctrl_limits
        self.data.ctrl[self._actuator_ids] = np.clip(target, low, high)

    def step(self, count: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=count)

    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos[self._qpos_addresses].copy()
