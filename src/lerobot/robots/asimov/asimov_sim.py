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

"""A MuJoCo Asimov 1 that answers the part of ``menlo.asimov.Robot`` that ``Asimov`` uses.

It models Asimov Edge's joint-command interface, not the onboard walking policy: it starts
in STAND holding a standing pose, follows ``trajectory()`` setpoints with per-joint PD, and
DAMPs two seconds after the last setpoint. State reports the ankles as motors A/B, and the
IMU in the pelvis frame, as the firmware does.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _menlo_sdk_available, _mujoco_available, require_package

from .asimov import joints_to_motors
from .asimov_kinematics import get_asimov_urdf

if TYPE_CHECKING or _menlo_sdk_available:
    import menlo.asimov as menlo
    from menlo.asimov import robots as menlo_robots
else:
    menlo = None
    menlo_robots = None

if TYPE_CHECKING or _mujoco_available:
    import mujoco
else:
    mujoco = None

logger = logging.getLogger(__name__)

# Asimov Edge DAMPs when trajectory setpoints stop for this long.
SETPOINT_TIMEOUT_S = 2.0
INIT_BASE_HEIGHT = 0.639
# Bent-knee standing pose, the one Menlo's locomotion policies are trained around.
STAND_POSE = {
    "L_Hip_Pitch": -0.15,
    "R_Hip_Pitch": 0.15,
    "L_Knee": 0.45,
    "R_Knee": -0.45,
    "L_Ankle_Pitch": -0.3,
    "R_Ankle_Pitch": 0.3,
    "L_Shoulder_Pitch": -0.25,
    "R_Shoulder_Pitch": 0.25,
    "L_Shoulder_Roll": -0.05,
    "R_Shoulder_Roll": 0.05,
    "L_Elbow": 0.4,
    "R_Elbow": -0.4,
}
# Stand-in for Edge's own gain table, used when a trajectory carries no gains and in STAND.
DEFAULT_GAINS = {
    "Hip": (150.0, 5.0),
    "Knee": (150.0, 5.0),
    "Ankle_Pitch": (224.42, 10.2),
    "Ankle_Roll": (35.2, 1.6),
    "Waist": (65.0, 5.0),
}
UPPER_GAINS = (60.0, 3.0)
DAMP_KD = 2.0


def _mjcf_joint(name: str) -> str:
    """Firmware joint name (ankles as Pitch/Roll) -> MJCF joint name."""
    side, _, rest = name.partition("_")
    if side in ("L", "R"):
        return f"{'left' if side == 'L' else 'right'}_{rest.lower()}_joint"
    return f"{name.lower()}_joint"


def _joint_space_names(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(n.replace("Ankle_A", "Ankle_Pitch").replace("Ankle_B", "Ankle_Roll") for n in names)


@dataclass(frozen=True)
class SimRobotInfo:
    joint_names: tuple[str, ...]
    dof: int


class AsimovMujocoClient:
    """Physics runs in real time on its own thread; ``onscreen`` opens the MuJoCo viewer."""

    def __init__(self, onscreen: bool = True) -> None:
        require_package("mujoco", extra="asimov", import_name="mujoco")
        xml = get_asimov_urdf(with_meshes=True).parent.parent / "xml" / "asimov_1_25dof.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.onscreen = onscreen
        self.joint_names = menlo_robots.ASIMOV_1_BIPED_JOINTS
        joint_ids = [self.model.joint(_mjcf_joint(n)).id for n in _joint_space_names(self.joint_names)]
        self._qadr = np.array([self.model.jnt_qposadr[j] for j in joint_ids])
        self._vadr = np.array([self.model.jnt_dofadr[j] for j in joint_ids])
        self._torque_limit = np.array([self.model.jnt_actfrcrange[j] for j in joint_ids])
        self._imu_site = self.model.site("imu_in_pelvis").id

        names = _joint_space_names(self.joint_names)
        self._stand = np.array([STAND_POSE.get(n, 0.0) for n in names])
        gains = [next((g for key, g in DEFAULT_GAINS.items() if key in n), UPPER_GAINS) for n in names]
        self._default_kp = np.array([kp for kp, _ in gains])
        self._default_kd = np.array([kd for _, kd in gains])

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mode = menlo.Mode.STAND
        self._targets = self._stand.copy()
        self._kp, self._kd = self._default_kp, self._default_kd
        self._last_setpoint = float("-inf")
        self._sequence = 0
        self._state: menlo.State | None = None
        self.info: SimRobotInfo | None = None

    @property
    def connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def connect(self, transport: str = "sim", timeout: float = 5.0, persist: bool = False) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = INIT_BASE_HEIGHT
        self.data.qpos[self._qadr] = self._stand
        mujoco.mj_forward(self.model, self.data)
        self._publish_state()
        self.info = SimRobotInfo(joint_names=self.joint_names, dof=len(self.joint_names))
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="asimov-mujoco", daemon=True)
        self._thread.start()

    def get_state(self) -> menlo.State:
        with self._lock:
            if self._state is None:
                raise RuntimeError("The Asimov simulation has not started.")
            return self._state

    def trajectory(self, positions, *, kp=None, kd=None, timeout: float = 0.0) -> None:
        """Hold ``positions`` (firmware order, ankles as motors A/B) until the next setpoint."""
        pos = tuple(float(p) for p in positions)
        if len(pos) != len(self.joint_names):
            raise ValueError(f"trajectory has {len(pos)} positions; this robot has {len(self.joint_names)}")
        if (kp is None) != (kd is None):
            raise ValueError("give both kp and kd, or neither")
        menlo_robots.check_ankle_limits(pos)
        with self._lock:
            if self._mode not in (menlo.Mode.STAND, menlo.Mode.MOVE):
                return  # Edge ignores joint commands while damped.
            self._targets = np.array(menlo_robots.trajectory_wire_positions(pos))
            if kp is not None:
                self._kp, self._kd = np.asarray(kp, dtype=np.float64), np.asarray(kd, dtype=np.float64)
            else:
                self._kp, self._kd = self._default_kp, self._default_kd
            self._last_setpoint = time.monotonic()

    def damp(self) -> None:
        with self._lock:
            self._mode = menlo.Mode.DAMP

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _step(self) -> None:
        with self._lock:
            # Before the first setpoint, STAND holds the standing pose indefinitely.
            since_setpoint = time.monotonic() - self._last_setpoint
            if self._mode == menlo.Mode.STAND and SETPOINT_TIMEOUT_S < since_setpoint < float("inf"):
                logger.warning("Asimov sim: no setpoint for 2 s, damping (as Asimov Edge does).")
                self._mode = menlo.Mode.DAMP
            mode, targets, kp, kd = self._mode, self._targets, self._kp, self._kd
        q = self.data.qpos[self._qadr]
        dq = self.data.qvel[self._vadr]
        holding = mode in (menlo.Mode.STAND, menlo.Mode.MOVE)
        tau = kp * (targets - q) - kd * dq if holding else -DAMP_KD * dq
        self.data.qfrc_applied[self._vadr] = np.clip(tau, self._torque_limit[:, 0], self._torque_limit[:, 1])
        mujoco.mj_step(self.model, self.data)

    def _publish_state(self) -> None:
        q = joints_to_motors(self.data.qpos[self._qadr])
        dq = joints_to_motors(self.data.qvel[self._vadr])
        rot = self.data.site_xmat[self._imu_site].reshape(3, 3)
        gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        gyro = rot.T @ self._world_angular_velocity()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._imu_site])
        joints = tuple(
            menlo.Joint(name=name, pos=float(p), vel=float(v), current=None, temp=None)
            for name, p, v in zip(self.joint_names, q, dq, strict=True)
        )
        with self._lock:
            self._sequence += 1
            self._state = menlo.State(
                mode=self._mode,
                joints=joints,
                gravity=tuple(float(x) for x in gravity),
                gyro=tuple(float(x) for x in gyro),
                quat=tuple(float(x) for x in quat),
                error_flags=0,
                alerts=(),
                sequence=self._sequence,
                fw_timestamp_us=int(self.data.time * 1e6),
                protocol_version=menlo_robots.PROTOCOL_VERSION,
            )

    def _world_angular_velocity(self) -> np.ndarray:
        # Pelvis angular velocity, which the IMU site rides on.
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self._imu_site, vel, 0)
        return vel[:3]

    def _run(self) -> None:
        viewer = None
        if self.onscreen:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = self.model.body("pelvis_link").id
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 150.0, -15.0
        dt = self.model.opt.timestep
        next_tick = time.monotonic()
        last_sync = 0.0
        try:
            while not self._stop.is_set():
                if viewer is not None and not viewer.is_running():
                    logger.info("Asimov sim viewer closed.")
                    break
                self._step()
                self._publish_state()
                now = time.monotonic()
                if viewer is not None and now - last_sync > 1 / 60:
                    viewer.sync()
                    last_sync = now
                next_tick += dt
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.1:
                    next_tick = time.monotonic()  # Fell behind; do not try to catch up.
        finally:
            if viewer is not None:
                viewer.close()
