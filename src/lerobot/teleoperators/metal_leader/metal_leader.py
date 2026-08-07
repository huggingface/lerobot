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

import logging
import threading
import time
from math import radians
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_metal_leader import MetalLeaderConfig
from .gravity import NUM_ARM_JOINTS, MetalGravityModel
from .gripper_friction import gripper_friction_torque
from .urdf import metal_urdf_path

logger = logging.getLogger(__name__)

# Damiao motor variant per joint, fixed by the hardware build. Duplicated from
# `robots/metal_follower/metal_follower.py` rather than imported: `teleoperators/` importing from
# `robots/` would invert the package layering. `test_metal_leader.py` asserts the two agree.
MOTOR_MODELS = {
    "shoulder_pan": "metal_jlo",
    "shoulder_lift": "metal_j2",
    "elbow_flex": "metal_jlo",
    "wrist_flex": "metal_jlo",
    "wrist_yaw": "metal_jhi",
    "wrist_roll": "metal_jhi",
    "gripper": "metal_jhi",
}

# The 6 revolute joints, in the order the URDF declares them (JOINT1..JOINT6). The gravity model
# indexes positionally, so this order is load-bearing.
ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)
GRIPPER_NAME = "gripper"


class MetalLeader(Teleoperator):
    """
    Metal arm leader/teleoperator: 6 joints + a permanent gripper, all driven as Damiao motors over
    classic CAN (`use_can_fd=False`) via the stock `DamiaoMotorsBus`.

    Unlike torque-disabled leaders, this arm stays powered while the human moves it. A background
    thread streams Pinocchio-computed gravity-compensation torque as the MIT feedforward term with
    `kp=0`, so the motors carry the arm's ~4 kg and the operator supplies only the force to move
    it. The gripper is left backdrivable (`kp=0`) with its own friction feedforward so it is easy
    to squeeze; its raw motor angle is read back by `get_action` and drives the follower gripper
    1:1, matching `MetalFollower.action_features`.
    """

    config_class = MetalLeaderConfig
    name = "metal_leader"

    def __init__(self, config: MetalLeaderConfig):
        super().__init__(config)
        self.config = config

        if not config.port:
            raise ValueError(
                "metal_leader requires `port`. With can_interface='slcan' (the default) it is "
                "the USB-CAN adapter's serial port — '/dev/ttyACM0' on Linux, "
                "'/dev/cu.usbmodem1101' on macOS, 'COM5' on Windows. With "
                "can_interface='socketcan' it is the interface name, e.g. 'can0'."
            )

        if config.can_interface not in ("socketcan", "slcan"):
            raise ValueError(
                f"metal_leader supports can_interface='socketcan' or 'slcan', got "
                f"'{config.can_interface}'. On Linux, bring the USB-CAN adapter up as a socketcan "
                "interface (`sudo slcand -o -f -s8 /dev/ttyACM0 can0 && sudo ip link set up can0 "
                "&& sudo ip link set can0 txqueuelen 1000`) and set port='can0'. On macOS/Windows, "
                "where SocketCAN does not exist, use can_interface='slcan' with the adapter's "
                "serial port as `port` (e.g. '/dev/cu.usbmodem1101' or 'COM5')."
            )

        # slcan is latency-bound, not bandwidth-bound: a full gravity tick (7 refresh + 7 state
        # replies + 7 MIT writes + 7 command replies) measures ~4.2 ms p50 on a CANable, against
        # a 5 ms period at gravity_hz=200 -- before Pinocchio's RNEA solve is added on top. At
        # gravity_hz=100 the same tick uses ~48% of the period, which leaves real headroom.
        if config.can_interface == "slcan" and config.gravity_hz > 100:
            logger.warning(
                f"gravity_hz={config.gravity_hz} over slcan: one tick measures ~4.2 ms p50 "
                f"against a {1000.0 / config.gravity_hz:.1f} ms period, so ticks will overrun and "
                "compensation will lag. Use gravity_hz=100 (or socketcan) unless you have "
                "measured this link."
            )

        missing = [name for name in ARM_JOINT_NAMES if name not in config.motor_can_ids]
        if missing:
            raise ValueError(
                f"metal_leader requires all {NUM_ARM_JOINTS} arm joints in motor_can_ids; "
                f"missing {missing}. The gravity model is built from the arm's URDF and cannot "
                "describe a partial arm."
            )

        # Build all 7 motors: the 6 joints plus the permanent gripper.
        motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id) in config.motor_can_ids.items():
            motor_type_str = MOTOR_MODELS[motor_name]
            motor = Motor(send_id, motor_type_str, MotorNormMode.DEGREES)
            motor.recv_id = recv_id
            motor.motor_type_str = motor_type_str
            motors[motor_name] = motor

        self._joint_motor_names = list(motors)

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

        # Built in connect(): Pinocchio is a heavy import and the URDF may need downloading, so
        # neither should happen just from importing this module.
        self._gravity: MetalGravityModel | None = None
        self._gravity_thread: threading.Thread | None = None
        self._gravity_stop_event = threading.Event()

        # DamiaoMotorsBus has no internal locking: the background gravity thread and the main
        # loop's get_action() both send a CAN request, poll the shared canbus.recv(), and mutate a
        # shared state cache. All bus access from this teleoperator is serialized behind this lock.
        self._bus_lock = threading.Lock()

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self._joint_motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        # Build the gravity model BEFORE enabling torque. If the URDF cannot be fetched or
        # Pinocchio fails to load it, the arm must never be left powered with no gravity thread
        # running behind it — the operator would be holding its full weight against live motors.
        try:
            urdf_path = self.config.urdf_path or str(metal_urdf_path())
            self._gravity = MetalGravityModel(urdf_path)
        except Exception:
            self.bus.disconnect(disable_torque=False)
            raise

        self.bus.enable_torque()

        self._gravity_stop_event.clear()
        self._gravity_thread = threading.Thread(
            target=self._gravity_loop, name=f"metal-gravity-{self.config.port}", daemon=True
        )
        self._gravity_thread.start()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # The Damiao motors use absolute encoders and the leader commands no positions, so there
        # is nothing to calibrate: it only ever reads angles and streams torque.
        return True

    def calibrate(self) -> None:
        """No-op: the metal leader requires no calibration procedure."""
        pass

    def configure(self) -> None:
        """No-op: gains (kp=0, kd=config.leader_kd) are applied per-tick, not at connect."""
        pass

    def _resolve_per_joint(self, setting: float | dict[str, float], motor: str) -> float:
        """Read a config field that accepts either one value for every joint or a per-joint dict.

        Motors absent from a dict get 0, so a dict that names only the joints you tuned leaves the
        rest at the safe default rather than inheriting someone else's value.
        """
        if isinstance(setting, dict):
            return float(setting.get(motor, 0.0))
        return float(setting)

    def _gravity_loop(self) -> None:
        period = 1.0 / self.config.gravity_hz
        while not self._gravity_stop_event.is_set():
            start = time.perf_counter()
            self._gravity_tick()
            remaining = period - (time.perf_counter() - start)
            if remaining > 0:
                self._gravity_stop_event.wait(remaining)

    def _gravity_tick(self) -> None:
        """One gravity-compensation cycle: read state, compute feedforward torque, write MIT.

        Factored out of `_gravity_loop` so tests can drive it directly without a running thread.
        A single bad tick (a dropped CAN reply, say) is logged and swallowed rather than killing
        the thread — losing the thread would silently drop the arm out of compensation.
        """
        try:
            with self._bus_lock:
                states = self.bus.sync_read_all_states()
                present = {motor: states[motor]["position"] for motor in self._joint_motor_names}

                q_rad = [radians(states[motor]["position"]) for motor in ARM_JOINT_NAMES]

                if self.config.use_velocity_feedforward:
                    deadzone = self.config.velocity_deadzone_rad_s
                    dq_rad = []
                    for motor in ARM_JOINT_NAMES:
                        velocity = radians(states[motor]["velocity"])
                        dq_rad.append(0.0 if abs(velocity) < deadzone else velocity)
                    scales = [
                        self._resolve_per_joint(self.config.friction_scale, motor)
                        for motor in ARM_JOINT_NAMES
                    ]
                    torques = self._gravity.blended_feedforward_torque(q_rad, dq_rad, scales)
                else:
                    torques = self._gravity.feedforward_torque(q_rad, [0.0] * NUM_ARM_JOINTS)

                # kp=0 on every motor: the arm must not pull toward any position, only hold its
                # own weight. Commanding the present position keeps the frame well-formed.
                commands: dict[str, tuple[float, float, float, float, float]] = {
                    motor: (
                        0.0,
                        self._resolve_per_joint(self.config.leader_kd, motor),
                        present[motor],
                        0.0,
                        torques[index],
                    )
                    for index, motor in enumerate(ARM_JOINT_NAMES)
                }

                gripper_torque = 0.0
                if self.config.use_velocity_feedforward and self.config.gripper_friction_scale > 0.0:
                    gripper_velocity = radians(states[GRIPPER_NAME]["velocity"])
                    gripper_torque = self.config.gripper_friction_scale * gripper_friction_torque(
                        gripper_velocity
                    )
                commands[GRIPPER_NAME] = (
                    0.0,
                    self._resolve_per_joint(self.config.leader_kd, GRIPPER_NAME),
                    present[GRIPPER_NAME],
                    0.0,
                    gripper_torque,
                )

                self.bus.sync_write_metal(commands)
        except Exception:
            logger.exception(f"{self}: gravity-compensation tick failed; continuing.")

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Read the leader's current joint positions (all 7 motors, degrees). The gripper value is
        raw motor degrees, matching `MetalFollower.action_features`, so squeezing the leader
        gripper drives the follower gripper 1:1.
        """
        start = time.perf_counter()

        with self._bus_lock:
            positions = self.bus.sync_read("Present_Position")
        action: dict[str, Any] = {f"{motor}.pos": positions[motor] for motor in self._joint_motor_names}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_action took: {dt_ms:.1f}ms")

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        raise NotImplementedError("Force feedback is not implemented for the metal leader.")

    @check_if_not_connected
    def disconnect(self) -> None:
        self._gravity_stop_event.set()
        if self._gravity_thread is not None:
            self._gravity_thread.join(timeout=1.0)
            if self._gravity_thread.is_alive():
                logger.warning(
                    f"{self} gravity-compensation thread did not stop within its timeout; "
                    "it may still be writing to the bus."
                )
            self._gravity_thread = None

        # Compensation has stopped, so the arm would sag on its last zero-kp command. Hold the
        # current pose instead: it stays up, no longer weightless. hold_kp=0 leaves it limp.
        if self.config.hold_kp_on_disconnect > 0.0:
            try:
                with self._bus_lock:
                    present = self.bus.sync_read("Present_Position")
                    self.bus.sync_write_metal(
                        {
                            motor: (
                                self.config.hold_kp_on_disconnect,
                                self.config.hold_kd_on_disconnect,
                                present[motor],
                                0.0,
                                0.0,
                            )
                            for motor in self._joint_motor_names
                        }
                    )
            except Exception:
                logger.exception(f"{self}: failed to command the hold pose on disconnect.")

        # Torque stays enabled so the hold persists instead of the arm free-falling. How long it
        # persists is set by the motors' own command-timeout configuration.
        self.bus.disconnect(disable_torque=False)

        logger.info(f"{self} disconnected.")
