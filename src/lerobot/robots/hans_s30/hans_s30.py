#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""LeRobot Robot adapter for the Hans Robot S30 6-DOF industrial arm.

The S30 exposes a 6-joint state/action interface over a TCP socket using
the CPS (Controller Programming System) protocol. No USB serial port or
Dynamixel/Feetech bus is required.

Typical usage::

    from lerobot.robots.hans_s30 import HansS30, HansS30RobotConfig

    config = HansS30RobotConfig(ip="192.168.115.11", id="my_hans_s30")
    robot = HansS30(config)
    robot.connect()

    obs = robot.get_observation()
    robot.send_action({"joint_1.pos": 0.0, ..., "joint_6.pos": 0.0})

    robot.disconnect()
"""

import logging
import time
from functools import cached_property

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_hans_s30 import HansS30RobotConfig
from .cps_client import CPSClient, RobotFSM

logger = logging.getLogger(__name__)

# Joint names match the S30 documentation: J1 (base) → J6 (wrist).
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# Number of degrees of freedom.
NUM_JOINTS = 6


class HansS30(Robot):
    """LeRobot interface for the Hans Robot S30 6-DOF arm.

    The S30 is an industrial manipulator that communicates via TCP/IP using
    the Hans CPS protocol. Joint positions are exchanged in **degrees**.

    Connection sequence (handled inside :pymeth:`connect`):

    1. ``HRIF_Connect``      – open TCP socket
    2. ``HRIF_Electrify``    – power on body (48 V), wait ~15 s
    3. ``HRIF_Connect2Controller`` – start EtherCAT master, wait ~20 s
    4. ``HRIF_GrpEnable``    – servo-on all joints

    Disconnection sequence (handled inside :pymeth:`disconnect`):

    1. ``HRIF_GrpDisable``   – servo-off
    2. ``HRIF_DisConnect``   – close TCP socket

    Observations contain 6 joint positions (deg) and optionally camera frames.
    Actions contain 6 joint-position targets (deg).
    """

    config_class = HansS30RobotConfig
    name = "hans_s30"

    def __init__(self, config: HansS30RobotConfig) -> None:
        super().__init__(config)
        self.config = config
        self._cps = CPSClient()
        self._is_connected = False
        self.cameras = make_cameras_from_configs(config.cameras)

    # ------------------------------------------------------------------
    # Feature descriptors (no hardware access required)
    # ------------------------------------------------------------------

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {f"{j}.pos": float for j in JOINT_NAMES}
        for cam_key, cam_cfg in self.config.cameras.items():
            features[cam_key] = (cam_cfg.height, cam_cfg.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in JOINT_NAMES}

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """The S30 uses absolute encoders; no external calibration is needed."""
        return True

    def calibrate(self) -> None:
        """No-op: the S30 ships factory-calibrated."""

    # ------------------------------------------------------------------
    # Connect / Disconnect
    # ------------------------------------------------------------------

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        """Establish communication and bring the robot to the *standby* state.

        The method is **idempotent with respect to controller state**: it reads
        the current FSM after opening the TCP socket and skips any initialisation
        steps that the controller has already completed.  This avoids the
        ``error 20018`` that occurs when, for example, ``HRIF_Electrify`` is
        called on a body that is already powered on.

        Typical FSM progression handled automatically:

        * FSM ≤ 7  (body unpowered)      → Electrify → Connect2Controller → GrpEnable
        * FSM 14   (controller offline)  → Connect2Controller → GrpEnable
        * FSM 24   (servo disabled)      → GrpEnable only
        * FSM 33   (standby / already on)→ nothing, just apply speed override

        The ``calibrate`` parameter is accepted for API compatibility but has
        no effect because the S30 uses absolute encoders.

        Raises:
            RuntimeError: if any step of the connection sequence fails.
        """
        cfg = self.config
        logger.info(f"Connecting to Hans S30 at {cfg.ip}:{cfg.port} …")
        CPSClient.raise_on_error(
            self._cps.HRIF_Connect(cfg.box_id, cfg.ip, cfg.port), "HRIF_Connect"
        )

        # Read current FSM to decide which init steps are still needed.
        fsm_result: list = []
        self._cps.HRIF_ReadCurFSM(cfg.box_id, cfg.robot_id, fsm_result)
        fsm = int(fsm_result[0]) if fsm_result else -1
        logger.info(f"Current FSM after TCP connect: {fsm}")

        # ── Step A: Power on body (only when body is unpowered) ──────────────
        # FSM ≤ 7 means the 48 V supply is off or just coming up.
        if fsm <= int(RobotFSM.BLACKOUT_48V):
            logger.info("Powering on robot body …")
            CPSClient.raise_on_error(self._cps.HRIF_Electrify(cfg.box_id), "HRIF_Electrify")
            time.sleep(cfg.electrify_wait_s)
            # Refresh FSM
            self._cps.HRIF_ReadCurFSM(cfg.box_id, cfg.robot_id, fsm_result)
            fsm = int(fsm_result[0]) if fsm_result else fsm
            logger.info(f"FSM after Electrify: {fsm}")
        else:
            logger.info(f"Body already powered (FSM={fsm}), skipping Electrify.")

        # ── Step B: Start EtherCAT master (only when controller is offline) ──
        # FSM 13–14 means EtherCAT master has not been started yet.
        if fsm <= int(RobotFSM.CONTROLLER_DISCONNECTED):
            logger.info("Starting EtherCAT master …")
            CPSClient.raise_on_error(
                self._cps.HRIF_Connect2Controller(cfg.box_id), "HRIF_Connect2Controller"
            )
            time.sleep(cfg.controller_init_wait_s)
            self._cps.HRIF_ReadCurFSM(cfg.box_id, cfg.robot_id, fsm_result)
            fsm = int(fsm_result[0]) if fsm_result else fsm
            logger.info(f"FSM after Connect2Controller: {fsm}")
        else:
            logger.info(f"Controller already initialised (FSM={fsm}), skipping Connect2Controller.")

        # ── Step C: Enable servo group (only when servos are off) ────────────
        # FSM 24 = DISABLED means servos are off but everything else is ready.
        if fsm <= int(RobotFSM.DISABLED):
            logger.info("Enabling servo group …")
            CPSClient.raise_on_error(
                self._cps.HRIF_GrpEnable(cfg.box_id, cfg.robot_id), "HRIF_GrpEnable"
            )
            fsm = self._cps.wait_for_fsm(
                cfg.box_id, cfg.robot_id, RobotFSM.STANDBY, timeout_s=30.0
            )
            logger.info(f"FSM after GrpEnable: {fsm}")
        else:
            logger.info(f"Servos already enabled (FSM={fsm}), skipping GrpEnable.")

        if fsm not in (int(RobotFSM.STANDBY), int(RobotFSM.DISABLED), int(RobotFSM.MOVING)):
            logger.warning(f"Unexpected FSM state after connect sequence: {fsm}")

        # Apply global speed override.
        self._cps.HRIF_SetOverride(cfg.box_id, cfg.robot_id, cfg.speed_override)

        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        logger.info(f"{self} connected (final FSM={fsm}).")

    @check_if_not_connected
    def disconnect(self) -> None:
        """Disable servos, release cameras, and close the TCP socket."""
        cfg = self.config
        logger.info(f"Disabling servo group for {self} …")
        self._cps.HRIF_GrpDisable(cfg.box_id, cfg.robot_id)

        for cam in self.cameras.values():
            cam.disconnect()

        self._cps.HRIF_DisConnect(cfg.box_id)
        self._is_connected = False
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------
    # Configuration (no-op for industrial arm)
    # ------------------------------------------------------------------

    def configure(self) -> None:
        """No additional runtime configuration is required for the S30."""

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read current joint positions and camera frames.

        Returns:
            dict with keys ``joint_1.pos`` … ``joint_6.pos`` (float, degrees)
            plus one key per camera returning an ``(H, W, 3)`` uint8 array.
        """
        cfg = self.config
        result: list = []
        start = time.perf_counter()
        ret = self._cps.HRIF_ReadActJointPos(cfg.box_id, cfg.robot_id, result)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read joints: {dt_ms:.1f} ms")

        if ret != 0 or len(result) < NUM_JOINTS:
            raise RuntimeError(f"Failed to read joint positions (error {ret})")

        obs: RobotObservation = {
            f"{j}.pos": float(result[i]) for i, j in enumerate(JOINT_NAMES)
        }

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f} ms")

        return obs

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command the arm to move to a target joint configuration.

        The action dict must contain keys ``joint_1.pos`` … ``joint_6.pos``
        with values in **degrees**.

        If ``config.max_relative_target`` is set, the displacement from the
        current position is clamped to that value per joint.

        Args:
            action: Target joint positions in degrees.

        Returns:
            The action actually sent (after optional safety clamping).
        """
        cfg = self.config
        goal = [float(action[f"{j}.pos"]) for j in JOINT_NAMES]

        if cfg.max_relative_target is not None:
            # Read current positions to apply safety clamping.
            result: list = []
            ret = self._cps.HRIF_ReadActJointPos(cfg.box_id, cfg.robot_id, result)
            if ret == 0 and len(result) >= NUM_JOINTS:
                current = [float(result[i]) for i in range(NUM_JOINTS)]
                delta = np.array(goal) - np.array(current)
                clipped = np.clip(delta, -cfg.max_relative_target, cfg.max_relative_target)
                goal = (np.array(current) + clipped).tolist()

        ret = self._cps.HRIF_MoveJ(
            box_id=cfg.box_id,
            rbt_id=cfg.robot_id,
            acs_pos=goal,
            tcp_name=cfg.tcp_name,
            ucs_name=cfg.ucs_name,
            velocity=cfg.velocity,
            acc=cfg.acc,
        )
        if ret != 0:
            logger.warning(f"{self} HRIF_MoveJ returned error {ret}")

        return {f"{j}.pos": goal[i] for i, j in enumerate(JOINT_NAMES)}

    # ------------------------------------------------------------------
    # Zero-force teaching (useful for manual teleoperation / dataset collection)
    # ------------------------------------------------------------------

    def _read_fsm(self) -> int:
        """Return the current FSM state code, or -1 on error."""
        result: list = []
        ret = self._cps.HRIF_ReadCurFSM(self.config.box_id, self.config.robot_id, result)
        if ret != 0 or not result:
            return -1
        return int(result[0])

    def _wait_for_standby(self, timeout_s: float = 15.0) -> int:
        """Block until the robot reaches STANDBY (FSM=33) or timeout.

        Returns the final FSM state code.
        """
        return self._cps.wait_for_fsm(
            self.config.box_id, self.config.robot_id, RobotFSM.STANDBY, timeout_s=timeout_s
        )

    @check_if_not_connected
    def enable_free_driver(self) -> None:
        """Switch to zero-force (gravity-compensated) teaching mode.

        ``GrpOpenFreeDriver`` requires the robot to be in **STANDBY** state
        (FSM=33). This method first waits up to 15 s for that state and raises
        a ``RuntimeError`` if the robot does not reach it in time.

        In this mode the arm can be manually guided. Use
        :pymeth:`disable_free_driver` to return to position-control mode.

        Raises:
            RuntimeError: if the robot is not in STANDBY after waiting, or if
                the controller rejects the command.
        """
        cfg = self.config

        # GrpOpenFreeDriver requires STANDBY (33); wait if not there yet.
        fsm = self._read_fsm()
        if fsm != int(RobotFSM.STANDBY):
            logger.info(f"FSM={fsm}, waiting for STANDBY before enabling free driver …")
            fsm = self._wait_for_standby(timeout_s=15.0)

        if fsm != int(RobotFSM.STANDBY):
            raise RuntimeError(
                f"Cannot enable free-driver mode: robot is in FSM state {fsm} "
                f"(expected STANDBY=33). Check robot for errors."
            )

        ret = self._cps.HRIF_GrpOpenFreeDriver(cfg.box_id, cfg.robot_id)
        if ret == 20606:
            raise RuntimeError(
                "Free-driver (zero-force teaching) mode is disabled on this controller "
                "(error 20606: 'Free drive function is disabled'). "
                "Please activate the zero-force teaching license on the Hans controller "
                "or enable it in the teach pendant's system settings."
            )
        CPSClient.raise_on_error(ret, "HRIF_GrpOpenFreeDriver")
        logger.info(f"{self} entered zero-force teaching mode.")

    @check_if_not_connected
    def disable_free_driver(self) -> None:
        """Exit zero-force teaching mode and return to position control."""
        cfg = self.config
        CPSClient.raise_on_error(
            self._cps.HRIF_GrpCloseFreeDriver(cfg.box_id, cfg.robot_id),
            "HRIF_GrpCloseFreeDriver",
        )
        # Wait for the robot to settle back to STANDBY after closing free driver.
        fsm = self._wait_for_standby(timeout_s=10.0)
        logger.info(f"{self} exited zero-force teaching mode (FSM={fsm}).")
