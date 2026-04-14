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

"""
Minimal CPS (Controller Programming System) client for Hans Robot arms.

Communicates with the Hans robot controller via TCP socket + XML-RPC.
This is a focused subset of the full CPS API covering the methods needed
for LeRobot integration: connection, power, enable/disable, state reading,
and joint-space motion.
"""

import socket
import time
import xmlrpc.client
from enum import IntEnum


class RobotFSM(IntEnum):
    """Hans robot finite state machine states."""

    UNINITIALIZED = 0
    INITIALIZED = 1
    BOX_DISCONNECTED = 2
    BOX_CONNECTING = 3
    EMERGENCY_STOP_HANDLING = 4
    EMERGENCY_STOP = 5
    BLACKOUT_48V = 7
    CONTROLLER_DISCONNECTED = 14
    CONTROLLER_CONNECTING = 15
    ETHERCAT_ERROR = 17
    ROBOT_ERROR = 22
    ENABLING = 23
    DISABLED = 24
    MOVING = 25
    STOPPING = 27
    DISABLING = 28
    FREE_DRIVER_OPENING = 29
    FREE_DRIVER_CLOSING = 30
    FREE_DRIVER = 31
    STANDBY = 33


class _TcpChannel:
    """Low-level TCP channel used to send/receive commands to the CPS controller."""

    TIMEOUT_S = 5.0

    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._xmlrpc: xmlrpc.client.ServerProxy | None = None

    def connect(self, host: str, port: int) -> int:
        self._sock = socket.socket()
        self._sock.settimeout(self.TIMEOUT_S)
        xmlrpc_url = f"http://{host}:20000"
        self._xmlrpc = xmlrpc.client.ServerProxy(xmlrpc_url)
        ret = self._sock.connect_ex((host, port))
        if ret != 0:
            self._sock.close()
            self._sock = None
        return ret

    def disconnect(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def send_recv(self, cmd: str, result: list) -> int:
        """Send a text command and parse the comma-separated response."""
        if self._sock is None:
            return 39500
        try:
            self._sock.send(cmd.encode())
            raw = self._sock.recv(4096).decode("utf-8", "ignore")
            parts = raw.split(",")
            if len(parts) < 3 or parts[0] == "errorcmd":
                return 39503
            if parts[1] == "Fail":
                return int(parts[2])
            # Strip the leading cmd-name and "OK" tokens, and the trailing ';'
            data = parts[2:]
            if data and data[-1].strip() in ("", ";"):
                data = data[:-1]
            result.clear()
            result.extend(data)
            return 0
        except (OSError, TimeoutError):
            return 39503


class CPSClient:
    """
    Client for the Hans Robot CPS (Controller Programming System) interface.

    Uses TCP socket communication on port 10003 for motion commands and
    XML-RPC on port 20000 for auxiliary operations.

    Supports up to ``MAX_BOXES`` independent controller boxes.
    """

    MAX_BOXES = 5

    def __init__(self) -> None:
        self._channels: list[_TcpChannel] = [_TcpChannel() for _ in range(self.MAX_BOXES)]
        self._connected: list[bool] = [False] * self.MAX_BOXES

    # ------------------------------------------------------------------
    # Part 1 – Connection & Initialization
    # ------------------------------------------------------------------

    def HRIF_Connect(self, box_id: int, host: str, port: int) -> int:
        """Connect to the robot controller TCP server."""
        if box_id >= self.MAX_BOXES:
            return 39501
        ret = self._channels[box_id].connect(host, port)
        if ret != 0:
            return 39504
        self._connected[box_id] = True
        return 0

    def HRIF_DisConnect(self, box_id: int) -> int:
        """Disconnect from the robot controller."""
        if box_id >= self.MAX_BOXES:
            return 39501
        self._channels[box_id].disconnect()
        self._connected[box_id] = False
        return 0

    def HRIF_IsConnected(self, box_id: int) -> bool:
        """Return whether this box is currently connected."""
        return self._connected[box_id]

    def HRIF_Electrify(self, box_id: int) -> int:
        """Power on the robot body (48 V)."""
        result: list = []
        return self._channels[box_id].send_recv("Electrify,;", result)

    def HRIF_BlackOut(self, box_id: int) -> int:
        """Power off the robot body."""
        result: list = []
        return self._channels[box_id].send_recv("BlackOut,;", result)

    def HRIF_Connect2Controller(self, box_id: int) -> int:
        """Start the EtherCAT master and initialise all drives."""
        result: list = []
        return self._channels[box_id].send_recv("StartMaster,;", result)

    # ------------------------------------------------------------------
    # Part 2 – Axis / Group Control
    # ------------------------------------------------------------------

    def HRIF_GrpEnable(self, box_id: int, rbt_id: int) -> int:
        """Enable (servo-on) the robot group."""
        result: list = []
        cmd = f"GrpPowerOn,{rbt_id},;"
        return self._channels[box_id].send_recv(cmd, result)

    def HRIF_GrpDisable(self, box_id: int, rbt_id: int) -> int:
        """Disable (servo-off) the robot group."""
        result: list = []
        cmd = f"GrpPowerOff,{rbt_id},;"
        return self._channels[box_id].send_recv(cmd, result)

    def HRIF_GrpReset(self, box_id: int, rbt_id: int) -> int:
        """Reset robot errors."""
        result: list = []
        cmd = f"GrpReset,{rbt_id},;"
        return self._channels[box_id].send_recv(cmd, result)

    def HRIF_GrpStop(self, box_id: int, rbt_id: int) -> int:
        """Stop all motion immediately."""
        result: list = []
        cmd = f"GrpStop,{rbt_id},;"
        return self._channels[box_id].send_recv(cmd, result)

    def HRIF_GrpOpenFreeDriver(self, box_id: int, rbt_id: int) -> int:
        """Enable zero-force (gravity-compensated) teaching mode."""
        result: list = []
        cmd = f"GrpOpenFreeDriver,{rbt_id},;"
        return self._channels[box_id].send_recv(cmd, result)

    def HRIF_GrpCloseFreeDriver(self, box_id: int, rbt_id: int) -> int:
        """Disable zero-force teaching mode."""
        result: list = []
        cmd = f"GrpCloseFreeDriver,{rbt_id},;"
        return self._channels[box_id].send_recv(cmd, result)

    # ------------------------------------------------------------------
    # Part 3 – State Reading
    # ------------------------------------------------------------------

    def HRIF_ReadCurFSM(self, box_id: int, rbt_id: int, result: list) -> int:
        """Read the current FSM state code and description."""
        cmd = f"ReadCurFSM,{rbt_id},;"
        return self._channels[box_id].send_recv(cmd, result)

    def HRIF_ReadRobotState(self, box_id: int, rbt_id: int, result: list) -> int:
        """Read detailed robot state flags (motion, enable, error, …)."""
        cmd = f"ReadRobotState,{rbt_id},;"
        return self._channels[box_id].send_recv(cmd, result)

    def HRIF_ReadActJointPos(self, box_id: int, rbt_id: int, result: list) -> int:
        """Read actual joint positions (degrees) for J1-J6."""
        cmd = f"ReadActACS,{rbt_id},;"
        return self._channels[box_id].send_recv(cmd, result)

    def HRIF_ReadActJointVel(self, box_id: int, rbt_id: int, result: list) -> int:
        """Read actual joint velocities for J1-J6."""
        cmd = f"ReadActACSVel,{rbt_id},;"
        return self._channels[box_id].send_recv(cmd, result)

    def HRIF_ReadActTcpPos(self, box_id: int, rbt_id: int, result: list) -> int:
        """Read actual TCP Cartesian position [X, Y, Z, RX, RY, RZ]."""
        cmd = f"ReadActPos,{rbt_id},;"
        return self._channels[box_id].send_recv(cmd, result)

    def HRIF_IsMotionDone(self, box_id: int, rbt_id: int, result: list) -> int:
        """Return True in result[0] when the robot has finished its current motion."""
        state: list = []
        err = self.HRIF_ReadRobotState(box_id, rbt_id, state)
        if err != 0:
            return err
        # result[11] == "1" → not moving; result[0] == "0" → no motion command pending
        result.append(len(state) > 11 and state[11] == "1" and state[0] == "0")
        return 0

    # ------------------------------------------------------------------
    # Part 4 – Speed
    # ------------------------------------------------------------------

    def HRIF_SetOverride(self, box_id: int, rbt_id: int, ratio: float) -> int:
        """Set global speed override ratio (0.01 – 1.0)."""
        result: list = []
        cmd = f"SetOverride,{rbt_id},{ratio},;"
        return self._channels[box_id].send_recv(cmd, result)

    # ------------------------------------------------------------------
    # Part 5 – Motion Commands
    # ------------------------------------------------------------------

    def HRIF_WayPoint(
        self,
        box_id: int,
        rbt_id: int,
        end_pos: list[float],
        acs_pos: list[float],
        tcp_name: str,
        ucs_name: str,
        velocity: float,
        acc: float,
        radius: float,
        move_type: int,
        is_joint: int,
        is_seek: int,
        io_bit: int,
        io_state: int,
        cmd_id: str,
    ) -> int:
        """
        Execute a waypoint motion command (WayPoint protocol).

        Note: ``velocity`` must be strictly less than ``acc``; otherwise the
        controller rejects the command with error 20073.

        Args:
            box_id: Controller box ID.
            rbt_id: Robot ID (usually 0).
            end_pos: Target Cartesian position [X, Y, Z, RX, RY, RZ] (mm / deg).
                     When ``is_joint=1`` the controller uses ``acs_pos`` as the
                     target; ``end_pos`` must still be 6 non-zero floats.
            acs_pos: Target joint angles [J1-J6] in degrees.
            tcp_name: Name of the tool-coordinate frame (e.g. "TCP").
            ucs_name: Name of the user-coordinate frame (e.g. "Base").
            velocity: Motion velocity in deg/s (joint) or mm/s (linear).
                      Must satisfy ``velocity < acc``.
            acc: Motion acceleration. Must satisfy ``acc > velocity``.
            radius: Blending radius in mm (0 = no blending).
            move_type: Motion type (0 = joint, 1 = linear, 2 = arc).
            is_joint: Use joint angles (``acs_pos``) as target when 1.
            is_seek: Enable DI-based seek stop when 1.
            io_bit: DI bit index for seek.
            io_state: DI state for seek stop.
            cmd_id: Waypoint ID string (e.g. "1").
        """
        result: list = []
        parts = ["WayPoint", str(rbt_id)]
        parts += [str(v) for v in end_pos[:6]]
        parts += [str(v) for v in acs_pos[:6]]
        parts += [tcp_name, ucs_name]
        parts += [str(velocity), str(acc), str(radius)]
        parts += [str(move_type), str(is_joint), str(is_seek)]
        parts += [str(io_bit), str(io_state), cmd_id]
        cmd = ",".join(parts) + ",;"
        return self._channels[box_id].send_recv(cmd, result)

    def HRIF_MoveJ(
        self,
        box_id: int,
        rbt_id: int,
        acs_pos: list[float],
        tcp_name: str = "TCP",
        ucs_name: str = "Base",
        velocity: float = 50.0,
        acc: float = 100.0,
        radius: float = 0.0,
        cmd_id: str = "0",
    ) -> int:
        """
        Convenience wrapper: move to a joint-space target.

        Uses the ``WayPoint`` command with ``is_joint=1``.  The same joint
        target is passed as both the Cartesian hint (``end_pos``) and the
        joint target (``acs_pos``); the controller uses ``acs_pos`` for actual
        positioning when ``is_joint=1``.

        Note: ``velocity`` must be strictly less than ``acc`` (controller
        enforces this constraint).
        """
        pos = list(acs_pos)
        return self.HRIF_WayPoint(
            box_id=box_id,
            rbt_id=rbt_id,
            end_pos=pos,
            acs_pos=pos,
            tcp_name=tcp_name,
            ucs_name=ucs_name,
            velocity=velocity,
            acc=acc,
            radius=radius,
            move_type=0,
            is_joint=1,
            is_seek=0,
            io_bit=0,
            io_state=0,
            cmd_id=cmd_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def wait_for_fsm(self, box_id: int, rbt_id: int, target_fsm: int, timeout_s: float = 60.0) -> int:
        """Block until the FSM reaches ``target_fsm`` or timeout expires.

        Returns the final FSM state code.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result: list = []
            self.HRIF_ReadCurFSM(box_id, rbt_id, result)
            if result and int(result[0]) == target_fsm:
                return target_fsm
            time.sleep(0.1)
        result = []
        self.HRIF_ReadCurFSM(box_id, rbt_id, result)
        return int(result[0]) if result else -1

    def wait_motion_done(self, box_id: int, rbt_id: int, timeout_s: float = 60.0) -> bool:
        """Block until motion is finished or timeout. Returns True on success."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result: list = []
            self.HRIF_IsMotionDone(box_id, rbt_id, result)
            if result and result[0] is True:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def raise_on_error(ret: int, context: str = "") -> None:
        """Raise a ``RuntimeError`` if *ret* is non-zero."""
        if ret != 0:
            msg = f"Hans CPS error {ret}"
            if context:
                msg += f" ({context})"
            raise RuntimeError(msg)
