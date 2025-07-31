#!/usr/bin/env python
from __future__ import annotations

import socket
import struct

from ..teleoperator import Teleoperator
from lerobot.net.transport import UDPReceiver
from .config_remote_receiver import RemoteReceiverConfig


class RemoteReceiver(Teleoperator):
    """Teleoperator that receives an action dict over UDP."""

    cfg: RemoteReceiverConfig
    name = "remote_receiver"
    config_class = RemoteReceiverConfig

    def __init__(self, cfg: RemoteReceiverConfig):
        super().__init__(cfg)
        self.receiver = UDPReceiver(cfg.port)
        self.receiver.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024)
        self._connected = False
        self._last_keys: list[str] | None = None  # remember keys for fallback
        self._last_action: dict[str, float] = {}
        self._stale = 0

    # --------------------------------------------------------------------- #
    #  Required abstract API – implemented as simple pass-throughs / stubs   #
    # --------------------------------------------------------------------- #

    # Connectivity --------------------------------------------------------- #
    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # Calibration / config ------------------------------------------------- #
    def calibrate(self) -> None:  # not needed for network wrapper
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        pass

    # Action / feedback ---------------------------------------------------- #
    @property
    def action_features(self) -> dict[str, type]:
        if self._last_keys:
            return {k: float for k in self._last_keys}
        return {}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}  # no haptic feedback path

    def get_action(self) -> dict[str, float]:
        buf = self.receiver.recv()

        # dropouts: reuse last action twice, then zero-out
        if buf is None or len(buf) != 20:
            # treat as dropout → reuse last action or zero
            self._stale += 1
            if self._stale <= 2:
                return self._last_action
            return {k: 0.0 for k in self._last_action}

        self._stale = 0

        # ---- unpack 20-byte binary payload ----
        pan, lift, elbow, wrist, grip = struct.unpack("<5f", buf)
        act = {
            "shoulder_pan.pos": pan,
            "shoulder_lift.pos": lift,
            "elbow_flex.pos": elbow,
            "wrist_flex.pos": wrist,
            "gripper.pos": grip,
        }
        self._last_action = act
        return act

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass  # no force-feedback channel
