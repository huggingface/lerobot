#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team.

from __future__ import annotations

import logging
import socket
import struct
from dataclasses import dataclass
from typing import Optional

from .config_remote_sender import RemoteSenderConfig
from ..teleoperator import Teleoperator
from ..config import TeleoperatorConfig
from ..utils import make_teleoperator_from_config
from lerobot.net.transport import UDPSender

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #


class RemoteSender(Teleoperator):
    """Wraps a *local* teleoperator and streams its actions over UDP."""

    cfg: RemoteSenderConfig
    name = "remote_sender"
    config_class = RemoteSenderConfig

    # ───────────────────────────────────────────────────────────────────── #
    #  Construction & connectivity                                         #
    # ───────────────────────────────────────────────────────────────────── #
    def __init__(self, cfg: RemoteSenderConfig):
        super().__init__(cfg)
        self.config = cfg

        # UDP socket that points at the follower
        self.sender = UDPSender(cfg.host, cfg.port)
        self.sender.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024)

        # Best-effort QoS: works on Linux; silently ignored on macOS/BSD
        try:
            self.sender.sock.setsockopt(
                socket.IPPROTO_IP, socket.IP_TOS, 0x2E
            )  # AF41 DSCP
        except (AttributeError, OSError):
            # IP_TOS not available on this platform – continue without DSCP
            pass

        # Resolve the *config class* for the chosen local teleop
        choices: dict[str, type[TeleoperatorConfig]] = (
            TeleoperatorConfig.get_known_choices()  # {"gamepad": GamepadConfig, ...}
        )
        if cfg.local_type not in choices:
            raise ValueError(
                f"Unknown local_type '{cfg.local_type}'. " f"Allowed: {sorted(choices)}"
            )
        LocalCfgClass = choices[cfg.local_type]

        # Instantiate its config (fill 'port' only if that field exists)
        kwargs = {}
        if "port" in LocalCfgClass.__dataclass_fields__ and cfg.local_port is not None:
            kwargs["port"] = cfg.local_port
        local_cfg = LocalCfgClass(**kwargs)

        # Build the *actual* teleoperator instance
        self.inner: Teleoperator = make_teleoperator_from_config(local_cfg)

        self._connected = False

    # Connectivity --------------------------------------------------------- #
    def connect(self) -> None:
        self.inner.connect()
        self._connected = True
        logger.info(
            "RemoteSender connected: local %s → %s:%d",
            self.config.local_type,
            self.config.host,
            self.config.port,
        )

    def disconnect(self) -> None:
        self.inner.disconnect()
        self._connected = False
        logger.info("RemoteSender disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # Calibration / configuration ----------------------------------------- #
    def calibrate(self) -> None:
        self.inner.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self.inner.is_calibrated

    def configure(self) -> None:
        self.inner.configure()

    # Action / feedback ---------------------------------------------------- #
    @property
    def action_features(self) -> dict[str, type]:
        return self.inner.action_features

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.inner.feedback_features

    def get_action(self) -> dict[str, float]:
        action = self.inner.get_action()

        # ---- pack 5 floats into 20-byte binary payload ----
        # ordering matches SO-100 joints (pan, lift, elbow, wrist, gripper)
        buf = struct.pack(
            "<5f",
            action.get("shoulder_pan.pos", 0.0),
            action.get("shoulder_lift.pos", 0.0),
            action.get("elbow_flex.pos", 0.0),
            action.get("wrist_flex.pos", 0.0),
            action.get("gripper.pos", 0.0),
        )
        self.sender.send(buf)
        return action  # echo for on-screen display

    def send_feedback(self, feedback: dict[str, float]) -> None:
        self.inner.send_feedback(feedback)


# --------------------------------------------------------------------------- #
#  Export alias so Draccus CLI can see "remote_sender"                        #
# --------------------------------------------------------------------------- #

remote_sender = RemoteSenderConfig
