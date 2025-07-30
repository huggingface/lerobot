#!/usr/bin/env python
from __future__ import annotations

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
        self._connected = False
        self._last_keys: list[str] | None = None  # remember keys for fallback

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
        msg = self.receiver.recv()
        if msg is None:
            # timeout → stop robot (all zeros) if we know the keys
            if self._last_keys is None:
                return {}
            return {k: self.cfg.default_action for k in self._last_keys}

        self._last_keys = list(msg)
        return msg

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass  # no force-feedback channel
