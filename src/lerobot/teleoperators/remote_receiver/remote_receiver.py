# lerobot/teleoperators/remote_receiver.py
from dataclasses import dataclass, field
from .. import Teleoperator
from lerobot.net.transport import UDPReceiver
from .config_remote_receiver import RemoteReceiverConfig


class RemoteReceiver(Teleoperator):
    cfg: RemoteReceiverConfig

    def __init__(self, cfg: RemoteReceiverConfig):
        super().__init__(cfg)
        self.receiver = UDPReceiver(cfg.port)
        self._last_keys: list[str] | None = None

    # optional, so teleoperate.py can size its action-display table
    @property
    def action_features(self) -> dict[str, type]:
        if self._last_keys:
            return {k: float for k in self._last_keys}
        return {}  # unknown until first packet arrives

    def connect(self):
        pass  # nothing to do

    def get_action(self):
        msg = self.receiver.recv()
        if msg is None:
            # timeout → fall back to zeros using previous key set
            if self._last_keys is None:
                return {}  # still waiting for first packet
            return {k: self.cfg.default_action for k in self._last_keys}

        # normal path
        self._last_keys = list(msg)  # remember key order
        return msg
