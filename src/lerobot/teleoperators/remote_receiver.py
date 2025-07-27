from dataclasses import dataclass
from . import Teleoperator, TeleoperatorConfig
from lerobot.net.transport import UDPReceiver


@dataclass
class RemoteReceiverConfig(TeleoperatorConfig):
    port: int = 5555
    default_action: float = 0.0  # safety: value when nothing received


class RemoteReceiver(Teleoperator):
    cfg: RemoteReceiverConfig

    def __init__(self, cfg: RemoteReceiverConfig, robot):
        super().__init__(cfg)
        self.receiver = UDPReceiver(cfg.port)
        self.robot = robot  # need action_features for fall‑back

    def connect(self):
        pass

    def get_action(self):
        msg = self.receiver.recv()
        if msg is None:
            # time‑out → fall back to safe zeros
            return {k: self.cfg.default_action for k in self.robot.action_features}
        return msg
