from dataclasses import dataclass
from . import Teleoperator, TeleoperatorConfig  # existing base classes
from lerobot.net.transport import UDPSender


@dataclass
class RemoteSenderConfig(TeleoperatorConfig):
    host: str
    port: int = 5555


class RemoteSender(Teleoperator):
    cfg: RemoteSenderConfig

    def __init__(self, cfg: RemoteSenderConfig):
        super().__init__(cfg)
        self.sender = UDPSender(cfg.host, cfg.port)

    def connect(self):
        pass  # nothing special

    def get_action(self):
        action = super().get_action()  # game‑pad → dict[str,float]
        self.sender.send(action)  # stream it out
        return action  # local echo for GUI/CLI
