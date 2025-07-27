import json, socket, struct, threading, queue, time
from typing import Dict, Any, Optional

_DEFAULT_PORT = 5555
_MAX_SIZE = 65535


class UDPSender:
    def __init__(self, host: str, port: int = _DEFAULT_PORT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, msg: Dict[str, Any]):
        data = json.dumps(msg).encode()
        self.sock.sendto(data, self.addr)


class UDPReceiver:
    def __init__(self, port: int = _DEFAULT_PORT, timeout_s: float = 0.02):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", port))
        self.sock.settimeout(timeout_s)

    def recv(self) -> Optional[Dict[str, Any]]:
        try:
            buf, _ = self.sock.recvfrom(_MAX_SIZE)
            return json.loads(buf.decode())
        except socket.timeout:
            return None
