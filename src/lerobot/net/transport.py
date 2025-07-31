import json, socket, struct, threading, queue, time
from typing import Dict, Any, Optional

_DEFAULT_PORT = 5555
_MAX_SIZE = 65535


class UDPSender:
    def __init__(self, host: str, port: int = _DEFAULT_PORT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, msg: Dict[str, Any]):
        if isinstance(msg, (bytes, bytearray)):
            data = msg
        else:
            data = json.dumps(msg).encode()
        self.sock.sendto(data, self.addr)


class UDPReceiver:
    def __init__(self, port: int, bufsize: int = 2048, timeout: float = 0.005):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", port))
        self.sock.settimeout(timeout)
        self.bufsize = bufsize

    def recv(self):
        try:
            data, _ = self.sock.recvfrom(self.bufsize)
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        except socket.timeout:
            return None
