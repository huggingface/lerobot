import json
import socket
from typing import Dict, Tuple

__all__ = [
    "UDPTransportSender",
    "UDPTransportReceiver",
]


class UDPTransportSender:
    """Lightweight, fire-and-forget UDP sender used by the teleoperation *leader* machine.

    This class is intentionally minimal: it serialises action dictionaries to JSON and pushes them
    over UDP to the *server* (robot/follower machine) with best-effort delivery. If packets are lost
    they are simply dropped – the next action will arrive a few milliseconds later anyway.
    """

    def __init__(self, server: str):
        """Args
        ----
        server: str
            Remote endpoint in the form "<ip>:<port>" (e.g. "10.10.10.10:5555").
        """
        ip, port_str = server.split(":")
        self._addr: Tuple[str, int] = (ip, int(port_str))

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Non-blocking send – we never wait for ACKs.
        self._sock.setblocking(False)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def send(self, action: Dict[str, float]) -> None:  # noqa: D401 – present tense OK
        """Serialise *action* to JSON and transmit it over UDP."""
        payload = json.dumps(action).encode("utf-8")
        # Best-effort – if the socket is not ready we'll simply drop the frame.
        try:
            self._sock.sendto(payload, self._addr)
        except (BlockingIOError, OSError):
            # Dropped – nothing to do, next frame will go through.
            pass


class UDPTransportReceiver:
    """Blocking UDP receiver used by the teleoperation *follower* machine."""

    def __init__(self, port: int, buffer_size: int = 65535):
        """Listen on *port* for incoming action packets."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow immediate rebinding after a restart.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind on all interfaces – users can rely on firewall rules for protection.
        self._sock.bind(("", port))
        self._buffer_size = buffer_size

    # ------------------------------------------------------------------
    def recv(self) -> Dict[str, float]:  # noqa: D401
        """Block until the next action packet is received and return it."""
        payload, _addr = self._sock.recvfrom(self._buffer_size)
        return json.loads(payload.decode("utf-8")) 